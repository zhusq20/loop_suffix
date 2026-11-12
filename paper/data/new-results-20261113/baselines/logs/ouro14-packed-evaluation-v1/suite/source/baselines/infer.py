"""Run AR, DFlash or LoopAhead on explicit JSONL prompts and local checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3Config

from .dflash.packed_target import PackedDFlashTarget
from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids, tokenize_prompt
from llm_memory_editability.loop_optimized_decode import SharedKVPool
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop

from .common import (append, check_draft_config, digest, environment, load_target,
                     save, target_arguments, token_ids)
from .dflash.decode import generate
from .dflash.projected_draft import inference_draft_class
from .loopahead.decode import LoopAheadConfig, decode_steps
from .loopahead.engine import LoopAheadExecutor
from .loopahead.lad import LADConfig, LookaheadDrafter
from .vendor.dflash_inference import DFlashDraftModel


def tokenizer_and_stops(model, path, prompt_format, explicit):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if explicit is not None:
        stops = set(explicit)
    else:
        gen_path = Path(path) / 'generation_config.json'
        gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
        stops, _ = resolve_stop_ids(MODELS[model], tokenizer, config.eos_token_id,
                                   gen.get('eos_token_id'), config.vocab_size, prompt_format)
    if any(type(t) is not int or not 0 <= t < config.vocab_size for t in stops):
        raise ValueError('Invalid stop IDs')
    return tokenizer, stops


def load_dflash(path, target):
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    check_draft_config(config, target)
    cls = inference_draft_class() if config.dflash_config.get('projection_adapter') else DFlashDraftModel
    return cls.from_pretrained(path, config=config, torch_dtype=target.dtype,
                               attn_implementation='sdpa', local_files_only=True).to(target.device).eval()


def load_loopahead(path, target):
    path = Path(path)
    if path.is_dir():
        path = path / 'lad.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    config = LADConfig(**saved['config'])
    if config.target_hidden != target.hidden_size or config.context != 64 or config.train_future != 7:
        raise ValueError('LAD width/context/future does not match the target execution interface')
    lad = LookaheadDrafter(config)
    lad.load_state_dict(saved['model'])
    return lad.to(device=target.device, dtype=target.dtype).eval(), saved['frequent_ids']


def loopahead_engine(target, lad, frequent, temperature):
    pool = SharedKVPool(1, pad_decode_rows=False)
    if target.family == 'ouro':
        backbone = OuroLoop(target.model, target.depth, target.max_seq, 32, kv_factory=pool)
        config = LoopAheadConfig(temperature=temperature)
    else:
        backbone = HuginnLoop(target.model, target.depth, target.max_seq, 32, 0,
                              target.max_seq, extra_depths=(8,), kv_factory=pool)
        backbone.init = target.noise
        config = LoopAheadConfig(total=32, first=4, second=8, future=2, guard=1,
                                 stride=2, family='huginn', temperature=temperature)
    backbone.position_limit = target.max_seq
    return backbone, LoopAheadExecutor(backbone, lad, frequent, temperature=temperature), config


@torch.inference_mode()
def generate_loopahead(backbone, engine, config, prompt, max_new_tokens, stops, seed=0,
                       admission=None):
    if len(prompt) + max_new_tokens > backbone.position_limit:
        raise ValueError('Prompt plus generation exceeds context capacity')
    engine.request(seed, stops)
    steps = decode_steps(backbone, prompt, config, max_new_tokens, stops,
                         generator=engine.generator, admission=admission)
    result = None
    while True:
        try:
            call = steps.send(result)
        except StopIteration as done:
            tokens, stats = done.value
            stats.update(decode_tokens=len(tokens) - 1,
                         decode_tok_s=(len(tokens) - 1) / stats['decode_s'] if stats['decode_s'] else None,
                         output_tokens=len(tokens), stop_reason='eos' if tokens[-1] in stops else 'length')
            return tokens, stats
        result = engine.execute(call.name, [(backbone, call)])[0]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    target_arguments(p)
    p.add_argument('--method', choices=('ar', 'dflash', 'loopahead'), required=True)
    p.add_argument('--backend', choices=('packed', 'native'), default='packed')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--inputs', type=Path, required=True, help='JSONL: source_id plus prompt or prompt_tokens')
    p.add_argument('--prompt-format', choices=('chat', 'raw'), default='raw')
    p.add_argument('--stop-ids', type=int, nargs='*', help='Override EOS; no values means no stop IDs')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, choices=(0., 1.), default=0.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--admission', type=Path, help='Optional matching LoopAhead calibration JSON')
    p.add_argument('--out', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.method != 'ar' and not args.checkpoint:
        raise ValueError('DFlash and LoopAhead require --checkpoint')
    if args.method == 'loopahead' and args.backend != 'packed':
        raise ValueError('LoopAhead uses the packed executor')
    if args.admission and args.method != 'loopahead':
        raise ValueError('--admission is a LoopAhead option')
    if min(args.max_new_tokens, args.max_seq) < 1:
        raise ValueError('Token capacities must be positive')
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if not rows or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty, unique input source IDs')
    tokenizer = None
    if args.stop_ids is None or any('prompt_tokens' not in r for r in rows):
        tokenizer, stops = tokenizer_and_stops(args.model, args.target_path, args.prompt_format, args.stop_ids)
    else:
        stops = set(args.stop_ids)
    torch.set_num_threads(1)
    target = load_target(args)
    packed = None
    try:
        vocab = target.model.config.vocab_size
        if stops:
            token_ids(sorted(stops), vocab_size=vocab)
        prompts = [token_ids(r['prompt_tokens'] if 'prompt_tokens' in r else
                             tokenize_prompt(MODELS[args.model], tokenizer, r['prompt'], args.prompt_format),
                             vocab_size=vocab) for r in rows]
        if any(len(p) + args.max_new_tokens > args.max_seq for p in prompts):
            raise ValueError('An input plus output budget exceeds --max-seq; no silent truncation')
        draft = load_dflash(args.checkpoint, target) if args.method == 'dflash' else None
        policy = None
        if args.method == 'loopahead':
            lad, frequent = load_loopahead(args.checkpoint, target)
            backbone, engine, config = loopahead_engine(target, lad, frequent, args.temperature)
            if args.admission:
                from .loopahead.admission import AdmissionPolicy
                policy = AdmissionPolicy.load(args.admission)
                ckpt = args.checkpoint / 'lad.pt' if args.checkpoint.is_dir() else args.checkpoint
                if policy.record['model'] != args.model or policy.record['checkpoint_sha256'] != digest(ckpt):
                    raise ValueError('Admission calibration belongs to another target/checkpoint')
        elif args.backend == 'packed':
            packed = PackedDFlashTarget(target)
        args.out.mkdir(parents=True, exist_ok=False)
        save(args.out / 'metadata.json', dict(
            method=args.method, backend=args.backend, model=args.model,
            target_path=str(args.target_path.resolve()),
            checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
            inputs_sha256=digest(args.inputs), stop_ids=sorted(stops), dtype=args.dtype,
            max_seq=args.max_seq, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            prompt_format=args.prompt_format, seed=args.seed, environment=environment(),
            scope='explicit input JSONL; no task scoring or formal benchmark aggregation'))
        decode_tokens, decode_s = 0, 0.
        for index, (row, prompt) in enumerate(zip(rows, prompts)):
            if args.method == 'loopahead':
                tokens, stats = generate_loopahead(backbone, engine, config, prompt,
                    args.max_new_tokens, stops, args.seed + index, policy)
            else:
                result = generate(packed or target, torch.tensor([prompt], device=target.device),
                    args.max_new_tokens, stops, draft=draft, temperature=args.temperature, seed=args.seed + index)
                tokens, stats = result.tokens, result.stats
            record = dict(source_id=row['source_id'], prompt_tokens=prompt, output_tokens=tokens, stats=stats)
            if tokenizer:
                record['text'] = tokenizer.decode(tokens, skip_special_tokens=False)
            append(args.out / 'outputs.jsonl', record)
            decode_tokens += stats['decode_tokens']
            decode_s += stats['decode_s']
        report = dict(requests=len(rows), decode_tokens=decode_tokens, decode_s=decode_s,
                      decode_tok_s=decode_tokens / decode_s if decode_s else None,
                      timing='decode only, excludes prefill and first token; no separate warmup')
        save(args.out / 'complete.json', report)
        print(json.dumps(report), flush=True)
    finally:
        if packed:
            packed.close()
        target.close()


if __name__ == '__main__':
    main()
