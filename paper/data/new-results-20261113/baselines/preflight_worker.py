"""Run AR/LoopSpec calibration using the immutable, published benchmark engine.

This is an environment check, not a DFlash or LoopAhead result. No W&B upload.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--model', choices=['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b'], required=True)
    ap.add_argument('--method', choices=['ar', 'loopspec-dual'], required=True)
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--cpus', required=True)
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.count <= 300 or not 2 <= args.max_new_tokens <= 512:
        ap.error('count must be 1..300 and max-new-tokens 2..512')
    args.out.mkdir(parents=True, exist_ok=False)
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                        '--format=csv,noheader'], text=True).splitlines()
    uuid = dict((int(i.strip()), u.strip()) for i, u in (x.split(',') for x in gpu_lines))[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupants = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                            '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() == uuid for line in occupants.splitlines()):
            raise RuntimeError(f'GPU occupied: {uuid}; refusing to co-time or terminate anything')
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        cpus = set()
        for part in args.cpus.split(','):
            if '-' in part:
                lo, hi = map(int, part.split('-'))
                cpus.update(range(lo, hi + 1))
            else:
                cpus.add(int(part))
        os.sched_setaffinity(0, cpus)
        run(args, uuid)


def run(args, uuid):
    suite = args.repo.resolve() / 'results/loopfill-latency512-20261107'
    p = read(suite / 'protocol.json')
    source = suite / 'source'
    for rel, expected in p['source_sha256'].items():
        if digest(source / rel) != expected:
            raise ValueError('Frozen source changed: ' + rel)
    input_file = suite / 'inputs/gsm8k.jsonl'
    if digest(input_file) != p['inputs']['gsm8k']['sha256']:
        raise ValueError('Frozen input changed')
    sys.path[:0] = [str(source / 'src'), str(source / 'scripts'), '/tmp/loopfill-assets/ArcticInference']
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from loop_batch_bench import arm_config, enforce_precision
    from loop_latency_worker import drive, summarize
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids, tokenize_prompt
    from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop, tree_decode_steps
    from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig, loopspec_decode_steps
    from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    spec = MODELS[args.model]
    path = p['models'][args.model]['path']
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    gp = Path(path) / 'generation_config.json'
    stop, stop_audit = resolve_stop_ids(spec, tok, getattr(cfg, 'eos_token_id', None),
        read(gp).get('eos_token_id') if gp.exists() else None, cfg.vocab_size, 'chat')
    all_rows = [json.loads(line) for line in input_file.read_text().split('\n') if line]
    # Match the full 300-question cell's cache capacity and Huginn noise-table shape.
    all_prompts = [tokenize_prompt(spec, tok, r['prompt'], 'chat') for r in all_rows]
    required = max(map(len, all_prompts)) + 512 + spec.depth_total + 4
    max_seq = min(spec.context_limit, ((required + 255) // 256) * 256)
    assert all(len(x) + 512 + spec.depth_total + 4 <= max_seq for x in all_prompts)
    texts, prompts = all_rows[:args.count], all_prompts[:args.count]
    config = arm_config(args.method, spec.depth_total, *spec.loopfill_readout_depths, 4,
        draft_node_budget=16, model_top_k=2, child_suffix_caps=(4, 4),
        suffix_c_initial=1., suffix_eta=.1, loopspec_depths=spec.loopspec_depths,
        wavefront_depth=spec.wavefront_depth)
    nodes = 1 if args.method == 'ar' else config.physical_nodes
    metadata = dict(stage='environment_calibration', model=args.model, method=args.method,
        count=args.count, max_new_tokens=args.max_new_tokens, max_seq=max_seq,
        input_sha256=digest(input_file), source_sha256=p['source_sha256'],
        model_revision=spec.revision, gpu_uuid=uuid, cpu_affinity=sorted(os.sched_getaffinity(0)),
        stop_audit=stop_audit, torch=torch.__version__, cuda=torch.version.cuda,
        transformers=__import__('transformers').__version__, seed=0,
        timing='same frozen drive(): synchronized decode only, excludes prefill and first token',
        full_benchmark=args.count == 300 and args.max_new_tokens == 512)
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('Loading', args.model, args.method, flush=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda:0').eval()
    model = enforce_precision(model, 'bfloat16')
    if spec.backend == 'ouro':
        model.model.total_ut_steps = spec.depth_total
        model.early_exit_step = model.early_exit_threshold = None
    pool = SharedKVPool(1, pad_decode_rows=False)
    if spec.backend == 'ouro':
        b = OuroLoop(model, spec.depth_total, max_seq, nodes, kv_factory=pool)
    else:
        b = HuginnLoop(model, spec.depth_total, max_seq, nodes, 0, max_seq,
            extra_depths=(spec.loopfill_readout_depths[1],), kv_factory=pool)
    b.position_limit = max_seq
    engine = PackedLoopExecutor([b], cuda_graph=False)

    def generate(prompt, limit):
        decoder = loopspec_decode_steps if isinstance(config, LoopSpecConfig) else tree_decode_steps
        with torch.inference_mode():
            return drive(b, engine, decoder(b, prompt, config, limit, stop, max_nodes=nodes))

    for prompt in all_prompts[:2]:
        generate(prompt, 32)
    engine.graphs.freeze()
    completed = []
    with (args.out / 'requests.jsonl').open('x') as f:
        for i, (row, prompt) in enumerate(zip(texts, prompts)):
            tokens, stats = generate(prompt, args.max_new_tokens)
            assert stats['tokens'] == len(tokens) - 1
            if args.method == 'ar':
                assert stats['calls'] == spec.depth_total * stats['tokens']
            record = dict(source_id=row['source_id'], prompt_sha256=row['prompt_sha256'],
                tokens=tokens, stats=stats, stop_reason='eos' if tokens[-1] in stop else 'length',
                completed_unix=time.time())
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())
            completed.append(record)
            summary = summarize(completed, spec.depth_total)
            (args.out / 'progress.json').write_text(json.dumps(summary, indent=2))
            print(f'{i+1}/{args.count}: {len(tokens)} tokens, {stats["decode_s"]:.3f}s', flush=True)
    (args.out / 'summary.json').write_text(json.dumps(dict(summary, **engine.metrics()), indent=2))
    (args.out / 'complete.json').write_text(json.dumps(dict(requests=args.count,
        requests_sha256=digest(args.out / 'requests.jsonl'), finished_unix=time.time()), indent=2))


if __name__ == '__main__':
    main()
