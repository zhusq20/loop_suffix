"""One isolated agentic decoding cell, with persistent causal completion history."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''):
            h.update(data)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


class RetrievalProbe:
    """Optional diagnostic only; the production timer must not be used as a speed result."""
    def __init__(self, cache):
        self.cache = cache
        self.queries = []

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def speculate(self, rid, context, **kwargs):
        draft = self.cache.speculate(rid, context, **kwargs)
        self.queries.append(dict(kind='root', confirmed=len(self.cache.responses[rid]),
            token_ids=list(draft.token_ids), parents=list(draft.parents), match_len=int(draft.match_len)))
        return draft

    def observe(self, event, **data):
        if event == 'child_suffix_query' and data['cap'] > 0:
            assert self.queries and self.queries[-1]['kind'] == 'root'
            self.queries[-1]['kind'] = 'child'

    def summarize(self, tokens):
        totals = {kind: Counter() for kind in ('root', 'child')}
        for query in self.queries:
            c = totals[query['kind']]
            c.update(queries=1, nonempty=int(bool(query['token_ids'])),
                     returned_nodes=len(query['token_ids']), matched_context_tokens=query['match_len'])
            if query['kind'] == 'root':
                future = tokens[query['confirmed']:]
                paths, depth = [], 0
                for token, parent in zip(query['token_ids'], query['parents']):
                    path = ([] if parent < 0 else paths[parent]) + [token]
                    paths.append(path)
                    if path == future[:len(path)]:
                        depth = max(depth, len(path))
                c.update(first_token_covered=int(depth > 0), matching_path_tokens=depth)
        self.queries.clear()
        return totals


def run(args):
    suite = args.suite.resolve()
    protocol = json.loads((suite/'protocol.json').read_text())
    source = suite/'source'
    for rel, expected in protocol['source_sha256'].items():
        if digest(source/rel) != expected:
            raise ValueError('Frozen source changed: '+rel)
    sys.path[:0] = [str(source), str(source/'src'), str(source/'scripts'), str(source/'suffix')]
    model_record=protocol['models'][args.model]
    input_path = suite/'inputs'/f'{args.model}.jsonl'
    if digest(input_path) != model_record['input_sha256']:
        raise ValueError('Input changed')
    for rel, expected in protocol['input_files_sha256'].items():
        if digest(suite/'inputs'/rel) != expected:
            raise ValueError('Frozen input/config changed: '+rel)
    for name, filename in [('loopahead', model_record['loopahead_checkpoint']),
                           ('dflash', str(Path(model_record['dflash_checkpoint'])/'model.safetensors')),
                           ('dflash_config', str(Path(model_record['dflash_checkpoint'])/'config.json'))]:
        if digest(filename) != model_record['checkpoint_sha256'][name]:
            raise ValueError('Checkpoint changed: '+name)
    rows = [json.loads(line) for line in input_path.open()]
    for row in rows:
        ids = row['prompt_token_ids']
        if (not ids or len(ids)+512+32 > protocol['max_seq'] or
                digest_tokens(ids) != row['prompt_sha256']):
            raise ValueError('Invalid tokenized prompt: '+row['source_id'])
    if args.smoke:
        # Exercise shortest and longest supported contexts before starting formal jobs.
        rows = [min(rows, key=lambda r:r['prompt_tokens']), max(rows, key=lambda r:r['prompt_tokens'])]
    if args.diagnostic:
        rows = rows[:12]
    limit = 24 if args.smoke else 512
    path = args.out.resolve()
    path.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    from loop_batch_bench import arm_config
    from loop_latency_worker import drive, load_journal, summarize
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids
    from llm_memory_editability.loop_sampling import TargetSampler, request_seed
    from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop, tree_decode_steps
    from llm_memory_editability.loop_budget_decode import budget_tree_decode_steps
    from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig, loopspec_decode_steps
    from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    from baselines.dflash.target import LoopTarget
    spec = MODELS[args.model]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    model_path = model_record['model_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    native = LoopTarget.load(model_path, spec.backend, protocol['max_seq'])
    model = native.model
    generation_path=Path(model_path)/'generation_config.json'
    generation=json.loads(generation_path.read_text()) if generation_path.exists() else {}
    stops, stop_audit = resolve_stop_ids(spec, tokenizer, model.config.eos_token_id,
        generation.get('eos_token_id'), model.config.vocab_size, 'chat')
    # Remove unused HF feature hooks; the packed adapter has its own observations.
    native.close()
    kwargs = dict(draft_node_budget=args.n, model_top_k=2, child_suffix_caps=(args.q,args.q),
                  suffix_c_initial=.7 if args.arm == 'budget-online' else 1., suffix_eta=.1)
    config = arm_config(args.arm if args.arm in ('ar','wavefront','loopspec-dual','budget-suffix','budget-online') else 'ar',
                        spec.depth_total,*spec.loopfill_readout_depths,args.q,
                        loopspec_depths=spec.loopspec_depths,wavefront_depth=spec.wavefront_depth,**kwargs)
    draft, policy = None, None
    if args.arm == 'dflash':
        from baselines.dflash.packed_target import PackedDFlashTarget
        from baselines.vendor.dflash_inference import DFlashDraftModel
        from baselines.dflash.decode import generate as dflash_generate
        from baselines.dflash.projected_draft import inference_draft_class
        target = PackedDFlashTarget(native)
        checkpoint = model_record['dflash_checkpoint']
        draft_cls=inference_draft_class() if spec.backend=='huginn' else DFlashDraftModel
        draft, info = draft_cls.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa', output_loading_info=True)
        if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError(f'Drafter load mismatch: {info}')
        draft = draft.to('cuda').eval()
        assert draft.block_size == 16 and draft.target_layer_ids == native.expanded_ids
        backbone, engine = target.backbone, target.engine
    else:
        nodes = (32 if args.arm == 'loopahead' else args.n+1 if args.arm.startswith('budget') else
                 config.physical_nodes if isinstance(config,LoopSpecConfig) else 1152 if args.arm=='wavefront' else 1)
        pool=SharedKVPool(1,pad_decode_rows=False)
        if spec.backend=='ouro':
            backbone=OuroLoop(model,spec.depth_total,protocol['max_seq'],nodes,kv_factory=pool)
        else:
            backbone=HuginnLoop(model,spec.depth_total,protocol['max_seq'],nodes,0,protocol['max_seq'],
                               extra_depths=(8,),kv_factory=pool)
            backbone.init=native.noise
        backbone.position_limit = protocol['max_seq']
        if args.arm == 'loopahead':
            from baselines.loopahead.lad import LADConfig, LookaheadDrafter
            from baselines.loopahead.engine import LoopAheadExecutor
            from baselines.loopahead.decode import LoopAheadConfig, decode_steps
            from baselines.loopahead.admission import AdmissionPolicy
            ckpt = torch.load(model_record['loopahead_checkpoint'], map_location='cpu', weights_only=True)
            draft = LookaheadDrafter(LADConfig(**ckpt['config']))
            draft.load_state_dict(ckpt['model'])
            draft = draft.to(device='cuda', dtype=torch.bfloat16).eval()
            engine = LoopAheadExecutor(backbone, draft, ckpt['frequent_ids'], temperature=args.temperature)
            config = (LoopAheadConfig(temperature=args.temperature) if spec.backend=='ouro' else
                      LoopAheadConfig(total=32,first=4,second=8,future=2,guard=1,stride=2,
                                      family='huginn',temperature=args.temperature))
            if args.temperature and spec.backend=='ouro':
                policy = AdmissionPolicy.load(suite/'inputs'/'loopahead-admission.json')
                assert policy.record['checkpoint_sha256'] == model_record['checkpoint_sha256']['loopahead']
        else:
            engine = PackedLoopExecutor([backbone], cuda_graph=False)

    suffix_enabled = args.arm in ('budget-suffix','budget-online')
    def new_cache():
        if not suffix_enabled:
            return None
        cache = CompletionSuffixCache.arctic()
        return RetrievalProbe(cache) if args.diagnostic else cache

    def generate(row, cache, cap, warmup=False):
        prompt = row['prompt_token_ids']
        seed = request_seed(0, args.model, row['dataset']+('/warmup' if warmup else ''), row['source_id'])
        sampler = TargetSampler(args.temperature, seed, protocol['max_seq']+2, 'cuda') if args.temperature else None
        backbone.target_sampler = sampler
        with torch.inference_mode():
            if args.arm == 'dflash':
                result = dflash_generate(target, torch.tensor([prompt], device='cuda'), cap, stops,
                    draft=draft, temperature=args.temperature, seed=seed, target_sampler=sampler)
                stats = dict(result.stats, tokens=result.stats['decode_tokens'],
                    calls=result.stats['loop_steps'],
                    accepted_lengths=result.stats['emitted_per_verification'])
                return result.tokens, stats
            if args.arm == 'loopahead':
                engine.request(seed, stops)
                steps = decode_steps(backbone, prompt, config, cap, stops, generator=engine.generator,
                    admission=policy, quality=policy.quality if policy else None)
            elif suffix_enabled:
                steps = budget_tree_decode_steps(backbone, prompt, config, cap, stops, cache=cache,
                    req_id=row['question_id'], max_nodes=nodes,
                    observer=cache.observe if args.diagnostic else None)
            elif isinstance(config,LoopSpecConfig):
                steps=loopspec_decode_steps(backbone,prompt,config,cap,stops,max_nodes=nodes)
            else:
                steps = tree_decode_steps(backbone, prompt, config, cap, stops, max_nodes=nodes)
            return drive(backbone, engine, steps)

    metadata = dict(model=args.model,arm=args.arm, temperature=args.temperature, q=args.q, n=args.n, cache_mode=args.cache_mode,
        config=asdict(config), batch=1, max_new_tokens=limit, max_seq=protocol['max_seq'],
        input_sha256=model_record['input_sha256'], source_sha256=protocol['source_sha256'],
        checkpoint_sha256=model_record['checkpoint_sha256'], stop_audit=stop_audit,
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), gpu=torch.cuda.get_device_name(0),
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch=torch.__version__,
        transformers=transformers.__version__, dtype='bfloat16', cuda_graph=False, compile=False,
        smoke=args.smoke, diagnostic=args.diagnostic, count=len(rows), sampling_seed=0,
        top_p=1., top_k=0, history='completed outputs only; isolated warmup; cold start per cell',
        dflash_trained_max_positions=draft.config.max_position_embeddings if args.arm=='dflash' else None,
        context_note='DFlash training configuration has 3072 positions; requests beyond that limit '
                     'are context extrapolation. No weight change or retuning. '
                     'Smoke checks exercise maximum selected prompt length.',
        timing='synchronized decode after prefill; includes drafting, sampling and suffix publication; '
               'prefill, loading, warmup, journal I/O excluded; diagnostic timings not reportable')
    if (path/'metadata.json').exists():
        previous = json.loads((path/'metadata.json').read_text())
        for key in ('model','arm','temperature','q','n','cache_mode','input_sha256','source_sha256',
                    'checkpoint_sha256','count','smoke','diagnostic','gpu_uuid'):
            if previous[key] != metadata[key]:
                raise ValueError('Resume identity mismatch: '+key)
    save(path/'metadata.json', metadata)
    for row in rows[:2]:
        generate(row, new_cache(), min(32,limit), warmup=True)
    engine.graphs.freeze()
    torch.cuda.reset_peak_memory_stats()
    journal = path/'requests.jsonl'
    prior = load_journal(journal, [r['source_id'] for r in rows])
    cache = new_cache()
    if cache is not None and args.cache_mode == 'stream':
        for row, record in zip(rows, prior):
            rid = row['question_id']
            cache.start_request(rid, np.asarray(row['prompt_token_ids'], dtype=np.int32))
            cache.add_active_response(rid, np.asarray(record['tokens'], dtype=np.int32))
            cache.stop_request(rid)
    with journal.open('a') as stream:
        for row in rows[len(prior):]:
            if args.cache_mode == 'request':
                cache = new_cache()
            tokens, stats = generate(row, cache, limit)
            record = dict(source_id=row['source_id'], question_id=row['question_id'],
                phase=row['phase'], instance_id=row['instance_id'], prompt_tokens=row['prompt_tokens'],
                prompt_sha256=row['prompt_sha256'], tokens=tokens, text=tokenizer.decode(tokens),
                stats=stats, stop_reason='eos' if tokens[-1] in stops else 'length',
                completed_unix=time.time())
            if args.diagnostic and cache is not None:
                record['retrieval'] = cache.summarize(tokens)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            prior.append(record)
            save(path/'progress.json', dict(completed=len(prior), total=len(rows), updated_unix=time.time()))
            print(f"{args.arm} T={args.temperature:g} {len(prior)}/{len(rows)} "
                  f"tokens={len(tokens)} decode={stats['decode_s']:.2f}s", flush=True)
    summary = summarize(prior, spec.depth_total)
    # Rejection-sampling baselines do not share candidate-coverage semantics.
    if args.arm in ('loopahead','dflash'):
        for key in ('draft_hits','mean_acceptance_rate','pooled_acceptance_rate'):
            summary.pop(key, None)
    summary.update(status='complete', requests_sha256=digest(journal), reportable=not(args.smoke or args.diagnostic),
                   peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_memory_reserved_bytes=torch.cuda.max_memory_reserved())
    save(path/'summary.json', summary)


def digest_tokens(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model',choices=['huginn-3.5b','ouro-2.6b'],required=True)
    parser.add_argument('--arm', choices=['ar','wavefront','loopspec-dual','budget-suffix','budget-online','loopahead','dflash'], required=True)
    parser.add_argument('--temperature', type=float, choices=[0.,1.], required=True)
    parser.add_argument('--q', type=int, default=4)
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--cache-mode', choices=['stream','request'], default='stream')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--diagnostic', action='store_true')
    run(parser.parse_args())
