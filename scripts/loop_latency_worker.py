"""One B=1 cell. All persistence and W&B operations occur outside decode timing."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drive(backbone, engine, steps, clock=time.perf_counter):
    """Time after synchronized prefill through completed decode and suffix publication."""
    value = None
    start = None
    prefill_s = 0.0
    while True:
        try:
            call = steps.send(value)
        except StopIteration as done:
            backbone.synchronize()
            end = clock()
            if start is None:
                raise ValueError('Missing prefill boundary')
            tokens, stats = done.value
            stats = dict(stats, decode_s=end - start, prefill_s=prefill_s,
                         timer_policy='synchronized_after_prefill_through_suffix_publication')
            return tokens, stats
        if call.name == 'prefill':
            if start is not None:
                raise ValueError('Multiple prefills in single request')
            backbone.synchronize()
            before = clock()
            value = engine.execute(call.name, [(backbone, call)])[0]
            backbone.synchronize()
            start = clock()
            prefill_s = start - before
        else:
            value = engine.execute(call.name, [(backbone, call)])[0]


def load_journal(path, ids):
    """Recover a durable prefix; only an incomplete final write may be discarded."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows, offset = [], 0
    for i, line in enumerate(lines):
        if not line.endswith(b'\n'):
            if i != len(lines) - 1:
                raise ValueError('Malformed journal')
            with path.open('r+b') as f:
                f.truncate(offset)
                f.flush()
                os.fsync(f.fileno())
            break
        row = json.loads(line)
        if len(rows) >= len(ids) or row['source_id'] != ids[len(rows)]:
            raise ValueError('Journal does not match frozen request order')
        rows.append(row)
        offset += len(line)
    return rows


def summarize(rows, depth):
    seconds = sum(r['stats']['decode_s'] for r in rows)
    tokens = sum(r['stats']['tokens'] for r in rows)
    calls = sum(r['stats']['calls'] for r in rows)
    hits = [r['stats'].get('draft_hits', r['stats'].get('covered_tokens',
        r['stats'].get('from_suffix', 0) + r['stats'].get('from_fill', 0))) for r in rows]
    rates = [h/r['stats']['tokens'] for h, r in zip(hits, rows) if r['stats']['tokens']]
    return dict(completed_requests=len(rows), decode_s=seconds, decode_tokens=tokens,
                mean_decode_s=seconds/len(rows) if rows else None, draft_hits=sum(hits),
                mean_acceptance_rate=sum(rates)/len(rates) if rates else None,
                pooled_acceptance_rate=sum(hits)/tokens if tokens else None,
                output_tokens=sum(len(r['tokens']) for r in rows), loop_steps=calls,
                decode_token_s=tokens / seconds if seconds else None,
                gamma=depth * tokens / calls if calls else None,
                cap_stopped=sum(r['stop_reason'] == 'length' for r in rows),
                prefill_s=sum(r['stats']['prefill_s'] for r in rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    suite = args.suite.resolve()
    protocol = read(suite / 'protocol.json')
    job = read(suite / 'jobs' / (args.job + '.json'))
    spec_name, arm, dataset = job['model'], job['arm'], job['dataset']
    target = suite / ('validation' if args.smoke else 'runs') / args.job
    target.mkdir(parents=True, exist_ok=True)
    for rel, expected in protocol['source_sha256'].items():
        if digest(Path(protocol['source']) / rel) != expected:
            raise ValueError(f'Frozen source changed: {rel}')
    input_path = suite / 'inputs' / (dataset + '.jsonl')
    if digest(input_path) != protocol['inputs'][dataset]['sha256']:
        raise ValueError('Input checksum changed')
    texts = [json.loads(line) for line in input_path.read_text().splitlines()]
    if args.smoke:
        texts = texts[:2]
    max_new_tokens = 16 if args.smoke else 512
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from loop_batch_bench import arm_config, enforce_precision
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids, tokenize_prompt
    from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop, tree_decode_steps
    from llm_memory_editability.loop_budget_decode import BudgetConfig, budget_tree_decode_steps
    from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig, loopspec_decode_steps
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
    import numpy as np
    spec = MODELS[spec_name]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    model_path = protocol['models'][spec_name]['path']
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    mc = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    gp = Path(model_path) / 'generation_config.json'
    generation_eos = read(gp).get('eos_token_id') if gp.exists() else None
    stop, stop_audit = resolve_stop_ids(spec, tok, getattr(mc, 'eos_token_id', None),
                                      generation_eos, mc.vocab_size, 'chat')
    prompts = [tokenize_prompt(spec, tok, row['prompt'], 'chat') for row in texts]
    required = max(map(len, prompts)) + 512 + spec.depth_total + 4
    max_seq = min(spec.context_limit, ((required + 255) // 256) * 256)
    invalid = [dict(source_id=r['source_id'], prompt_tokens=len(ids), required=len(ids) + 512)
               for r, ids in zip(texts, prompts) if len(ids) + 512 + spec.depth_total + 4 > max_seq]
    if invalid:
        save(target / 'context-errors.json', invalid)
        raise ValueError(f'{len(invalid)} inputs exceed context; no truncation or deletion')
    config = arm_config(arm, spec.depth_total, *spec.loopfill_readout_depths, 4,
        draft_node_budget=16, model_top_k=2, child_suffix_caps=(4, 4),
        suffix_c_initial=.7 if arm == 'budget-online' else 1., suffix_eta=.1,
        loopspec_depths=spec.loopspec_depths, wavefront_depth=spec.wavefront_depth)
    nodes = 1 if arm == 'ar' else 17 if isinstance(config, BudgetConfig) else (
        config.physical_nodes if isinstance(config, LoopSpecConfig) else 1152)
    sampling = {k: protocol.get(k, default) for k, default in [('temperature', 0.),
        ('top_p', 1.), ('top_k', 0), ('sampling_seed', 0)]}
    if sampling['top_p'] != 1. or sampling['top_k'] != 0:
        raise ValueError('Only untruncated target sampling supported')
    metadata = dict(model=spec.to_dict(), arm=arm, config=asdict(config), sampling=sampling,
        dataset=dataset, batch_size=1, max_new_tokens=max_new_tokens,
        cuda_graph=False, pad_decode_rows=False, torch_compile=False,
        dtype='bfloat16', torch=torch.__version__, transformers=__import__('transformers').__version__,
        cuda=torch.version.cuda, gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        gpu=torch.cuda.get_device_name(0), cpu_affinity=sorted(os.sched_getaffinity(0)),
        max_seq=max_seq, physical_nodes=nodes, stop_audit=stop_audit,
        inputs_sha256=digest(input_path), source_sha256=protocol['source_sha256'],
        timing='B1 synchronized wall-clock; prefill excluded; includes suffix setup/query/update/publication; '
               'journal/W&B/loading/warmup excluded',
        history='cold per dataset/method; only completed prior selected outputs; warmup isolated',
        prompt_format='released chat template; EOS/chat-end stops; no task-specific text delimiters')
    prior_meta = target / 'metadata.json'
    if prior_meta.exists():
        prior = read(prior_meta)
        for key in ['model', 'arm', 'config', 'dataset', 'max_new_tokens', 'inputs_sha256', 'source_sha256']:
            if prior[key] != metadata[key]:
                raise ValueError('Resume identity mismatch: ' + key)
    if prior_meta.exists() and read(prior_meta).get('sampling', sampling) != sampling:
        raise ValueError('Resume sampling identity mismatch')
    save(target / 'metadata.json', metadata)
    print(f'Loading {spec_name} {arm} {dataset}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda:0').eval()
    model = enforce_precision(model, 'bfloat16')
    if spec.backend == 'ouro':
        model.model.total_ut_steps = spec.depth_total
        model.early_exit_step = model.early_exit_threshold = None
    pool = SharedKVPool(1, pad_decode_rows=False)
    if spec.backend == 'ouro':
        b = OuroLoop(model, spec.depth_total, max_seq, nodes, kv_factory=pool)
        b.position_limit = max_seq
    else:
        b = HuginnLoop(model, spec.depth_total, max_seq, nodes, 0, max_seq,
                       extra_depths=(spec.loopfill_readout_depths[1],), kv_factory=pool)
        b.position_limit = max_seq
    if b.kv.K.dtype != torch.bfloat16:
        raise ValueError('KV dtype is not BF16')
    engine = PackedLoopExecutor([b], cuda_graph=False)

    def cache_new():
        return CompletionSuffixCache.arctic() if config.use_suffix else None

    def generate(prompt, limit, cache, rid, warmup=False):
        from llm_memory_editability.loop_sampling import TargetSampler, request_seed
        if sampling['temperature']:
            seed = request_seed(sampling['sampling_seed'], spec_name,
                dataset + ('/warmup' if warmup else ''), texts[rid]['source_id'])
            b.target_sampler = TargetSampler(sampling['temperature'], seed, max_seq+2, b.kv.K.device)
        else:
            b.target_sampler = None
        if isinstance(config, BudgetConfig):
            steps = budget_tree_decode_steps(b, prompt, config, limit, stop, cache=cache, req_id=rid, max_nodes=nodes)
        elif isinstance(config, LoopSpecConfig):
            steps = loopspec_decode_steps(b, prompt, config, limit, stop, cache=cache, req_id=rid, max_nodes=nodes)
        else:
            steps = tree_decode_steps(b, prompt, config, limit, stop, cache=cache, req_id=rid, max_nodes=nodes)
        with torch.inference_mode():
            return drive(b, engine, steps)

    for i, prompt in enumerate(prompts[:2]):
        generate(prompt, 16 if args.smoke else 32, cache_new(), i, warmup=True)
    engine.graphs.freeze()
    cache = cache_new()
    journal = target / 'requests.jsonl'
    completed = load_journal(journal, [r['source_id'] for r in texts])
    # Rebuild completion history without rerunning model generation.
    if cache is not None:
        for i, row in enumerate(completed):
            cache.start_request(i, np.asarray(prompts[i], dtype=np.int32))
            cache.add_active_response(i, np.asarray(row['tokens'], dtype=np.int32))
            cache.stop_request(i)
    run = None
    if not args.smoke:
        import wandb
        os.environ['WANDB_API_KEY'] = Path(protocol['wandb_key_file']).read_text().strip()
        run = wandb.init(project=protocol['wandb_project'], entity=protocol['wandb_entity'],
            group=protocol['name'], name=args.job, id=job['wandb_id'], resume='allow',
            dir=str(target), config=metadata,
            settings=wandb.Settings(init_timeout=90))
        save(target / 'wandb.json', {'id': run.id, 'url': run.url, 'mode': run.settings.mode})
        journal.touch(exist_ok=True)
        run.save(str(journal), base_path=str(target), policy='live')
        run.save(str(target / 'metadata.json'), base_path=str(target), policy='now')
    torch.cuda.reset_peak_memory_stats()
    try:
        with journal.open('a') as f:
            for i in range(len(completed), len(texts)):
                row, prompt = texts[i], prompts[i]
                tokens, stats = generate(prompt, max_new_tokens, cache, i)
                assert stats['tokens'] == len(tokens) - 1
                if arm == 'ar':
                    assert stats['calls'] == spec.depth_total * stats['tokens']
                record = dict(source_id=row['source_id'], question_id=i, category=row['category'],
                    prompt_tokens=len(prompt), prompt_sha256=row['prompt_sha256'],
                    tokens=tokens, text=tok.decode(tokens, skip_special_tokens=False), stats=stats,
                    stop_reason='eos' if tokens[-1] in stop else 'length', completed_unix=time.time(),
                    gpu_uuid=metadata['gpu_uuid'], sampling=sampling)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                completed.append(record)
                summary = summarize(completed, spec.depth_total)
                save(target / 'progress.json', dict(summary, total_requests=len(texts), updated_unix=time.time()))
                if run is not None:
                    run.log({**{'aggregate/' + k: v for k, v in summary.items() if v is not None},
                        'request/decode_s': stats['decode_s'], 'request/decode_tokens': stats['tokens'],
                        'request/loop_steps': stats['calls'], 'request/cap_stopped': record['stop_reason'] == 'length'},
                        step=i + 1)
                print(f"{args.job} {i+1}/{len(texts)} decode={stats['decode_s']:.3f}s tokens={len(tokens)}", flush=True)
        summary = dict(summarize(completed, spec.depth_total), **engine.metrics(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), total_requests=len(texts))
        assert not summary['cuda_graph'] and summary['cuda_graph_replays'] == 0
        save(target / 'summary.json', summary)
        # Measurement completion is durable even when the upload service is unavailable.
        save(target / 'complete.json', {'requests': len(completed), 'finished_unix': time.time(),
                                      'requests_sha256': digest(journal)})
        if run is not None:
            try:
                run.summary.update(summary)
                artifact = wandb.Artifact(args.job, type='latency-results', metadata={'complete': True})
                for name in ['metadata.json', 'requests.jsonl', 'summary.json']:
                    artifact.add_file(str(target / name))
                logged = run.log_artifact(artifact)
                logged.wait(timeout=180)
                run.finish()
                save(target / 'upload-status.json', {'state': 'complete', 'updated_unix': time.time()})
            except Exception as exc:
                save(target / 'upload-status.json', {'state': 'pending', 'error_type': type(exc).__name__,
                                                   'updated_unix': time.time()})
                print('Measurement complete; upload pending: ' + type(exc).__name__, flush=True)
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            run = None

    finally:
        if run is not None:
            run.finish(exit_code=1)


if __name__ == '__main__':
    main()
