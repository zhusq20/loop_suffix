"""Batched full-depth teacher continuations from a frozen, decontaminated manifest.

Ouro uses left-padded, length-sorted batches with masks and real per-example positions.
Huginn batches have identical prompt lengths. No benchmark timing configuration is
used to estimate production speed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def append(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ouro-2.6b', 'huginn-3.5b', 'ouro-1.4b'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--max-batch', type=int, default=32)
    parser.add_argument('--kv-budget-gib', type=int, default=None,
                        help='Logical KV ceiling; defaults to32 for dictionary-cache Huginn,70 for Ouro')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--production-cache-proof', type=Path,
                        help='completed real Huginn equivalence probe; enables teacher-only preallocated KV')
    parser.add_argument('--packed-graph-proof', type=Path,
                        help='completed same-engine graph and native FP32 equivalence probe; production only')
    args = parser.parse_args()
    if args.production_cache_proof and args.packed_graph_proof:
        parser.error('Choose one production cache backend')
    if not 0 <= args.shard < args.shards or args.max_batch < 1 or (args.kv_budget_gib is not None and args.kv_budget_gib < 1):
        parser.error('Invalid sharding or batch size')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'dflash_impl/target.py', BASE/'dflash_impl/production_cache.py',
                   BASE/'provenance.py', BASE/'gpu_startup.py', BASE/'gpu_allocation.py']
                   + ([BASE/'teacher_graph.py'] if args.packed_graph_proof else []))
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (r.split(',') for r in devices.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi + 1)))
        started = time.time()
        status = 'failed'
        try:
            from gpu_startup import acquire_cuda_context
            lease, acquisition = acquire_cuda_context(args.gpu, uuid)
            save(args.out/'cuda-acquisition.json', acquisition)
            started = acquisition['acquired_unix']
            run(args, uuid)
            status = 'complete'
        except Exception:
            save(args.out / 'failure.json', dict(error=traceback.format_exc(), time_unix=time.time()))
            raise
        finally:
            append(args.out / 'reservation-cost.jsonl', dict(started_unix=started, finished_unix=time.time(),
                reserved_gpu_seconds=time.time() - started, gpu_uuid=uuid, status=status))


def run(args, uuid):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoConfig
    from dflash_impl.target import LoopTarget
    sys.path.insert(0, str(ROOT / 'results/loopfill-latency512-20261107/source/src'))
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt, resolve_stop_ids
    torch.set_num_threads(1)
    torch.manual_seed(0)
    manifest = json.loads(args.manifest.with_name('manifest.json').read_text())
    expected = manifest['files'][args.manifest.stem]
    assert digest(args.manifest) == expected['sha256']
    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    assert len(rows) == expected['count']
    protocol = json.loads((ROOT / 'results/loopfill-latency512-20261107/protocol.json').read_text())
    spec = MODELS[args.model]
    proof_hash = None
    graph_proof_hash = None
    if args.production_cache_proof:
        assert spec.backend == 'huginn'
        proof_path = args.production_cache_proof/'report.json'
        proof = json.loads(proof_path.read_text())
        hashes = json.loads((args.production_cache_proof/'source-manifest.json').read_text())
        assert proof['status'] == 'complete' and proof['exact_logit_steps'] == proof['decode_steps'] == 128
        assert proof['top1_matches'] == proof['comparisons'] == 2048
        assert proof['max_abs_logit_difference'] == 0.
        for name in ['dflash_impl/production_cache.py','dflash_impl/target.py']:
            assert hashes[name] == digest(BASE/name), 'Cache proof belongs to a different source revision'
        proof_hash = digest(proof_path)
    if args.packed_graph_proof:
        proof_path=args.packed_graph_proof/'report.json'
        proof=json.loads(proof_path.read_text())
        hashes=json.loads((args.packed_graph_proof/'source-manifest.json').read_text())
        assert proof['status']=='complete'
        checks=[r for r in proof['results'] if r['model']==args.model]
        assert {r['dtype'] for r in checks}=={'bfloat16','float32'}
        assert all(r['graph_off_on_exact_sequences']==r['requests']==8 for r in checks)
        assert next(r for r in checks if r['dtype']=='float32')['native_fp32_exact_sequences']==8
        for name in ['teacher_graph.py','dflash_impl/target.py']:
            assert hashes[name]==digest(BASE/name), 'Graph proof belongs to another source revision'
        for name,expected_hash in protocol['source_sha256'].items():
            assert digest(ROOT/'results/loopfill-latency512-20261107/source'/name)==expected_hash
        graph_proof_hash=digest(proof_path)
    kv_budget_gib = args.kv_budget_gib if args.kv_budget_gib is not None else (70 if spec.backend == 'ouro' else 32)
    path = Path(protocol['models'][args.model]['path'])
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    generation_config = json.loads((path / 'generation_config.json').read_text()) if (path / 'generation_config.json').exists() else {}
    formats = {r['prompt_format'] for r in rows}
    assert len(formats) == 1
    stop, stop_audit = resolve_stop_ids(spec, tok, cfg.eos_token_id, generation_config.get('eos_token_id'),
                                       cfg.vocab_size, next(iter(formats)))
    by_length, prepared = defaultdict(list), {}
    for row in rows:
        tokens = tokenize_prompt(spec, tok, row['prompt'], row['prompt_format'])
        assert len(tokens) == row['prompt_lengths'][args.model]
        assert 0 < len(tokens) <= 2560 and row['max_new_tokens'] == 512
        prepared[row['source_id']] = dict(row=row, prompt_tokens=tokens)
        by_length[len(tokens)].append(row['source_id'])
    batches = []
    # Huginn's per-position dictionary views retain backing storage and repeatedly
    # stack caches. Its measured physical peak needs a lower logical KV ceiling.
    bytes_per_position = 2 * (192 if args.model == 'ouro-2.6b' else 96 if spec.backend == 'ouro' else 132) * cfg.num_key_value_heads * spec.head_dim * 2
    if spec.backend == 'ouro':
        batch = []
        for length, ids in sorted(by_length.items()):
            for identity_id in ids:
                limit = min(args.max_batch, max(1, (kv_budget_gib * 1024**3) // (bytes_per_position * (length + 512))))
                if len(batch) >= limit:
                    batches.append(batch)
                    batch = []
                batch.append(identity_id)
        if batch:
            batches.append(batch)
    else:
        for length, ids in sorted(by_length.items()):
            batch_size = min(args.max_batch, max(1, (kv_budget_gib * 1024**3) // (bytes_per_position * (length + 512))))
            for offset in range(0, len(ids), batch_size):
                batches.append(ids[offset:offset + batch_size])
    plan_hash = hashlib.sha256(json.dumps(batches).encode()).hexdigest()
    owned = [(i, batch) for i, batch in enumerate(batches) if i % args.shards == args.shard]
    assignment_owned = owned
    if graph_proof_hash:
        # Preserve the frozen global assignment and exact shard ID sets, but
        # exploit the packed engine's independently positioned ragged requests.
        # Native Huginn previously needed equal prompt lengths; keeping those
        # tiny groups would waste the validated packed producer's capacity.
        assigned_ids=[s for _,batch in owned for s in batch]
        assigned_ids.sort(key=lambda s:(len(prepared[s]['prompt_tokens']),s))
        execution_batches=[];batch=[]
        for identity_id in assigned_ids:
            length=len(prepared[identity_id]['prompt_tokens'])
            capacity=((length+512+127)//128)*128
            limit=min(args.max_batch,max(1,(kv_budget_gib*1024**3)//(bytes_per_position*capacity)))
            if len(batch)>=limit:
                execution_batches.append(batch);batch=[]
            batch.append(identity_id)
        if batch:execution_batches.append(batch)
        owned=[(i*args.shards+args.shard,batch) for i,batch in enumerate(execution_batches)]
        assert {s for _,batch in owned for s in batch}==set(assigned_ids)
        assert sum(len(batch) for _,batch in owned)==len(assigned_ids)
    execution_plan_hash=hashlib.sha256(json.dumps(owned).encode()).hexdigest()
    identity = dict(model=args.model, model_revision=spec.revision, manifest_sha256=expected['sha256'],
        batch_plan_sha256=plan_hash, shards=args.shards, shard=args.shard, max_batch=args.max_batch,
        execution_batch_plan_sha256=execution_plan_hash,
        packed_regrouping=bool(graph_proof_hash),
        kv_budget_gib=kv_budget_gib,
        max_new_tokens=512, precision='bfloat16', full_depth=spec.depth_total, seed=0,
        noise_capacity=3072, prompt_format=next(iter(formats)), stop_audit=stop_audit,
        batching='packed_ragged_length_sorted_same_frozen_shard_assignment' if graph_proof_hash else
            ('left_padded_real_positions' if spec.backend == 'ouro' else 'identical_prompt_length'),
        teacher_cache='packed_graph' if graph_proof_hash else ('huginn_preallocated' if proof_hash else 'checkpoint_native'),
        production_cache_proof_sha256=proof_hash,
        packed_graph_proof_sha256=graph_proof_hash,
        torch=torch.__version__, transformers=transformers.__version__)
    meta_path = args.out / 'metadata.json'
    if meta_path.exists():
        assert json.loads(meta_path.read_text())['identity'] == identity
    else:
        save(meta_path, dict(identity=identity, gpu_uuid=uuid, planned_requests=sum(map(len, (b for _, b in owned))),
                             planned_batches=len(owned), timing='production generation, not benchmark',
                             manifest=str(args.manifest.resolve())))
    if graph_proof_hash:
        save(args.out/'batch-plans.json',dict(assignment=assignment_owned,execution=owned,
            global_assignment_sha256=plan_hash,execution_sha256=execution_plan_hash))
    journal_path = args.out / 'teachers.jsonl'
    completed = {}
    if journal_path.exists():
        data = journal_path.read_bytes()
        complete_bytes = data.rfind(b'\n') + 1
        if complete_bytes < len(data):
            with journal_path.open('r+b') as f:
                f.truncate(complete_bytes)
        for line in data[:complete_bytes].splitlines():
            row = json.loads(line)
            assert row['source_id'] not in completed
            completed[row['source_id']] = row
    allowed = {s for _, batch in owned for s in batch}
    assert set(completed) <= allowed
    print('Loading teacher', args.model, 'requests', len(allowed), 'remaining', len(allowed - set(completed)), flush=True)
    target = LoopTarget.load(path, spec.backend, 3072)
    graph_engine=None

    @torch.inference_mode()
    def graph_generate(batch):
        nonlocal graph_engine
        from teacher_graph import GraphTeacher
        sequences=[prepared[s]['prompt_tokens'] for s in batch]
        capacity=((max(map(len,sequences))+512+127)//128)*128
        torch.cuda.synchronize();started=time.perf_counter()
        if graph_engine is None or (graph_engine.batch_size,graph_engine.max_seq)!=(len(batch),capacity):
            if graph_engine is not None:graph_engine.close()
            graph_engine=GraphTeacher(target,len(batch),capacity,graph=True)
            warm=graph_engine.warm(sequences)
            append(args.out/'graph-warmups.jsonl',dict(batch_size=len(batch),max_seq=capacity,warmup_s=warm,
                finished_unix=time.time(),metrics=graph_engine.engine.metrics()))
        outputs,summary=graph_engine.generate(sequences,512,stop)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        assert summary['cuda_graph']
        if any(len(tokens)>1 for tokens in outputs):assert summary['cuda_graph_replays']>0
        append(args.out/'graph-batches.jsonl',dict(source_ids=batch,inference_wall_s=elapsed,
            summary=summary,finished_unix=time.time()))
        return outputs,elapsed

    @torch.inference_mode()
    def batch_generate(batch):
        sequences = [prepared[s]['prompt_tokens'] for s in batch]
        lengths = torch.tensor([len(s) for s in sequences], device=target.device)
        width = max(map(len, sequences))
        inputs = torch.full((len(batch), width), min(stop), device=target.device, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            inputs[i, -len(sequence):] = torch.tensor(sequence, device=target.device)
        mask = (torch.arange(width, device=target.device)[None, :] >= (width - lengths)[:, None]).long()
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        if proof_hash:
            from dflash_impl.production_cache import HuginnProductionCache
            cache = HuginnProductionCache(width+512)
        else:
            cache = target.new_cache()
        torch.cuda.synchronize()
        started = time.perf_counter()
        options = dict(attention_mask=mask, position_ids=positions) if spec.backend == 'ouro' else {}
        output = target.forward(inputs, cache, features=False, **options)
        next_tokens = output.logits[:, -1].argmax(-1)
        generated = [[x] for x in next_tokens.tolist()]
        del output
        live = [seq[-1] not in stop for seq in generated]
        for step in range(1, 512):
            if not any(live):
                break
            if spec.backend == 'ouro':
                mask = torch.cat([mask, torch.ones((len(batch), 1), device=target.device, dtype=mask.dtype)], -1)
                options = dict(attention_mask=mask, position_ids=(lengths + step - 1)[:, None])
            output = target.forward(next_tokens[:, None], cache, features=False, **options)
            next_tokens = output.logits[:, -1].argmax(-1)
            values = next_tokens.tolist()
            del output
            for i, token in enumerate(values):
                if live[i]:
                    generated[i].append(token)
                    if token in stop:
                        live[i] = False
        torch.cuda.synchronize()
        return generated, time.perf_counter() - started

    for batch_id, batch in owned:
        if all(s in completed for s in batch):
            continue
        outputs, elapsed = graph_generate(batch) if graph_proof_hash else batch_generate(batch)
        append(args.out / 'batches.jsonl', dict(batch_id=batch_id, requests=len(batch),
            output_tokens=sum(map(len, outputs)), prompt_tokens=[len(prepared[s]['prompt_tokens']) for s in batch],
            inference_wall_s=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(), finished_unix=time.time()))
        for identity_id, generated in zip(batch, outputs):
            entry = prepared[identity_id]
            if identity_id in completed:
                assert completed[identity_id]['output_tokens'] == generated, 'Resumed batch outputs changed'
                continue
            record = dict(source_id=identity_id, pool=entry['row']['pool'], batch_id=batch_id,
                prompt_tokens=entry['prompt_tokens'], output_tokens=generated,
                prompt_sha256=sha_text(entry['row']['prompt']), question_sha256=entry['row']['question_sha256'],
                stop_reason='eos' if generated[-1] in stop else 'length',
                output_text=tok.decode(generated, skip_special_tokens=False))
            append(journal_path, record)
            completed[identity_id] = record
        progress = dict(completed_requests=len(completed), planned_requests=len(allowed), last_batch=batch_id,
                        last_batch_wall_s=elapsed, last_batch_size=len(batch), updated_unix=time.time())
        save(args.out / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
    assert set(completed) == allowed
    if graph_engine is not None:graph_engine.close()
    target.close()
    save(args.out / 'complete.json', dict(requests=len(completed), teachers_sha256=digest(journal_path),
        output_tokens=sum(len(r['output_tokens']) for r in completed.values()), finished_unix=time.time()))


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == '__main__':
    main()
