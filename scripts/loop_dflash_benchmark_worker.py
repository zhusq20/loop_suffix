"""One frozen DFlash validation, latency, or continuous-throughput cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def run(suite, job, out):
    import torch
    from baselines.dflash.batch import BatchedDFlashTarget, batch_decode
    from baselines.dflash.decode import generate
    from baselines.dflash.packed_target import PackedDFlashTarget
    from baselines.dflash.target import LoopTarget
    from baselines.infer import load_dflash
    from llm_memory_editability.loop_batch_decode import BatchRequest
    from llm_memory_editability.loop_sampling import request_seed, TargetSampler

    protocol = read(suite/'protocol.json')
    model = protocol['models'][job['model']]
    for rel, sha in protocol['source_sha256'].items():
        if digest(suite/'source'/rel) != sha:
            raise ValueError(f'Frozen source changed: {rel}')
    path = suite/job['inputs']
    if digest(path) != job['inputs_sha256']:
        raise ValueError('Frozen input changed')
    checkpoint = Path(model['checkpoint'])
    for name, sha in model['checkpoint_sha256'].items():
        if digest(checkpoint/name) != sha:
            raise ValueError(f'Checkpoint changed: {name}')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    native = LoopTarget.load(Path(model['path']), model['family'], job['max_seq'],
                            dtype=torch.float32 if job['kind'] == 'validate' else torch.bfloat16)
    draft = load_dflash(checkpoint, native)
    # from_pretrained must never silently fill missing checkpoint tensors.
    from safetensors import safe_open
    with safe_open(checkpoint/'model.safetensors', framework='pt', device='cpu') as f:
        if set(f.keys()) != set(draft.state_dict()):
            raise ValueError('Drafter checkpoint keys do not match the loaded architecture')
    metadata = dict(job=job, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        cpu_affinity=sorted(os.sched_getaffinity(0)), torch=str(torch.__version__), cuda=torch.version.cuda,
        checkpoint=model['checkpoint'], checkpoint_sha256=model['checkpoint_sha256'],
        source_sha256=protocol['source_sha256'], graph=False, compile=False,
        target_dtype=str(native.dtype), reportable=job['kind'] != 'validate',
        denominator='original table AR, historical same-host measurement; no drift correction')
    save(out/'metadata.json', metadata)
    stops = set(model['stop_ids'])
    if job['kind'] == 'latency':
        packed = PackedDFlashTarget(native)
        def one(row, limit):
            seed = request_seed(0, job['model'], job['dataset'], row['source_id'])
            sampler = TargetSampler(job['temperature'], seed, job['max_seq'], native.device) if job['temperature'] else None
            return generate(packed, torch.tensor([row['prompt_tokens']], device=native.device),
                limit, stops, draft=draft, temperature=job['temperature'], seed=seed, target_sampler=sampler)
        for row in [rows[0], max(rows, key=lambda r: len(r['prompt_tokens']))]:
            one(row, 16)
        save(out/'preflight.json', dict(status='passed', requests=2, max_new_tokens=16))
        for row in rows[:2]:
            one(row, 32)
        records = []
        with (out/'requests.jsonl').open('x') as stream:
            for row in rows:
                result = one(row, 512)
                record = dict(source_id=row['source_id'], prompt_sha256=row['prompt_sha256'],
                    tokens=result.tokens, stats=result.stats, stop_reason=result.stats['stop_reason'],
                    finished_unix=time.time())
                append(stream, record)
                records.append(record)
                save(out/'progress.json', dict(requests=len(records), expected=len(rows), updated_unix=time.time()))
        tokens = sum(r['stats']['decode_tokens'] for r in records)
        seconds = sum(r['stats']['decode_s'] for r in records)
        calls = sum(r['stats']['target_verifications'] for r in records)
        summary = dict(requests=len(records), decode_tokens=tokens, decode_s=seconds,
            decode_token_s=tokens/seconds, gamma=tokens/calls if calls else None,
            output_tokens=sum(len(r['tokens']) for r in records), target_verifications=calls,
            cap_stopped=sum(r['stop_reason'] == 'length' for r in records))
        packed.close()
    else:
        if job['kind'] == 'validate':
            rows = rows[:6]
            reqs = [BatchRequest(r['source_id'], r['prompt_tokens'][:8+3*i],
                                [33, 18, 2, 17, 1, 9][i], set()) for i, r in enumerate(rows)]
        else:
            reqs = [BatchRequest(r['source_id'], r['prompt_tokens'], 1024, stops) for r in rows]
            random.Random(0).shuffle(reqs)  # Original Table 2 order-seed=0.
        batch = BatchedDFlashTarget(native, job['batch'])
        if job['kind'] == 'validate':
            serial = PackedDFlashTarget(native)
            expected = [generate(serial, torch.tensor([r.prompt], device=native.device),
                r.max_new_tokens, r.stop, draft=draft).tokens for r in reqs]
            actual, summary = batch_decode(reqs, batch, draft=draft, prefill_chunk_size=8)
            save(out/'comparison.json', dict(expected=expected, actual=actual, summary=summary))
            assert [r['tokens'] for r in actual] == expected, 'Full-weight FP32 serial/batch divergence'
            assert summary['max_verify_batch'] > 1 and summary['max_draft_batch'] > 1
            serial.close()
            with (out/'requests.jsonl').open('x') as stream:
                for record in actual:
                    append(stream, record)
            summary['status'] = 'passed_full_weight_fp32'
        else:
            warm = [BatchRequest(r.request_id, r.prompt, 32, r.stop) for r in reqs[:2*job['batch']]]
            _, smoke = batch_decode(warm, batch, draft=draft)
            save(out/'preflight.json', smoke)
            torch.cuda.reset_peak_memory_stats()
            io_seconds = 0.
            with (out/'requests.jsonl').open('x') as stream:
                def progress(completed, elapsed):
                    nonlocal io_seconds
                    before = time.perf_counter()
                    append(stream, dict(completed[-1], finished_unix=time.time()))
                    io_seconds += time.perf_counter() - before
                    save(out/'progress.json', dict(requests=len(completed), expected=len(reqs),
                        wall_s=elapsed, updated_unix=time.time()))
                records, summary = batch_decode(reqs, batch, draft=draft, progress_callback=progress)
            assert summary['prefill_chunk_tokens'] == sum(len(r.prompt) for r in reqs)
            if job['batch'] > 1:
                assert summary['max_verify_batch'] > 1 and summary['max_draft_batch'] > 1
                assert summary['refilled_requests'] > 0
            summary.update(result_persistence_s=io_seconds,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        batch.close()
    native.close()
    summary.update(model=job['model'], kind=job['kind'], dataset=job['dataset'],
        temperature=job['temperature'], batch=job['batch'], reportable=job['kind'] != 'validate')
    save(out/'summary.json', summary)
    save(out/'complete.json', dict(requests=summary['requests'],
        requests_sha256=digest(out/'requests.jsonl'), summary_sha256=digest(out/'summary.json'),
        finished_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    os.fstat(args.lock_fd)  # Inherited shared GPU lock survives controller termination.
    args.out.mkdir(parents=True, exist_ok=False)
    job = next(j for j in read(args.suite/'jobs.json') if j['id'] == args.job)
    started = time.time()
    try:
        import torch
        # Acquire the exclusive context before loading models to close startup races.
        torch.cuda.set_device(0)
        lease = torch.empty(1, device='cuda')
        run(args.suite, job, args.out)
        del lease
    except BaseException as exc:
        import torch
        status = 'oom' if isinstance(exc, torch.OutOfMemoryError) else 'failed'
        save(args.out/'failure.json', dict(outcome=status, error=traceback.format_exc(),
            started_unix=started, finished_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
