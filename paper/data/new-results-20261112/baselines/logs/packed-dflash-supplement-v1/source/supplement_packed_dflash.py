"""Measure missing own-stack AR and draft profiling without repeating timed DFlash."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from packed_dflash_evidence import (check_frozen, check_plan, check_profile, check_run,
                                    read, save, sha, totals)


def run(args):
    plan = check_plan(args.plan)
    suite = Path(plan['suite'])
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    protocol, inputs = check_frozen(suite, job, weights=True)
    binding = read(args.binding)
    assert binding['job'] == job and binding['stage'] == args.stage
    assert binding['plan_sha256'] == sha(args.plan)
    timed = check_run(binding['timed']['path'], suite, job)
    assert timed['hashes'] == binding['timed']['hashes']
    source = suite / 'source'
    sys.path[:0] = [str(source), str(source / 'src')]
    import torch
    from safetensors import safe_open
    from baselines.dflash.decode import generate
    from baselines.dflash.packed_target import PackedDFlashTarget
    from baselines.dflash.target import LoopTarget
    from baselines.infer import load_dflash
    from llm_memory_editability.loop_sampling import request_seed, TargetSampler
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    lease = torch.empty(1, device='cuda')
    model = protocol['models'][job['model']]
    native = LoopTarget.load(Path(model['path']), model['family'], job['max_seq'], dtype=torch.bfloat16)
    draft = load_dflash(Path(model['checkpoint']), native) if args.stage == 'profile' else None
    if draft is not None:
        with safe_open(Path(model['checkpoint']) / 'model.safetensors', framework='pt', device='cpu') as f:
            assert set(f.keys()) == set(draft.state_dict())
    packed = PackedDFlashTarget(native)
    meta = dict(job=job, stage=args.stage, target_backend='PackedDFlashTarget',
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch=str(torch.__version__), cuda=torch.version.cuda, target_dtype=str(native.dtype),
        checkpoint=model['checkpoint'], checkpoint_sha256=model['checkpoint_sha256'],
        source_sha256=protocol['source_sha256'], supplement_source_sha256=plan['source_sha256'],
        plan_sha256=sha(args.plan), binding_sha256=sha(args.binding),
        graph=False, compile=False, profiling=args.stage == 'profile', reportable=args.stage == 'ar',
        denominator='Fresh AR on identical frozen PackedDFlashTarget and generate(draft=None); max512 and same inputs.',
        timing='Synchronized decode only; load, prefill, warmup and persistence excluded.',
        timed_run=binding['timed'], limitations=plan['limitations'])
    assert meta['torch'] == timed['metadata']['torch'] and meta['cuda'] == timed['metadata']['cuda']
    save(args.out / 'metadata.json', meta)

    def one(row, budget, profile=False):
        seed = request_seed(0, job['model'], job['dataset'], row['source_id'])
        sampler = TargetSampler(job['temperature'], seed, job['max_seq'], native.device) if job['temperature'] else None
        return generate(packed, torch.tensor([row['prompt_tokens']], device=native.device), budget,
            set(model['stop_ids']), draft=draft, temperature=job['temperature'], seed=seed,
            target_sampler=sampler, profile_draft=profile)

    for row in [inputs[0], max(inputs, key=lambda r: len(r['prompt_tokens']))]:
        one(row, 16)
    for row in inputs[:2]:
        one(row, 32)
    save(args.out / 'preflight.json', dict(status='passed', preflight_requests=2,
        preflight_tokens=16, warmup_requests=2, warmup_tokens=32))
    records = []
    with (args.out / 'requests.jsonl').open('x') as stream:
        for index, row in enumerate(inputs):
            result = one(row, job['max_new_tokens'], args.stage == 'profile')
            record = dict(source_id=row['source_id'], prompt_sha256=row['prompt_sha256'],
                tokens=result.tokens, stats=result.stats, stop_reason=result.stats['stop_reason'],
                finished_unix=time.time())
            if args.stage == 'profile':
                check_profile([record], [timed['rows'][index]])
            import json
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            save(args.out / 'progress.json', dict(requests=len(records), expected=len(inputs), updated_unix=time.time()))
    summary = totals(records, job, profile=args.stage == 'profile', ar=args.stage == 'ar')
    save(args.out / 'summary.json', summary)
    packed.close()
    native.close()
    del lease
    save(args.out / 'complete.json', dict(requests=len(records),
        requests_sha256=sha(args.out / 'requests.jsonl'), summary_sha256=sha(args.out / 'summary.json'),
        finished_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--stage', choices=['ar', 'profile'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    os.fstat(args.lock_fd)
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run(args)
    except BaseException:
        save(args.out / 'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        raise
    finally:
        save(args.out / 'reservation-cost.json', dict(gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
            started_unix=started, finished_unix=time.time(), reserved_gpu_seconds=time.time() - started))


if __name__ == '__main__':
    main()
