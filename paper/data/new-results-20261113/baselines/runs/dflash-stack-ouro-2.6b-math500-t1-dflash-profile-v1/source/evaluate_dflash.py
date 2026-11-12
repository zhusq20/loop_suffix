"""DFlash and its same-stack AR denominator on the immutable LoopFill inputs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
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
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['ouro-2.6b', 'huginn-3.5b', 'ouro-1.4b'], required=True)
    p.add_argument('--method', choices=['ar', 'dflash'], required=True)
    p.add_argument('--dataset', choices=['gsm8k', 'math500', 'bbh', 'humanevalplus', 'mbppplus', 'specbench'], required=True)
    p.add_argument('--temperature', type=float, choices=[0., 1.], default=0.)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--count', type=int, default=0, help='0 means complete frozen task')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    from gpu_allocation import allowed_gpus
    assert args.gpu in allowed_gpus(), 'GPU released by the user'
    if args.method == 'dflash' and args.checkpoint is None:
        p.error('DFlash requires an actual trained checkpoint')
    # An adopted worker can predate the controller's driver-environment fix.
    # Re-exec before importing Torch or creating the run, so its next stage
    # also loads driver libraries matching the already loaded kernel module.
    from native_cuda_env import native_cuda_environment
    native_env, driver_evidence = native_cuda_environment()
    if os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] != driver_evidence['directory']:
        command = [sys.executable, '-u', *sys.argv]
        driver_evidence['command'] = command
        record = BASE/'logs'/f'{args.out.name}-native-driver.json'
        with record.open('x') as stream:
            stream.write(json.dumps(driver_evidence, indent=2) + '\n')
        os.execve(sys.executable, command, native_env)
    args.out.mkdir(parents=True, exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'dflash_impl/target.py', BASE/'dflash_impl/decode.py',
        BASE/'provenance.py', BASE/'gpu_startup.py', BASE/'gpu_allocation.py', BASE/'native_cuda_env.py'])
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (line.split(',') for line in lines.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi + 1)))
        started, status = time.time(), 'failed'
        try:
            from gpu_startup import acquire_cuda_context
            lease, acquisition = acquire_cuda_context(args.gpu, uuid)
            save(args.out/'cuda-acquisition.json', acquisition)
            started = acquisition['acquired_unix']
            run(args, uuid)
            status = 'complete'
        except Exception:
            save(args.out / 'failure.json', dict(error=traceback.format_exc(), finished_unix=time.time()))
            raise
        finally:
            save(args.out / 'reservation-cost.json', dict(gpu_uuid=uuid, started_unix=started,
                finished_unix=time.time(), reserved_gpu_seconds=time.time()-started, status=status))


def run(args, uuid):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoConfig
    from dflash_impl.target import LoopTarget
    from dflash_impl.decode import generate
    suite = ROOT / 'results/loopfill-latency512-20261107'
    protocol = json.loads((suite / 'protocol.json').read_text())
    sys.path.insert(0, str(suite / 'source/src'))
    sys.path.insert(0, str(BASE / 'third_party/dflash'))
    from dflash.model import DFlashDraftModel
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids, tokenize_prompt
    sampling_path = ROOT / 'results/loopfill-latency512-t1-20261108/source/src/llm_memory_editability/loop_sampling.py'
    loader_spec = importlib.util.spec_from_file_location('frozen_target_sampling', sampling_path)
    sampling = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(sampling)
    spec = MODELS[args.model]
    path = Path(protocol['models'][args.model]['path'])
    input_path = suite / f'inputs/{args.dataset}.jsonl'
    assert digest(input_path) == protocol['inputs'][args.dataset]['sha256']
    rows = [json.loads(line) for line in input_path.open()]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    generation = json.loads((path / 'generation_config.json').read_text()) if (path / 'generation_config.json').exists() else {}
    stops, stop_audit = resolve_stop_ids(spec, tok, config.eos_token_id, generation.get('eos_token_id'), config.vocab_size, 'chat')
    prompts = [tokenize_prompt(spec, tok, row['prompt'], 'chat') for row in rows]
    required = max(map(len, prompts)) + 512 + spec.depth_total + 4
    capacity = min(spec.context_limit, ((required + 255) // 256) * 256)
    assert all(len(prompt) + 512 + spec.depth_total + 4 <= capacity for prompt in prompts)
    all_count = len(rows)
    count = args.count or all_count
    assert 0 < count <= all_count
    rows, prompts = rows[:count], prompts[:count]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    target = LoopTarget.load(path, spec.backend, capacity)
    draft, checkpoint_hashes = None, {}
    if args.method == 'dflash':
        assert (args.checkpoint / 'complete.json').exists()
        checkpoint_hashes = {f.name: digest(f) for f in sorted(args.checkpoint.glob('*.safetensors'))}
        checkpoint_hashes['config.json'] = digest(args.checkpoint / 'config.json')
        draft, info = DFlashDraftModel.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', output_loading_info=True, local_files_only=True)
        assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'], info
        draft = draft.to(target.device).eval()
        assert draft.target_layer_ids == target.expanded_ids and draft.block_size == 16
        save(args.out / 'checkpoint-loading.json', info)
    metadata = dict(model=args.model, model_revision=spec.revision, method=args.method, dataset=args.dataset,
        temperature=args.temperature, top_p=1., top_k=0, seed=0, count=count, full_count=all_count,
        max_new_tokens=512, max_seq=capacity, input_sha256=digest(input_path), stop_audit=stop_audit,
        gpu_uuid=uuid, cpu_affinity=sorted(os.sched_getaffinity(0)), torch=torch.__version__,
        dtype='bfloat16',
        transformers=transformers.__version__, cuda=torch.version.cuda, graph=False, compile=False,
        checkpoint=str(args.checkpoint) if args.checkpoint else None, checkpoint_sha256=checkpoint_hashes,
        stage='pilot' if args.pilot else 'formal', trace=args.trace, profiling=args.profile,
        reportable=not (args.pilot or args.trace or args.profile or count != all_count),
        denominator='full-depth HF LoopTarget AR, same target version and precision',
        sampling='frozen positional target uniforms for AR; DFlash softmax-q rejection with request-hashed seed',
        source_sha256={str(p.relative_to(BASE)): digest(p) for p in [Path(__file__), BASE/'dflash_impl/target.py', BASE/'dflash_impl/decode.py']})
    save(args.out / 'metadata.json', metadata)

    def run_one(row, prompt, limit, trace=None):
        seed = sampling.request_seed(0, args.model, args.dataset, row['source_id'])
        sampler = sampling.TargetSampler(args.temperature, seed, capacity, target.device) if args.temperature else None
        return generate(target, torch.tensor([prompt], device=target.device), limit, stops, draft=draft,
                        temperature=args.temperature, seed=seed, target_sampler=sampler,
                        profile_draft=args.profile, trace=trace)

    for row, prompt in list(zip(rows, prompts))[:2]:
        run_one(row, prompt, 32)
    completed = []
    with (args.out / 'requests.jsonl').open('x') as f:
        for row, prompt in zip(rows, prompts):
            trace = [] if args.trace else None
            result = run_one(row, prompt, 512, trace)
            record = dict(source_id=row['source_id'], prompt_sha256=row['prompt_sha256'], tokens=result.tokens,
                stats=result.stats, stop_reason=result.stats['stop_reason'], finished_unix=time.time())
            if trace is not None:
                record['trace'] = trace
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
            completed.append(record)
            seconds = sum(r['stats']['decode_s'] for r in completed)
            tokens = sum(r['stats']['decode_tokens'] for r in completed)
            calls = sum(r['stats']['target_verifications'] for r in completed)
            summary = dict(requests=len(completed), output_tokens=sum(len(r['tokens']) for r in completed),
                decode_tokens=tokens, decode_s=seconds, tok_s=tokens/seconds, target_verifications=calls,
                loop_steps=calls*spec.depth_total, gamma=tokens/calls if calls else None)
            save(args.out / 'progress.json', summary)
            print(json.dumps(dict(request=len(completed), tokens=len(result.tokens), decode_s=result.stats['decode_s'],
                                  gamma=result.stats['gamma'])), flush=True)
    save(args.out / 'summary.json', summary)
    save(args.out / 'complete.json', dict(requests=count, requests_sha256=digest(args.out/'requests.jsonl'), finished_unix=time.time()))
    target.close()


if __name__ == '__main__':
    main()
