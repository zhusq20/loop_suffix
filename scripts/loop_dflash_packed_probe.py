"""Validate and time DFlash on the frozen LoopFill packed target backend.

CPU --tiny checks use the released model classes at reduced width. GPU runs
check full-weight FP32 features, then compare native/packed AR and DFlash in
BF16 on the same frozen prompts. Results are diagnostic, never formal-table rows.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def adapter_class():
    # The adapter is new; all executor/backbone imports resolve to --engine-source.
    path = REPO/'baselines/dflash/packed_target.py'
    spec = importlib.util.spec_from_file_location('packed_dflash_adapter', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PackedDFlashTarget


def check_target(native, packed, tolerance):
    import torch
    nc, pc = native.new_cache(), packed.new_cache()
    differences = []
    for tokens, retained in [([1, 2, 3, 4, 5, 6], None), ([7, 8, 9, 10], 7),
                             ([11, 12, 13], 8), ([14], None)]:
        ids = torch.tensor([tokens], device=native.device)
        a, b = native.forward(ids, nc), packed.forward(ids, pc)
        for kind in ['logits', 'features']:
            x, y = getattr(a, kind), getattr(b, kind)
            assert x.shape == y.shape and bool(torch.isfinite(y).all())
            torch.testing.assert_close(x.float(), y.float(), atol=tolerance, rtol=tolerance)
            differences.append(dict(kind=kind, width=len(tokens),
                                    max_abs=float((x.float()-y.float()).abs().max())))
        assert torch.equal(a.logits.argmax(-1), b.logits.argmax(-1))
        assert nc.get_seq_length() == pc.get_seq_length()
        if retained is not None:
            native.crop(nc, retained)
            packed.crop(pc, retained)
    assert all(v == pc.get_seq_length() for v in packed.cache_lengths(pc))
    # The adapter must not retain selected feature/logit tensors between calls.
    assert not packed._features and packed._logits is None
    return dict(status='passed', cases=differences, tolerance=tolerance,
                logits_top1_equal=True, partial_rejection_and_recompute=True)


def tiny_native(path, family):
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from dflash_impl.target import LoopTarget
    config = json.loads((path/'config.json').read_text())
    cls = get_class_from_dynamic_module(config['auto_map']['AutoModelForCausalLM'],
                                       str(path), local_files_only=True)
    if family == 'ouro':
        cfg = cls.config_class(vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=24, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, tie_word_embeddings=False)
    else:
        cfg = cls.config_class(n_embd=32, n_heads=4, n_layers=8, block_size=128,
            vocab_size=97, padding_multiple=1, intermediate_size=64,
            mean_recurrence=32, n_layers_in_prelude=2, n_layers_in_recurrent_block=4,
            n_layers_in_coda=2, qk_bias=True, torch_dtype='float32')
    cfg._attn_implementation = 'sdpa'
    torch.manual_seed(71)
    return LoopTarget(cls(cfg).float().eval(), family, 128)


def tiny_draft(native):
    from dflash.model import DFlashDraftModel
    from dflash_impl.projected_draft import projected_config, inference_draft_class
    cfg = native.draft_config(0)
    if native.family == 'huginn':
        cfg = projected_config(cfg, 16, 48, 2)
        cls = inference_draft_class()
    else:
        cls = DFlashDraftModel
    cfg._attn_implementation = 'sdpa'
    return cls(cfg).float().eval()


def run_tiny(args, report):
    import torch
    from dflash_impl.decode import generate
    cls = adapter_class()
    for name in args.models:
        family = 'huginn' if name.startswith('huginn') else 'ouro'
        native = tiny_native(Path(report['models'][name]['path']), family)
        packed = cls(native)
        result = check_target(native, packed, 5e-5)
        draft = tiny_draft(native)
        prompt = torch.tensor([[1, 2, 3, 4]])
        result['decode'] = []
        for temperature in [0., 1.]:
            for budget in [1, 2, 17, 33]:
                for drafter in [None, draft]:
                    kwargs = dict(draft=drafter, temperature=temperature, seed=27)
                    a = generate(native, prompt, budget, [], **kwargs)
                    b = generate(packed, prompt, budget, [], **kwargs)
                    assert a.tokens == b.tokens
                    assert a.stats['target_verifications'] == b.stats['target_verifications']
                    result['decode'].append(dict(temperature=temperature, budget=budget,
                        method='ar' if drafter is None else 'dflash', tokens_equal=True))
        report['checks'][name] = result
        save(args.out/'report.json', report)
        packed.close()
        native.close()


def run_gpu(args, report):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from dflash.model import DFlashDraftModel
    from dflash_impl.target import LoopTarget
    from dflash_impl.decode import generate
    from dflash_impl.projected_draft import inference_draft_class
    import dflash_training_registry as registry
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt, resolve_stop_ids
    cls = adapter_class()
    suite = args.baseline_root.parents[1]/'results/loopfill-latency512-20261107'
    report['environment'] = dict(python=sys.version, torch=torch.__version__,
        transformers=transformers.__version__, cuda=torch.version.cuda,
        executable=sys.executable, cpu_affinity=sorted(os.sched_getaffinity(0)))
    for name in args.models:
        family = 'huginn' if name.startswith('huginn') else 'ouro'
        path = Path(report['models'][name]['path'])
        native = LoopTarget.load(path, family, 128, dtype=torch.float32)
        packed = cls(native)
        check = check_target(native, packed, 5e-4)
        packed.close()
        native.close()
        del packed, native
        gc.collect()
        torch.cuda.empty_cache()
        report['checks'][name] = check
        save(args.out/'report.json', report)

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        model_spec = MODELS[name]
        selected = []
        for task in ['gsm8k', 'humanevalplus']:
            input_file = suite/'inputs'/f'{task}.jsonl'
            rows = [json.loads(line) for line in input_file.open()]
            for row in rows[:args.per_task]:
                selected.append(dict(dataset=task, source_id=row['source_id'],
                    prompt=tokenize_prompt(model_spec, tok, row['prompt'], 'chat'),
                    input_sha256=digest(input_file)))
        required = max(len(r['prompt']) for r in selected) + args.max_new_tokens + model_spec.depth_total + 4
        capacity = ((required + 255)//256)*256
        native = LoopTarget.load(path, family, capacity)
        packed = cls(native)
        generation = json.loads((path/'generation_config.json').read_text()) if (path/'generation_config.json').exists() else {}
        stops, audit = resolve_stop_ids(model_spec, tok, native.model.config.eos_token_id,
            generation.get('eos_token_id'), native.model.config.vocab_size, 'chat')
        root = registry.training_root(name)
        checkpoint = Path(json.loads((root/'latest.json').read_text())['checkpoint'])
        assert (root/'complete.json').exists() and (checkpoint/'complete.json').exists()
        assert json.loads((root/'metadata.json').read_text())['epochs'] == 6
        if family == 'huginn':
            registry.validate_projected_completion()
        draft_cls = inference_draft_class() if family == 'huginn' else DFlashDraftModel
        draft, loading = draft_cls.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa', output_loading_info=True)
        assert all(not loading[k] for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
        draft = draft.to(native.device).eval()
        assert draft.target_layer_ids == native.expanded_ids
        save(args.out/f'{name}-metadata.json', dict(inputs=selected, checkpoint=str(checkpoint),
            weights_sha256=digest(checkpoint/'model.safetensors'), stop_audit=audit,
            dtype='bfloat16', max_seq=capacity, max_new_tokens=args.max_new_tokens,
            temperature=0, batch_size=1, graph=False, compile=False, reportable=False))
        methods = [('native-ar', native, None), ('packed-ar', packed, None),
                   ('native-dflash', native, draft), ('packed-dflash', packed, draft)]
        # Equal warmup outside timing, using the same first two selected prompts.
        for _, target, drafter in methods:
            for row in selected[:2]:
                generate(target, torch.tensor([row['prompt']], device=native.device),
                         16, stops, draft=drafter, seed=0)
        measured = []
        journal = args.out/f'{name}-requests.jsonl'
        with journal.open('x') as stream:
            for index, row in enumerate(selected):
                order = methods[index % len(methods):] + methods[:index % len(methods)]
                for method, target, drafter in order:
                    result = generate(target, torch.tensor([row['prompt']], device=native.device),
                        args.max_new_tokens, stops, draft=drafter, temperature=0, seed=0)
                    record = dict(model=name, dataset=row['dataset'], source_id=row['source_id'],
                        method=method, tokens=result.tokens, stats=result.stats, finished_unix=time.time())
                    measured.append(record)
                    stream.write(json.dumps(record)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    save(args.out/'progress.json', dict(model=name, completed=len(measured),
                        expected=len(selected)*4, latest_method=method, updated_unix=time.time()))
        summary = {}
        for method, _, _ in methods:
            subset = [r for r in measured if r['method'] == method]
            tokens = sum(r['stats']['decode_tokens'] for r in subset)
            seconds = sum(r['stats']['decode_s'] for r in subset)
            calls = sum(r['stats']['target_verifications'] for r in subset)
            summary[method] = dict(requests=len(subset), decode_tokens=tokens, decode_s=seconds,
                                   token_s=tokens/seconds, gamma=tokens/calls)
        for method, ar in [('native-dflash', 'native-ar'), ('packed-dflash', 'packed-ar')]:
            summary[method]['eta_latency'] = summary[ar]['decode_s']/summary[method]['decode_s']
        agreement = {}
        keyed = {(r['dataset'], r['source_id'], r['method']):r for r in measured}
        for left, right in [('native-ar', 'packed-ar'), ('native-dflash', 'packed-dflash'),
                            ('packed-ar', 'packed-dflash')]:
            agreement[left+'/'+right] = sum(
                keyed[r['dataset'], r['source_id'], left]['tokens'] ==
                keyed[r['dataset'], r['source_id'], right]['tokens'] for r in selected)
        report['benchmarks'][name] = dict(summary=summary, exact_sequences=agreement,
            requests=len(selected), journal_sha256=digest(journal))
        save(args.out/'report.json', report)
        packed.close()
        native.close()
        del packed, native, draft, methods, target, drafter
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path,
        default=Path('/ossfs/workspace/sz/loopfill/experiments/baselines'))
    parser.add_argument('--engine-source', type=Path)
    parser.add_argument('--models', nargs='+', default=['huginn-3.5b', 'ouro-2.6b'],
        choices=['huginn-3.5b', 'ouro-2.6b'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cpus', default='8-11')
    parser.add_argument('--wait-seconds', type=float, default=3600)
    parser.add_argument('--per-task', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--release-reservation', action='store_true')
    args = parser.parse_args()
    if args.engine_source is None:
        args.engine_source = args.baseline_root.parents[1]/'results/loopfill-latency512-20261107/source/src'
    if not args.tiny and args.gpu is None:
        parser.error('--gpu is required for real-weight checks')
    if args.per_task < 1 or not 1 <= args.max_new_tokens <= 512:
        parser.error('positive per-task count and 1..512 token budget required')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.baseline_root), str(args.engine_source),
                   str(args.baseline_root/'third_party/dflash')]
    from native_cuda_env import native_cuda_environment
    env, driver = native_cuda_environment()
    os.environ['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']
    # Set environment before importing torch; the caller should use native_cuda_env
    # for real GPU runs so driver lookup is fixed from process startup as well.
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.tiny:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        from gpu_allocation import allowed_gpus
        assert args.gpu in allowed_gpus()
        uuids = {int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
        os.environ['CUDA_VISIBLE_DEVICES'] = uuids[args.gpu]
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
    import torch
    torch.set_num_threads(1)
    protocol = json.loads((args.baseline_root.parents[1]/'results/loopfill-latency512-20261107/protocol.json').read_text())
    report = dict(status='running', tiny=args.tiny, reportable=False, started_unix=time.time(),
        engine_source=str(args.engine_source), models=protocol['models'], checks={}, benchmarks={},
        limitations=['Bounded diagnostics only; not a full benchmark matrix.',
                     'BF16 outputs may differ by execution shape; report exact agreement separately.',
                     'No claim of a proportional speed benefit or task-quality equivalence.'])
    source = args.out/'source'
    source.mkdir()
    files = [Path(__file__), REPO/'baselines/dflash/packed_target.py',
             args.baseline_root/'dflash_impl/target.py', args.baseline_root/'dflash_impl/decode.py',
             args.baseline_root/'dflash_impl/projected_draft.py',
             args.baseline_root/'third_party/dflash/dflash/model.py']
    files += list((args.engine_source/'llm_memory_editability').glob('*.py'))
    manifest = {}
    for index, path in enumerate(files):
        dest = source/f'{index:03d}-{path.name}'
        shutil.copyfile(path, dest)
        manifest[str(path)] = dict(snapshot=dest.name, sha256=digest(path))
    save(args.out/'source-manifest.json', manifest)
    save(args.out/'report.json', report)
    lease, lock = None, None
    acquired = None
    try:
        if not args.tiny:
            deadline = time.monotonic()+args.wait_seconds
            lock = open(f'/tmp/loopfill-{uuids[args.gpu]}.lock', 'a+')
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                        '--format=csv,noheader'], text=True)
                    if uuids[args.gpu] not in busy:
                        torch.cuda.set_device(0)
                        lease = torch.empty(1, device='cuda')
                        acquired = time.time()
                        break
                    fcntl.flock(lock, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
                save(args.out/'progress.json', dict(stage='waiting_for_existing_job',
                    gpu=args.gpu, updated_unix=time.time()))
                if time.monotonic() >= deadline:
                    raise TimeoutError('Assigned GPU did not become free; no job interrupted')
                time.sleep(2)
        with torch.inference_mode():
            (run_tiny if args.tiny else run_gpu)(args, report)
        report['status'] = 'passed_cpu_tiny_checks' if args.tiny else 'complete_gpu_diagnostic'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        if acquired is not None:
            report['gpu'] = args.gpu
            report['reserved_gpu_seconds'] = time.time()-acquired
            report['cuda_lease_device'] = str(lease.device)
        save(args.out/'report.json', report)
        if lock:
            lock.close()
        if args.release_reservation:
            policy_path = args.baseline_root/'configs/gpu-allocation.json'
            policy = json.loads(policy_path.read_text())
            policy['reserved_for_development'] = [g for g in policy['reserved_for_development'] if g != args.gpu]
            policy['updated_unix'] = time.time()
            save(policy_path, policy)
        print(json.dumps({'status':report['status'], 'report':str(args.out/'report.json')}), flush=True)


if __name__ == '__main__':
    main()
