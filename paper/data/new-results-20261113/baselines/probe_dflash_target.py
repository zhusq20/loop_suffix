"""Short A1 compatibility probe of unmodified target checkpoints. No training/benchmark."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from provenance import freeze_sources
    base = Path(__file__).resolve().parent
    freeze_sources(args.out, [Path(__file__), base/'provenance.py', base/'gpu_startup.py', base/'gpu_allocation.py'])
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                      '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (r.split(',') for r in devices.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi + 1)))
        report = dict(stage='target_compatibility_only', model=args.model, gpu_uuid=uuid,
                      started_unix=time.time(), status='running')
        try:
            from gpu_startup import acquire_cuda_context
            lease, acquisition = acquire_cuda_context(args.gpu, uuid)
            report['cuda_acquisition'] = acquisition
            probe(args, report)
            report['status'] = 'probe_complete_not_baseline_validation'
        except Exception:
            report['status'] = 'failed'
            report['error'] = traceback.format_exc()
            raise
        finally:
            report['finished_unix'] = time.time()
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)


def probe(args, report):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = Path(__file__).resolve().parent
    root = base.parents[1]
    protocol = json.loads((root / 'results/loopfill-latency512-20261107/protocol.json').read_text())
    path = Path(protocol['models'][args.model]['path'])
    sys.path.insert(0, str(base / 'third_party/dflash'))
    from dflash.model import DFlashDraftModel
    report.update(torch=torch.__version__, transformers=transformers.__version__,
                  target_path=str(path), official_draft_import=DFlashDraftModel.__name__,
                  dflash_commit=subprocess.check_output(['git', '-C', str(base / 'third_party/dflash'),
                    'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.py')})
    print('Loading', args.model, flush=True)
    kwargs = dict(trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16)
    if args.model.startswith('ouro'):
        kwargs['attn_implementation'] = 'sdpa'
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to('cuda').eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    ids = tok('A short compatibility check: one two three four five six seven eight nine ten. '
              'We check causal attention and cache rollback with a fixed continuation.',
              return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')[:, :40]
    assert ids.shape[1] >= 24
    # Nonbenchmark text; no outputs from this probe enter performance tables or training.
    ids = ids[:, :24]
    prefix, suffix = ids[:, :16], ids[:, 16:]
    records, captured, handles = {}, {}, []
    if args.model.startswith('ouro'):
        model.model.total_ut_steps = 4
        model.early_exit_step = model.early_exit_threshold = None
        blocks = list(model.model.layers)
        selected = [1, 12, 23, 34, 45] if len(blocks)==48 else [1, 6, 11, 16, 21]
        assert len(blocks) in [24,48]
        expected_calls = {str(i): 4 for i in range(len(blocks))}
        cache_factory = DynamicCache
        extras = {}
        noise = None
    else:
        blocks = list(model.transformer.prelude) + list(model.transformer.core_block) + list(model.transformer.coda)
        # Physical layers 1,2,4,6,7; retaining each selected layer's last invocation selects final recurrence.
        selected = [1, 2, 4, 6, 7]
        expected_calls = {str(i): (32 if 2 <= i < 6 else 1) for i in range(len(blocks))}
        module = importlib.import_module(type(model).__module__)
        cache_factory = module.HuginnDynamicCache
        extras = dict(num_steps=32, output_details=dict(return_logits=True, return_latents=False,
                                                       return_head=False, return_stats=False))
        gen = torch.Generator(device='cpu').manual_seed(0)
        noise = torch.empty((4096, model.config.n_embd), dtype=torch.float32)
        std = model.config.init_values['std']
        torch.nn.init.trunc_normal_(noise, mean=0., std=std, a=-3 * std, b=3 * std, generator=gen)
        noise = (noise * model.emb_scale).to(device='cuda', dtype=torch.bfloat16)
    for i, block in enumerate(blocks):
        def hook(module, inputs, output, index=i):
            key = str(index)
            records[key] = records.get(key, 0) + 1
            if index in selected:
                captured[index] = output.detach()
        handles.append(block.register_forward_hook(hook))

    def lengths(cache):
        if args.model.startswith('huginn'):
            return {str(k): len(v) for k, v in cache.key_cache.items()}
        return {str(i): int(layer.get_seq_length()) for i, layer in enumerate(cache.layers)}

    def safe_crop(cache, keep):
        if args.model.startswith('huginn'):
            cache.clear_last_k_entries(cache.get_seq_length() - keep)
        else:
            cache.crop(keep)
        assert all(v == keep for v in lengths(cache).values())
        assert cache.get_seq_length() == keep

    @torch.inference_mode()
    def forward(tokens, cache):
        records.clear()
        captured.clear()
        start = cache.get_seq_length()
        pos = torch.arange(start, start + tokens.shape[1], device='cuda')
        kw = dict(extras)
        if noise is not None:
            kw['input_states'] = noise[pos].unsqueeze(0)
        out = model(tokens, past_key_values=cache, use_cache=True, cache_position=pos, **kw)
        assert records == expected_calls, (records, expected_calls)
        assert len(captured) == 5
        assert out.past_key_values is cache, 'Target replaced the supplied cache'
        feature = torch.cat([captured[i] for i in selected], dim=-1)
        return out.logits, feature

    cache = cache_factory()
    logits, feat = forward(prefix, cache)
    report.update(prefill_logits_shape=list(logits.shape), feature_shape=list(feat.shape),
                  cache_class=type(cache).__name__, physical_layer_calls_per_forward=dict(records),
                  cache_slots=len(lengths(cache)), embedding_accessor='model.embed_tokens' if noise is None else 'transformer.wte')
    expected_slots = 4*len(blocks) if noise is None else 132
    assert len(lengths(cache)) == expected_slots
    assert all(v == 16 for v in lengths(cache).values())
    first, features = forward(suffix, cache)
    first, features = first.clone(), features.clone()
    report['after_block_cache_lengths'] = sorted(set(lengths(cache).values()))
    # Demonstrate whether inherited crop actually touches this checkpoint's own storage.
    cache.crop(16)
    report['native_crop_lengths'] = sorted(set(lengths(cache).values()))
    report['native_crop_effective'] = all(v == 16 for v in lengths(cache).values())
    safe_crop(cache, 16)
    repeat, repeat_features = forward(suffix, cache)
    report['rollback_repeat_logits_exact'] = torch.equal(first, repeat)
    report['rollback_repeat_features_exact'] = torch.equal(features, repeat_features)
    assert report['rollback_repeat_logits_exact'] and report['rollback_repeat_features_exact']
    safe_crop(cache, 16)
    changed = suffix.clone()
    changed[:, 4:] = (changed[:, 4:] + 7) % model.config.vocab_size
    other, other_features = forward(changed, cache)
    report['future_perturbation_prefix_logits_exact'] = torch.equal(first[:, :4], other[:, :4])
    report['future_perturbation_prefix_features_exact'] = torch.equal(features[:, :4], other_features[:, :4])
    assert report['future_perturbation_prefix_logits_exact'] and report['future_perturbation_prefix_features_exact']
    safe_crop(cache, 20)
    next_logits, _ = forward(suffix[:, 4:5], cache)
    report['single_token_after_partial_accept_shape'] = list(next_logits.shape)
    report['final_cache_lengths'] = sorted(set(lengths(cache).values()))
    assert all(v == 21 for v in lengths(cache).values())
    for h in handles:
        h.remove()
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['limitations'] = ['short fixed text only; not a trained DFlash pilot',
        'no speed or task score reported', 'no AR versus speculative output equivalence claim',
        'Huginn fixed noise table follows benchmark construction but uses capacity4096 for this diagnostic']


if __name__ == '__main__':
    main()
