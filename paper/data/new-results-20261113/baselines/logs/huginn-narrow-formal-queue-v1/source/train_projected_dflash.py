"""Projected narrow Huginn DFlash; official SpecForge blocks/loss/optimizer.

Derived from train_dflash.py; the original full-width trainer remains unchanged.

No target/code/teacher answers are finetuned. This launcher consumes audited token IDs
directly, uses DDP instead of the official SGLang+FSDP launcher, and saves FP32 master
parameters as well as optimizer/RNG state for faithful resumption.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def append(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_teachers(args):
    rows, sources = {}, []
    for path in sorted(args.teachers.glob('shard*/teachers.jsonl')):
        meta = json.loads(path.with_name('metadata.json').read_text())
        assert meta['identity']['model'] == args.model
        if not args.smoke:
            complete = json.loads(path.with_name('complete.json').read_text())
            assert sha(path) == complete['teachers_sha256']
        sources.append(dict(path=str(path.resolve()), sha256=sha(path), metadata=meta['identity']))
        for line in path.open():
            row = json.loads(line)
            assert row['source_id'] not in rows
            assert 0 < len(row['prompt_tokens']) <= 2560
            assert 0 < len(row['output_tokens']) <= 512
            rows[row['source_id']] = row
    manifest = json.loads((BASE / 'data/manifests/manifest.json').read_text())
    manifest_name = sources[0]['metadata']['manifest_sha256']
    assert all(s['metadata']['manifest_sha256'] == manifest_name for s in sources)
    if not args.smoke:
        first = sources[0]['metadata']
        assert len(sources) == first['shards']
        assert {s['metadata']['shard'] for s in sources} == set(range(first['shards']))
        assert all(s['metadata']['batch_plan_sha256'] == first['batch_plan_sha256'] for s in sources)
    expected = next(v for v in manifest['files'].values() if v['sha256'] == manifest_name)
    manifest_path = next(BASE / 'data/manifests' / (k + '.jsonl') for k, v in manifest['files'].items()
                         if v['sha256'] == manifest_name)
    assert sha(manifest_path) == manifest_name
    allowed = {json.loads(line)['source_id'] for line in manifest_path.open()}
    assert set(rows) <= allowed
    if not args.smoke:
        assert len(rows) == expected['count'] and set(rows) == allowed
    if args.selection_manifest:
        assert args.selection_manifest.resolve() == (BASE/'data/manifests/dflash-pilot.jsonl').resolve()
        assert sha(args.selection_manifest) == manifest['files']['dflash-pilot']['sha256']
        selected = {json.loads(line)['source_id'] for line in args.selection_manifest.open()}
        assert len(selected) == 2000 and selected <= set(rows)
        rows = {key:row for key,row in rows.items() if key in selected}
    excluded = [s for s, row in rows.items() if len(row['output_tokens']) < 32]
    excluded_ids = set(excluded)
    rows = [row for s, row in sorted(rows.items()) if s not in excluded_ids]
    if args.smoke:
        # Exercise the longest currently available examples in a bounded dry run.
        rows = sorted(rows, key=lambda r: -len(r['prompt_tokens']) - len(r['output_tokens']))[:32]
    assert rows
    return rows, sources, excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b'], required=True)
    p.add_argument('--teachers', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--cpu-groups', required=True, help='one lo-hi group per local rank, comma separated')
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--checkpoint-layers', action='store_true')
    p.add_argument('--selection-manifest', type=Path, help='Only the frozen2000-request pilot subset is accepted')
    p.add_argument('--projected-config', type=Path, default=BASE/'configs/dflash-huginn-narrow-2048-v1.json')
    args = p.parse_args()
    assert args.model == 'huginn-3.5b'
    from gpu_allocation import allowed_gpus
    allowed_uuids = {int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert set(visible) <= {allowed_uuids[g] for g in allowed_gpus()}
    rank, world, local = (int(os.environ.get(k, d)) for k, d in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    groups = args.cpu_groups.split(',')
    assert len(groups) == world
    lo, hi = map(int, groups[local].split('-'))
    os.sched_setaffinity(0, set(range(lo, hi + 1)))
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    deadline = time.monotonic() + 60
    while True:
        try:
            torch.cuda.set_device(local)
            lease = torch.empty(1, device=torch.device('cuda',local))
            break
        except torch.AcceleratorError as exc:
            if 'busy or unavailable' not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(.2)
    uuid = str(torch.cuda.get_device_properties(local).uuid)
    if not uuid.startswith('GPU-'):
        uuid = 'GPU-' + uuid
    args.out.mkdir(parents=True, exist_ok=True)
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started, status = time.time(), 'failed'
        try:
            if world > 1:
                dist.init_process_group('nccl', device_id=torch.device('cuda', local))
            train(args, rank, world, local)
            status = 'complete'
        except Exception:
            save(args.out / f'failure-rank{rank}.json', dict(error=traceback.format_exc(), time_unix=time.time()))
            raise
        finally:
            append(args.out / f'reservation-cost-rank{rank}.jsonl', dict(started_unix=started, finished_unix=time.time(),
                reserved_gpu_seconds=time.time() - started, gpu_uuid=uuid, status=status))
            if dist.is_initialized():
                dist.destroy_process_group()


def train(args, rank, world, local):
    import torch
    import transformers
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from specforge.core.dflash import OnlineDFlashModel
    from dflash_impl.projected_draft import training_draft_class
    from transformers import Qwen3Config
    from specforge.optimizer import BF16Optimizer
    from dflash_impl.target import LoopTarget
    sys.path.insert(0, str(ROOT / 'results/loopfill-latency512-20261107/source/src'))
    from llm_memory_editability.loop_model_registry import MODELS

    if rank == 0:
        from provenance import freeze_sources
        freeze_sources(args.out, [Path(__file__), BASE/'dflash_impl/target.py',
            BASE/'dflash_impl/projected_draft.py', BASE/'gpu_allocation.py', BASE/'provenance.py', args.projected_config])

    rows, sources, excluded = read_teachers(args)
    spec = MODELS[args.model]
    protocol = json.loads((ROOT / 'results/loopfill-latency512-20261107/protocol.json').read_text())
    torch.manual_seed(42)
    target = LoopTarget.load(protocol['models'][args.model]['path'], spec.backend, 3072,
                             device=torch.device('cuda', local))
    mask_id = 0 if spec.backend == 'ouro' else 65513
    cfg = Qwen3Config.from_pretrained(args.projected_config, local_files_only=True)
    assert cfg.hidden_size == 2048 and cfg.intermediate_size == 5632 and cfg.num_hidden_layers == 5
    assert cfg.num_attention_heads == cfg.num_key_value_heads == 16 and cfg.head_dim == 128
    assert cfg.dflash_config['target_hidden_size'] == target.hidden_size == 5280
    assert cfg.dflash_config['target_layer_ids'] == target.expanded_ids
    assert cfg.dflash_config['mask_token_id'] == mask_id and cfg.block_size == 16
    cfg._attn_implementation = 'flex_attention'
    draft = training_draft_class()(cfg).to(device=target.device, dtype=torch.bfloat16)
    if args.checkpoint_layers:
        draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapper = OnlineDFlashModel(draft, target.head, target.embedding, mask_id,
                               block_size=16, attention_backend='flex_attention',
                               num_anchors=512, loss_decay_gamma=7.)
    assert all(not p.requires_grad for p in target.model.parameters())
    model = DistributedDataParallel(wrapper, device_ids=[local], broadcast_buffers=False) if world > 1 else wrapper
    global_batch = world * args.micro_batch * args.accumulation
    updates_epoch = math.ceil(len(rows) / global_batch)
    total_updates = args.epochs * updates_epoch
    if args.smoke:
        total_updates = 1
    identity = dict(model=args.model, revision=spec.revision, seed=42, epochs=args.epochs,
        draft_architecture='target-width-projections-v1', target_hidden_size=target.hidden_size,
        draft_hidden_size=cfg.hidden_size, draft_intermediate_size=cfg.intermediate_size,
        draft_config_sha256=sha(args.projected_config),
        selection_manifest=str(args.selection_manifest) if args.selection_manifest else None,
        selection_manifest_sha256=sha(args.selection_manifest) if args.selection_manifest else None,
        training_scope='pilot2000' if args.selection_manifest else 'full50000',
        world_size=world, micro_batch=args.micro_batch, accumulation=args.accumulation, global_batch=global_batch,
        examples=len(rows), excluded_under_32_tokens=excluded, sources=sources, max_seq=3072,
        num_anchors=512, block_size=16, loss_decay_gamma=7., lr=6e-4, warmup_ratio=.04, max_grad_norm=1.,
        weight_decay=0., adam_betas=[.9, .999], total_updates=total_updates,
        target_features=target.expanded_ids, mask_token_id=mask_id,
        draft_parameters=sum(p.numel() for p in draft.parameters()), smoke=args.smoke,
        torch=torch.__version__, transformers=transformers.__version__, gradient_checkpointing=args.checkpoint_layers,
        distributed='DDP', final_global_batch='deterministically pad from shuffled epoch start',
        target_generation_software='recorded in teacher sources; target features recomputed in training environment',
        source_sha256=sha(Path(__file__)))
    if rank == 0:
        metadata = args.out / 'metadata.json'
        if metadata.exists():
            assert json.loads(metadata.read_text()) == identity
        else:
            save(metadata, identity)
        print(json.dumps({k: v for k, v in identity.items() if k not in ['sources', 'excluded_under_32_tokens']}), flush=True)
    optimizer = BF16Optimizer(draft, lr=6e-4, weight_decay=0., max_grad_norm=1.,
                               total_steps=total_updates, warmup_ratio=.04)
    step = 0
    torch.manual_seed(42 + rank)
    if args.resume:
        state = torch.load(args.resume / 'training-state.pt', map_location='cpu', weights_only=False)
        assert state['identity'] == identity
        draft.load_state_dict(state['draft'])
        optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for mp, saved in zip(optimizer.fp32_params, state['master']):
                mp.copy_(saved)
        step = state['step']
        rng = torch.load(args.resume / f'rng-rank{rank}.pt', map_location='cpu', weights_only=False)
        torch.set_rng_state(rng['cpu'])
        torch.cuda.set_rng_state(rng['cuda'])
        del state, rng

    def checkpoint():
        dest = args.out / f'checkpoint-{step:06d}'
        dest.mkdir(exist_ok=True)
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()), dest / f'rng-rank{rank}.pt')
        if world > 1:
            dist.barrier()
        if rank == 0:
            draft.save_pretrained(dest)
            state = dict(identity=identity, step=step, draft={k: v.cpu() for k, v in draft.state_dict().items()},
                master=[p.detach().cpu() for p in optimizer.fp32_params], optimizer=optimizer.state_dict())
            torch.save(state, dest / 'training-state.pt')
            save(dest / 'complete.json', dict(step=step, finished_unix=time.time()))
            save(args.out / 'latest.json', dict(checkpoint=str(dest.resolve()), step=step))
            del state
        if world > 1:
            dist.barrier()

    for epoch in range(args.epochs):
        indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        padding = updates_epoch * global_batch - len(indices)
        indices += (indices * math.ceil(padding / len(indices)))[:padding]
        for update in range(updates_epoch):
            absolute_step = epoch * updates_epoch + update
            if absolute_step < step:
                continue
            if step >= total_updates:
                break
            started = time.perf_counter()
            metrics = torch.zeros(2, device=target.device, dtype=torch.float32)
            for micro in range(args.accumulation):
                offset = update * global_batch + micro * world * args.micro_batch + rank * args.micro_batch
                chosen = [rows[i] for i in indices[offset:offset + args.micro_batch]]
                width = max(len(r['prompt_tokens']) + len(r['output_tokens']) for r in chosen)
                ids = torch.full((len(chosen), width), mask_id, device=target.device, dtype=torch.long)
                loss_mask = torch.zeros_like(ids, dtype=torch.float32)
                for i, row in enumerate(chosen):
                    prompt, output = row['prompt_tokens'], row['output_tokens']
                    ids[i, :len(prompt) + len(output)] = torch.tensor(prompt + output, device=target.device)
                    loss_mask[i, len(prompt):len(prompt) + len(output)] = 1.
                # Right padding cannot influence real prefix states in a causal target.
                # No inference_mode: the trainable fc backward must save these features.
                target_output = target.forward(ids, features=True, use_cache=False)
                features = target_output.features
                del target_output
                sync = model.no_sync() if world > 1 and micro + 1 < args.accumulation else contextlib.nullcontext()
                with sync:
                    loss, accuracy = model(ids, features, loss_mask)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite loss at update {absolute_step}, micro {micro}')
                    (loss / args.accumulation).backward()
                metrics += torch.stack([loss.detach().float(), accuracy.detach().float()]) / args.accumulation
                del features, loss, accuracy, ids, loss_mask
            grad_norm = optimizer.step()
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f'Nonfinite gradient at update {absolute_step}')
            step += 1
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            torch.cuda.synchronize()
            if rank == 0:
                report = dict(step=step, total_updates=total_updates, epoch=epoch, loss=metrics[0].item(),
                    accuracy=metrics[1].item(), lr=optimizer.get_learning_rate(), grad_norm=grad_norm.item(),
                    update_wall_s=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    finished_unix=time.time())
                append(args.out / 'metrics.jsonl', report)
                save(args.out / 'progress.json', report)
                print(json.dumps(report), flush=True)
            if step % args.save_every == 0 or step == total_updates:
                checkpoint()
        if step >= total_updates:
            break
    assert step == total_updates
    target.close()
    if rank == 0:
        save(args.out / 'complete.json', dict(step=step, examples=len(rows), epochs=args.epochs,
            smoke=args.smoke, finished_unix=time.time(), limitation='dry run only' if args.smoke else None))


if __name__ == '__main__':
    main()
