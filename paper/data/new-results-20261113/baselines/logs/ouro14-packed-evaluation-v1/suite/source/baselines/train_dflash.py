"""Portable DFlash training with frozen targets, FP32 masters and resumable DDP."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import Qwen3Config
from safetensors.torch import save_file

from .common import (append, check_draft_config, environment, load_target, read_teachers,
                     save, sync, target_arguments, token_ids)
from .dflash.projected_draft import training_draft_class
from .vendor.dflash_loss import OnlineDFlashModel
from .vendor.dflash_train import DFlashDraftModel
from .vendor.optimizer import BF16Optimizer


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    target_arguments(p)
    p.add_argument('--teachers', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--stage', help='Split key in the optional manifest')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--draft-config', type=Path,
                   help='Defaults to Huginn 332.6M or target-width Ouro config')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--micro-batch', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--num-anchors', type=int, default=512)
    p.add_argument('--attention', choices=('sdpa', 'flex_attention'), default='flex_attention')
    p.add_argument('--checkpoint-layers', action='store_true')
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--resume', type=Path, help='New portable trainer checkpoint directory')
    p.add_argument('--max-updates', type=int, help='Bounded experiment; recorded as such')
    p.add_argument('--stop-after', type=int, help='Checkpoint and stop without changing the LR schedule')
    return p


def train(args, rank=0, world=1):
    if min(args.epochs, args.micro_batch, args.accumulation, args.num_anchors,
           args.save_every, args.max_seq) < 1:
        raise ValueError('Training dimensions and intervals must be positive')
    if any(v is not None and v < 1 for v in (args.max_updates, args.stop_after)):
        raise ValueError('Update limits must be positive')
    rows, sources = read_teachers(args.teachers, args.model, args.max_seq,
                                  args.manifest, args.stage)
    excluded = [r['source_id'] for r in rows if len(r['output_tokens']) < 32]
    rows = [r for r in rows if len(r['output_tokens']) >= 32]
    if not rows:
        raise ValueError('DFlash requires teacher continuations of at least 32 tokens')
    torch.manual_seed(42)
    target = load_target(args)
    try:
        return optimize(args, target, rows, sources, excluded, rank, world)
    finally:
        target.close()


def optimize(args, target, rows, sources, excluded, rank=0, world=1):
    """Kept separate so training/checkpoint tests can use a small frozen target."""
    for row in rows:
        token_ids(row['prompt_tokens'] + row['output_tokens'],
                  vocab_size=target.model.config.vocab_size)
    config_path = args.draft_config
    if config_path is None and target.family == 'huginn':
        config_path = Path(__file__).parent / 'configs/dflash-huginn-332m.json'
    cfg = (Qwen3Config.from_pretrained(config_path, local_files_only=True) if config_path
           else target.draft_config(0))
    check_draft_config(cfg, target)
    cfg._attn_implementation = args.attention
    cls = training_draft_class() if cfg.dflash_config.get('projection_adapter') else DFlashDraftModel
    draft = cls(cfg).to(device=target.device, dtype=target.dtype)
    if args.checkpoint_layers:
        draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    mask_id = cfg.dflash_config['mask_token_id']
    wrapper = OnlineDFlashModel(draft, target.head, target.embedding, mask_id, block_size=16,
                               attention_backend=args.attention, num_anchors=args.num_anchors,
                               loss_decay_gamma=7.)
    if any(p.requires_grad for p in target.model.parameters()):
        raise ValueError('Target parameters must remain frozen')
    model = torch.nn.parallel.DistributedDataParallel(
        wrapper, device_ids=[target.device.index], broadcast_buffers=False) if world > 1 else wrapper
    global_batch = world * args.micro_batch * args.accumulation
    updates_epoch = math.ceil(len(rows) / global_batch)
    total = args.epochs * updates_epoch
    if args.max_updates:
        total = min(total, args.max_updates)
    identity = dict(model=args.model, target_path=str(args.target_path.resolve()),
                    config=cfg.to_dict(), seed=42, epochs=args.epochs, max_seq=args.max_seq,
                    world_size=world, micro_batch=args.micro_batch, accumulation=args.accumulation,
                    examples=len(rows), sources=sources, excluded_under_32_tokens=excluded,
                    num_anchors=args.num_anchors, loss_decay_gamma=7., lr=6e-4,
                    warmup_ratio=.04, max_grad_norm=1., total_updates=total,
                    bounded_experiment=args.max_updates is not None, dtype=str(target.dtype),
                    gradient_checkpointing=args.checkpoint_layers, attention=args.attention,
                    draft_parameters=sum(p.numel() for p in draft.parameters()),
                    environment=environment())
    # Normalize torch dtype/config objects in the same way as the on-disk JSON.
    identity = json.loads(json.dumps(identity))
    if rank == 0:
        if (args.out / 'metadata.json').exists():
            if json.loads((args.out / 'metadata.json').read_text()) != identity:
                raise ValueError('Output directory belongs to a different training run')
        else:
            save(args.out / 'metadata.json', identity)
    optimizer = BF16Optimizer(draft, lr=6e-4, max_grad_norm=1., total_steps=total, warmup_ratio=.04)
    step = 0
    torch.manual_seed(42 + rank)
    if args.resume:
        state = torch.load(args.resume / 'training-state.pt', map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume recipe, input data, or source code changed')
        draft.load_state_dict(state['draft'])
        optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for master, saved in zip(optimizer.fp32_params, state['master']):
                master.copy_(saved)
        step = state['step']
        rng = torch.load(args.resume / f'rng-rank{rank}.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['cpu'])
        if target.device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda'], target.device)

    def checkpoint():
        dest = args.out / f'checkpoint-{step:06d}'
        dest.mkdir(exist_ok=True)
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(target.device)
                        if target.device.type == 'cuda' else None), dest / f'rng-rank{rank}.pt')
        if world > 1:
            dist.barrier()
        if rank == 0:
            # Save the ordinary HF checkpoint layout directly. This plain module
            # is not wrapped; HF's generic unwrap helper can import an unrelated
            # installed DeepSpeed and initialize its GPU kernels even on CPU.
            draft.config.architectures = [type(draft).__name__]
            draft.config.dtype = str(target.dtype).split('.')[-1]
            draft.config.save_pretrained(dest)
            save_file({k: v.detach().cpu().contiguous() for k, v in draft.state_dict().items()},
                      dest / 'model.safetensors', metadata={'format': 'pt'})
            torch.save(dict(identity=identity, step=step, draft=draft.state_dict(),
                            master=optimizer.fp32_params, optimizer=optimizer.state_dict()),
                       dest / 'training-state.pt')
            save(dest / 'complete.json', dict(step=step))
            save(args.out / 'latest.json', dict(checkpoint=str(dest.resolve()), step=step))
        if world > 1:
            dist.barrier()

    for epoch in range(args.epochs):
        indices = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        padding = updates_epoch * global_batch - len(indices)
        indices += (indices * math.ceil(padding / len(indices)))[:padding]
        for update in range(updates_epoch):
            if epoch * updates_epoch + update < step:
                continue
            if step >= total or (args.stop_after and step >= args.stop_after):
                break
            started = time.perf_counter()
            metrics = torch.zeros(2, device=target.device)
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
                features = target.forward(ids, features=True, use_cache=False).features
                ctx = model.no_sync() if world > 1 and micro + 1 < args.accumulation else contextlib.nullcontext()
                with ctx:
                    loss, accuracy = model(ids, features, loss_mask)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f'Nonfinite loss at step {step}')
                    (loss / args.accumulation).backward()
                metrics += torch.stack([loss.detach().float(), accuracy.detach().float()]) / args.accumulation
            # Check before updating, so a failed update never corrupts the last good state.
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in draft.parameters()):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            norm = optimizer.step()
            step += 1
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            sync(target.device)
            if rank == 0:
                report = dict(step=step, total_updates=total, loss=metrics[0].item(),
                              accuracy=metrics[1].item(), grad_norm=norm.item(),
                              lr=optimizer.get_learning_rate(), wall_s=time.perf_counter() - started)
                append(args.out / 'metrics.jsonl', report)
                print(json.dumps(report), flush=True)
            if step % args.save_every == 0 or step == total or step == args.stop_after:
                checkpoint()
        if step >= total or (args.stop_after and step >= args.stop_after):
            break
    if rank == 0:
        save(args.out / ('complete.json' if step == total else 'interrupted.json'),
             dict(step=step, total_updates=total, bounded_experiment=args.max_updates is not None))
    return draft


def main(argv=None):
    args = parser().parse_args(argv)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(1)
    if world > 1:
        local = int(os.environ['LOCAL_RANK'])
        args.device = f'cuda:{local}'
        torch.cuda.set_device(local)
        dist.init_process_group('nccl')
    try:
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=args.resume is not None)
        if world > 1:
            dist.barrier()
        train(args, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
