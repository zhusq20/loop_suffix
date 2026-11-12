"""Portable base-LAD training with held-out selection and explicit feature manifests."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

from .common import MODEL_IDS, append, environment, save, sync


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODEL_IDS, required=True)
    p.add_argument('--features', type=Path, required=True,
                   help='Root containing loopahead-{train,dev,confirmation}/MODEL/shard*')
    p.add_argument('--manifest', type=Path, help='Defaults to FEATURES/manifest.json')
    p.add_argument('--target-path', type=Path, help='Local target tokenizer/config for EOS resolution')
    p.add_argument('--stop-ids', type=int, nargs='*', help='Explicit EOS IDs instead of reading the tokenizer')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--lad-config', type=Path, help='Optional LADConfig JSON; context64/future7 are required')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--anchors-per-request', type=int, default=32)
    p.add_argument('--micro-batch', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--max-updates', type=int, help='Bounded experiment, explicitly recorded')
    p.add_argument('--stop-after', type=int, help='Checkpoint and stop while preserving the LR schedule')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.epochs, args.anchors_per_request, args.micro_batch, args.accumulation, args.eval_every) < 1:
        raise ValueError('Training dimensions and intervals must be positive')
    if any(v is not None and v < 1 for v in (args.max_updates, args.stop_after)):
        raise ValueError('Update limits must be positive')
    if args.stop_ids is None and args.target_path is None:
        raise ValueError('Provide --target-path or explicit --stop-ids')
    args.out.mkdir(parents=True, exist_ok=args.resume is not None)
    train(args)


def train(args):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file
    from .loopahead.lad import LADConfig, LookaheadDrafter, frequent_training_ids
    from .loopahead.training_data import FeatureStore
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    manifest = args.manifest or args.features/'manifest.json'
    stores = {stage: FeatureStore(args.features/f'loopahead-{stage}'/args.model, f'loopahead-{stage}', args.model, manifest)
              for stage in ['train', 'dev', 'confirmation']}
    if len({s.component_hash for s in stores.values()}) != 1:
        raise ValueError('Feature splits use different frozen target components')
    seen = set()
    for store in stores.values():
        ids = {r['source_id'] for r in store.rows}
        if seen & ids:
            raise ValueError('Train/dev/confirmation source IDs must be disjoint')
        seen.update(ids)
    components = {k:v.to(device=device, dtype=dtype) for k,v in load_file(stores['train'].components).items()}
    embedding = torch.nn.Embedding.from_pretrained(components['embedding'], freeze=True)
    head_weight, head_bias = components['head'], components.get('head_bias')
    config = LADConfig(**json.loads(args.lad_config.read_text())) if args.lad_config else LADConfig(target_hidden=head_weight.shape[1])
    if config.target_hidden != head_weight.shape[1] or config.context != 64 or config.train_future != 7:
        raise ValueError('LAD config must match target width, context64 and future7')
    model = LookaheadDrafter(config).to(device=device, dtype=dtype)
    params = list(model.parameters())
    master = [p.detach().float().clone().requires_grad_(True) for p in params]
    optimizer = torch.optim.AdamW(master, lr=2e-4, betas=(.9,.95), weight_decay=.01)
    frequent = frequent_training_ids(stores['train'].counts)
    frequent_gpu = torch.tensor(frequent, device=device)
    effective = args.micro_batch * args.accumulation
    if not stores['train'].eligible:
        raise ValueError('Training split has no real64-state contexts with future labels')
    updates_epoch = math.ceil(len(stores['train'].eligible)*args.anchors_per_request/effective)
    total = updates_epoch*args.epochs
    if args.max_updates:
        total = min(total, args.max_updates)
    identity = dict(model=args.model, config=config.to_dict(), seed=42, epochs=args.epochs, anchors_per_request_per_epoch=args.anchors_per_request,
        microbatch=args.micro_batch, accumulation=args.accumulation, effective_batch=effective, total_updates=total,
        dtype=args.dtype, bounded_experiment=args.max_updates is not None, environment=environment(),
        excluded_no_real_64_state_context=stores['train'].ineligible,
        anchor_sampling='without replacement if >=32 available, otherwise with replacement',
        final_batch='repeat shuffled epoch prefix to fill effective batch',
        optimizer='AdamW FP32 master', betas=[.9,.95], weight_decay=.01,
        peak_lr=2e-4, min_lr=2e-5, warmup_fraction=.05, max_grad_norm=1.,
        eval_interval=args.eval_every, eval_anchors_per_request=8, eval_future=3, eval_counts_anchor=True,
        dev_ineligible_policy='native-only accepted length0/1 at fixed fallback anchors; excluded from future CE',
        dev_anchor='full-vocabulary shallow argmax; prefix length0 when anchor differs from teacher',
        dev_ce='teacher-anchor future CE, used only as checkpoint tie-break',
        dev_vocabulary='training top8192 plus prompt/teacher-committed prefix/EOS, no future labels',
        feature_sources={k:s.provenance for k,s in stores.items()}, components_sha256=stores['train'].component_hash,
        torch=torch.__version__, parameters=sum(p.numel() for p in params))
    # Tokenizer access is optional when explicit stop IDs are supplied.
    if args.stop_ids is None:
        from .infer import tokenizer_and_stops
        _, eos = tokenizer_and_stops(args.model, args.target_path, 'raw', None)
    else:
        eos = set(args.stop_ids)
    if any(i < 0 or i >= head_weight.shape[0] for i in eos):
        raise ValueError('Invalid stop token IDs')
    identity['stop_ids'] = sorted(eos)
    if (args.out/'metadata.json').exists():
        assert json.loads((args.out/'metadata.json').read_text()) == identity
    else:
        save(args.out/'metadata.json', identity)
        save(args.out/'frequent-token-ids.json', frequent)
    eos_gpu = torch.tensor(sorted(eos), device=device, dtype=torch.long)
    eval_windows = {k:stores[k].anchors(8, 0, False) for k in ['dev','confirmation']}

    @torch.inference_mode()
    def evaluate(stage):
        model.eval()
        store = stores[stage]
        domain = defaultdict(lambda: dict(prefix_sum=0., windows=0, ce_sum=0., labels=0))
        for row in store.rows:
            if row['anchor_start'] == row['anchor_stop']:
                import hashlib
                key = f'42/short-dev/{row["source_id"]}'.encode()
                rng = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1))
                first, last = len(row['prompt_tokens']), len(row['tokens'])
                anchors = torch.randint(first, last, (8,), generator=rng)
                states = row['states'][anchors-1].to(device=device, dtype=dtype)
                native = F.linear(states, head_weight, head_bias).argmax(-1).cpu()
                truth = torch.tensor([row['tokens'][int(t)] for t in anchors])
                domain[row['pool']]['prefix_sum'] += int((native == truth).sum())
                domain[row['pool']]['windows'] += 8
        for offset in range(0, len(eval_windows[stage]), 8):
            batch = eval_windows[stage][offset:offset+8]
            context, anchors, labels, rows = store.batch(batch, device)
            context = context.to(dtype)
            logits = F.linear(model(context, embedding(anchors)), head_weight, head_bias).float()
            native_correct = F.linear(context[:, -1], head_weight, head_bias).argmax(-1) == anchors
            ce = F.cross_entropy(logits.flatten(0,1), labels.flatten(), ignore_index=-100, reduction='none').view_as(labels)
            for j, ((_, t), row) in enumerate(zip(batch, rows)):
                valid = torch.zeros(head_weight.shape[0], device=device, dtype=torch.bool)
                valid[frequent_gpu] = True
                valid[eos_gpu] = True
                valid[torch.tensor(row['tokens'][:t], device=device)] = True
                predicted = logits[j,:3].masked_fill(~valid, float('-inf')).argmax(-1)
                accepted = ((predicted == labels[j,:3]) & (labels[j,:3] != -100)).int().cumprod(-1).sum()
                entry = domain[row['pool']]
                entry['prefix_sum'] += (1 + int(accepted)) * int(native_correct[j])
                entry['windows'] += 1
                entry['ce_sum'] += float(ce[j].sum())
                entry['labels'] += int((labels[j] != -100).sum())
        metrics = {name: dict(accepted_prefix=d['prefix_sum']/d['windows'],
                    ce=d['ce_sum']/d['labels'] if d['labels'] else None, **d) for name,d in domain.items()}
        assert metrics and sum(d['windows'] for d in metrics.values()) == len(store.rows)*8
        result = dict(domains=metrics, accepted_prefix=sum(d['accepted_prefix'] for d in metrics.values())/len(metrics),
                      ce=sum(d['ce'] for d in metrics.values() if d['ce'] is not None)/max(1,sum(d['ce'] is not None for d in metrics.values())), requests=len(store.rows),
                      ineligible_requests=len(store.ineligible))
        model.train()
        return result

    step, best = 0, None
    if args.resume:
        saved = torch.load(args.resume/'training-state.pt', map_location='cpu', weights_only=False)
        assert saved['identity'] == identity
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        with torch.no_grad():
            for p,x in zip(master,saved['master']):
                p.copy_(x)
        step, best = saved['step'], saved['best']
        del saved
    warmup = max(1, int(.05*total))
    for epoch in range(args.epochs):
        windows = stores['train'].anchors(args.anchors_per_request, epoch, True)
        pad = updates_epoch*effective - len(windows)
        windows += (windows*math.ceil(pad/len(windows)))[:pad]
        for update in range(updates_epoch):
            if epoch*updates_epoch+update < step:
                continue
            if step >= total or (args.stop_after and step >= args.stop_after):
                break
            started = time.perf_counter()
            group = windows[update*effective:(update+1)*effective]
            denom = sum(min(7, len(stores['train'].rows[i]['tokens'])-t-1) for i,t in group)
            if step < warmup:
                lr = 2e-4*(step+1)/warmup
            else:
                phase = (step-warmup)/max(1,total-warmup-1)
                lr = 2e-5 + .5*(2e-4-2e-5)*(1+math.cos(math.pi*phase))
            optimizer.param_groups[0]['lr'] = lr
            loss_sum = torch.zeros((), device=device)
            model.train()
            for micro in range(args.accumulation):
                context, anchors, labels, _ = stores['train'].batch(group[micro*args.micro_batch:(micro+1)*args.micro_batch], device)
                context = context.to(dtype)
                logits = F.linear(model(context, embedding(anchors)), head_weight, head_bias).float()
                loss = F.cross_entropy(logits.flatten(0,1), labels.flatten(), ignore_index=-100, reduction='sum')
                (loss/denom).backward()
                loss_sum += loss.detach()
                del logits, context, anchors, labels, loss
            with torch.no_grad():
                for p,mp in zip(params,master):
                    mp.grad = p.grad.detach().float()
            norm = torch.nn.utils.clip_grad_norm_(master, 1.)
            if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(loss_sum)):
                raise FloatingPointError(f'Nonfinite optimization at {step}')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p,mp in zip(params,master):
                    p.copy_(mp.to(p.dtype))
                    p.grad = None
            step += 1
            if step == 1 or step % 25 == 0 or step == total:
                sync(device)
                metric = dict(step=step,total_updates=total,epoch=epoch,loss=float(loss_sum)/denom,
                    lr=lr,grad_norm=float(norm),update_wall_s=time.perf_counter()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,finished_unix=time.time())
                append(args.out/'metrics.jsonl',metric)
                save(args.out/'progress.json',metric)
                print(json.dumps(metric),flush=True)
            if step % args.eval_every == 0 or step == total or step == args.stop_after:
                # An interruption must not add a new checkpoint-selection trial.
                result = evaluate('dev') if step % args.eval_every == 0 or step == total else None
                if result is not None:
                    append(args.out/'dev-metrics.jsonl',dict(step=step,**result))
                dest = args.out/f'checkpoint-{step:06d}'
                dest.mkdir(exist_ok=True)
                model_state = {k:v.cpu() for k,v in model.state_dict().items()}
                torch.save(dict(config=config.to_dict(),model=model_state,frequent_ids=frequent,step=step),dest/'lad.pt')
                chosen = result is not None and (best is None or
                    (result['accepted_prefix'],-result['ce']) > (best['accepted_prefix'],-best['ce']))
                if chosen:
                    best = dict(step=step,checkpoint=str(dest.resolve()),**result)
                    save(args.out/'best.json',best)
                torch.save(dict(identity=identity,step=step,best=best,model=model_state,
                    master=[p.detach().cpu() for p in master],optimizer=optimizer.state_dict()),dest/'training-state.pt')
                save(dest/'complete.json',dict(step=step,dev=result,finished_unix=time.time()))
                save(args.out/'latest.json',dict(step=step,checkpoint=str(dest.resolve())))
                if result is not None:
                    print('dev',step,json.dumps(result),flush=True)
        if step >= total or (args.stop_after and step >= args.stop_after):
            break
    if step < total:
        save(args.out/'interrupted.json', dict(step=step, total_updates=total))
        return
    assert step == total and best is not None
    selected = torch.load(Path(best['checkpoint'])/'lad.pt',map_location=device,weights_only=True)
    model.load_state_dict(selected['model'])
    # Confirmation is evaluated exactly once after selection and never feeds back.
    if (args.out/'confirmation.json').exists():
        raise RuntimeError('Confirmation has already been consumed; inspect completion state')
    confirmation = evaluate('confirmation')
    save(args.out/'confirmation.json',dict(selected_step=best['step'],**confirmation))
    save(args.out/'complete.json',dict(step=step,selected_checkpoint=best['checkpoint'],
        bounded_experiment=args.max_updates is not None,
        dev_accepted_prefix=best['accepted_prefix'],confirmation_accepted_prefix=confirmation['accepted_prefix'],
        finished_unix=time.time()))


if __name__ == '__main__':
    main()
