"""Paper base-LAD optimization and held-out selection, with frozen target heads."""
import argparse
from collections import defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

BASE = Path(__file__).resolve().parent


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def append(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj)+'\n')
        f.flush()
        os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b'], required=True)
    p.add_argument('--features', type=Path, required=True, help='contains loopahead-train/dev/confirmation/model/shard*')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.resume is not None)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'loopahead/lad.py', BASE/'loopahead/training_data.py', BASE/'provenance.py',
        BASE/'gpu_startup.py', BASE/'gpu_allocation.py'])
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i,u in (line.split(',') for line in lines.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        started, status = time.time(), 'failed'
        try:
            from gpu_startup import acquire_cuda_context
            lease, acquisition = acquire_cuda_context(args.gpu, uuid)
            save(args.out/'cuda-acquisition.json', acquisition)
            started = acquisition['acquired_unix']
            train(args)
            status = 'complete'
        except Exception:
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
            raise
        finally:
            append(args.out/'reservation-cost.jsonl', dict(gpu_uuid=uuid, started_unix=started,
                finished_unix=time.time(), reserved_gpu_seconds=time.time()-started, status=status))


def train(args):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file
    from loopahead.lad import LADConfig, LookaheadDrafter, frequent_training_ids
    from loopahead.training_data import FeatureStore
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device('cuda')
    stores = {stage: FeatureStore(args.features/f'loopahead-{stage}'/args.model, f'loopahead-{stage}', args.model)
              for stage in ['train', 'dev', 'confirmation']}
    assert len({s.component_hash for s in stores.values()}) == 1
    components = load_file(stores['train'].components, device='cuda')
    embedding = torch.nn.Embedding.from_pretrained(components['embedding'], freeze=True)
    head_weight, head_bias = components['head'], components.get('head_bias')
    config = LADConfig(target_hidden=head_weight.shape[1])
    model = LookaheadDrafter(config).to(device=device, dtype=torch.bfloat16)
    params = list(model.parameters())
    master = [p.detach().float().clone().requires_grad_(True) for p in params]
    optimizer = torch.optim.AdamW(master, lr=2e-4, betas=(.9,.95), weight_decay=.01)
    frequent = frequent_training_ids(stores['train'].counts)
    frequent_gpu = torch.tensor(frequent, device=device)
    updates_epoch = math.ceil(len(stores['train'].eligible)*32/64)
    total = updates_epoch*3
    identity = dict(model=args.model, config=config.to_dict(), seed=42, epochs=3, anchors_per_request_per_epoch=32,
        microbatch=8, accumulation=8, effective_batch=64, total_updates=total,
        excluded_no_real_64_state_context=stores['train'].ineligible,
        anchor_sampling='without replacement if >=32 available, otherwise with replacement',
        final_batch='repeat shuffled epoch prefix to fill64 windows',
        optimizer='AdamW FP32 master with BF16 model', betas=[.9,.95], weight_decay=.01,
        peak_lr=2e-4, min_lr=2e-5, warmup_fraction=.05, max_grad_norm=1.,
        eval_interval=500, eval_anchors_per_request=8, eval_future=3, eval_counts_anchor=True,
        dev_ineligible_policy='native-only accepted length0/1 at fixed fallback anchors; excluded from future CE',
        dev_anchor='full-vocabulary shallow argmax; prefix length0 when anchor differs from teacher',
        dev_ce='teacher-anchor future CE, used only as checkpoint tie-break',
        dev_vocabulary='training top8192 plus prompt/teacher-committed prefix/EOS, no future labels',
        feature_sources={k:s.provenance for k,s in stores.items()}, components_sha256=stores['train'].component_hash,
        torch=torch.__version__, parameters=sum(p.numel() for p in params))
    if (args.out/'metadata.json').exists():
        assert json.loads((args.out/'metadata.json').read_text()) == identity
    else:
        save(args.out/'metadata.json', identity)
        save(args.out/'frequent-token-ids.json', frequent)
    # Stop IDs are included in the vocabulary even before they occur in a prefix.
    # Use the same pinned stop resolver as inference, rather than a hand-written list.
    import sys
    root = BASE.parents[1]
    sys.path.insert(0, str(root/'results/loopfill-latency512-20261107/source/src'))
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids
    from transformers import AutoConfig, AutoTokenizer
    protocol = json.loads((root/'results/loopfill-latency512-20261107/protocol.json').read_text())
    target_path = Path(protocol['models'][args.model]['path'])
    target_cfg = AutoConfig.from_pretrained(target_path, trust_remote_code=True, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True, local_files_only=True)
    generation = json.loads((target_path/'generation_config.json').read_text()) if (target_path/'generation_config.json').exists() else {}
    eos, _ = resolve_stop_ids(MODELS[args.model], tok, target_cfg.eos_token_id, generation.get('eos_token_id'),
                              target_cfg.vocab_size, 'completion')
    eos_gpu = torch.tensor(sorted(eos), device=device)
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
                states = row['states'][anchors-1].to(device)
                native = F.linear(states, head_weight, head_bias).argmax(-1).cpu()
                truth = torch.tensor([row['tokens'][int(t)] for t in anchors])
                domain[row['pool']]['prefix_sum'] += int((native == truth).sum())
                domain[row['pool']]['windows'] += 8
        for offset in range(0, len(eval_windows[stage]), 8):
            batch = eval_windows[stage][offset:offset+8]
            context, anchors, labels, rows = store.batch(batch, device)
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
        assert len(metrics) == 5 and sum(d['windows'] for d in metrics.values()) == len(store.rows)*8
        result = dict(domains=metrics, accepted_prefix=sum(d['accepted_prefix'] for d in metrics.values())/5,
                      ce=sum(d['ce'] for d in metrics.values())/5, requests=len(store.rows),
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
            for p,x in zip(master,saved['master']): p.copy_(x)
        step, best = saved['step'], saved['best']
        del saved
    warmup = max(1, int(.05*total))
    for epoch in range(3):
        windows = stores['train'].anchors(32, epoch, True)
        pad = updates_epoch*64 - len(windows)
        windows += (windows*math.ceil(pad/len(windows)))[:pad]
        for update in range(updates_epoch):
            if epoch*updates_epoch+update < step:
                continue
            started = time.perf_counter()
            group = windows[update*64:(update+1)*64]
            denom = sum(min(7, len(stores['train'].rows[i]['tokens'])-t-1) for i,t in group)
            if step < warmup:
                lr = 2e-4*(step+1)/warmup
            else:
                phase = (step-warmup)/max(1,total-warmup-1)
                lr = 2e-5 + .5*(2e-4-2e-5)*(1+math.cos(math.pi*phase))
            optimizer.param_groups[0]['lr'] = lr
            loss_sum = torch.zeros((), device=device)
            model.train()
            for micro in range(8):
                context, anchors, labels, _ = stores['train'].batch(group[micro*8:(micro+1)*8], device)
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
                torch.cuda.synchronize()
                metric = dict(step=step,total_updates=total,epoch=epoch,loss=float(loss_sum)/denom,
                    lr=lr,grad_norm=float(norm),update_wall_s=time.perf_counter()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_unix=time.time())
                append(args.out/'metrics.jsonl',metric)
                save(args.out/'progress.json',metric)
                print(json.dumps(metric),flush=True)
            if step % 500 == 0 or step == total:
                result = evaluate('dev')
                append(args.out/'dev-metrics.jsonl',dict(step=step,**result))
                dest = args.out/f'checkpoint-{step:06d}'
                dest.mkdir(exist_ok=True)
                model_state = {k:v.cpu() for k,v in model.state_dict().items()}
                torch.save(dict(config=config.to_dict(),model=model_state,frequent_ids=frequent,step=step),dest/'lad.pt')
                chosen = best is None or (result['accepted_prefix'],-result['ce']) > (best['accepted_prefix'],-best['ce'])
                if chosen:
                    best = dict(step=step,checkpoint=str(dest.resolve()),**result)
                    save(args.out/'best.json',best)
                torch.save(dict(identity=identity,step=step,best=best,model=model_state,
                    master=[p.detach().cpu() for p in master],optimizer=optimizer.state_dict()),dest/'training-state.pt')
                save(dest/'complete.json',dict(step=step,dev=result,finished_unix=time.time()))
                save(args.out/'latest.json',dict(step=step,checkpoint=str(dest.resolve())))
                print('dev',step,json.dumps(result),flush=True)
    assert step == total and best is not None
    selected = torch.load(Path(best['checkpoint'])/'lad.pt',map_location=device,weights_only=True)
    model.load_state_dict(selected['model'])
    # Confirmation is evaluated exactly once after selection and never feeds back.
    if (args.out/'confirmation.json').exists():
        raise RuntimeError('Confirmation has already been consumed; inspect completion state')
    confirmation = evaluate('confirmation')
    save(args.out/'confirmation.json',dict(selected_step=best['step'],**confirmation))
    save(args.out/'complete.json',dict(step=step,selected_checkpoint=best['checkpoint'],
        dev_accepted_prefix=best['accepted_prefix'],confirmation_accepted_prefix=confirmation['accepted_prefix'],
        finished_unix=time.time()))


if __name__ == '__main__':
    main()
