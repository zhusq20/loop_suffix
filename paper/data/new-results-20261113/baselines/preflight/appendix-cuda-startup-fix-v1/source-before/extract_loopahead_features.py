"""Persist actual shallow target readout states from frozen teacher token journals."""
import argparse
from collections import Counter
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
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b'], required=True)
    p.add_argument('--teacher-shard', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=args.resume)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'dflash_impl/target.py', BASE/'provenance.py'])
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (line.split(',') for line in lines.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
        assert uuid not in busy
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        started, status = time.time(), 'failed'
        try:
            run(args)
            status = 'complete'
        except Exception:
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
            raise
        finally:
            with (args.out/'reservation-cost.jsonl').open('a') as f:
                f.write(json.dumps(dict(gpu_uuid=uuid, started_unix=started, finished_unix=time.time(),
                    reserved_gpu_seconds=time.time()-started, status=status))+'\n')


def run(args):
    import torch
    import transformers
    from safetensors.torch import save_file
    from dflash_impl.target import LoopTarget
    sys.path.insert(0, str(ROOT/'results/loopfill-latency512-20261107/source/src'))
    from llm_memory_editability.loop_model_registry import MODELS
    spec = MODELS[args.model]
    protocol = json.loads((ROOT/'results/loopfill-latency512-20261107/protocol.json').read_text())
    meta = json.loads((args.teacher_shard/'metadata.json').read_text())
    complete = json.loads((args.teacher_shard/'complete.json').read_text())
    assert meta['identity']['model'] == args.model and meta['identity']['model_revision'] == spec.revision
    teacher_path = args.teacher_shard/'teachers.jsonl'
    assert digest(teacher_path) == complete['teachers_sha256']
    rows = [json.loads(line) for line in teacher_path.open()]
    assert len(rows) == complete['requests']
    rows.sort(key=lambda r: (len(r['prompt_tokens']) + len(r['output_tokens']), r['source_id']))
    depth = 1 if spec.backend == 'ouro' else 4
    identity = dict(model=args.model, revision=spec.revision, depth=depth, max_seq=3072,
        teacher_sha256=complete['teachers_sha256'], teacher_identity=meta['identity'],
        dtype='bfloat16', torch=torch.__version__, transformers=transformers.__version__,
        feature='LM head input at d1, full real causal teacher prefix, independent shallow coda for Huginn',
        chunk_size=64, batch_size=8, source_sha256=digest(Path(__file__)), adapter_sha256=digest(BASE/'dflash_impl/target.py'))
    if (args.out/'metadata.json').exists():
        assert json.loads((args.out/'metadata.json').read_text()) == identity
    else:
        save(args.out/'metadata.json', identity)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    target = LoopTarget.load(protocol['models'][args.model]['path'], spec.backend, 3072)
    components = args.out/'frozen-components.safetensors'
    if not components.exists():
        weights = {'embedding': target.embedding.weight.detach().cpu().contiguous().clone(),
                   'head': target.head.weight.detach().cpu().contiguous().clone()}
        if getattr(target.head, 'bias', None) is not None:
            weights['head_bias'] = target.head.bias.detach().cpu().contiguous().clone()
        save_file(weights, components)
        del weights
    index, counts = [], Counter()
    for chunk_id, offset in enumerate(range(0, len(rows), 64)):
        batch_rows = rows[offset:offset+64]
        dest = args.out/f'chunk-{chunk_id:05d}.pt'
        info_path = dest.with_suffix('.json')
        if info_path.exists():
            info = json.loads(info_path.read_text())
            assert digest(dest) == info['sha256']
            assert info['source_ids'] == [r['source_id'] for r in batch_rows]
            index.append(info)
        else:
            chunk = []
            started = time.perf_counter()
            for begin in range(0, len(batch_rows), 8):
                micro = batch_rows[begin:begin+8]
                lengths = [len(r['prompt_tokens']) + len(r['output_tokens']) for r in micro]
                ids = torch.zeros((len(micro), max(lengths)), device=target.device, dtype=torch.long)
                for i, row in enumerate(micro):
                    ids[i, :lengths[i]] = torch.tensor(row['prompt_tokens']+row['output_tokens'], device=target.device)
                features = target.shallow_readout_states(ids, depth)
                for i, row in enumerate(micro):
                    first, last = max(64, len(row['prompt_tokens'])), lengths[i]-1
                    chunk.append(dict(source_id=row['source_id'], pool=row['pool'],
                        prompt_tokens=row['prompt_tokens'], output_tokens=row['output_tokens'],
                        states=features[i, :lengths[i]].cpu().contiguous().clone(),
                        anchor_start=first, anchor_stop=max(first, last)))
                del features, ids
            temp = dest.with_suffix('.pt.tmp')
            torch.save(chunk, temp)
            temp.replace(dest)
            info = dict(file=dest.name, sha256=digest(dest), requests=len(chunk),
                source_ids=[r['source_id'] for r in chunk], lengths=[r['states'].shape[0] for r in chunk],
                eligible_anchors=[r['anchor_stop']-r['anchor_start'] for r in chunk],
                wall_s=time.perf_counter()-started, bytes=dest.stat().st_size)
            save(info_path, info)
            index.append(info)
            del chunk
        for row in batch_rows:
            counts.update(row['output_tokens'])
        save(args.out/'progress.json', dict(requests=sum(c['requests'] for c in index), planned=len(rows)))
        print('features', offset+len(batch_rows), '/', len(rows), flush=True)
    target.close()
    save(args.out/'index.json', dict(identity=identity, chunks=index, output_token_counts=dict(counts),
        frozen_components_sha256=digest(components), finished_unix=time.time()))
    save(args.out/'complete.json', dict(requests=len(rows), bytes=sum(c['bytes'] for c in index),
        no_eligible_anchors=sum(n == 0 for c in index for n in c['eligible_anchors']),
        index_sha256=digest(args.out/'index.json'), finished_unix=time.time()))


if __name__ == '__main__':
    main()
