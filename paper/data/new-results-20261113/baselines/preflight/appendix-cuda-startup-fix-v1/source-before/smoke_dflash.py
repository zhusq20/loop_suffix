"""Untrained official drafter compatibility run; never a reportable baseline."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (r.split(',') for r in rows.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if any(r.split(',')[0].strip() == uuid for r in occupied.splitlines()):
            raise RuntimeError('GPU occupied')
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi + 1)))
        report = dict(model=args.model, gpu_uuid=uuid, stage='untrained_drafter_compatibility', started_unix=time.time())
        try:
            run(args, report)
            report['status'] = 'smoke_complete_not_trained_pilot'
        except Exception:
            report['status'], report['error'] = 'failed', traceback.format_exc()
            raise
        finally:
            report['finished_unix'] = time.time()
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)


def run(args, report):
    import torch
    import transformers
    from transformers import AutoTokenizer
    from dflash_impl.decode import generate
    from dflash_impl.target import LoopTarget
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base / 'third_party/dflash'))
    from dflash.model import DFlashDraftModel
    protocol = json.loads((base.parents[1] / 'results/loopfill-latency512-20261107/protocol.json').read_text())
    path = protocol['models'][args.model]['path']
    family = 'ouro' if args.model.startswith('ouro') else 'huginn'
    target = LoopTarget.load(path, family, 4096)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    prompt = 'Write a short Python function that adds two integers.\n'
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(target.device)
    mask = 0 if family == 'ouro' else 65513
    config = target.draft_config(mask)
    config.save_pretrained(args.out / 'draft-config')
    with torch.device(target.device):
        draft = DFlashDraftModel(config).to(dtype=torch.bfloat16).eval()
    report.update(torch=torch.__version__, transformers=transformers.__version__,
                  target_path=path, draft_parameters=sum(p.numel() for p in draft.parameters()),
                  mask_token_id=mask, mask_token=tok.convert_ids_to_tokens(mask),
                  features=target.expanded_ids, seed=0, prompt=prompt)
    stop = [tok.eos_token_id]
    outputs = {}
    for name, d, temp in [('ar', None, 0.), ('dflash', draft, 0.),
                          ('dflash_repeat', draft, 0.), ('dflash_sampling', draft, 1.)]:
        print('Running', args.model, name, flush=True)
        output = generate(target, ids, 32, stop, draft=d, temperature=temp)
        record = dict(tokens=output.tokens, stats=output.stats)
        outputs[name] = record
        (args.out / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    report['repeat_exact'] = outputs['dflash']['tokens'] == outputs['dflash_repeat']['tokens']
    assert report['repeat_exact']
    a, b = outputs['ar']['tokens'], outputs['dflash']['tokens']
    report['greedy_ar_equal'] = a == b
    report['first_divergence'] = next((dict(position=i, ar=x, dflash=y) for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                      None if len(a) == len(b) else dict(position=min(len(a), len(b)), length_difference=True))
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['limitations'] = ['random drafter; no training or quality claim', 'one short synthetic prompt only',
                            'wall times are diagnostic and must not enter paper tables',
                            'greedy differences, if any, require common-prefix numerical diagnosis']
    target.close()


if __name__ == '__main__':
    main()
