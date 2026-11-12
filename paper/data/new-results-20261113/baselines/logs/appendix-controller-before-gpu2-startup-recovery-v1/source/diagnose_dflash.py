"""Replay every greedy first divergence on its common prefix in BF16 and FP32."""
import argparse
import fcntl
import gc
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


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.open()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ar', type=Path, required=True)
    p.add_argument('--dflash', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert 0 <= args.shard < args.shards
    args.out.mkdir(parents=True, exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'provenance.py',
        *sorted((BASE/'dflash_impl').glob('*.py')),
        BASE/'third_party/dflash/dflash/model.py'])
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (line.split(',') for line in lines.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
        assert uuid not in busy
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        started = time.time()
        try:
            run(args)
        except Exception:
            save(args.out/'failure.json', dict(error=traceback.format_exc()))
            raise
        finally:
            save(args.out/'reservation-cost.json', dict(gpu_uuid=uuid, started_unix=started,
                finished_unix=time.time(), reserved_gpu_seconds=time.time()-started))


def run(args):
    import torch
    from transformers import AutoTokenizer
    from dflash_impl.target import LoopTarget
    from dflash_impl.decode import generate
    suite = ROOT/'results/loopfill-latency512-20261107'
    sys.path.insert(0, str(suite/'source/src'))
    sys.path.insert(0, str(BASE/'third_party/dflash'))
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt
    from dflash.model import DFlashDraftModel
    meta = json.loads((args.dflash/'metadata.json').read_text())
    ar_meta = json.loads((args.ar/'metadata.json').read_text())
    for key in ['model', 'model_revision', 'dataset', 'input_sha256', 'max_seq',
                'max_new_tokens', 'stop_audit', 'temperature', 'dtype']:
        assert meta[key] == ar_meta[key], key
    assert meta['temperature'] == 0 and meta['dtype'] == 'bfloat16'
    assert meta['method'] == 'dflash' and ar_meta['method'] == 'ar'
    for folder in [args.ar, args.dflash]:
        marker = json.loads((folder/'complete.json').read_text())
        assert digest(folder/'requests.jsonl') == marker['requests_sha256']
        assert len(rows(folder/'requests.jsonl')) == marker['requests']
    for name, sha in meta['checkpoint_sha256'].items():
        assert digest(Path(meta['checkpoint'])/name) == sha
    for rel, sha in json.loads((args.dflash/'source-manifest.json').read_text()).items():
        assert digest(BASE/rel) == sha, f'Live DFlash source changed: {rel}'
    spec = MODELS[meta['model']]
    protocol = json.loads((suite/'protocol.json').read_text())
    for rel, sha in protocol['source_sha256'].items():
        assert digest(suite/'source'/rel) == sha
    assert spec.revision == meta['model_revision']
    assert digest(suite/f'inputs/{meta["dataset"]}.jsonl') == meta['input_sha256']
    path = protocol['models'][meta['model']]['path']
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    prompts = {r['source_id']: tokenize_prompt(spec, tok, r['prompt'], 'chat') for r in rows(suite/f'inputs/{meta["dataset"]}.jsonl')}
    cases = []
    for ar, df in zip(rows(args.ar/'requests.jsonl'), rows(args.dflash/'requests.jsonl'), strict=True):
        assert ar['source_id'] == df['source_id']
        assert ar['prompt_sha256'] == df['prompt_sha256']
        if ar['tokens'] != df['tokens']:
            i = next((i for i, (a, b) in enumerate(zip(ar['tokens'], df['tokens'])) if a != b), None)
            assert i is not None, 'Length-only divergence requires separate EOS diagnosis'
            cases.append(dict(source_id=ar['source_id'], position=i, ar_token=ar['tokens'][i], dflash_token=df['tokens'][i],
                common_output=ar['tokens'][:i], expected_dflash=df['tokens'], seed=df['stats']['seed']))
    cases = cases[args.shard::args.shards]
    provenance = dict(model=meta['model'], dataset=meta['dataset'], shard=args.shard, shards=args.shards,
        checkpoint_sha256=meta['checkpoint_sha256'],
        source_journals={label:dict(path=str(folder.resolve()), sha256=digest(folder/'requests.jsonl'))
                         for label,folder in [('ar',args.ar),('dflash',args.dflash)]})
    save(args.out/'metadata.json', provenance)
    if not cases:
        save(args.out/'diagnosis.json', [])
        save(args.out/'complete.json', dict(cases=0, all_bf16_divergences_reproduced=True,
            all_fp32_argmax_agree=True, finished_unix=time.time()))
        return
    torch.set_num_threads(1)
    torch.manual_seed(0)
    target = LoopTarget.load(path, spec.backend, meta['max_seq'])
    draft = DFlashDraftModel.from_pretrained(meta['checkpoint'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).to(target.device).eval()
    for case in cases:
        trace = []
        prompt = prompts[case['source_id']]
        output = generate(target, torch.tensor([prompt], device=target.device), 512, meta['stop_audit']['stop_ids'],
            draft=draft, seed=case['seed'], trace=trace, diagnostic_stop_after=case['position']+1)
        assert output.tokens == case['expected_dflash'][:len(output.tokens)], 'Instrumented replay changed tokens'
        case['trace'] = trace
        del case['expected_dflash']
        print('Traced', case['source_id'], case['position'], flush=True)
    target.close()
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()

    @torch.inference_mode()
    def replay(target, case, mode):
        prompt = prompts[case['source_id']]
        cache = target.new_cache()
        prefill = target.forward(torch.tensor([prompt], device=target.device), cache, features=False)
        if case['position'] == 0:
            return prefill.logits[0, -1].float().cpu()
        del prefill
        if mode == 'ar':
            result = None
            for token in case['common_output']:
                result = target.forward(torch.tensor([[token]], device=target.device), cache, features=False)
            logits = result.logits[0, -1].float().cpu()
            del result
            return logits
        for entry in case['trace']:
            assert cache.get_seq_length() == entry['cache_start']
            result = target.forward(torch.tensor([entry['block']], device=target.device), cache, features=False)
            local = case['position'] - entry['anchor_output_index'] - 1
            if 0 <= local < len(entry['emitted']):
                return result.logits[0, local].float().cpu()
            target.crop(cache, entry['cache_start'] + len(entry['emitted']))
            del result
        raise RuntimeError('Trace did not cover the first divergence')

    for label, dtype in [('bf16', torch.bfloat16), ('fp32', torch.float32)]:
        target = LoopTarget.load(path, spec.backend, meta['max_seq'], dtype=dtype)
        for case in cases:
            a = replay(target, case, 'ar')
            d = replay(target, case, 'dflash')
            def describe(logits):
                values, ids = logits.topk(3)
                p = logits.softmax(-1)
                return dict(top_ids=ids.tolist(), top_logits=values.tolist(), top_probs=p[ids].tolist(),
                    top2_probability_margin=float(p[ids[0]]-p[ids[1]]),
                    top1_margin=float(values[0]-values[1]), observed_ar_logit=float(logits[case['ar_token']]),
                    observed_dflash_logit=float(logits[case['dflash_token']]))
            case[label] = dict(ar=describe(a), dflash=describe(d), argmax_equal=int(a.argmax()) == int(d.argmax()),
                max_abs_logit_difference=float((a-d).abs().max()))
            if label == 'bf16':
                assert int(a.argmax()) == case['ar_token'] and int(d.argmax()) == case['dflash_token'], 'Replay failed to reproduce actual divergence'
            save(args.out/'progress.json', cases)
            print(label, case['source_id'], json.dumps(case[label]), flush=True)
        target.close()
        del target
        gc.collect()
        torch.cuda.empty_cache()
    save(args.out/'diagnosis.json', cases)
    save(args.out/'complete.json', dict(cases=len(cases), all_bf16_divergences_reproduced=True,
        all_fp32_argmax_agree=all(c['fp32']['argmax_equal'] for c in cases), finished_unix=time.time()))


if __name__ == '__main__':
    main()
