"""Replay packed DFlash greedy divergences on the observed common prefix."""
import argparse
import fcntl
import gc
import os
from pathlib import Path
import sys
import time
import traceback

from packed_dflash_evidence import (check_frozen, check_plan, check_run, compare,
                                    read, save, sha)


def run(args):
    plan = check_plan(args.plan)
    suite = Path(plan['suite'])
    job = next(j for j in plan['jobs'] if j['id'] == args.job)
    assert job['temperature'] == 0
    protocol, inputs = check_frozen(suite, job, weights=True)
    ar = check_run(args.ar, suite, job, stage='ar')
    binding = ar['metadata']['timed_run']
    method = check_run(binding['path'], suite, job)
    assert method['hashes'] == binding['hashes']
    agreement = compare(method['rows'], ar['rows'])
    prompts = {row['source_id']: row['prompt_tokens'] for row in inputs}
    cases = []
    for a, d in zip(ar['rows'], method['rows'], strict=True):
        if a['tokens'] == d['tokens']:
            continue
        position = next((i for i, (x, y) in enumerate(zip(a['tokens'], d['tokens'])) if x != y), None)
        assert position is not None, 'Length-only disagreement requires EOS investigation'
        cases.append(dict(source_id=a['source_id'], position=position, ar_token=a['tokens'][position],
            dflash_token=d['tokens'][position], common_output=a['tokens'][:position],
            expected_dflash=d['tokens'], seed=d['stats']['seed']))
    metadata = dict(job=job, model=job['model'], dataset=job['dataset'],
        target_backend='PackedDFlashTarget', checkpoint_sha256=method['metadata']['checkpoint_sha256'],
        source_sha256=protocol['source_sha256'], diagnostic_source_sha256=sha(Path(__file__)),
        evidence_source_sha256=sha(Path(__file__).with_name('packed_dflash_evidence.py')),
        original_timed_exit=binding['exit_evidence'], reportable=False, max_new_tokens=512,
        source_journals={name: dict(path=r['path'], sha256=r['hashes']['requests.jsonl'])
                         for name, r in [('ar', ar), ('dflash', method)]})
    save(args.out / 'metadata.json', metadata)
    save(args.out / 'agreement.json', agreement)
    if cases:
        source = suite / 'source'
        sys.path[:0] = [str(source), str(source / 'src')]
        import torch
        from baselines.dflash.decode import generate
        from baselines.dflash.target import LoopTarget
        from baselines.dflash.packed_target import PackedDFlashTarget
        from baselines.infer import load_dflash
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.cuda.set_device(0)
        model = protocol['models'][job['model']]

        def load(dtype):
            native = LoopTarget.load(Path(model['path']), model['family'], job['max_seq'], dtype=dtype)
            return native, PackedDFlashTarget(native)

        native, target = load(torch.bfloat16)
        draft = load_dflash(Path(model['checkpoint']), native)
        for case in cases:
            trace = []
            output = generate(target, torch.tensor([prompts[case['source_id']]], device=target.device),
                512, set(model['stop_ids']), draft=draft, seed=case['seed'], trace=trace,
                diagnostic_stop_after=case['position'] + 1)
            assert output.tokens == case['expected_dflash'][:len(output.tokens)], 'Instrumented trace changed tokens'
            case['trace'] = trace
            del case['expected_dflash']
            save(args.out / 'trace-progress.json', cases)
        target.close()
        native.close()
        del target, native, draft
        gc.collect()
        torch.cuda.empty_cache()

        @torch.inference_mode()
        def replay(target, case, mode):
            cache = target.new_cache()
            result = target.forward(torch.tensor([prompts[case['source_id']]], device=target.device), cache, features=False)
            if case['position'] == 0:
                return result.logits[0, -1].float().cpu()
            del result
            if mode == 'ar':
                for token in case['common_output']:
                    result = target.forward(torch.tensor([[token]], device=target.device), cache, features=False)
                return result.logits[0, -1].float().cpu()
            for entry in case['trace']:
                assert cache.get_seq_length() == entry['cache_start']
                result = target.forward(torch.tensor([entry['block']], device=target.device), cache, features=False)
                local = case['position'] - entry['anchor_output_index'] - 1
                if 0 <= local < len(entry['emitted']):
                    return result.logits[0, local].float().cpu()
                target.crop(cache, entry['cache_start'] + len(entry['emitted']))
                del result
            raise RuntimeError('Block trace did not cover first divergence')

        def describe(logits, case):
            values, ids = logits.topk(3)
            p = logits.softmax(-1)
            return dict(top_ids=ids.tolist(), top_logits=values.tolist(), top_probs=p[ids].tolist(),
                top2_probability_margin=float(p[ids[0]] - p[ids[1]]),
                top1_margin=float(values[0] - values[1]), observed_ar_logit=float(logits[case['ar_token']]),
                observed_dflash_logit=float(logits[case['dflash_token']]))

        for label, dtype in [('bf16', torch.bfloat16), ('fp32', torch.float32)]:
            native, target = load(dtype)
            for case in cases:
                a, d = replay(target, case, 'ar'), replay(target, case, 'dflash')
                case[label] = dict(ar=describe(a, case), dflash=describe(d, case),
                    argmax_equal=int(a.argmax()) == int(d.argmax()),
                    max_abs_logit_difference=float((a - d).abs().max()))
                if label == 'bf16':
                    assert int(a.argmax()) == case['ar_token'] and int(d.argmax()) == case['dflash_token'], \
                        'Replay did not reproduce both original greedy choices'
                save(args.out / 'progress.json', cases)
            target.close()
            native.close()
            del target, native
            gc.collect()
            torch.cuda.empty_cache()
    save(args.out / 'diagnosis.json', cases)
    save(args.out / 'complete.json', dict(cases=len(cases),
        all_bf16_divergences_reproduced=True,
        all_fp32_argmax_agree=all(c['fp32']['argmax_equal'] for c in cases),
        diagnosis_sha256=sha(args.out / 'diagnosis.json'), finished_unix=time.time(),
        limitation='Common-prefix numerical evidence; original BF16 whole-sequence disagreements remain reported.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--ar', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    os.fstat(args.lock_fd)
    fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run(args)
    except BaseException:
        save(args.out / 'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        raise
    finally:
        save(args.out / 'reservation-cost.json', dict(gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
            started_unix=started, finished_unix=time.time(), reserved_gpu_seconds=time.time() - started))


if __name__ == '__main__':
    main()
