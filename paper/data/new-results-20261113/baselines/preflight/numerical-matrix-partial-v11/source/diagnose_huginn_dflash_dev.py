"""Replay the frozen max128 Huginn width comparison; never a formal table cell."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from evaluate_dflash import digest, save
from gpu_allocation import allowed_gpus
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE = BASE/'preflight/huginn-dflash-dev-compare-v1'
REPORT_SHA = 'c3cbc9734de017f011f1458651c1b55471f1747cb6c8cb07acb02cdce00d5b0e'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.open()]


def prepare(label):
    assert label in {'narrow', 'wide'}
    source = SOURCE
    meta, done, report = (read(source/name) for name in ['metadata.json', 'complete.json', 'report.json'])
    assert digest(source/'report.json') == done['report_sha256'] == REPORT_SHA
    assert digest(source/'metadata.json') == report['metadata_sha256']
    plan = meta['plan']
    assert plan['model'] == 'huginn-3.5b' and plan['stage'] == 'heldout_feasibility'
    assert plan['count'] == done['requests'] == 20 and not plan['reportable']
    assert plan['max_new_tokens'] == 128 and plan['temperature'] == 0
    assert not plan['graph'] and not plan['compile'] and meta['dtype'] == 'bfloat16'
    assert meta['max_seq'] == 3072 and meta['target_depth'] == 32
    assert meta['target_features'] == [1, 126, 128, 130, 131]
    manifest = read(source/'source-manifest.json')
    assert manifest == meta['source_sha256']
    from dflash_source_identity import OLD_ENTRY, CONTEXT_ENTRY, NEW_ENTRY, without_main
    for relative, expected in manifest.items():
        archived, live = source/'source'/relative, BASE/relative
        assert digest(archived) == expected, f'Changed archived development source: {relative}'
        if relative == 'configs/gpu-allocation.json':
            historical = read(archived)
            assert historical['allowed_gpus'] == [2, 3, 4, 6]
            assert historical['released_gpus'] == [0, 1, 5, 7]
        elif relative == 'evaluate_dflash.py':
            assert expected in {OLD_ENTRY, CONTEXT_ENTRY, NEW_ENTRY}
            assert digest(live) == NEW_ENTRY and without_main(live) == without_main(archived)
        else:
            assert digest(live) == expected, f'Changed development implementation: {relative}'
    plan_path = BASE/'preflight/huginn-dflash-dev-plan-v1/plan.json'
    assert read(plan_path) == plan and digest(plan_path) == meta['plan_sha256']
    from compare_huginn_dflash_dev import selected_rows
    selected, current_plan = selected_rows()
    assert current_plan == plan
    assert digest(source/'inputs.jsonl') == meta['inputs_sha256']
    inputs = rows(source/'inputs.jsonl')
    assert [r['source_id'] for r in inputs] == [r['source_id'] for r in selected]
    for stored, expected in zip(inputs, plan['rows'], strict=True):
        assert all(stored[k] == v for k, v in expected.items())
        assert all(type(t) is int and 0 <= t < 65536 for t in stored['prompt_tokens'])
        assert len(stored['prompt_tokens']) + 128 + 32 + 4 <= meta['max_seq']
    data = {}
    assert set(done['journals']) == {'ar', 'narrow', 'wide', 'narrow_profile', 'wide_profile'}
    for name, expected in done['journals'].items():
        path = source/f'{name}.jsonl'
        assert digest(path) == expected
        data[name] = rows(path)
        assert [r['source_id'] for r in data[name]] == [r['source_id'] for r in inputs]
        for record in data[name]:
            assert 1 <= len(record['tokens']) <= 128
            assert record['stats']['decode_tokens'] == len(record['tokens']) - 1
            assert record['stats']['temperature'] == 0 and record['stats']['seed'] == 0
    for name in ['narrow', 'wide']:
        assert [r['tokens'] for r in data[name]] == [r['tokens'] for r in data[name+'_profile']]
        checkpoint = meta['checkpoints'][name]
        path = Path(checkpoint['path'])
        assert read(path/'complete.json')['step'] == checkpoint['step'] == (60 if name == 'narrow' else 4000)
        assert checkpoint['parameters'] == (332621056 if name == 'narrow' else 2116288320)
        assert not any(checkpoint['loading'].values())
        assert digest(path/'config.json') == checkpoint['config_sha256']
        assert digest(path.parent/'metadata.json') == checkpoint['training_metadata_sha256']
        for filename, expected in checkpoint['hashes'].items():
            assert digest(path/filename) == expected
    cases = []
    for ar, method in zip(data['ar'], data[label], strict=True):
        if ar['tokens'] == method['tokens']:
            continue
        position = next((i for i, (a, d) in enumerate(zip(ar['tokens'], method['tokens'])) if a != d), None)
        assert position is not None, 'Length-only development divergence requires separate diagnosis'
        cases.append(dict(source_id=ar['source_id'], position=position, ar_token=ar['tokens'][position],
            dflash_token=method['tokens'][position], common_output=ar['tokens'][:position],
            expected_dflash=method['tokens']))
    assert [{k: c[k] for k in ['source_id', 'position', 'ar_token', 'dflash_token']} for c in cases] == report['comparisons'][label]['first_differences']
    assert len(cases) == (8 if label == 'narrow' else 7)
    launch, finish = (read(BASE/f'logs/huginn-dflash-dev-compare-v1-{suffix}.json') for suffix in ['launcher', 'finished'])
    assert finish['exit_code'] == 0 and all(finish[k] == v for k, v in launch.items())
    assert finish['argv'][finish['argv'].index('--out')+1] == str(source)
    identity = dict(model='huginn-3.5b', dataset='heldout-huginn-dflash-dev20', label=label,
        development_only=True, max_new_tokens=128, count=20, cases=len(cases),
        source_report_sha256=REPORT_SHA, source_metadata_sha256=digest(source/'metadata.json'),
        inputs_sha256=meta['inputs_sha256'], plan_sha256=meta['plan_sha256'],
        checkpoint=meta['checkpoints'][label],
        source_journals={name: dict(path=str(source/f'{name}.jsonl'), sha256=done['journals'][name]) for name in ['ar', label]})
    return meta, inputs, cases, identity


def run(args, prepared):
    import torch
    import transformers
    from dflash_impl.target import LoopTarget
    from dflash_impl.decode import generate
    from dflash_impl.projected_draft import inference_draft_class
    meta, inputs, cases, identity = prepared
    assert str(torch.__version__) == meta['torch'] and transformers.__version__ == meta['transformers']
    suite = ROOT/'results/loopfill-latency512-20261107'
    protocol = read(suite/'protocol.json')
    for relative, expected in protocol['source_sha256'].items():
        assert digest(suite/'source'/relative) == expected
    assert protocol['models']['huginn-3.5b']['revision'] == meta['model_revision']
    sys.path[:0] = [str(suite/'source/src'), str(BASE/'third_party/dflash')]
    from llm_memory_editability.loop_model_registry import MODELS
    from dflash.model import DFlashDraftModel
    spec, model_path = MODELS['huginn-3.5b'], protocol['models']['huginn-3.5b']['path']
    prompts = {r['source_id']: r['prompt_tokens'] for r in inputs}
    stops = meta['stop_audit']['stop_ids']
    torch.set_num_threads(1)
    torch.manual_seed(0)
    save(args.out/'metadata.json', identity)
    target = LoopTarget.load(model_path, spec.backend, meta['max_seq'])
    drafts = {'ar': None}
    for label, cls in [('narrow', inference_draft_class()), ('wide', DFlashDraftModel)]:
        draft, info = cls.from_pretrained(meta['checkpoints'][label]['path'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
        assert not any(info.values())
        drafts[label] = draft.to(target.device).eval()
    # Preserve the original two resident drafters and its warmup order.
    for draft in drafts.values():
        for row in inputs[:2]:
            generate(target, torch.tensor([row['prompt_tokens']], device=target.device), 32,
                     stops, draft=draft, temperature=0., seed=0)
    for case in cases:
        trace = []
        result = generate(target, torch.tensor([prompts[case['source_id']]], device=target.device),
            128, stops, draft=drafts[args.label], temperature=0., seed=0, trace=trace,
            diagnostic_stop_after=case['position']+1)
        assert len(result.tokens) >= case['position']+1
        assert result.tokens == case['expected_dflash'][:len(result.tokens)], 'Development trace changed the actual prefix'
        case['trace'] = trace
        del case['expected_dflash']
    target.close()
    del target, drafts, draft, result
    gc.collect()
    torch.cuda.empty_cache()

    @torch.inference_mode()
    def replay(target, case, mode, check_bf16):
        cache = target.new_cache()
        result = target.forward(torch.tensor([prompts[case['source_id']]], device=target.device), cache, features=False)
        if case['position'] == 0:
            return result.logits[0, -1].float().cpu()
        if mode == 'ar':
            for token in case['common_output']:
                if check_bf16:
                    assert int(result.logits[0, -1].argmax()) == token, 'AR prefix was not reproduced'
                result = target.forward(torch.tensor([[token]], device=target.device), cache, features=False)
            return result.logits[0, -1].float().cpu()
        del result
        for entry in case['trace']:
            assert cache.get_seq_length() == entry['cache_start']
            result = target.forward(torch.tensor([entry['block']], device=target.device), cache, features=False)
            local = case['position'] - entry['anchor_output_index'] - 1
            if 0 <= local < len(entry['emitted']):
                return result.logits[0, local].float().cpu()
            target.crop(cache, entry['cache_start']+len(entry['emitted']))
            del result
        raise RuntimeError('Development trace did not cover its first divergence')

    def describe(logits):
        assert torch.isfinite(logits).all()
        values, ids = logits.topk(3)
        probs = logits.softmax(-1)
        return dict(top_ids=ids.tolist(), top_logits=values.tolist(), top_probs=probs[ids].tolist(),
                    top2_probability_margin=float(probs[ids[0]]-probs[ids[1]]))

    for precision, dtype in [('bf16', torch.bfloat16), ('fp32', torch.float32)]:
        target = LoopTarget.load(model_path, spec.backend, meta['max_seq'], dtype=dtype)
        for case in cases:
            a = replay(target, case, 'ar', precision == 'bf16')
            d = replay(target, case, 'dflash', False)
            if precision == 'bf16':
                assert int(a.argmax()) == case['ar_token'] and int(d.argmax()) == case['dflash_token']
                case['both_bf16_prefixes_reproduced'] = True
            case[precision] = dict(ar=describe(a), dflash=describe(d),
                argmax_equal=int(a.argmax()) == int(d.argmax()), max_abs_logit_difference=float((a-d).abs().max()))
            save(args.out/'progress.json', cases)
            print(precision, case['source_id'], case['position'], flush=True)
        target.close()
        del target
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(identity, status='complete', results=cases,
        all_bf16_prefixes_reproduced=all(c['both_bf16_prefixes_reproduced'] for c in cases),
        all_fp32_argmax_agree=all(c['fp32']['argmax_equal'] for c in cases), finished_unix=time.time(),
        limitations=['Frozen20/max128 development comparison only; not a formal benchmark cell.',
                    'Narrow pilot60 and wide4000 have unequal training budgets.',
                    'FP32 uses the observed common prefixes; it is not a full FP32 benchmark.'])
    save(args.out/'report.json', report)
    save(args.out/'complete.json', dict(cases=len(cases), report_sha256=digest(args.out/'report.json'), finished_unix=time.time()))


def validate(label, folder):
    _, _, cases, identity = prepare(label)
    report, done = read(folder/'report.json'), read(folder/'complete.json')
    assert report['status'] == 'complete' and all(report[k] == v for k, v in identity.items())
    assert digest(folder/'report.json') == done['report_sha256'] and done['cases'] == len(cases)
    assert report['all_bf16_prefixes_reproduced'] and report['all_fp32_argmax_agree']
    assert [(c['source_id'], c['position']) for c in report['results']] == [(c['source_id'], c['position']) for c in cases]
    assert all(c['both_bf16_prefixes_reproduced'] and c['fp32']['argmax_equal'] for c in report['results'])
    assert all('top2_probability_margin' in c[p][side] for c in report['results'] for p in ['bf16', 'fp32'] for side in ['ar', 'dflash'])
    name = f'formal-greedy-dev-huginn-dflash-{label}-v1'
    launch, finish = (read(BASE/f'logs/{name}-{suffix}.json') for suffix in ['launcher', 'finished'])
    assert finish['exit_code'] == 0 and all(finish[k] == v for k, v in launch.items())
    assert finish['argv'][finish['argv'].index('--out')+1] == str(folder)
    assert report['finished_unix'] <= done['finished_unix'] <= finish['finished_unix']
    for relative, expected in read(folder/'source-manifest.json').items():
        assert digest(folder/'source'/relative) == digest(BASE/relative) == expected
    return dict(model='huginn-3.5b', benchmark=identity['dataset'], method='dflash', label=label,
        development_only=True, cases=len(cases), folder=str(folder), result_sha256=digest(folder/'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', choices=['narrow', 'wide'], required=True)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cpus')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    prepared = prepare(args.label)
    if args.plan:
        print(json.dumps(prepared[3], indent=2))
        return
    assert args.gpu in allowed_gpus() and args.cpus and args.out
    uuids = {int(i): u.strip() for i, u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
    uuid = uuids[args.gpu]
    args.out.mkdir(parents=True, exist_ok=False)
    freeze_sources(args.out, [Path(__file__), BASE/'gpu_startup.py', BASE/'gpu_allocation.py',
        BASE/'provenance.py', BASE/'dflash_source_identity.py', BASE/'dflash_impl/target.py',
        BASE/'dflash_impl/decode.py', BASE/'dflash_impl/projected_draft.py', BASE/'third_party/dflash/dflash/model.py'])
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        started = time.time()
        try:
            from gpu_startup import acquire_cuda_context
            lease, acquisition = acquire_cuda_context(args.gpu, uuid)
            save(args.out/'cuda-acquisition.json', acquisition)
            started = acquisition['acquired_unix']
            run(args, prepared)
        except Exception:
            save(args.out/'failure.json', dict(error=traceback.format_exc(), finished_unix=time.time()))
            raise
        finally:
            save(args.out/'reservation-cost.json', dict(gpu_uuid=uuid, started_unix=started,
                finished_unix=time.time(), reserved_gpu_seconds=time.time()-started))


if __name__ == '__main__':
    main()
