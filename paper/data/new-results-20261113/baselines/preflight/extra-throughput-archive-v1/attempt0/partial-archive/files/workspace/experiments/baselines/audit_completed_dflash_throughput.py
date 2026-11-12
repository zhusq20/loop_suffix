"""Independently verify all completed packed DFlash throughput measurements."""
import argparse
import csv
import math
from pathlib import Path
import sys
import time

from packed_dflash_evidence import read, rows, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    suite = args.suite.resolve()
    protocol = read(suite/'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert sha(suite/'source'/name) == expected, name
    for name, expected in protocol['reference_sha256'].items():
        assert sha(suite/name) == expected, name
    for model in protocol['models'].values():
        for name, expected in model['checkpoint_sha256'].items():
            assert sha(Path(model['checkpoint'])/name) == expected, name
    sys.path.insert(0, str(suite/'source/scripts'))
    from loop_dflash_benchmark_suite import audit
    table = list(csv.DictReader((suite/'measurements.csv').open()))
    historical = list(csv.DictReader((suite/'references/table2-measurements.csv').open()))
    dispatch = read(suite/'dispatch.json')
    jobs = [j for j in read(suite/'jobs.json') if j['kind'] == 'throughput']
    assert len(jobs) == 10
    assert {(j['model'],j['batch']) for j in jobs} == {(m,b) for m in protocol['models'] for b in [1,2,4,8,16]}
    results, evidence = [], []
    for job in jobs:
        state = dispatch[job['id']]
        assert state['outcome'] == 'complete' and state['exit_code'] == 0
        out = suite/state['out']
        summary = audit(suite,job,out)
        meta, done = read(out/'metadata.json'), read(out/'complete.json')
        receipt_path = suite/f"logs/{job['id']}-attempt{state['attempt']}-exit.json"
        receipt = read(receipt_path)
        assert receipt['exit_code'] == 0
        assert all(receipt[k] == state[k] for k in ['pid','argv','started_unix'])
        assert receipt['finished_unix'] >= done['finished_unix']
        script = suite/'source/scripts/loop_dflash_benchmark_worker.py'
        assert str(script) in receipt['argv']
        assert len(meta['cpu_affinity']) == 4 and meta['target_dtype'] == 'torch.bfloat16'
        assert not meta['graph'] and not meta['compile'] and meta['reportable']
        assert meta['checkpoint_sha256'] == protocol['models'][job['model']]['checkpoint_sha256']
        assert job['max_new_tokens'] == 1024 and job['temperature'] == 0
        assert sha(suite/job['inputs']) == job['inputs_sha256']
        inputs, records = rows(suite/job['inputs']), rows(out/'requests.jsonl')
        assert len(records) == summary['requests'] == job['count'] == 480
        assert {r['request_id'] for r in records} == {r['source_id'] for r in inputs}
        assert len({r['request_id'] for r in records}) == 480
        for record in records:
            assert 1 <= len(record['tokens']) <= 1024
            assert receipt['started_unix'] <= record['finished_unix'] <= done['finished_unix'] <= receipt['finished_unix']
        assert summary['concurrency'] == job['batch'] and summary['schedule'] == 'continuous'
        assert summary['input_tokens'] == summary['prefill_chunk_tokens'] == sum(len(r['prompt_tokens']) for r in inputs)
        assert not summary['cuda_graph'] and not summary['pad_decode_rows'] and summary['core_padding_rows'] == 0
        assert math.isfinite(summary['wall_s']) and summary['wall_s'] > 0
        assert summary['output_tput'] == summary['output_tokens']/summary['wall_s']
        assert summary['output_tokens'] == sum(len(r['tokens']) for r in records)
        assert summary['gamma'] == sum(len(r['tokens'])-1 for r in records)/sum(r['stats']['target_verifications'] for r in records)
        if job['batch'] > 1:
            assert summary['max_verify_batch'] > 1 and summary['max_draft_batch'] > 1 and summary['refilled_requests'] > 0
        reported = next(r for r in table if r['id'] == job['id'])
        assert float(reported['output_tput']) == summary['output_tput']
        old_ar = next(r for r in historical if r['model'] == job['model'] and int(r['batch']) == job['batch'] and r['arm'] == 'ar')
        old_lf = next(r for r in historical if r['model'] == job['model'] and int(r['batch']) == job['batch'] and r['arm'] == 'budget-online')
        assert float(reported['historical_ar_output_tput']) == float(old_ar['output_tput'])
        assert float(reported['throughput_ratio_vs_historical_ar']) == summary['output_tput']/float(old_ar['output_tput'])
        result = dict(method='dflash', model=job['model'], batch=job['batch'], requests=480,
            output_tokens=summary['output_tokens'], wall_s=summary['wall_s'], output_tok_s=summary['output_tput'],
            gamma=summary['gamma'], historical_ar_tok_s=float(old_ar['output_tput']),
            historical_loopfill_tok_s=float(old_lf['output_tput']),
            ratio_to_historical_ar=summary['output_tput']/float(old_ar['output_tput']),
            ratio_to_historical_loopfill=summary['output_tput']/float(old_lf['output_tput']),
            run=str(out), actual_exit_code=0)
        results.append(result)
        evidence.append(dict(id=job['id'],artifacts={str(p):sha(p) for p in
            [out/'metadata.json',out/'summary.json',out/'complete.json',out/'requests.jsonl',receipt_path]},
            checkpoint_sha256=meta['checkpoint_sha256'],actual_exit_code=0))
    with (args.out/'measurements.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    save(args.out/'report.json',dict(status='complete_throughput_artifact_audit',cells=10,requests=4800,
        results=results,evidence=evidence,protocol_sha256=sha(suite/'protocol.json'),
        source_sha256=sha(Path(__file__)),checked_unix=time.time(),new_gpu_runs=0,
        limitations=['Ratios use original same-host historical AR/LoopFill measurements; no fresh batch-throughput AR and no drift correction.',
            'Single timing repeat, shared host, BF16 output differences; not a decoding-equivalence or task-quality validation.',
            'Max1024 continuous end-to-end throughput includes prefill/scheduling/persistence; separate from max512 decode-only latency.']))
    print({'status':'passed','cells':10,'requests':4800,'new_gpu_runs':0})


if __name__ == '__main__':
    main()
