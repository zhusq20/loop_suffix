"""Preserve completed stdout records from an interrupted diagnostic; never resume it.

The derived records are partial evidence, not a successful diagnostic report.
No model, CUDA library, inference worker, or production queue is started.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

BASE, OUT = (Path(x).resolve() for x in sys.argv[1:])
assert not (OUT / 'report.json').exists()
ROOT = BASE / 'logs/baseline-diagnostics-resume-v1'
NAME = 'loopahead-ouro-2.6b-math500-v1'
OBSERVED = BASE / 'preflight/formal-greedy-margins' / NAME
INTERRUPTED = OBSERVED / 'fp32-common-prefix'
ARTIFACTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path, copy=False):
    path = Path(path).resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    item = dict(sha256=digest)
    if copy:
        dest = OUT / 'evidence' / path.relative_to(BASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        item['snapshot'] = str(dest.relative_to(OUT))
    ARTIFACTS[str(path)] = item
    return data


def receipt(step, expected):
    launcher = json.loads(bind(ROOT / 'logs' / f'{step}-launcher.json', True))
    exited = json.loads(bind(ROOT / 'logs' / f'{step}-exit.json', True))
    assert exited['exit_code'] == expected and exited['exit_evidence'] == 'supervisor_waitpid'
    for key in ['pid', 'start_ticks', 'argv', 'started_unix']:
        assert launcher[key] == exited[key]
    assert exited['finished_unix'] >= exited['started_unix']
    return exited


fp32_exit = receipt(NAME + '-fp32', -15)
bf16_exit = receipt(NAME + '-observed', 0)
assert not (INTERRUPTED / 'report.json').exists()
proc = Path(f'/proc/{fp32_exit["pid"]}/stat')
assert not proc.exists() or proc.read_text().rsplit(')', 1)[1].split()[19] != str(fp32_exit['start_ticks'])
manifest = json.loads(bind(INTERRUPTED / 'source-manifest.json', True))
assert manifest['diagnose_trained_loopahead.py'] == '8e4c18ebc54ef0c0b1475ead9573ec80ca4da24c8af18c670231d28526addc7f'
for name, digest in manifest.items():
    data = bind(INTERRUPTED / 'source' / name, True)
    assert hashlib.sha256(data).hexdigest() == digest
plan = json.loads(bind(ROOT / 'plan.json'))
assert sha(ROOT / 'plan.json') == fp32_exit['plan_sha256']
group = next(g for g in plan['groups'] if g['id'] == NAME)
assert str(INTERRUPTED) in group['steps'][-1]['argv']
margins = json.loads(bind(OBSERVED / 'report.json', True))
assert margins['status'] == 'complete' and margins['cases'] == 429
assert margins['finished_unix'] <= bf16_exit['finished_unix'] <= fp32_exit['started_unix']
observed = {(r['source_id'], r['position']): r for r in margins['results']}
assert len(observed) == 429 and all(r['both_original_prefixes_reproduced'] for r in observed.values())
rows = {}
for label, source in margins['source_journals'].items():
    folder = Path(source['path'])
    data = bind(folder / 'requests.jsonl')
    assert hashlib.sha256(data).hexdigest() == source['sha256']
    rows[label] = [json.loads(line) for line in data.splitlines()]
    meta = json.loads(bind(folder / 'metadata.json'))
    done = json.loads(bind(folder / 'complete.json'))
    assert done['requests'] == len(rows[label]) == 500
    assert done['requests_sha256'] == source['sha256']
    assert meta['model'] == 'ouro-2.6b' and meta['dataset'] == 'math500'
    assert meta['temperature'] == 0 and meta['max_new_tokens'] == 512
    if label == 'loopahead':
        assert meta['checkpoint_sha256'] == margins['checkpoint_sha256']
cases = []
for ar, la in zip(rows['ar'], rows['loopahead']):
    assert (ar['source_id'], ar['prompt_sha256']) == (la['source_id'], la['prompt_sha256'])
    if ar['tokens'] == la['tokens']:
        continue
    position = next(i for i, (a, d) in enumerate(zip(ar['tokens'], la['tokens'])) if a != d)
    margin = observed[(ar['source_id'], position)]
    assert margin['ar']['argmax'] == ar['tokens'][position]
    assert margin['loopahead']['argmax'] == la['tokens'][position]
    cases.append(dict(source_id=ar['source_id'], position=position,
        ar_token=ar['tokens'][position], loopahead_token=la['tokens'][position],
        common_prefix=ar['tokens'][:position], observed_bf16=margin))
assert len(cases) == 429
log = bind(ROOT / 'logs' / f'{NAME}-fp32.log', True)
partial = [json.loads(line) for line in log.splitlines() if line.strip()]
assert len(partial) == 262 < len(cases)
for record, case in zip(partial, cases):
    for key in ['source_id', 'position', 'ar_token', 'loopahead_token']:
        assert record[key] == case[key]
    assert math.isfinite(record['max_abs_logit_difference']) and record['max_abs_logit_difference'] >= 0
    for label in ['serial', 'loopahead']:
        top = record[label]
        assert len(top['top_ids']) == len(set(top['top_ids'])) == 3
        assert all(isinstance(i, int) and 0 <= i < 49152 for i in top['top_ids'])
        assert len(top['top_logits']) == len(top['top_probs']) == 3
        assert all(math.isfinite(x) for x in top['top_logits'] + top['top_probs'])
        assert top['top_logits'] == sorted(top['top_logits'], reverse=True)
        assert all(0 <= x <= 1 for x in top['top_probs']) and sum(top['top_probs']) <= 1.000001
        margin = struct.unpack('f', struct.pack('f', top['top_probs'][0] - top['top_probs'][1]))[0]
        assert margin == top['top2_probability_margin']
    assert record['same_top1'] == (record['serial']['top_ids'][0] == record['loopahead']['top_ids'][0])
    record['common_prefix'] = case['common_prefix']
    record['observed_bf16'] = case['observed_bf16']
with (OUT / 'recovered-records.partial.jsonl').open('x') as stream:
    for record in partial:
        stream.write(json.dumps(record) + '\n')
csv = BASE / 'combined-results/results.csv'
assert sha(csv) == 'ed407418e235c10bf37b7b4b729e1486f5ac7c3026a9ebbec3c7c598e6621ff4'
report = dict(status='interrupted_partial_stdout_preserved_not_complete_diagnostic',
    expected_cases=len(cases), recovered_cases=len(partial), remaining_cases=len(cases)-len(partial),
    recovered_same_top1=sum(r['same_top1'] for r in partial), actual_original_exit_code=-15,
    counts_toward_completed_numerical_groups=False, new_gpu_runs=0,
    no_production_files_changed=True, unchanged_csv_sha256=sha(csv),
    artifact_bindings=ARTIFACTS, recovered_records_sha256=sha(OUT / 'recovered-records.partial.jsonl'),
    source_sha256=sha(Path(__file__)), checked_unix=time.time(),
    limitations=['Only emitted stdout records are preserved; the interrupted worker produced no final report.',
        'Common prefixes and BF16 records are joined from the original inputs, not newly generated.',
        'Full logits were not saved; the reported max-logit difference cannot be independently recomputed.',
        'This is not a resumed job, a complete FP32 benchmark, an exit-0 diagnostic, or approval to restart.'])
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'recovered_cases', 'remaining_cases', 'recovered_same_top1']}))
