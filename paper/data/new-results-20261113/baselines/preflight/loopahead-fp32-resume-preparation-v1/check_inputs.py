"""Check the real interrupted inputs and negative bindings without importing CUDA."""
from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

BASE = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from loopahead_partial_resume import load_partial, reconcile

assert not (OUT / 'inputs-report.json').exists()
proof = BASE / 'preflight/interrupted-loopahead-math-fp32-v1/report.json'
observed = BASE / 'preflight/formal-greedy-margins/loopahead-ouro-2.6b-math500-v1/report.json'
margin = json.loads(observed.read_text())
args = Namespace(ar=Path(margin['source_journals']['ar']['path']),
    loopahead=Path(margin['source_journals']['loopahead']['path']),
    observed_margin_report=observed, paper_check=False,
    out=BASE / 'preflight/formal-greedy-margins/loopahead-ouro-2.6b-math500-resume-v1/fp32-common-prefix')
partial = load_partial(proof, args)
rows = {label: [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
        for label, folder in [('ar', args.ar), ('loopahead', args.loopahead)]}
by_key = {(r['source_id'], r['position']): r for r in margin['results']}
cases = []
for ar, la in zip(rows['ar'], rows['loopahead'], strict=True):
    assert ar['source_id'] == la['source_id'] and ar['prompt_sha256'] == la['prompt_sha256']
    if ar['tokens'] == la['tokens']: continue
    position = next(i for i, (a, d) in enumerate(zip(ar['tokens'], la['tokens'])) if a != d)
    m = by_key[(ar['source_id'], position)]
    cases.append(dict(source_id=ar['source_id'], position=position, common_prefix=ar['tokens'][:position],
        ar_token=ar['tokens'][position], loopahead_token=la['tokens'][position],
        observed_bf16={label: m[label] for label in ['ar', 'loopahead']}))
reused, pending = reconcile(cases, partial)
assert len(reused) == 262 and len(pending) == 167
assert pending[0] == cases[262] and reused[-1]['source_id'] != pending[0]['source_id']
rejections = []
for field in ['ar', 'loopahead', 'observed_margin_report', 'paper_check']:
    invalid = copy.copy(args)
    setattr(invalid, field, True if field == 'paper_check' else BASE / 'nonmatching-input')
    try: load_partial(proof, invalid)
    except AssertionError: rejections.append(field)
    else: raise AssertionError('Accepted invalid '+field)
with tempfile.TemporaryDirectory() as directory:
    changed = Path(directory) / 'report.json'
    changed.write_text(proof.read_text() + ' ')
    try: load_partial(changed, args)
    except AssertionError: rejections.append('changed_capsule')
    else: raise AssertionError('Accepted changed evidence capsule')
    invalid = copy.copy(args)
    invalid.out = Path(directory)
    try: load_partial(proof, invalid)
    except AssertionError: rejections.append('existing_output')
    else: raise AssertionError('Accepted an existing output directory')
assert len(rejections) == 6 and 'torch' not in sys.modules
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = dict(status='prepared_resume_inputs_verified_no_gpu_run', reused_cases=len(reused),
    pending_cases=len(pending), first_pending_source_id=pending[0]['source_id'],
    negative_bindings_rejected=rejections, torch_imported=False, new_gpu_runs=0,
    partial_evidence=partial['evidence'], source_sha256={name: sha(BASE / name) for name in
        ['diagnose_trained_loopahead.py', 'diagnose_trained_loopahead_resume.py',
         'loopahead_partial_resume.py', 'test_loopahead_partial_resume.py']},
    checker_sha256=sha(Path(__file__)), checked_unix=time.time(),
    limitations=['Prepared only. No resumed GPU worker or new complete numerical group exists.',
        'A new attempt and explicit audit integration must preserve the interrupted -15 receipt.'])
(OUT / 'inputs-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'reused_cases', 'pending_cases', 'negative_bindings_rejected']}))
