"""CPU audit of B4 input routing and existing formal diagnostic compatibility.

This does not run a model or establish numerical agreement.
"""
import argparse
from copy import deepcopy
import csv
import difflib
import json
import os
from pathlib import Path
import sys

from diagnose_formal_loopahead import prepare
from loopahead_diagnostic_inputs import BASE, SUITE, diagnostic_input, digest, read
from orchestrate_greedy_diagnostics import paper_check_plan, validate
from provenance import freeze_sources


def rejected(call):
    try:
        call()
    except AssertionError:
        return True
    raise AssertionError('Invalid diagnostic scope was accepted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use CUDA_VISIBLE_DEVICES= for this CPU-only audit'
    args.out.mkdir(parents=True, exist_ok=False)
    freeze_sources(args.out, [Path(__file__), BASE/'loopahead_diagnostic_inputs.py',
        BASE/'diagnose_formal_loopahead.py', BASE/'diagnose_trained_loopahead.py',
        BASE/'orchestrate_greedy_diagnostics.py', BASE/'provenance.py'])
    row, agreement, provenance = paper_check_plan()
    ar, target = Path(row['ar_run']), Path(row['method_run'])
    meta, ar_meta = read(target/'metadata.json'), read(ar/'metadata.json')
    _, cases = prepare(ar, target, paper_check=True)
    checks = {'implicit_B4_scope_rejected': rejected(lambda: prepare(ar, target))}
    for field, wrong in [('max_new_tokens', 512), ('prompt_format', 'chat'),
                         ('input_sha256', '0'*64), ('model', 'ouro-1.4b'), ('count', 20)]:
        a, m = deepcopy(ar_meta), deepcopy(meta)
        a[field] = m[field] = wrong
        checks[field+'_mutation_rejected'] = rejected(lambda: diagnostic_input(m, a, True))

    # All already-collected formal greedy LA cells must retain their exact case
    # lists, and the previously completed Huginn83 diagnostic must remain usable.
    formal = []
    agreements = read(BASE/'formal-agreement.json')
    for item in csv.DictReader((BASE/'results.csv').open()):
        if item['method'] != 'loopahead' or item['temperature'] != '0':
            continue
        m, observed = prepare(Path(item['ar_run']), Path(item['method_run']))
        key = '/'.join([item['engine'], item['model'], item['benchmark'], '0', 'loopahead'])
        expected = agreements[key]
        assert [(r['source_id'], r['position']) for r in observed] == [
            (r['source_id'], r['position']) for r in expected['first_divergences']]
        formal.append(dict(model=m['model'], dataset=m['dataset'], cases=len(observed)))
        checks['formal_as_B4_rejected'] = rejected(lambda: diagnostic_input(
            m, read(Path(item['ar_run'])/'metadata.json'), True))
        if m['model'] == 'huginn-3.5b' and m['dataset'] == 'gsm8k':
            reused = validate(item, BASE/'preflight/formal-greedy-margins/loopahead-huginn-3.5b-gsm8k-v1', expected)
    assert len(formal) >= 12 and reused['cases'] == 83

    # The FP32 helper also supports the earlier traced20 preflight; do not impose
    # formal/full-count requirements on that path while adding explicit B4 scope.
    prefix = BASE/'preflight'
    traced = {label: read(prefix/f'loopahead-trained-huginn-3.5b-gsm8k-{label}-bfloat16-n20-v1/metadata.json')
              for label in ['ar', 'loopahead']}
    assert all(m['trace'] and m['count'] == 20 for m in traced.values())
    traced_input = diagnostic_input(traced['loopahead'], traced['ar'])

    # Confirm actual raw-completion tokenization and the original cache budget.
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(SUITE/'source/src'))
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt
    spec = MODELS[meta['model']]
    protocol = read(SUITE/'protocol.json')
    tok = AutoTokenizer.from_pretrained(protocol['models'][meta['model']]['path'],
        trust_remote_code=True, local_files_only=True)
    inputs = [json.loads(line) for line in diagnostic_input(meta, ar_meta, True).open()]
    lengths = []
    for source in inputs:
        tokens = tokenize_prompt(spec, tok, source['prompt'], meta['prompt_format'])
        assert tokens == tok(source['prompt'], add_special_tokens=False).input_ids
        assert len(tokens)+meta['max_new_tokens']+spec.depth_total+4 <= meta['max_seq']
        lengths.append(len(tokens))
    assert len(lengths) == 164 and not torch.cuda.is_initialized()
    diff = ''.join(difflib.unified_diff(
        (target/'source/evaluate_loopahead.py').read_text().splitlines(True),
        (BASE/'evaluate_loopahead.py').read_text().splitlines(True),
        fromfile='B4-LA-evaluator', tofile='current-evaluator'))
    diff_path = args.out/'historical-evaluator.diff'
    diff_path.write_text(diff)
    report = dict(status='passed_cpu_plan_audit', b4=provenance, checks=checks,
        b4_case_positions=[dict(source_id=c['source_id'], position=c['position']) for c in cases],
        b4_requests=agreement['requests'], b4_exact_sequences=agreement['exact_sequences'],
        b4_prompt_length_min=min(lengths), b4_prompt_length_max=max(lengths),
        completion_without_added_special_tokens=True, formal_cases=formal,
        reused_completed_formal_diagnostic=reused, traced20_input_sha256=digest(traced_input),
        historical_evaluator_diff_sha256=digest(diff_path), cuda_initialized=torch.cuda.is_initialized(),
        limitation='CPU planning/input/source checks only. The additional67 B4 BF16/FP32 replays remain pending.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], b4_cases=len(cases),
        formal_cells_checked=len(formal), reused_fp32_cases=reused['cases'],
        cuda_initialized=report['cuda_initialized'])), flush=True)


if __name__ == '__main__':
    main()
