"""Acceptance regressions using existing completed evidence, without inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from audit_numerical_matrix import (
    BASE, EXPECTED, audit_group, check_receipt, completion_gate, inspect_cases,
    read, rows, scope_key, sha,
)


class NumericalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {}
        for method in ['dflash','loopahead']:
            folder = BASE/'preflight/formal-greedy-margins'/f'{method}-huginn-3.5b-gsm8k-v1'
            meta = read(folder/('metadata.json' if method == 'dflash' else 'report.json'))
            cases = read(folder/'diagnosis.json') if method == 'dflash' else meta['results']
            precise = [] if method == 'dflash' else read(folder/'fp32-common-prefix/report.json')['results']
            source = {('ar' if label == 'ar' else 'method'):rows(Path(item['path'])/'requests.jsonl')
                      for label,item in meta['source_journals'].items()}
            first = cases[0]
            source = {label:[r for r in records if r['source_id'] == first['source_id']]
                      for label,records in source.items()}
            cls.examples[method] = ([first],precise[:1],source)

    def test_real_saved_examples(self):
        for method,args in self.examples.items():
            with self.subTest(method=method):
                result = inspect_cases(method,*args)
                self.assertEqual(result['first_divergences'],1)
                self.assertEqual(result['fp32_argmax_equal'],1)

    def test_omitted_divergence_rejected(self):
        for method,(_,precise,source) in self.examples.items():
            with self.subTest(method=method), self.assertRaises(AssertionError):
                inspect_cases(method,[],precise,source)

    def test_corrupt_prefix_or_fp32_claim_rejected(self):
        for method,args in self.examples.items():
            for corruption in ['prefix','fp32_claim','nan_margin','fp32_token']:
                cases,precise,source = deepcopy(args)
                case = cases[0]
                if corruption == 'prefix':
                    prefix = precise[0]['common_prefix'] if method == 'loopahead' else case['common_output']
                    prefix[0] += 1
                elif corruption == 'fp32_claim':
                    if method == 'loopahead':
                        precise[0]['same_top1'] = False
                    else:
                        case['fp32']['argmax_equal'] = False
                elif corruption == 'nan_margin':
                    side = case['ar'] if method == 'loopahead' else case['bf16']['ar']
                    side['top2_probability_margin'] = float('nan')
                else:
                    side = precise[0]['loopahead'] if method == 'loopahead' else case['fp32']['dflash']
                    self.assertGreater(side['top_logits'][0],side['top_logits'][1])
                    side['top_ids'][0] = 999999
                with self.subTest(method=method,corruption=corruption), self.assertRaises(AssertionError):
                    inspect_cases(method,cases,precise,source)

    def test_full_scope_cannot_be_inferred_from_partial(self):
        complete = {key:dict(status='validated_actual_diagnostic') for key in EXPECTED}
        completion_gate(complete,{}, {})
        for missing in ['formal/dflash/ouro-1.4b/gsm8k',
                        'paper/loopahead/ouro-2.6b/humaneval164','development/wide']:
            partial = dict(complete)
            del partial[missing]
            with self.subTest(missing=missing), self.assertRaises(AssertionError):
                completion_gate(partial,{missing:dict(status='pending_actual_exit')},{})

    def test_failed_stage_cannot_be_hidden_by_pending_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            (queue/'logs').mkdir()
            (queue/'plan.json').write_text('{}')
            steps = [dict(id=name,out=str(queue/name),argv=['--out',str(queue/name)])
                     for name in ['observed','fp32']]
            group = dict(id='example',steps=steps)
            launch = dict(argv=steps[0]['argv'],started_unix=1.,group='example',step='observed',
                          plan_sha256=sha(queue/'plan.json'))
            (queue/'logs/observed-launcher.json').write_text(json.dumps(launch))
            (queue/'logs/observed-exit.json').write_text(json.dumps(dict(launch,exit_code=1,finished_unix=2.)))
            with self.assertRaises(AssertionError):
                audit_group(queue,dict(reused_groups={}),group,{})
            (queue/'logs/observed-exit.json').write_text(json.dumps(dict(launch,exit_code=0,finished_unix=2.)))
            result = audit_group(queue,dict(reused_groups={}),group,{})
            self.assertEqual(result['status'],'pending_actual_exit')

    def test_receipt_identity_and_arguments(self):
        step = dict(id='observed',out='/output',argv=['--ar','/original-ar','--out','/output'])
        launch = dict(argv=step['argv'],started_unix=1.,group='example',step='observed',plan_sha256='expected')
        receipt = dict(launch,finished_unix=2.,exit_code=0)
        check_receipt(launch,receipt,step,dict(id='example'),'expected',False)
        for mutation in [dict(plan_sha256='wrong'),dict(exit_code=None),dict(finished_unix=.5),
                         dict(argv=['--ar','/different-ar','--out','/output'])]:
            changed = dict(receipt,**mutation)
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                check_receipt(launch,changed,step,dict(id='example'),'expected',False)

    def test_frozen_scope_mapping(self):
        scopes = []
        for name in ['baseline-diagnostics-resume-v1','appendix-loopahead-diagnostics-v1']:
            plan = read(BASE/'logs'/name/'plan.json')
            scopes += [scope_key(plan,group) for group in plan['groups']]
        self.assertEqual(len(scopes),len(set(scopes)))
        expected = EXPECTED-{f'formal/dflash/ouro-1.4b/{task}' for task in
                            ['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench']}
        self.assertEqual(set(scopes),expected)


if __name__ == '__main__':
    unittest.main()
