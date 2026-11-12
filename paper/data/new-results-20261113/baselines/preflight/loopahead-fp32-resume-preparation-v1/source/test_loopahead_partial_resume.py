import ast
import copy
from pathlib import Path
import unittest

from loopahead_partial_resume import reconcile

BASE = Path(__file__).resolve().parent


def fixture():
    cases = [dict(source_id=str(i), position=i, common_prefix=[1] * i,
        observed_bf16={'ar': {'argmax': 1}, 'loopahead': {'argmax': 2}},
        ar_token=1, loopahead_token=2) for i in range(429)]
    return cases, {'records': copy.deepcopy(cases[:262])}


class ResumeBoundaryTests(unittest.TestCase):
    def test_exact_prefix_leaves_only_167_cases(self):
        cases, partial = fixture()
        reused, pending = reconcile(cases, partial)
        self.assertEqual(reused, cases[:262])
        self.assertEqual(pending, cases[262:])
        reused[0]['common_prefix'].append(99)
        self.assertEqual(partial['records'][0]['common_prefix'], [])

    def test_reordered_records_rejected(self):
        cases, partial = fixture()
        partial['records'][0], partial['records'][1] = partial['records'][1], partial['records'][0]
        with self.assertRaises(AssertionError): reconcile(cases, partial)

    def test_wrong_forced_prefix_rejected(self):
        cases, partial = fixture()
        partial['records'][10]['common_prefix'][-1] = 99
        with self.assertRaises(AssertionError): reconcile(cases, partial)

    def test_wrong_bf16_trace_rejected(self):
        cases, partial = fixture()
        partial['records'][10]['observed_bf16']['ar']['argmax'] = 99
        with self.assertRaises(AssertionError): reconcile(cases, partial)

    def test_wrong_divergence_position_rejected(self):
        cases, partial = fixture()
        partial['records'][10]['position'] += 1
        with self.assertRaises(AssertionError): reconcile(cases, partial)

    def test_missing_or_extra_records_rejected(self):
        for count in [261, 263]:
            with self.subTest(count=count):
                cases, partial = fixture()
                partial['records'] = copy.deepcopy(cases[:count])
                with self.assertRaises(AssertionError): reconcile(cases, partial)

    def test_duplicate_full_case_rejected(self):
        cases, partial = fixture()
        cases[-1] = copy.deepcopy(cases[-2])
        with self.assertRaises(AssertionError): reconcile(cases, partial)

    def test_no_inference_algorithm_change(self):
        original = ast.parse((BASE / 'diagnose_trained_loopahead.py').read_text())
        resumed = ast.parse((BASE / 'diagnose_trained_loopahead_resume.py').read_text())
        old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        new = next(n for n in resumed.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        additions = """
reused, pending_cases = reconcile(cases, args.partial)
report['results'].extend(reused)
report['resume_provenance'] = args.partial['evidence']
report['executed_case_count'] = 0
report['executed_case_count'] += 1
assert len(report['results']) == len(cases)
assert report['executed_case_count'] == len(cases) - len(reused)
report['limitations'].append('First 262 results reuse the preserved interrupted attempt; only the remaining 167 are executed here. Original exit -15 remains explicit in resume_provenance.')
"""
        allowed = {ast.dump(n): 0 for n in ast.parse(additions).body}
        class RemoveBookkeeping(ast.NodeTransformer):
            changed_loops = 0
            def visit(self, node):
                key = ast.dump(node)
                if isinstance(node, ast.stmt) and key in allowed:
                    allowed[key] += 1
                    return None
                return super().visit(node)
            def visit_For(self, node):
                if isinstance(node.iter, ast.Name) and node.iter.id == 'pending_cases':
                    self.changed_loops += 1
                    node.iter.id = 'cases'
                return self.generic_visit(node)
        transformer = RemoveBookkeeping()
        normalized = transformer.visit(new)
        self.assertEqual(transformer.changed_loops, 1)
        self.assertTrue(all(n == 1 for n in allowed.values()))
        self.assertEqual(ast.dump(old), ast.dump(normalized))
        for name in ['digest', 'read']:
            a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
            d = next(n for n in resumed.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(d))


if __name__ == '__main__':
    unittest.main()
