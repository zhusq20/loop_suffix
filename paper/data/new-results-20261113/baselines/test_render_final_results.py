"""No inference: reject incomplete evidence and preserve measured CSV values."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest

from audit_numerical_matrix import EXPECTED as NUMERICAL_SCOPE
from render_final_results import BASE, EXPECTED, TASKS, UNKNOWN_EXITS, key, normalize, render, table_text, validate_matrix


def synthetic_matrix():
    """Arithmetic fixtures only; never written as measured results."""
    rows = []
    for method,model,task,temp in sorted(EXPECTED):
        dflash = method == 'dflash'
        backend = 'legacy_HF_full_loop' if dflash and task == 'gsm8k' else 'packed_full_loop' if dflash else 'packed_LoopAhead'
        rows.append(dict(method=method,model=model,benchmark=task,temperature=str(temp),
            eta='2.0',eta_latency='2.0',eta_throughput='2.0',gamma='3.0',
            method_tok_s='40.0',ar_tok_s='20.0',method_decode_tokens='120',ar_decode_tokens='120',
            method_decode_s='3.0',ar_decode_s='6.0',match_rate='1.0',aligned_token_match_rate='1.0',
            drafter_time_frac='0.1' if dflash else 'NA',profile_run='/fixture/profile' if dflash else '',
            engine=('official' if backend == 'legacy_HF_full_loop' else 'official_dflash_with_packed_loop_target') if dflash else 'ours',
            target_backend=backend,method_exit_evidence='unavailable_artifacts_verified' if (method,model,task,temp) in UNKNOWN_EXITS else 'actual_exit0',
            status='complete_with_declared_limitations',numerical_diagnostics='not_all_completed',
            notes='Synthetic fixture; numerical diagnostics not all completed. '))
    return rows


def synthetic_numerical():
    groups = {scope:dict(status='validated_actual_diagnostic') for scope in NUMERICAL_SCOPE}
    for scope,record in groups.items():
        if scope.startswith('formal/'):
            record.update(whole_sequence_match_rate_bf16=1.,requests=TASKS[scope.split('/')[-1]],
                          first_divergences=0,fp32_argmax_equal=0)
    return dict(results=groups,pending={},errors={})


class FinalTableChecks(unittest.TestCase):
    def test_requires_exact_72_keys(self):
        rows = synthetic_matrix()
        validate_matrix(rows)
        for altered in [rows[:-1],rows[:-1]+[rows[0]]]:
            with self.assertRaises(AssertionError):
                validate_matrix(altered)

    def test_arithmetic_and_exit_scope_are_not_silently_fixed(self):
        rows = synthetic_matrix()
        for field,value in [('eta_latency','0.5'),('method_tok_s','41'),('match_rate','nan')]:
            altered = deepcopy(rows);altered[0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_matrix(altered)
        unknown = next(row for row in rows if key(row) in UNKNOWN_EXITS)
        unknown['method_exit_evidence'] = 'actual_exit0'
        with self.assertRaises(AssertionError):
            validate_matrix(rows)

    def test_annotations_do_not_change_any_measurement(self):
        rows = synthetic_matrix();before = deepcopy(rows)
        normalized = normalize(rows,synthetic_numerical())
        self.assertEqual(rows,before)
        self.assertEqual({r['engine'] for r in normalized},{'official','ours'})
        for original,result in zip(rows,normalized,strict=True):
            for field,value in original.items():
                if field in ['engine','notes','numerical_diagnostics']:
                    self.assertEqual(result['collected_'+field],value)
                else:
                    self.assertEqual(result[field],value)
        text = table_text(normalized)
        self.assertEqual(sum(line.startswith('| ') for line in text.splitlines()),13)
        self.assertIn('混合实现',text)
        self.assertEqual(text.count(' P ‡'),3)
        self.assertIn('跨日期时间漂移未校正',text)
        self.assertIn('不能用来证明采样分布',text)

    def test_missing_additional_diagnostics_or_mismatched_agreement_rejected(self):
        rows = synthetic_matrix()
        for scope in ['development/wide','paper/loopahead/ouro-2.6b/humaneval164']:
            numerical = synthetic_numerical();del numerical['results'][scope]
            with self.subTest(scope=scope),self.assertRaises(AssertionError):
                normalize(rows,numerical)
        numerical = synthetic_numerical()
        numerical['results']['formal/dflash/huginn-3.5b/gsm8k']['whole_sequence_match_rate_bf16'] = .9
        with self.assertRaises(AssertionError):
            normalize(rows,numerical)

    def test_real_partial_collection_cannot_produce_final_files(self):
        # This regression binds the real 58-cell snapshot, not a permissive fake.
        collection = BASE/'preflight/combined-results-cell58-v1'
        with (collection/'before-results.csv').open() as stream:
            historical_rows = list(csv.DictReader(stream))
        self.assertEqual(len(historical_rows),57)
        with self.assertRaises(AssertionError):
            validate_matrix(historical_rows)
        # The actual current matrix is still incomplete. Later completion should
        # use the preserved snapshot above rather than reject valid final data.
        with (BASE/'combined-results/results.csv').open() as stream:
            current = list(csv.DictReader(stream))
        if len(current) < 72:
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)/'final'
                with self.assertRaisesRegex(AssertionError,'72 unique'):
                    render(collection/'report.json',BASE/'preflight/numerical-matrix-partial-v2/report.json',
                           BASE/'logs/audit-numerical-matrix-partial-v2-exit.json',out)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
