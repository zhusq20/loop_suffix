import copy
import unittest

from loopahead_resume_attempt import verify_join
from queue_loopahead_math_continuation import ready_to_launch


def fixture():
    records = [dict(source_id=str(i), same_top1=True, common_prefix=[i],
        observed_bf16={'ar': {'argmax': i}, 'loopahead': {'argmax': i+1}}) for i in range(429)]
    report = dict(status='complete', all_fp32_same_top1=True, cases=429,
        results=copy.deepcopy(records), executed_case_count=167, resume_provenance={'original_interrupted_exit_code': -15})
    old = copy.deepcopy(records[:262])
    for r in old: r['observed_bf16']['extra_archived_field'] = True
    emitted = [{k: v for k, v in r.items() if k not in ['common_prefix', 'observed_bf16']} for r in records[262:]]
    return report, old, emitted, copy.deepcopy(report['resume_provenance'])


class ContinuationTests(unittest.TestCase):
    def test_exact_join(self):
        verify_join(*fixture())

    def test_incomplete_new_suffix_rejected(self):
        report, old, emitted, evidence = fixture()
        with self.assertRaises(AssertionError): verify_join(report, old, emitted[:-1], evidence)

    def test_changed_reused_result_rejected(self):
        report, old, emitted, evidence = fixture()
        report['results'][25]['common_prefix'] = [999]
        with self.assertRaises(AssertionError): verify_join(report, old, emitted, evidence)

    def test_changed_new_result_rejected(self):
        report, old, emitted, evidence = fixture()
        report['results'][300]['source_id'] = 'changed'
        with self.assertRaises(AssertionError): verify_join(report, old, emitted, evidence)

    def test_false_original_success_rejected(self):
        report, old, emitted, evidence = fixture()
        report['resume_provenance']['original_interrupted_exit_code'] = 0
        with self.assertRaises(AssertionError): verify_join(report, old, emitted, evidence)

    def test_failed_report_rejected(self):
        report, old, emitted, evidence = fixture()
        report['status'] = 'failed'
        with self.assertRaises(AssertionError): verify_join(report, old, emitted, evidence)

    def test_measurements_and_stop_block_launch(self):
        for parent, ready, stopped in [(['x'], [], False), ([], ['x'], False), ([], [], True)]:
            self.assertFalse(ready_to_launch({'pending': parent}, {'ready': ready, 'failures': {}}, stopped))
        self.assertTrue(ready_to_launch({'pending': []}, {'ready': [], 'failures': {}}, False))

    def test_measurement_failure_requires_inspection(self):
        with self.assertRaises(AssertionError):
            ready_to_launch({'pending': []}, {'ready': [], 'failures': {'x': 'failed'}}, False)


if __name__ == '__main__':
    unittest.main()
