"""CPU accounting tests. Synthetic timestamps are never measurement evidence."""
from copy import deepcopy
import unittest

from external_dflash_costs import STAGES, intervals


class ClosedIntervalsTest(unittest.TestCase):
    def setUp(self):
        self.selection = dict(gpu=0, selected_unix=10.)
        self.launches, self.receipts = {}, {}
        for i, stage in enumerate(STAGES):
            launch = dict(stage=stage, gpu=0, pid=100+i, started_unix=12.+20*i, argv=[stage])
            self.launches[stage] = launch
            self.receipts[stage] = dict(launch, finished_unix=22.+20*i, exit_code=0)

    def test_complete_lease_is_partitioned_once(self):
        result = intervals(self.selection, self.launches, self.receipts, finished=90.)
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result), 80.)
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result if r['category']=='generation'), 10.)
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result if r['category']=='training'), 10.)
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result if r['category']=='training_smoke'), 10.)
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result if r['category']=='held_gpu_cpu_audit'), 10.)
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result if r['category']=='held_gpu_between_stages'), 40.)

    def test_running_teacher_has_no_estimated_stage_cost(self):
        result = intervals(self.selection, {'teachers':self.launches['teachers']}, {})
        self.assertEqual([(r['stage'],r['reserved_gpu_seconds']) for r in result], [('before_teachers',2.)])

    def test_failed_stage_is_not_success_or_future_cost(self):
        launch = {'teachers':self.launches['teachers']}
        receipt = {'teachers':dict(self.receipts['teachers'],exit_code=1)}
        result = intervals(self.selection, launch, receipt)
        self.assertEqual(result[-1]['status'], 'failed')
        self.assertEqual(sum(r['reserved_gpu_seconds'] for r in result), 12.)
        with self.assertRaises(AssertionError):
            intervals(self.selection, self.launches, {**self.receipts, **receipt})

    def test_overlapping_stages_rejected(self):
        launches, receipts = deepcopy(self.launches), deepcopy(self.receipts)
        launches['smoke']['started_unix'] = receipts['smoke']['started_unix'] = 21.
        with self.assertRaises(AssertionError):
            intervals(self.selection, launches, receipts)

    def test_exit_identity_mismatch_rejected(self):
        receipts = deepcopy(self.receipts)
        receipts['train']['pid'] += 1
        with self.assertRaises(AssertionError):
            intervals(self.selection, self.launches, receipts)

    def test_final_report_requires_all_real_exits(self):
        with self.assertRaises(AssertionError):
            intervals(self.selection, self.launches, {k:v for k,v in self.receipts.items() if k!='audit'}, finished=90.)


if __name__ == '__main__':
    unittest.main()
