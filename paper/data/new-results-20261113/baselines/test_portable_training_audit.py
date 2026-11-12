"""Reject corrupt/incomplete portable logs without manufacturing old fields."""
import copy
import unittest

from audit_portable_dflash_training import inspect_portable_metrics


class PortableMetricsTest(unittest.TestCase):
    def setUp(self):
        self.meta = dict(epochs=6, examples=33, world_size=1, micro_batch=4,
                         accumulation=8, total_updates=12, bounded_experiment=False)
        self.rows = [dict(step=i, total_updates=12, loss=1., accuracy=.5,
                          lr=.0001, grad_norm=.2, wall_s=.1) for i in range(1,13)]

    def test_actual_portable_schema(self):
        result = inspect_portable_metrics(self.rows, self.meta)
        self.assertEqual(result['updates_per_epoch_from_recipe'], 2)
        self.assertIsNone(result['peak_allocated_bytes'])
        self.assertNotIn('epoch_update_counts', result)

    def test_missing_or_duplicate_updates(self):
        for rows in [self.rows[:-1], self.rows[:-1]+[self.rows[-2]]]:
            with self.subTest(rows=rows[-1]), self.assertRaises(AssertionError):
                inspect_portable_metrics(rows, self.meta)

    def test_invalid_measurements(self):
        for key, value in [('loss',float('nan')), ('accuracy',1.01),
                           ('grad_norm',float('inf')), ('lr',-.01), ('wall_s',0.),
                           ('total_updates',11)]:
            changed = copy.deepcopy(self.rows)
            changed[3][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                inspect_portable_metrics(changed, self.meta)

    def test_smoke_and_wrong_recipe_rejected(self):
        for key, value in [('bounded_experiment',True), ('epochs',1),
                           ('accumulation',4), ('total_updates',11), ('examples',0)]:
            changed = dict(self.meta, **{key:value})
            with self.subTest(key=key), self.assertRaises(AssertionError):
                inspect_portable_metrics(self.rows, changed)


if __name__ == '__main__':
    unittest.main()
