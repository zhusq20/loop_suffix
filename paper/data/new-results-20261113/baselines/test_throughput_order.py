"""Compare the reusable evaluator's admission rule with the actual archived table."""
import json
import unittest

from evaluate_loopahead_throughput import TABLE2, ordered_requests


class AdmissionOrderRegression(unittest.TestCase):
    def test_formal_order_matches_all_archived_ar_and_loopfill_cells(self):
        protocol = json.loads((TABLE2/'protocol.json').read_text())
        for model in ['huginn-3.5b', 'ouro-2.6b']:
            with (TABLE2/f'inputs/{model}.jsonl').open() as stream:
                original = [json.loads(line)['question_id'] for line in stream]
            actual = ordered_requests(original, 'formal', protocol['order_seed'])
            self.assertNotEqual(actual, original)
            for batch in [1, 2, 4, 8, 16]:
                for arm in ['ar', 'budget-online']:
                    with self.subTest(model=model, batch=batch, arm=arm):
                        archive = json.loads((TABLE2/f'jobs/{model}-b{batch}-{arm}.json').read_text())
                        self.assertEqual(archive['outcome'], 'complete')
                        self.assertEqual(actual, archive['summary']['request_order'])

    def test_probes_keep_their_fixed_shape_cases(self):
        cases = ['length33', 'length18', 'length2', 'length17']
        for mode in ['validate', 'smoke']:
            self.assertEqual(ordered_requests(cases, mode, 0), cases)


if __name__ == '__main__':
    unittest.main()
