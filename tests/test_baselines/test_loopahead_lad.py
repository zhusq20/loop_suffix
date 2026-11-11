import unittest

import torch

from baselines.loopahead.lad import LADConfig, LookaheadDrafter, RequestVocabulary, frequent_training_ids


class LADChecks(unittest.TestCase):
    def test_future_queries_do_not_leak_backward(self):
        torch.manual_seed(42)
        model = LookaheadDrafter(LADConfig(target_hidden=16, width=16, heads=2, intermediate=24)).eval()
        context, anchor = torch.randn(2, 64, 16), torch.randn(2, 16)
        with torch.no_grad():
            before = model(context, anchor)
            model.future_position[3:].add_(1000 * torch.randn_like(model.future_position[3:]))
            after = model(context, anchor)
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)

    def test_real_context_requirement_and_frozen_head_rows(self):
        model = LookaheadDrafter(LADConfig(target_hidden=16, width=16, heads=2, intermediate=24))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 63, 16), torch.zeros(1, 16))
        head = torch.nn.Linear(16, 20, bias=False).requires_grad_(False)
        vocab = RequestVocabulary(head, [0, 1], [3, 4], [2])
        self.assertEqual(vocab.ids.tolist(), [0, 1, 2, 3, 4])
        vocab.commit([7, 7, 0])
        self.assertEqual(vocab.ids.tolist(), [0, 1, 2, 3, 4, 7])
        self.assertFalse(vocab.weight.requires_grad)
        self.assertEqual(frequent_training_ids({3: 2, 1: 2, 2: 4}, 2), [2, 1])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
