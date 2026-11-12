"""Distribution and stopping checks independent of model quality."""
import unittest

import torch

from .decode import generate, rejection_sample, sample_probabilities


class SamplingChecks(unittest.TestCase):
    def test_rejection_recovers_target_distribution(self):
        # q deliberately puts most mass where p puts little: residual correction matters.
        generator = torch.Generator().manual_seed(1234)
        p = torch.tensor([.05, .35, .60])
        q = torch.tensor([.70, .25, .05])
        counts = torch.zeros(3)
        for _ in range(20000):
            proposed = sample_probabilities(q, generator).reshape(1, 1)
            accepted, bonus = rejection_sample(proposed, p.repeat(1, 2, 1), q.reshape(1, 1, -1), generator)
            counts[int(proposed.item()) if accepted else bonus] += 1
        self.assertLess(float((counts / counts.sum() - p).abs().max()), .015)

    def test_equal_distributions_always_accept_and_sample_bonus(self):
        generator = torch.Generator().manual_seed(99)
        probs = torch.tensor([[[.2, .8], [.7, .3]]])
        for _ in range(100):
            token = sample_probabilities(probs[:, :1], generator)
            accepted, _ = rejection_sample(token, probs, probs[:, :1], generator)
            self.assertEqual(accepted, 1)

    def test_disjoint_support_uses_residual(self):
        generator = torch.Generator().manual_seed(0)
        accepted, token = rejection_sample(torch.tensor([[0]]),
            torch.tensor([[[0., 1.], [1., 0.]]]), torch.tensor([[[1., 0.]]]), generator)
        self.assertEqual((accepted, token), (0, 1))


class CounterCache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length


class FixedTarget:
    device = torch.device('cpu')
    max_seq = 100
    depth = 4

    def __init__(self, eos_position):
        self.eos_position = eos_position

    def new_cache(self):
        return CounterCache()

    def crop(self, cache, length):
        cache.length = length

    def embedding(self, tokens):
        return torch.nn.functional.one_hot(tokens, 3).float()

    def head(self, hidden):
        return hidden

    def forward(self, tokens, cache, features=True):
        from types import SimpleNamespace
        n = tokens.shape[1]
        labels = torch.tensor([2 if i >= self.eos_position else 1 for i in range(cache.length, cache.length + n)])
        logits = torch.full((1, n, 3), -100.)
        logits.scatter_(-1, labels.reshape(1, n, 1), 100.)
        cache.length += n
        return SimpleNamespace(logits=logits, features=torch.zeros((1, n, 3)), cache=cache)


class FixedDraft:
    block_size = 16
    mask_token_id = 0

    def __init__(self, token):
        self.token = token

    def __call__(self, *, target_hidden, noise_embedding, position_ids, past_key_values, **kwargs):
        n = noise_embedding.shape[1]
        size = target_hidden.shape[1] + n
        self_before = past_key_values.get_seq_length()
        assert position_ids.shape[1] == size
        assert int(position_ids[0, 0]) == self_before
        key = torch.zeros((1, 1, size, 1))
        past_key_values.update(key, key.clone(), 0)
        logits = torch.full((1, n, 3), -100.)
        logits[..., self.token] = 100.
        return logits


class StopAndBudgetChecks(unittest.TestCase):
    def test_rejected_eos_does_not_stop(self):
        result = generate(FixedTarget(4), torch.tensor([[0]]), 10, [2], draft=FixedDraft(2))
        self.assertEqual(result.tokens, [1, 1, 1, 1, 2])
        self.assertEqual(result.stats['decode_tokens'], 4)

    def test_eos_inside_accepted_block_counts_only_emitted_tokens(self):
        result = generate(FixedTarget(4), torch.tensor([[0]]), 30, [2], draft=FixedDraft(1))
        self.assertEqual(result.tokens, [1, 1, 1, 1, 2])
        self.assertEqual(result.stats['emitted_per_verification'], [4])
        self.assertEqual(result.stats['gamma'], 4.)

    def test_partial_last_block_and_prefill_only(self):
        for budget in [1, 2, 8, 16, 17, 18, 31, 33]:
            result = generate(FixedTarget(99), torch.tensor([[0]]), budget, [2], draft=FixedDraft(1))
            self.assertEqual(result.tokens, [1] * budget)
            self.assertEqual(result.stats['decode_tokens'], budget - 1)
            self.assertEqual(sum(result.stats['emitted_per_verification']), budget - 1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
