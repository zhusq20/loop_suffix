import unittest
import torch
from .sampling import Proposal, probabilities, residual, second_proposal, verify


class SamplingTest(unittest.TestCase):
    def test_exact_conditional_enumeration(self):
        # Enumerate the entire two-stage stochastic procedure, including the
        # dependence of the second proposal distribution on the first token.
        for p, q1, raw in [([.05, .65, .3], [.7, .2, .1], [.1, .4, .5]),
                           ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
                           ([.2, .3, .5], [.2, .3, .5], [.4, .3, .3])]:
            p, q1, raw = map(lambda x: torch.tensor(x, dtype=torch.float64), (p, q1, raw))
            outcome = torch.zeros_like(p)
            for a in range(3):
                if q1[a] == 0:
                    continue
                acceptance = min(1., float(p[a] / q1[a]))
                outcome[a] += q1[a] * acceptance
                reject_mass = q1[a] * (1 - acceptance)
                if reject_mass == 0:
                    continue
                remain = residual(p, q1)
                if raw[a] >= q1[a]:
                    outcome += reject_mass * remain
                    continue
                q2 = residual(raw, q1)
                for b in range(3):
                    if q2[b] == 0:
                        continue
                    acceptance2 = min(1., float(remain[b] / q2[b]))
                    mass = reject_mass * q2[b]
                    outcome[b] += mass * acceptance2
                    if acceptance2 < 1:
                        outcome += mass * (1 - acceptance2) * residual(remain, q2)
            torch.testing.assert_close(outcome, p, rtol=1e-13, atol=1e-13)

    def test_monte_carlo_and_duplicate_candidates(self):
        p = torch.tensor([.1, .6, .3])
        q1 = torch.tensor([.65, .1, .25])
        q2 = torch.tensor([.15, .6, .25])
        g = torch.Generator().manual_seed(42)
        counts = torch.zeros(3)
        for _ in range(20000):
            a = int(torch.multinomial(q1, 1, generator=g))
            first = Proposal(a, q1, 1)
            second = second_proposal(first, q2, 2, g)
            result, _ = verify(p, [first] + ([] if second is None else [second]), g)
            counts[result] += 1
        torch.testing.assert_close(counts / counts.sum(), p, rtol=0, atol=.012)
        # Equal p=q is accepted with probability one, even if a duplicate later
        # proposal exists; absent proposals sample from the original p directly.
        for token in range(3):
            self.assertEqual(verify(p, [Proposal(token, p, 1), Proposal(token, p, 2)], g)[0], token)
        self.assertEqual(verify(torch.tensor([0., 1., 0.]), [], g)[0], 1)

    def test_top_p_and_zero_residual(self):
        p = probabilities(torch.tensor([3., 2., 1.]), top_p=.6)
        torch.testing.assert_close(p, torch.tensor([1., 0., 0.]))
        self.assertIsNone(residual(p, p))


if __name__ == '__main__':
    unittest.main()
