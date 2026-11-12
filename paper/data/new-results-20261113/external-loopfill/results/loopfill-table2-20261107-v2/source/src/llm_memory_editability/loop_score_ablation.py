"""Three scoring choices with identical decoding mechanics."""
from collections import defaultdict
from .loop_prob_tree import ABSENT, PROB_EDGES, _bin


class RawScores:
    def shallow_prob(self, depth, rank, prob, in_suffix):
        return 0.0 if rank == ABSENT else float(prob)

    def joint_prob(self, deep, rank_d, rank, prob, flag):
        return self.shallow_prob(deep, rank, prob, flag)

    def suffix_prob(self, q, match_len):
        return float(q)

    def suffix_prob_insitu(self, q, match_len, rival, lead):
        return float(q)


class SimpleScores(RawScores):
    """Two reduced feature families, fitted from external64 event counts only.

    Readout keys: (depth, probability bin), excluding ABSENT pseudo-probabilities.
    Suffix keys: probability bin, pooled from the same insitu event population.
    Beta(1,1) smoothing: (hits+1)/(count+2). Empty bins use their family's
    pooled counts, with the same smoothing; no 50-event threshold.
    """
    def __init__(self, base):
        self.readout = defaultdict(lambda: [0, 0])
        self.depth_totals = defaultdict(lambda: [0, 0])
        self.suffix = defaultdict(lambda: [0, 0])
        self.suffix_total = [0, 0]
        for (depth, rank, bucket, flag), (h, n) in base.shallow.items():
            if rank == ABSENT:
                continue
            for target in (self.readout[(depth, bucket)], self.depth_totals[depth]):
                target[0] += h
                target[1] += n
        for (bucket, match, rival, lead), (h, n) in base.insitu.items():
            for target in (self.suffix[bucket], self.suffix_total):
                target[0] += h
                target[1] += n
        if not self.suffix_total[1]:
            raise ValueError('SimpleScores requires external insitu calibration counts')

    @staticmethod
    def estimate(counts):
        h, n = counts
        return (h + 1) / (n + 2)

    def shallow_prob(self, depth, rank, prob, in_suffix):
        if rank == ABSENT:
            return 0.0
        pair = self.readout.get((depth, _bin(prob, PROB_EDGES)))
        return self.estimate(pair if pair is not None else self.depth_totals[depth])

    def suffix_prob(self, q, match_len):
        pair = self.suffix.get(_bin(q, PROB_EDGES))
        return self.estimate(pair if pair is not None else self.suffix_total)

    def suffix_prob_insitu(self, q, match_len, rival, lead):
        return self.suffix_prob(q, match_len)

    def to_json(self):
        return {'schema':'simple-v1', 'smoothing':'Beta(1,1)',
                'readout':[[*k,*v] for k,v in sorted(self.readout.items())],
                'depth_totals':[[k,*v] for k,v in sorted(self.depth_totals.items())],
                'suffix':[[k,*v] for k,v in sorted(self.suffix.items())],
                'suffix_total':self.suffix_total,
                'empty_bin':'pooled depth/source counts with Beta(1,1)',
                'absent_top8':'zero raw probability; rank 8 is not fitted as a softmax observation'}
