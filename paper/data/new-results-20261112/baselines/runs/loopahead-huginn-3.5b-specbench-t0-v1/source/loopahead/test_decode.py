"""Scheduler oracle: adverse proposals, suspended branches, KV readiness, EOS/caps."""
import sys
from pathlib import Path
import types
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'results/loopfill-latency512-20261107/source/src'))
from .decode import LoopAheadConfig, decode_steps


def gold(prefix):
    return (sum(prefix[-3:]) + 2 * len(prefix) + 3) % 13


class Oracle:
    def __init__(self, cfg, bad_lad=False):
        self.cfg, self.bad_lad = cfg, bad_lad
        self.backbone = types.SimpleNamespace(length=0)
        self.ready = {}
        self.actual = []
        self.peak = 0

    def execute(self, call):
        name, args = call.name, call.args
        if name == 'synchronize':
            return None
        if name == 'prefill':
            self.prompt = args[0]
            self.backbone.length = len(self.prompt)
            self.actual = [gold(self.prompt)]
            return self.actual[0]
        tree = args[0]
        if name == 'embed':
            for node in args[1]:
                self.ready[node.idx] = 0
            return None
        if name == 'advance':
            nodes = args[1]
            assert len(nodes) <= self.cfg.execute
            identities = {n.idx for n in nodes}
            for node in nodes:
                assert self.ready[node.idx] == node.step
                parent = node.parent
                while parent >= 0:
                    ancestor = tree.nodes[parent]
                    assert ancestor.alive
                    assert self.ready[parent] > node.step or (self.ready[parent] == node.step and parent in identities)
                    parent = ancestor.parent
            for node in nodes:
                self.ready[node.idx] += 1
            return None
        if name == 'lad_readout':
            _, groups, full = args
            reads, final = {}, {}
            for depth, group in groups:
                for node in group:
                    prefix = self.prompt + list(node.path)
                    token = gold(prefix)
                    # Purposefully conflicting d1/d2 feedback on different paths.
                    if len(prefix) % 4 == 0 and depth == self.cfg.first:
                        token = (token + 1) % 13
                    if len(prefix) % 5 == 0 and depth == self.cfg.second:
                        token = (token + 2) % 13
                    logits = torch.full((16,), -40.)
                    logits[token] = 40.
                    if depth == self.cfg.first:
                        node.shallow_state = torch.tensor([len(prefix)])
                    reads[node.idx] = (depth, logits)
            for node in full:
                logits = torch.full((16,), -40.)
                logits[gold(self.prompt + list(node.path))] = 40.
                final[node.idx] = logits
            return reads, final
        if name == 'lad_propose':
            _, parent, anchor, count = args
            prefix = self.prompt + list(anchor.path)
            tokens = []
            for _ in range(count):
                token = 15 if self.bad_lad else gold(prefix)
                tokens.append(token)
                prefix.append(token)
            return tokens, [.9] * count
        if name == 'lad_commit':
            _, node, token = args
            assert node.step == self.cfg.total and node.shallow_state is not None
            assert token == gold(self.prompt + self.actual)
            self.actual.append(token)
            self.backbone.length += 1
            return None
        raise AssertionError(name)

    def run(self, cap, stop):
        def observe(event, **data):
            if event != 'advance':
                return
            tree, selected, pool = (data[k] for k in ['tree','selected','pool'])
            assert len(pool) <= self.cfg.retain
            self.peak = max(self.peak, len(pool))

        steps = decode_steps(self.backbone, [2, 3, 7], self.cfg, cap, stop,
                             generator=torch.Generator().manual_seed(42), observer=observe)
        value, calls = None, 0
        while True:
            try:
                call = steps.send(value)
            except StopIteration as done:
                tokens, stats = done.value
                assert tokens == self.actual
                return tokens, stats
            value = self.execute(call)
            calls += 1
            assert calls <= cap * self.cfg.total * 8 + 30


class SchedulerTest(unittest.TestCase):
    def test_native_only_development_reference(self):
        tokens, stats = Oracle(LoopAheadConfig(future=0)).run(80, {15})
        self.assertEqual(len(tokens),80)
        self.assertEqual(stats['lad_calls'],0)
        self.assertEqual(stats['lad_tokens'],0)

    def test_adversarial_drafts_and_tight_budgets(self):
        for retain, execute in [(1, 1), (3, 2), (8, 4), (32, 16)]:
            for bad in [False, True]:
                cfg = LoopAheadConfig(retain=retain, execute=execute)
                tokens, stats = Oracle(cfg, bad).run(80, {15})
                self.assertEqual(len(tokens), 80)
                self.assertEqual(stats['tokens'], 79)
                self.assertLessEqual(stats['max_rows'], execute)
                self.assertLessEqual(stats['peak_live_nodes'], retain)

    def test_cascade_and_cap(self):
        for cap in [1, 2, 3, 4, 16, 17, 33, 80]:
            tokens, stats = Oracle(LoopAheadConfig()).run(cap, {15})
            self.assertEqual(len(tokens), cap)
            self.assertEqual(stats['tokens'], cap - 1)
            if cap == 80:
                self.assertGreater(stats['cascade_commits'], 0)
                self.assertGreater(stats['lad_calls'], 0)

    def test_verified_eos_and_huginn_migration(self):
        for cfg in [LoopAheadConfig(), LoopAheadConfig(total=32, first=4, second=8,
                    future=2, guard=1, stride=2, family='huginn')]:
            tokens, stats = Oracle(cfg).run(80, {4})
            self.assertEqual(tokens[-1], 4)
            self.assertLess(len(tokens), 80)

    def test_sampling_with_near_deterministic_target(self):
        tokens, stats = Oracle(LoopAheadConfig(temperature=1.)).run(80, {15})
        self.assertEqual(len(tokens), 80)
        self.assertFalse(stats['sampling_calibrated'])


if __name__ == '__main__':
    unittest.main()
