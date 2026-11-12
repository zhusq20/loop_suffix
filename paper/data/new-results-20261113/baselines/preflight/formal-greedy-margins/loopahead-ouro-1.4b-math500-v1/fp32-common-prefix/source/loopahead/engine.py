"""Readout-state access around the immutable PackedLoopExecutor, Graph OFF only."""
from __future__ import annotations

import torch

from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor
from llm_memory_editability.loop_tree_decode import HuginnLoop
from llm_memory_editability.loop_decode_steps import ModelCall

from .lad import RequestVocabulary
from .sampling import probabilities, draw


class LoopAheadExecutor(PackedLoopExecutor):
    def __init__(self, backbone, lad, frequent_ids, *, temperature=0., top_p=1.):
        super().__init__([backbone], cuda_graph=False)
        self.lad = lad
        self.frequent_ids = frequent_ids
        self.temperature, self.top_p = temperature, top_p
        self.head = backbone.model.lm_head if isinstance(backbone, HuginnLoop) else backbone.lm.lm_head
        self.embedding = backbone.model.transformer.wte if isinstance(backbone, HuginnLoop) else backbone.inner.embed_tokens
        self.prefix_states = None
        self.vocab = None

    def request(self, seed, stops):
        self.generator = torch.Generator(device=self.first.hidden.device).manual_seed(seed)
        self.stops = stops

    def execute(self, name, jobs):
        if len(jobs) != 1 or jobs[0][0] is not self.first:
            raise ValueError('LoopAhead latency runner requires one request')
        b, call = jobs[0]
        if name == 'prefill':
            return [self._prefill(b, *call.args)]
        if name == 'lad_readout':
            return [self._readout(b, *call.args)]
        if name == 'lad_propose':
            tree, parent, anchor, count = call.args
            context = self.context(tree, parent)
            if context is None:
                return [([], [])]
            emb = self.embedding(torch.tensor([anchor.token], device=b.hidden.device))
            tokens, confidence = self.lad.propose(context[None], emb, self.vocab, count)
            return [(tokens[0].tolist(), confidence[0].tolist())]
        if name == 'lad_commit':
            tree, node, output_token = call.args
            if getattr(node, 'shallow_state', None) is None:
                raise RuntimeError('Cannot commit a node without its real d1 state')
            self.prefix_states = torch.cat([self.prefix_states, node.shallow_state[None]], 0)[-64:]
            self.vocab.commit([output_token])
            b.commit(tree, [node])
            return [None]
        return super().execute(name, jobs)

    def _prefill(self, b, prompt, shallow_depth):
        self.draft_depth = shallow_depth
        captured, logits = [], []
        head_hook = self.head.register_forward_hook(lambda module, inp, out: logits.append(out.detach()))
        if isinstance(b, HuginnLoop):
            # The first coda call in the frozen prefill is exactly the d1 coda;
            # d2 and full-depth coda follow and use their own cache slots.
            original = b._coda

            def coda(*args, **kwargs):
                result = original(*args, **kwargs)
                if not captured:
                    captured.append(b.model.transformer.ln_f(result)[0].detach().clone())
                return result

            b._coda = coda
            hook = None
        else:
            def norm_hook(module, inp, out):
                captured.append(out[0].detach().clone())
            hook = b.inner.norm.register_forward_hook(norm_hook)
        try:
            first = b.prefill(prompt, shallow_depth)
        finally:
            head_hook.remove()
            if hook is not None:
                hook.remove()
            else:
                b._coda = original
        state = captured[0] if isinstance(b, HuginnLoop) else captured[shallow_depth - 1]
        self.prefix_states = state[-64:]
        if getattr(self, 'capture_prefill_logits', False):
            self.prefill_logits = logits[-1].detach().reshape(-1, logits[-1].shape[-1])[-1]
        if self.temperature > 0 and getattr(b, 'target_sampler', None) is None:
            first = draw(probabilities(logits[-1].reshape(-1, logits[-1].shape[-1])[-1],
                                       self.temperature, self.top_p), self.generator)
        self.vocab = RequestVocabulary(self.head, self.frequent_ids, prompt, self.stops)
        self.vocab.commit([first])
        return first

    def _readout(self, b, tree, groups, full):
        nodes = [n for _, group in groups for n in group] + full
        if not nodes:
            return {}, {}
        if isinstance(b, HuginnLoop):
            bases = [b.coda_extra.get(d, b.coda_shallow) for d, group in groups for _ in group]
            bases += [b.coda_full] * len(full)
        else:
            bases = [0] * len(nodes)
        sorted_nodes = [nodes[i] for i in sorted(range(len(nodes)), key=lambda i: bases[i])]
        captured = []

        def capture(module, inputs, output):
            captured.append((inputs[0].detach(), output.detach().float()))

        hook = self.head.register_forward_hook(capture)
        sampler = getattr(b, 'target_sampler', None)
        b.target_sampler = None  # The wrapper consumes raw logits for native rejection correction.
        try:
            super().execute('readout_top', [(b, ModelCall('readout_top', (tree, groups, full, 0)))])
        finally:
            hook.remove()
            b.target_sampler = sampler
        if len(captured) != 1 or captured[0][0].shape[0] != len(nodes):
            raise RuntimeError('Unexpected packed readout layout')
        states, logits = captured[0]
        by_node = {n.idx: (states[i], logits[i]) for i, n in enumerate(sorted_nodes)}
        reads, final = {}, {}
        for depth, group in groups:
            for node in group:
                state, logit = by_node[node.idx]
                reads[node.idx] = (depth, logit)
                if depth == self.draft_depth:
                    node.shallow_state = state.clone()
        for node in full:
            final[node.idx] = by_node[node.idx][1]
        return reads, final

    def context(self, tree, node):
        branch = []
        while node is not None:
            state = getattr(node, 'shallow_state', None)
            if state is None:
                return None
            branch.append(state)
            node = tree.nodes[node.parent] if node.parent >= 0 else None
        available = self.prefix_states.shape[0] + len(branch)
        if available < 64:
            return None
        return torch.cat([self.prefix_states, torch.stack(branch[::-1])], 0)[-64:]
