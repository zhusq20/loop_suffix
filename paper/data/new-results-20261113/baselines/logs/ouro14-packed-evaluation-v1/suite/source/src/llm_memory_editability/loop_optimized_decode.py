"""Chunked prefill, shared-storage attention, and warmed decode CUDA graphs.

Metadata remains request-specific even when a single attention call serves
multiple requests. Optional execution padding writes to an isolated KV sink.
Graph capture is allowed only in explicit warmup; the timed run replays known
shapes or records eager misses. Token decoders and scheduling stay unchanged.
"""
from __future__ import annotations

import time
from collections import Counter
from types import SimpleNamespace

from .loop_sampling import target_tokens

import numpy as np
import torch
import torch.nn.functional as F

from .loop_batch_decode import FusedLoopExecutor
from .loop_tree_decode import DraftTree, HuginnLoop, LoopKV, OuroLoop, _split, _top, _top_probs
from .loop_decode_steps import ModelCall
from .loop_tree_decode import grouped_scores, grouped_values


def bucket(n):
    return 1 << (max(1, n) - 1).bit_length()


class SharedKVPool:
    """Allocate every request's dense cache once, as disjoint contiguous views."""
    def __init__(self, batch_size, pad_decode_rows=False):
        self.batch_size = batch_size
        self.pad_decode_rows = pad_decode_rows
        self.used = 0
        self.signature = None

    def __call__(self, slots, max_seq, max_nodes, heads, dim, dtype, device):
        signature = (slots, max_seq, max_nodes, heads, dim, dtype, device)
        if self.signature is None:
            self.signature = signature
            self.slots, self.max_seq, self.max_nodes = slots, max_seq, max_nodes
            self.K = torch.zeros(self.batch_size * slots, max_seq, heads, dim, dtype=dtype, device=device)
            self.V = torch.zeros_like(self.K)
            real = self.batch_size * slots * max_nodes
            # Padded rows write only into this private sink. Different padded
            # rows have distinct addresses within an operation. No attention
            # reads the sink, so different layers may reuse its storage.
            reserve = slots * max_nodes + bucket(self.batch_size * max_nodes) if self.pad_decode_rows else 0
            self.tree_k_storage = torch.zeros(real + reserve, heads, dim, dtype=dtype, device=device)
            self.tree_v_storage = torch.zeros_like(self.tree_k_storage)
            self.TK = self.tree_k_storage[:real].view(self.batch_size * slots, max_nodes, heads, dim)
            self.TV = self.tree_v_storage[:real].view_as(self.TK)
        if self.signature != signature or self.used >= self.batch_size:
            raise ValueError('shared KV pool shape/capacity mismatch')
        index = self.used
        self.used += 1
        start, end = index * slots, (index + 1) * slots
        kv = LoopKV(slots, max_seq, max_nodes, heads, dim, dtype, device,
                    storage=(self.K[start:end], self.V[start:end], self.TK[start:end], self.TV[start:end]))
        kv.pool, kv.pool_index = self, index
        return kv


def chunk_attention(kv, slot, start, end):
    """Bottom-right causal attention: queries [start,end), keys [0,end)."""
    def attend(q, k, v):
        kv.K[slot, start:end] = k
        kv.V[slot, start:end] = v
        mask = torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
        return F.scaled_dot_product_attention(q.transpose(0, 1)[None],
            kv.K[slot, :end].transpose(0, 1)[None], kv.V[slot, :end].transpose(0, 1)[None],
            attn_mask=mask, enable_gqa=q.shape[1] != kv.K.shape[-2])[0].transpose(0, 1)
    return attend


@torch.no_grad()
def prefill_chunk(b, prompt, depth, start, chunk_size):
    """Process one prompt chunk through every recurrent depth, preserving its KV."""
    if start == 0:
        b.kv.length = 0
    if start != b.kv.length or not 0 <= start < len(prompt):
        raise ValueError('noncontiguous prefill chunk')
    end = min(start + chunk_size, len(prompt))
    dev = b.kv.K.device
    ids = torch.tensor([prompt[start:end]], device=dev)
    positions = torch.arange(start, end, device=dev)
    if isinstance(b, OuroLoop):
        x = b.inner.embed_tokens(ids)
        cos, sin = b.inner.rotary_emb(x, positions[None])
        for step in range(b.T):
            for li, layer in enumerate(b.layers):
                x = b._layer(layer, x, cos, sin, chunk_attention(b.kv, step * b.n_layers + li, start, end))
            x = b.inner.norm(x)
        logits = b.lm.lm_head(x[:, -1]) if end == len(prompt) else None
    else:
        freqs = b._positions(positions)
        x = b._embed_tokens(ids)
        for li, block in enumerate(b.prelude):
            x = b._block(block, x, freqs, chunk_attention(b.kv, li, start, end))
        h = b.init[start:end][None]
        for step in range(b.T):
            h = b.model.transformer.adapter(torch.cat([h, x], dim=-1))
            for li, block in enumerate(b.core):
                h = b._block(block, h, freqs, chunk_attention(b.kv, b.n_pre + step * b.n_core + li, start, end))
            if step + 1 == depth:
                b._coda(h, freqs, lambda li: chunk_attention(b.kv, b.coda_shallow + li, start, end))
            if step + 1 in b.coda_extra:
                base = b.coda_extra[step + 1]
                b._coda(h, freqs, lambda li: chunk_attention(b.kv, base + li, start, end))
        y = b._coda(h, freqs, lambda li: chunk_attention(b.kv, b.coda_full + li, start, end))
        logits = b.model.lm_head(b.model.transformer.ln_f(y[:, -1])) if end == len(prompt) else None
    b.kv.length = end
    return end, (target_tokens(b, logits, [len(prompt)])[0] if logits is not None else None)


class DecodeGraphs:
    """Capture execution shapes during warmup; never capture inside timing."""
    def __init__(self, enabled=False, max_graphs=24):
        self.enabled = enabled
        self.max_graphs = max_graphs
        self.warming = True
        self.entries = {}
        self.observed = Counter()
        self.candidates = {}
        self.capture_s = 0.0
        self.replays = self.eager_calls = 0
        self.required = set()
        self.prioritize = False

    @torch.no_grad()
    def freeze(self):
        # Capture the frequently used steady-state shapes, not only the first
        # pipeline-fill shapes. Candidate tensors are bounded by max_graphs.
        if not self.warming:
            return
        for key, (static, fn) in self.candidates.items():
            started = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    fn(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = fn(static)
            self.entries[key] = (static, graph, output)
            self.capture_s += time.perf_counter() - started
        self.candidates.clear()
        self.warming = False

    def reset_counts(self):
        self.replays = self.eager_calls = 0

    def run(self, key, inputs, fn):
        if not self.enabled:
            self.eager_calls += 1
            return fn(inputs)
        if inputs['x'].device.type != 'cuda':
            raise ValueError('CUDA graphs require CUDA tensors')
        if self.warming:
            self.observed[key] += 1
            if key not in self.candidates and len(self.candidates) == self.max_graphs:
                eligible = [k for k in self.candidates if k not in self.required]
                if not eligible and self.prioritize:
                    raise ValueError('max_graphs cannot cover required verifier shapes')
                least = min(eligible, key=lambda k: self.observed[k]) if eligible else None
                if least is not None and (self.prioritize or self.observed[key] > self.observed[least]):
                    del self.candidates[least]
            if key not in self.candidates and len(self.candidates) < self.max_graphs:
                self.candidates[key] = ({k: v.clone() for k, v in inputs.items()}, fn)
            elif key in self.candidates:
                static, _ = self.candidates[key]
                for name, value in inputs.items():
                    static[name].copy_(value)
            if self.prioritize:
                self.required.add(key)
            self.eager_calls += 1
            return fn(inputs)
        if key in self.entries:
            static, graph, output = self.entries[key]
            for name, value in inputs.items():
                static[name].copy_(value)
            graph.replay()
            self.replays += 1
            return output
        self.eager_calls += 1
        return fn(inputs)


class PackedLoopExecutor(FusedLoopExecutor):
    """One prefix attention invocation and one tree attention batch per layer."""
    def __init__(self, backbones, cuda_graph=False, max_graphs=24):
        super().__init__(backbones)
        self.core_execution_rows = 0
        self.pool = getattr(backbones[0].kv, 'pool', None)
        self.backbones = backbones
        if self.pool is None or any(getattr(b.kv, 'pool', None) is not self.pool for b in backbones):
            raise ValueError('packed attention requires a shared KV pool')
        self.graphs = DecodeGraphs(cuda_graph, max_graphs)
        self.rope = None
        if isinstance(self.first, OuroLoop):
            with torch.no_grad():
                self.rope = self.first.inner.rotary_emb(
                    self.first.hidden[:1][None],
                    torch.arange(self.pool.max_seq, device=self.pool.K.device)[None])
        self.attention_batches = 0
        self.prefill_chunks = 0
        self.chunk_tokens = 0

    def reset_metrics(self):
        self.core_calls = self.core_rows = self.max_core_rows = 0
        self.core_execution_rows = 0
        self.attention_batches = self.prefill_chunks = self.chunk_tokens = 0
        self.graphs.reset_counts()

    def metrics(self):
        return dict(attention_backend='packed_prefix_and_tree', cuda_graph=self.graphs.enabled,
                    cuda_graph_shapes=len(self.graphs.entries), cuda_graph_replays=self.graphs.replays,
                    cuda_graph_eager_misses=self.graphs.eager_calls, cuda_graph_capture_s=self.graphs.capture_s,
                    attention_policy='packed_prefix_and_tree', attention_batches=self.attention_batches,
                    pad_decode_rows=self.pool.pad_decode_rows,
                    core_execution_rows=self.core_execution_rows,
                    core_padding_rows=self.core_execution_rows-self.core_rows,
                    verifier_graph_shapes=len(self.graphs.required),
                    padding_sink_bytes=(self.pool.tree_k_storage.numel()-self.pool.TK.numel())
                    * self.pool.TK.element_size() * 2)

    @torch.no_grad()
    def prime_verifier_shapes(self, max_query_nodes):
        """Cover bounded full-depth verifier shapes without reading test outputs.

        Pure suffix and AR have one recurrence group per active request. Enumerate
        reachable row/query buckets from declared capacity alone. Synthetic states
        never enter the suffix cache or measurements; normal request admission
        overwrites them. Prioritize recurrent graphs within the existing graph cap.
        """
        if not self.graphs.enabled or not self.pool.pad_decode_rows:
            return
        if not self.graphs.warming or not 1 <= max_query_nodes <= self.pool.max_nodes:
            raise ValueError('invalid verifier warmup state or node bound')
        if self.pool.max_nodes > 65:
            raise ValueError('verifier shape priming requires a bounded tree of at most 65 slots')
        representatives = {}
        for largest in range(1, max_query_nodes + 1):
            for active in range(1, len(self.backbones) + 1):
                # A largest group plus at least one row in every other group.
                low, high = largest + active - 1, largest * active
                total = low
                while total <= high:
                    counts = [largest] + [1] * (active - 1)
                    extra = total - low
                    for i in range(1, active):
                        take = min(extra, largest - 1)
                        counts[i] += take
                        extra -= take
                    representatives.setdefault((bucket(total), bucket(largest)), counts)
                    total = bucket(total) + 1
        if len(representatives) > self.graphs.max_graphs:
            raise ValueError(f'verifier needs {len(representatives)} graphs; increase --max-cuda-graphs')
        self.graphs.prioritize = True
        try:
            for counts in representatives.values():
                jobs = []
                for b, count in zip(self.backbones, counts):
                    tree = DraftTree(self.pool.max_nodes, max_query_nodes)
                    tree.reset(min(b.kv.length, b.kv.max_seq - max_query_nodes))
                    parent = None
                    for _ in range(count):
                        parent = tree.add(0, parent, 0, 'warmup')
                    jobs.append((b, ModelCall('advance', (tree, tree.nodes))))
                self._rows('advance', jobs)
        finally:
            self.graphs.prioritize = False

    def _rows(self, name, jobs):
        plans = []
        b0, pool = self.first, self.pool
        for j, (b, call) in enumerate(jobs):
            tree = call.args[0]
            if name in ('advance', 'embed'):
                nodes = call.args[1]
                bases = [0 if name == 'embed' else (n.step * b.n_layers if isinstance(b, OuroLoop)
                         else b.n_pre + n.step * b.n_core) for n in nodes]
                sizes = None
            elif name == 'readout_top':
                _, groups, full, m = call.args[:4]
                sizes = [len(g) for _, g in groups]
                nodes = [n for _, g in groups for n in g] + full
                bases = ([b.coda_extra.get(d, b.coda_shallow) for d, g in groups for _ in g]
                         + [b.coda_full] * len(full)) if isinstance(b, HuginnLoop) else [0] * len(nodes)
            elif name == 'readout':
                _, shallow, full, m, shared = call.args
                sizes, nodes = [len(shallow)], shallow + full
                bases = ([b.coda_full if shared else b.coda_shallow] * len(shallow)
                         + [b.coda_full] * len(full)) if isinstance(b, HuginnLoop) else [0] * len(nodes)
            else:
                raise ValueError(name)
            if nodes:
                order = sorted(range(len(nodes)), key=lambda i: bases[i])
                sorted_nodes = [nodes[i] for i in order]
                sorted_bases = [bases[i] for i in order]
                groups, first = [], 0
                for i in range(1, len(nodes) + 1):
                    if i == len(nodes) or sorted_bases[i] != sorted_bases[first]:
                        groups.append((sorted_bases[first], first, i))
                        first = i
                rows = SimpleNamespace(nodes=sorted_nodes, groups=groups)
                plans.append((j, b, rows, nodes, sizes, tree))
        results = [None] * len(jobs)
        if name == 'readout_top':
            results = [([[] for _ in c.args[1]], []) for _, c in jobs]
        elif name == 'readout':
            results = [([], []) for _ in jobs]
        if not plans:
            return results
        counts = [len(r.nodes) for _, _, r, _, _, _ in plans]
        R = sum(counts)
        group_info = []
        plan_offsets = {}
        base, idx, pos, tokens = [], [], [], []
        offset = 0
        for _, b, rows, _, _, tree in plans:
            plan_offsets[id(b)] = offset
            global_offset = b.kv.pool_index * pool.slots
            for localbase, a, z in rows.groups:
                group_info.append((global_offset + localbase, offset + a, offset + z, b, rows, tree))
                base.extend([global_offset + localbase] * (z-a))
            idx.extend(n.idx for n in rows.nodes)
            pos.extend(tree.position(n) for n in rows.nodes)
            tokens.extend(n.token for n in rows.nodes)
            offset += len(rows.nodes)
        G = bucket(len(group_info))
        group_info.sort(key=lambda group: group[0])
        Q = bucket(max(z-a for _, a, z, *_ in group_info))
        N = bucket(max(len(t.nodes)-t.lo for *_, t in group_info))
        padded_R = bucket(R) if pool.pad_decode_rows else R
        if pool.pad_decode_rows:
            G = max(G, bucket(pool.batch_size))
            # Small bounded trees use a stable attention width. Large WFD
            # buffers retain their existing dynamic width to avoid huge masks.
            if pool.max_nodes <= 65:
                N = bucket(pool.max_nodes)
        query_map = np.zeros((G, Q), dtype=np.int64)
        output_map = np.zeros(padded_R, dtype=np.int64)
        anc = np.zeros((G, Q, N), dtype=bool)
        anc[:, :, 0] = True
        tree_idx = np.zeros((G, N), dtype=np.int64)
        prefix_lengths, gslots = [], []
        for g, (slot, a, z, b, rows, tree) in enumerate(group_info):
            query_map[g, :z-a] = np.arange(a, z)
            output_map[a:z] = g * Q + np.arange(z-a)
            # Dummy query rows see one tree key, preventing all-masked softmax.
            anc[g, :, 0] = True
            n = len(tree.nodes)-tree.lo
            local_offset = plan_offsets[id(b)]
            node_ids = [rows.nodes[i-local_offset].idx for i in range(a,z)]
            anc[g, :z-a, :n] = tree.anc[node_ids, tree.lo:len(tree.nodes)]
            tree_idx[g] = np.minimum(tree.lo+np.arange(N), pool.max_nodes-1)
            prefix_lengths.append(b.kv.length)
            gslots.append(slot)
        # Padded groups read a valid one-token prefix but contribute no output
        # rows or KV writes. Real queries still write exactly R cache entries.
        missing = G - len(group_info)
        prefix_lengths.extend([1] * missing)
        gslots.extend([gslots[-1]] * missing)
        padding = padded_R - R
        base.extend([pool.batch_size * pool.slots] * padding)
        idx.extend(range(padding))
        pos.extend([0] * padding)
        tokens.extend([0] * padding)
        dev = pool.K.device
        # Three packed metadata uploads, independent of request count. Graph
        # replay also copies these packed tensors rather than dozens of views.
        parts64 = dict(pos=pos, tokens=tokens, base=base, idx=idx, gslots=gslots,
                       tree_idx=tree_idx.reshape(-1), query_map=query_map.reshape(-1), output_map=output_map)
        parts32 = dict(cuq=np.arange(G+1)*Q,
                       cuk=[slot*pool.max_seq for slot in gslots] + [gslots[-1]*pool.max_seq+prefix_lengths[-1]],
                       lengths=prefix_lengths)
        inputs = dict(meta64=b0.kv.upload(np.concatenate(list(parts64.values())).astype(np.int64)),
                      meta32=b0.kv.upload(np.concatenate(list(parts32.values())).astype(np.int32)),
                      anc=b0.kv.upload(anc))
        slices = {}
        for source, parts in [('meta64', parts64), ('meta32', parts32)]:
            start = 0
            for label, values in parts.items():
                slices[label] = (source, start, start + len(values))
                start += len(values)

        def metadata(v):
            data = dict(v)
            for label, (source, a, z) in slices.items():
                data[label] = v[source][a:z]
            data['tree_idx'] = data['tree_idx'].view(G,N)
            return data

        data = metadata(inputs)
        for _, b, rows, _, _, _ in plans:
            a = plan_offsets[id(b)]
            rows.idx = data['idx'][a:a+len(rows.nodes)]
            rows.pos = data['pos'][a:a+len(rows.nodes)]
        inputs['x'] = torch.cat([b.hidden[rows.idx] for _, b, rows, _, _, _ in plans])[None]
        if isinstance(b0, HuginnLoop):
            inputs['static'] = torch.cat([b.static[r.idx] for _, b, r, _, _, _ in plans])[None]
        if padding:
            inputs['x'] = F.pad(inputs['x'], (0, 0, 0, padding))
            if 'static' in inputs:
                inputs['static'] = F.pad(inputs['static'], (0, 0, 0, padding))
        key = (name, padded_R, G, Q, N)
        layer_count = (len(b0.layers) if isinstance(b0, OuroLoop) else len(b0.core)) if name == 'advance' else (
            len(b0.prelude) if name == 'embed' and isinstance(b0, HuginnLoop) else
            len(b0.coda) if isinstance(b0, HuginnLoop) else 0)
        self.attention_batches += layer_count

        def forward(v):
            v = metadata(v)
            def attend(layer):
                def attention(q, k, val):
                    write_idx = (v['base']+layer)*pool.max_nodes + v['idx']
                    pool.tree_k_storage[write_idx] = k
                    pool.tree_v_storage[write_idx] = val
                    qp = q[v['query_map']].view(G,Q,*q.shape[1:])
                    kt = pool.TK[(v['gslots']+layer)[:,None], v['tree_idx']].float()
                    vt = pool.TV[(v['gslots']+layer)[:,None], v['tree_idx']].float()
                    scores = grouped_scores(qp.float(), kt) * q.shape[-1]**-0.5
                    scores = scores.masked_fill(~v['anc'][:,None], float('-inf'))
                    tree_lse = torch.logsumexp(scores, -1)
                    tree_out = grouped_values(torch.exp(scores-tree_lse[...,None]), vt)
                    tree_out = tree_out.reshape(G*Q,*q.shape[1:])[v['output_map']]
                    tree_lse = tree_lse.transpose(1,2).reshape(G*Q,q.shape[1])[v['output_map']]
                    if pool.K.device.type == 'cuda' and pool.K.dtype in (torch.float16,torch.bfloat16):
                        prefix, lse = torch.ops.aten._flash_attention_forward(
                            qp.reshape(G*Q,*q.shape[1:]), pool.K.flatten(0,1),
                            pool.V.flatten(0,1), v['cuq'], v['cuk']+layer*pool.max_seq,
                            Q, pool.max_seq, 0.0, False, False, seqused_k=v['lengths'])[:2]
                        if lse.dim() == 3:
                            lse = lse[:, :, :Q].permute(0,2,1).reshape(G*Q,q.shape[1])
                        else:
                            lse = lse.transpose(0,1)
                        prefix = prefix.float()[v['output_map']]
                        lse = lse[v['output_map']]
                    else:
                        # Dense CPU oracle; CUDA production uses the no-copy varlen path.
                        kp = pool.K[v['gslots']+layer].float()
                        vp = pool.V[v['gslots']+layer].float()
                        scores = grouped_scores(qp.float(),kp)*q.shape[-1]**-0.5
                        valid = torch.arange(pool.max_seq,device=dev)[None,:] < v['lengths'][:,None]
                        scores = scores.masked_fill(~valid[:,None,None,:],float('-inf'))
                        lse = torch.logsumexp(scores,-1)
                        prefix = grouped_values(torch.exp(scores-lse[...,None]),vp)
                        prefix = prefix.reshape(G*Q,*q.shape[1:])[v['output_map']]
                        lse = lse.transpose(1,2).reshape(G*Q,q.shape[1])[v['output_map']]
                    maximum = torch.maximum(lse,tree_lse)
                    wp, wt = torch.exp(lse-maximum)[...,None], torch.exp(tree_lse-maximum)[...,None]
                    return ((prefix*wp+tree_out*wt)/(wp+wt)).to(q.dtype)
                return attention
            x = v['x']
            if name == 'embed':
                x = b0._embed_tokens(v['tokens'][None])
                freqs = b0._positions(v['pos'])
                for li, block in enumerate(b0.prelude):
                    x = b0._block(block,x,freqs,attend(li))
                return x
            if name == 'advance':
                if isinstance(b0,OuroLoop):
                    cos,sin=(t[:,v['pos']] for t in self.rope)
                    for li,layer in enumerate(b0.layers):
                        x=b0._layer(layer,x,cos,sin,attend(li))
                    return b0.inner.norm(x)
                x=b0.model.transformer.adapter(torch.cat([x,v['static']],dim=-1))
                freqs = b0._positions(v['pos'])
                for li,block in enumerate(b0.core):
                    x=b0._block(block,x,freqs,attend(li))
                return x
            if isinstance(b0,HuginnLoop):
                x=b0._coda(x,b0._positions(v['pos']),attend)
                return b0.model.lm_head(b0.model.transformer.ln_f(x[0])).float()
            return b0.lm.lm_head(x[0]).float()

        output = self.graphs.run(key, inputs, forward)
        if name == 'advance':
            self.core_calls += 1
            self.core_rows += R
            self.core_execution_rows += padded_R
            self.max_core_rows = max(self.max_core_rows,R)
        if name in ('advance','embed'):
            for (_,b,rows,_,_,_),part in zip(plans,output[0,:R].split(counts),strict=True):
                if name == 'embed':
                    b.static[rows.idx]=part
                    b.hidden[rows.idx]=b.init[rows.pos]
                else:
                    b.hidden[rows.idx]=part
            return results
        for (j,b,rows,nodes,group_sizes,tree),part in zip(plans,output[:R].split(counts),strict=True):
            at={n.idx:i for i,n in enumerate(rows.nodes)}
            order=b.kv.upload(np.asarray([at[n.idx] for n in nodes],dtype=np.int64))
            part=part[order]
            split=sum(group_sizes)
            m=jobs[j][1].args[3]
            if name == 'readout_top':
                extra = jobs[j][1].args[4] if len(jobs[j][1].args) > 4 else None
                logs = jobs[j][1].args[5] if len(jobs[j][1].args) > 5 else False
                results[j]=(_split(part[:split].argmax(-1).tolist() if m==0 else _top_probs(part[:split],m,extra,logs),group_sizes),target_tokens(b,part[split:],[tree.position(n)+1 for n in nodes[split:]]))
            else:
                results[j]=(_top(part[:split],m), [[t] for t in target_tokens(
                    b,part[split:],[tree.position(n)+1 for n in nodes[split:]])])
        return results
