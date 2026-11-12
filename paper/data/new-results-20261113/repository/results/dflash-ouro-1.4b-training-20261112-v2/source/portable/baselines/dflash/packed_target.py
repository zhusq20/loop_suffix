"""Full-depth DFlash target using LoopFill's packed executor and owned KV.

The caller supplies the same frozen target/feature specification used to train
the drafter. Only the target execution backend changes; the drafter and rejection
sampler remain external. One cache may be active per target (batch size one).
"""
from dataclasses import dataclass

import torch

from llm_memory_editability.loop_decode_steps import ModelCall
from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, SharedKVPool, chunk_attention
from llm_memory_editability.loop_tree_decode import DraftTree, HuginnLoop, OuroLoop


@dataclass
class PackedTargetOutput:
    logits: torch.Tensor
    features: torch.Tensor | None
    cache: object


class PackedTargetCache:
    def __init__(self, owner):
        self.owner = owner

    def get_seq_length(self):
        self.owner._check_cache(self)
        return self.owner.backbone.kv.length


class PackedDFlashTarget:
    """Implement the existing DFlash target interface without HF dynamic caches."""

    def __init__(self, target, block_size=16, *, pool=None):
        if block_size < 1:
            raise ValueError('block_size must be positive')
        for key in ('model', 'family', 'max_seq', 'depth', 'device', 'dtype',
                    'embedding', 'head', 'hidden_size', 'physical_ids', 'expanded_ids', 'capture_at'):
            setattr(self, key, getattr(target, key))
        self.block_size = block_size
        pool = pool if pool is not None else SharedKVPool(1, pad_decode_rows=False)
        if self.family == 'ouro':
            b = OuroLoop(self.model, self.depth, self.max_seq, block_size, kv_factory=pool)
            layers = b.layers
            operation = '_layer'
        elif self.family == 'huginn':
            b = HuginnLoop(self.model, self.depth, self.max_seq, block_size, 0,
                          self.max_seq, extra_depths=(), kv_factory=pool)
            b.init = target.noise
            layers = b.prelude + b.core + b.coda
            operation = '_block'
        else:
            raise ValueError(f'Unsupported family: {self.family}')
        self.backbone = b
        self.engine = PackedLoopExecutor([b], cuda_graph=False)
        self.tree = DraftTree(block_size, block_size - 1)
        self._cache = None
        self._active = self._capture = False
        self._counts, self._features = {}, {}
        self._logits = None
        indices = {id(layers[i]): i for i in self.physical_ids}
        original = getattr(b, operation)

        def observed(layer, *args):
            result = original(layer, *args)
            index = indices.get(id(layer))
            if self._capture and index is not None:
                count = self._counts.get(index, 0) + 1
                self._counts[index] = count
                if count == self.capture_at[index]:
                    self._features[index] = result.detach()
            return result

        # Instance-local observation: no changes to the shared executor or model
        # blocks. Graph capture is off, so every call observes fresh features.
        setattr(b, operation, observed)

        def read_logits(module, args, output):
            if self._active:
                self._logits = output.detach()

        self._head_hook = self.head.register_forward_hook(read_logits)

    def _check_cache(self, cache):
        if not isinstance(cache, PackedTargetCache) or cache is not self._cache or cache.owner is not self:
            raise ValueError('Stale or foreign packed cache; only one request can be active')

    def new_cache(self):
        self.backbone.kv.length = 0
        self._cache = PackedTargetCache(self)
        return self._cache

    def crop(self, cache, length):
        self._check_cache(cache)
        if not 0 <= length <= self.backbone.kv.length:
            raise ValueError('Crop must retain an existing prefix')
        # Rejected positions become inaccessible. Subsequent commits overwrite
        # them in every recurrent slot before increasing the visible length.
        self.backbone.kv.length = length

    def cache_lengths(self, cache):
        self._check_cache(cache)
        slots = self.depth * self.backbone.n_layers if self.family == 'ouro' else (
            self.backbone.coda_full + len(self.backbone.coda))
        return [self.backbone.kv.length] * slots

    def close(self):
        self._head_hook.remove()
        self._cache = None
        self._features.clear()
        self._logits = None

    def _prefill(self, ids, start=0, *, chunked=False):
        """Same dense prefill operators as LoopFill, retaining all feature rows."""
        b = self.backbone
        size = ids.shape[1]
        positions = torch.arange(start, start + size, device=self.device)
        def attend(slot):
            return chunk_attention(b.kv, slot, start, start + size) if chunked else b.kv.dense(slot)
        if self.family == 'ouro':
            x = b.inner.embed_tokens(ids)
            cos, sin = b.inner.rotary_emb(x, positions[None])
            for step in range(self.depth):
                for index, layer in enumerate(b.layers):
                    x = b._layer(layer, x, cos, sin, attend(step * b.n_layers + index))
                x = b.inner.norm(x)
            self.head(x)
        else:
            freqs = b._positions(positions)
            x = b._embed_tokens(ids)
            for index, layer in enumerate(b.prelude):
                x = b._block(layer, x, freqs, attend(index))
            h = b.init[start:start + size][None]
            for step in range(self.depth):
                h = self.model.transformer.adapter(torch.cat([h, x], dim=-1))
                for index, layer in enumerate(b.core):
                    h = b._block(layer, h, freqs,
                                 attend(b.n_pre + step * b.n_core + index))
            y = b._coda(h, freqs, lambda index: attend(b.coda_full + index))
            self.head(self.model.transformer.ln_f(y))
        b.kv.length = start + size

    @torch.inference_mode()
    def prefill_chunk(self, input_ids, cache, *, features=True):
        """Append a contiguous prompt chunk, retaining all training feature rows."""
        self._check_cache(cache)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError('Expected nonempty batch-one prompt chunk')
        start, size = cache.get_seq_length(), input_ids.shape[1]
        if start + size > self.max_seq:
            raise ValueError('Prompt exceeds target context capacity')
        self._counts, self._features, self._logits = {}, {}, None
        self._active, self._capture = True, features
        try:
            self._prefill(input_ids, start, chunked=True)
            if features and self._counts != self.capture_at:
                raise RuntimeError('Incomplete prompt chunk features')
            selected = torch.cat([self._features[i] for i in self.physical_ids], -1) if features else None
            return PackedTargetOutput(self._logits.reshape(1, size, -1), selected, cache)
        except Exception:
            self._cache = None
            raise
        finally:
            self._active = self._capture = False
            self._features.clear()
            self._logits = None

    def _verify(self, ids):
        b, tree = self.backbone, self.tree
        tree.reset(b.kv.length)
        parent = None
        for token in ids[0].tolist():
            parent = tree.add(token, parent, 0, 'dflash')
        self.engine.execute('embed', [(b, ModelCall('embed', (tree, tree.nodes)))])
        for _ in range(self.depth):
            self.engine.execute('advance', [(b, ModelCall('advance', (tree, tree.nodes)))])
            for node in tree.nodes:
                node.step += 1
        self.engine.execute('readout_top', [
            (b, ModelCall('readout_top', (tree, [], tree.nodes, 0)))])
        b.commit(tree, tree.nodes)

    @torch.inference_mode()
    def forward(self, input_ids, cache=None, *, features=True, use_cache=True,
                attention_mask=None, position_ids=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError('Packed DFlash requires nonempty batch-one token IDs')
        if not use_cache or attention_mask is not None or position_ids is not None:
            raise ValueError('Only unpadded contiguous cached inference is supported')
        if cache is None:
            cache = self.new_cache()
        self._check_cache(cache)
        start, size = cache.get_seq_length(), input_ids.shape[1]
        if start + size > self.max_seq or (start and size > self.block_size):
            raise ValueError('Input exceeds target context or verification block capacity')
        self._counts, self._features = {}, {}
        self._logits = None
        self._active, self._capture = True, features
        try:
            if start == 0:
                self._prefill(input_ids)
            else:
                self._verify(input_ids)
            if self._logits is None:
                raise RuntimeError('Packed execution did not produce target logits')
            logits = self._logits.reshape(1, size, -1)
            if features:
                if self._counts != self.capture_at or set(self._features) != set(self.physical_ids):
                    raise RuntimeError(f'Incomplete target features: {self._counts}')
                selected = torch.cat([self._features[i] for i in self.physical_ids], dim=-1)
            else:
                selected = None
            return PackedTargetOutput(logits, selected, cache)
        except Exception:
            # A partially executed call cannot expose a usable cache.
            self._cache = None
            raise
        finally:
            self._active = self._capture = False
            self._features.clear()
            self._logits = None
