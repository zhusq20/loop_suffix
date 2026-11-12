"""Pinned model identities and explicit input policies for LoopFill experiments.

Historical models retain their benchmark defaults. Expansion models use tokenizer
EOS semantics (the Raven config's special IDs are inherited Huginn placeholders).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    repo_id: str
    revision: str
    backend: str
    depth_total: int
    loopspec_depths: tuple[int, int]
    wavefront_depth: int
    loopfill_readout_depths: tuple[int, int]
    prompt_format: str
    stop_token: str | None
    context_limit: int
    prelude_layers: int
    core_layers: int
    coda_layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    expansion: bool = False

    def to_dict(self):
        data = asdict(self)
        for name in ('loopspec_depths', 'loopfill_readout_depths'):
            data[name] = list(data[name])
        return data


MODELS = {s.model_id: s for s in (
    ModelSpec('ouro-2.6b', 'ByteDance/Ouro-2.6B',
              '1ed04250da1a9936042725d302e81c8fa2ab5abd', 'ouro', 4, (1, 2), 1,
              (1, 2), 'chat', '<|im_end|>', 65536, 0, 48, 0, 16, 16, 128),
    ModelSpec('huginn-3.5b', 'tomg-group-umd/huginn-0125',
              'bb6621b65e90b6a4b9b29ef88dc83866d450470c', 'huginn', 32, (4, 8), 4, (4, 8), 'chat', '<|end_turn|>',
              4096, 2, 4, 2, 55, 55, 96),
    ModelSpec('ouro-1.4b', 'ByteDance/Ouro-1.4B',
              '574fa66cb8bf5abdc979642d01cf2b79b16bfab1', 'ouro', 4, (1, 2), 1,
              (1, 2), 'chat', '<|im_end|>', 65536, 0, 24, 0, 16, 16, 128, True),
    ModelSpec('raven-llama-3.2', 'smcleish/Recurrent-Llama-3.2-train-recurrence-32',
              'a5f6f126e9d9302445346844f1d7d29d111a06eb', 'raven', 32, (2, 8), 2,
              (2, 8), 'raw', None, 131072, 4, 6, 4, 32, 8, 64, True),
    ModelSpec('raven-olmo-2-0425', 'smcleish/Recurrent-OLMo-2-0425-train-recurrence-32',
              'e1208e9af61d9e2fe1be95857e3490efb524294f', 'raven', 32, (2, 8), 2,
              (2, 8), 'raw', None, 4096, 4, 6, 4, 16, 16, 128, True),
    ModelSpec('raven-tinyllama-3t', 'smcleish/Recurrent-TinyLlama-3T-train-recurrence-32',
              '2579b6a0e9837dcea0f9fcaa85788f74d4a1f2b4', 'raven', 32, (2, 8), 2,
              (2, 8), 'raw', None, 2048, 4, 8, 4, 32, 4, 64, True),
)}
EXPANSION_MODELS = tuple(k for k, v in MODELS.items() if v.expansion)


def _ids(value):
    return [] if value is None else list(value) if isinstance(value, (list, tuple)) else [value]


def resolve_stop_ids(spec, tokenizer, model_eos, generation_eos, vocab_size, prompt_format):
    sources = dict(tokenizer=_ids(tokenizer.eos_token_id),
                   generation_config=_ids(generation_eos), model_config=_ids(model_eos))
    invalid = {name: [i for i in ids if type(i) is not int or not 0 <= i < vocab_size]
               for name, ids in sources.items()}
    source = 'tokenizer' if spec.expansion or spec.backend == 'ouro' else 'model_config'
    chosen = sources[source]
    if invalid[source] or (spec.expansion and not chosen):
        raise ValueError(f'{spec.model_id}: missing/invalid {source} EOS: {chosen}')
    stop = set(chosen)
    if prompt_format == 'chat' and spec.stop_token in tokenizer.get_vocab():
        stop.add(tokenizer.convert_tokens_to_ids(spec.stop_token))
    if any(type(i) is not int or not 0 <= i < vocab_size for i in stop):
        raise ValueError(f'{spec.model_id}: invalid stop IDs {stop}')
    return stop, dict(sources=sources, invalid_ids=invalid, selected_source=source,
                      stop_ids=sorted(stop), conflict=any(v != chosen for v in sources.values()),
                      reason=('Tokenizer defines released token semantics; Raven config/generation '
                              'special IDs are inherited placeholders.' if spec.backend == 'raven' else
                              'Released tokenizer EOS plus chat end token.') if spec.expansion else
                             'Preserve historical benchmark stopping policy.')


def tokenize_prompt(spec, tokenizer, text, prompt_format=None):
    fmt = prompt_format or spec.prompt_format
    if fmt == 'chat':
        if not getattr(tokenizer, 'chat_template', None):
            raise ValueError(f'{spec.model_id}: no released chat template')
        return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                             add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def special_token_audit(tokenizer, config, generation_config):
    """Retain BOS/PAD conflicts even when raw prompts deliberately use neither."""
    vocab = config['vocab_size']
    result = {}
    for name in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        values = dict(tokenizer=getattr(tokenizer, name, None), model_config=config.get(name),
                      generation_config=generation_config.get(name))
        result[name] = dict(sources=values, invalid_ids={
            source: [i for i in _ids(value) if type(i) is not int or not 0 <= i < vocab]
            for source, value in values.items()})
    return result


def context_audit(spec, request_id, prompt, max_new_tokens, *, max_seq=None, draft_margin=0):
    if not prompt or type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError(f'request {request_id}: empty prompt or invalid output budget')
    required = len(prompt) + max_new_tokens
    # Expansion drafts are clipped at the model/cache position boundary, never
    # by reducing the requested output budget or truncating the prompt.
    capacity = min(spec.context_limit, max_seq or spec.context_limit)
    return dict(request_id=request_id, prompt_tokens=len(prompt), max_new_tokens=max_new_tokens,
                required_context=required, model_context_limit=spec.context_limit,
                physical_capacity=max_seq, uncapped_draft_highest_position=required + draft_margin - 1,
                draft_highest_position=min(required + draft_margin, capacity) - 1,
                context_ok=required <= spec.context_limit,
                capacity_ok=max_seq is None or required <= max_seq)


def checkpoint_manifest(spec, path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file())
    return dict(model=spec.to_dict(), path=str(path.resolve()),
                file_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                             if p.suffix in ('.json', '.py', '.model', '.txt')},
                weights=[dict(name=p.name, bytes=p.stat().st_size) for p in files
                         if p.suffix in ('.safetensors', '.bin')])
