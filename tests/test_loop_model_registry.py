from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_memory_editability.loop_model_registry import (
    MODELS, EXPANSION_MODELS, context_audit, resolve_stop_ids, tokenize_prompt,
)
from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig


class Tokenizer:
    eos_token_id = 2
    chat_template = None

    def get_vocab(self):
        return {'</s>': 2, 'word': 3}

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[3])


def test_pinned_identity_and_separate_readout_policies():
    assert len(MODELS) == 6 and len(EXPANSION_MODELS) == 4
    for name in EXPANSION_MODELS:
        s = MODELS[name]
        assert len(s.revision) == 40 and all(c in '0123456789abcdef' for c in s.revision)
        cfg = LoopSpecConfig('loopspec-dual', s.depth_total, *s.loopspec_depths)
        assert cfg.physical_nodes == (15 if s.backend == 'ouro' else 499)
        changed = replace(s, loopfill_readout_depths=(4, 8))
        assert changed.loopspec_depths == s.loopspec_depths


def test_tinyllama_invalid_config_eos_and_raw_no_bos():
    spec = MODELS['raven-tinyllama-3t']
    tok = Tokenizer()
    stops, audit = resolve_stop_ids(spec, tok, 65505, 65505, 32000, 'raw')
    assert stops == {2} and audit['invalid_ids']['model_config'] == [65505]
    assert tokenize_prompt(spec, tok, 'word') == [3]
    with pytest.raises(ValueError, match='no released chat template'):
        tokenize_prompt(spec, tok, 'word', 'chat')
    tok.eos_token_id = 32000
    with pytest.raises(ValueError, match='invalid'):
        resolve_stop_ids(spec, tok, 65505, 65505, 32000, 'raw')


def test_context_and_physical_capacity_are_distinct():
    s = MODELS['raven-tinyllama-3t']
    audit = context_audit(s, 42, [3] * 1100, 1024, max_seq=2816, draft_margin=36)
    assert not audit['context_ok'] and audit['capacity_ok']
    assert audit['required_context'] == 2124 and audit['draft_highest_position'] == 2047
    exact = context_audit(s, 43, [3] * 1024, 1024, max_seq=2048, draft_margin=36)
    assert exact['context_ok'] and exact['draft_highest_position'] == 2047


def test_checkpoint_model_identity_survives_json_roundtrip():
    import json
    for spec in MODELS.values():
        assert json.loads(json.dumps(spec.to_dict())) == spec.to_dict()
