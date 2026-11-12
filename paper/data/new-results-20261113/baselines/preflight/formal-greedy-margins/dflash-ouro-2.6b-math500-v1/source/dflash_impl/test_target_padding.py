"""Check actual Ouro checkpoint code under variable-length padded cached decoding."""
import json
from pathlib import Path
import unittest

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .target import LoopTarget


class PaddingChecks(unittest.TestCase):
    def test_cached_padding_matches_unpadded_positions(self):
        torch.manual_seed(17)
        base = Path(__file__).resolve().parents[1]
        protocol = json.loads((base.parents[1] / 'results/loopfill-latency512-20261107/protocol.json').read_text())
        cfg = AutoConfig.from_pretrained(protocol['models']['ouro-2.6b']['path'], trust_remote_code=True, local_files_only=True)
        cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 24, 16, 32
        cfg.num_attention_heads = cfg.num_key_value_heads = 2
        cfg.head_dim, cfg.vocab_size = 8, 32
        cfg.layer_types = ['full_attention'] * 24
        cfg._attn_implementation = 'sdpa'
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()
        target = LoopTarget(model, 'ouro', 64)
        full_states = []
        hook = model.model.norm.register_forward_hook(lambda module, inputs, output: full_states.append(output.detach().clone()))
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        target.forward(ids, features=False, use_cache=False)
        hook.remove()
        shallow = target.shallow_readout_states(ids, 1)
        torch.testing.assert_close(shallow, full_states[0], atol=0, rtol=0)
        self.assertEqual(model.model.total_ut_steps, 4)
        changed = target.shallow_readout_states(torch.tensor([[3, 4, 5, 14, 18]]), 1)
        torch.testing.assert_close(shallow[:, :3], changed[:, :3], atol=0, rtol=0)
        padded = torch.tensor([[0, 0, 3, 4, 5], [6, 7, 8, 9, 10]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        lengths = mask.sum(-1)
        cache = target.new_cache()
        out = target.forward(padded, cache, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp_min(0))
        references = [target.new_cache(), target.new_cache()]
        for i, length in enumerate(lengths):
            single = target.forward(padded[i:i + 1, -int(length):], references[i])
            torch.testing.assert_close(out.logits[i, -int(length):], single.logits[0], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(out.features[i, -int(length):], single.features[0], atol=2e-6, rtol=2e-5)
        for step in range(3):
            token = torch.tensor([[11 + step], [15 + step]])
            mask = torch.cat([mask, torch.ones((2, 1), dtype=mask.dtype)], -1)
            out = target.forward(token, cache, attention_mask=mask, position_ids=(lengths + step)[:, None])
            for i in range(2):
                single = target.forward(token[i:i + 1], references[i])
                torch.testing.assert_close(out.logits[i], single.logits[0], atol=2e-6, rtol=2e-5)
        target.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
