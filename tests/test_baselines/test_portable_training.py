"""Actual optimizer/checkpoint/decoder integration with tiny frozen target layers.

No downloads, model snapshots, external repos, or GPU are needed. These validate
the training plumbing and packed inference, not quality of trained full-size models.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from baselines import extract_features, infer, train_dflash, train_loopahead
from baselines.common import save
from baselines.dflash.decode import generate
from baselines.dflash.projected_draft import training_draft_class
from baselines.loopahead.lad import LADConfig
from baselines.loopahead.training_data import FeatureStore
from baselines.dflash.packed_target import PackedDFlashTarget
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop
from test_loop_batch_decode import TinyHuginn, TinyOuro


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyTarget:
    def __init__(self, family='ouro'):
        torch.manual_seed(731)
        self.family, self.max_seq = family, 128
        self.depth = 4 if family == 'ouro' else 32
        self.model = (TinyOuro() if family == 'ouro' else TinyHuginn()).eval().requires_grad_(False)
        self.model.config.vocab_size = 19
        self.device, self.dtype, self.hidden_size = torch.device('cpu'), torch.float32, 16
        self.embedding = self.model.model.embed_tokens if family == 'ouro' else self.model.transformer.wte
        self.head = self.model.lm_head
        self.capture_at = {0: 4, 1: 4} if family == 'ouro' else {0: 1, 1: 32, 2: 32, 3: 1}
        self.physical_ids = self.expanded_ids = list(self.capture_at)
        self.noise = None if family == 'ouro' else HuginnLoop(self.model, 32, 128, 16, 0, 128).init
        self.packed = PackedDFlashTarget(self)
        self.frozen_weights = deepcopy(self.model.state_dict())

    def forward(self, ids, features=True, use_cache=False):
        assert not use_cache
        outputs = [self.packed.forward(row[None], self.packed.new_cache()) for row in ids]
        # Clone outside inference mode: the frozen feature tensor must be saveable for backward.
        return SimpleNamespace(features=torch.cat([o.features.clone() for o in outputs]),
                               logits=torch.cat([o.logits.clone() for o in outputs]))

    @torch.no_grad()
    def shallow_readout_states(self, ids, depth):
        backbone = (OuroLoop(self.model, depth, 128, 16) if self.family == 'ouro' else
                    HuginnLoop(self.model, depth, 128, 16, 0, 128))
        if self.family == 'huginn':
            backbone.init = self.noise
        states = []
        if self.family == 'ouro':
            hook = self.model.model.norm.register_forward_hook(
                lambda m, inputs, output: states.append(output.clone()))
        else:
            original = backbone._coda

            def coda(*args):
                result = original(*args)
                states.append(self.model.transformer.ln_f(result).clone())
                return result

            backbone._coda = coda
            hook = None
        try:
            for row in ids:
                backbone.prefill(row.tolist(), None)
        finally:
            if hook:
                hook.remove()
        return torch.cat(states)

    def close(self):
        for key, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, self.frozen_weights[key], rtol=0, atol=0)
        assert not any(p.requires_grad or p.grad is not None for p in self.model.parameters())
        self.packed.close()


def tiny_config(target, projected):
    width = 8 if projected else 16
    details = dict(mask_token_id=0, target_layer_ids=target.expanded_ids)
    if projected:
        details.update(projection_adapter='target-width-projections-v1', target_hidden_size=16)
    return Qwen3Config(hidden_size=width, intermediate_size=24, num_hidden_layers=5,
        num_attention_heads=2, num_key_value_heads=2, head_dim=width // 2,
        num_target_layers=132, vocab_size=19, max_position_embeddings=128,
        layer_types=['full_attention'] * 5, block_size=16, dflash_config=details,
        tie_word_embeddings=False)


@pytest.mark.parametrize('projected', [False, True])
def test_dflash_training_resume_and_packed_inference(tmp_path, monkeypatch, projected):
    target = TinyTarget()
    config = tiny_config(target, projected)
    config_path = tmp_path / 'config.json'
    config.to_json_file(config_path)
    target.close()
    teachers = tmp_path / 'teachers.jsonl'
    teachers.write_text(json.dumps(dict(source_id='train/0', prompt_tokens=[1, 2, 3],
                                        output_tokens=[i % 19 for i in range(40)])) + '\n')
    monkeypatch.setattr(train_dflash, 'load_target', lambda args: TinyTarget())
    base = ['--model', 'ouro-2.6b', '--target-path', str(tmp_path), '--device', 'cpu',
            '--dtype', 'float32', '--attention', 'sdpa', '--teachers', str(teachers),
            '--draft-config', str(config_path), '--epochs', '2', '--micro-batch', '1',
            '--num-anchors', '2', '--save-every', '1', '--max-seq', '128']
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    train_dflash.main(base + ['--out', str(full)])
    train_dflash.main(base + ['--out', str(resumed), '--stop-after', '1'])
    assert not (resumed / 'complete.json').exists()
    train_dflash.main(base + ['--out', str(resumed), '--resume', str(resumed / 'checkpoint-000001')])
    a = torch.load(full / 'checkpoint-000002/training-state.pt', weights_only=False)
    b = torch.load(resumed / 'checkpoint-000002/training-state.pt', weights_only=False)
    for name in a['draft']:
        torch.testing.assert_close(a['draft'][name], b['draft'][name], rtol=0, atol=0)
    first = torch.load(full / 'checkpoint-000001/training-state.pt', weights_only=False)
    assert any(not torch.equal(a['draft'][k], first['draft'][k]) for k in a['draft'])
    target = TinyTarget()
    frozen = deepcopy(target.model.state_dict())
    draft = infer.load_dflash(full / 'checkpoint-000002', target)
    prompt = torch.tensor([[1, 2, 3]])
    greedy = generate(target.packed, prompt, 19, set()).tokens
    assert generate(target.packed, prompt, 19, set(), draft=draft).tokens == greedy
    for temperature in (0., 1.):
        for budget in (1, 2, 17, 19):
            result = generate(target.packed, prompt, budget, set(), draft=draft, temperature=temperature)
            assert len(result.tokens) == budget
            assert sum(result.stats['emitted_per_verification']) == budget - 1
    for key, tensor in target.model.state_dict().items():
        torch.testing.assert_close(tensor, frozen[key], rtol=0, atol=0)
    target.close()
    inputs = tmp_path / 'inputs.jsonl'
    inputs.write_text(json.dumps(dict(source_id='test/0', prompt_tokens=[1, 2, 3])) + '\n')
    monkeypatch.setattr(infer, 'load_target', lambda args: TinyTarget())
    infer.main(['--model', 'ouro-2.6b', '--target-path', str(tmp_path), '--method', 'dflash',
                '--checkpoint', str(full / 'checkpoint-000002'), '--inputs', str(inputs),
                '--device', 'cpu', '--dtype', 'float32', '--stop-ids', '--max-new-tokens', '19',
                '--max-seq', '128', '--out', str(tmp_path / 'decoded')])
    record = json.loads((tmp_path / 'decoded/outputs.jsonl').read_text())
    assert record['output_tokens'] == greedy


def test_default_huginn_parameter_count():
    path = Path(train_dflash.__file__).parent / 'configs/dflash-huginn-332m.json'
    config = Qwen3Config.from_pretrained(path, local_files_only=True)
    with torch.device('meta'):
        model = training_draft_class()(config)
    assert sum(p.numel() for p in model.parameters()) == 332_621_056


@pytest.mark.parametrize('family,model_id', [('ouro', 'ouro-2.6b'), ('huginn', 'huginn-3.5b')])
def test_loopahead_extraction_training_and_inference(tmp_path, monkeypatch, family, model_id):
    features = tmp_path / 'features'
    monkeypatch.setattr(extract_features, 'load_target', lambda args: TinyTarget(family))
    for stage in ('train', 'dev', 'confirmation'):
        teachers = tmp_path / f'{stage}.jsonl'
        teachers.write_text(json.dumps(dict(source_id=f'{stage}/0', pool='math',
            prompt_tokens=[i % 19 for i in range(65)], output_tokens=[(i + 7) % 19 for i in range(10)])) + '\n')
        extract_features.main(['--model', model_id, '--target-path', str(tmp_path),
            '--device', 'cpu', '--dtype', 'float32', '--teachers', str(teachers),
            '--features', str(features), '--stage', f'loopahead-{stage}', '--max-seq', '128'])
    store = FeatureStore(features / 'loopahead-train' / model_id, 'loopahead-train', model_id,
                         features / 'manifest.json')
    assert len(store.rows) == 1 and store.eligible == [0]
    cfg = tmp_path / 'lad-config.json'
    save(cfg, LADConfig(target_hidden=16, width=16, heads=2, intermediate=24).to_dict())
    out = tmp_path / 'trained'
    train_args = ['--model', model_id, '--features', str(features), '--device', 'cpu',
        '--dtype', 'float32', '--stop-ids', '--lad-config', str(cfg),
        '--epochs', '1', '--anchors-per-request', '2', '--micro-batch', '1', '--accumulation', '1']
    train_loopahead.main(train_args + ['--out', str(out)])
    resumed = tmp_path / 'resumed'
    train_loopahead.main(train_args + ['--out', str(resumed), '--stop-after', '1'])
    assert not (resumed / 'dev-metrics.jsonl').exists()
    assert not (resumed / 'confirmation.json').exists()
    train_loopahead.main(train_args + ['--out', str(resumed), '--resume', str(resumed / 'checkpoint-000001')])
    full_state = torch.load(out / 'checkpoint-000002/lad.pt', weights_only=True)
    resumed_state = torch.load(resumed / 'checkpoint-000002/lad.pt', weights_only=True)
    for key in full_state['model']:
        torch.testing.assert_close(full_state['model'][key], resumed_state['model'][key], rtol=0, atol=0)
    completed = json.loads((out / 'complete.json').read_text())
    assert completed['step'] == 2 and (out / 'confirmation.json').exists()
    target = TinyTarget(family)
    lad, frequent = infer.load_loopahead(completed['selected_checkpoint'], target)
    prompt = [i % 19 for i in range(65)]
    greedy = generate(target.packed, torch.tensor([prompt]), 17, set()).tokens
    for temperature in (0., 1.):
        backbone, engine, cfg = infer.loopahead_engine(target, lad, frequent, temperature)
        for budget in (1, 2, 17):
            tokens, stats = infer.generate_loopahead(backbone, engine, cfg, prompt, budget, set())
            assert len(tokens) == budget and stats['decode_tokens'] == budget - 1
            if temperature == 0:
                assert tokens == greedy[:budget]
    target.close()
    monkeypatch.setattr(infer, 'load_target', lambda args: TinyTarget(family))
    inputs = tmp_path / 'inputs.jsonl'
    inputs.write_text(json.dumps(dict(source_id='test/0', prompt_tokens=prompt)) + '\n')
    infer.main(['--model', model_id, '--target-path', str(tmp_path), '--method', 'loopahead',
                '--checkpoint', completed['selected_checkpoint'], '--inputs', str(inputs),
                '--device', 'cpu', '--dtype', 'float32', '--stop-ids', '--max-new-tokens', '17',
                '--max-seq', '128', '--out', str(tmp_path / 'decoded')])
    assert json.loads((tmp_path / 'decoded/outputs.jsonl').read_text())['output_tokens'] == greedy
    # Corrupted serialized features must not reach training.
    chunk = next((features / 'loopahead-train' / model_id).glob('shard*/chunk*.pt'))
    with chunk.open('ab') as f:
        f.write(b'corruption')
    with pytest.raises((ValueError, AssertionError)):
        FeatureStore(features / 'loopahead-train' / model_id, 'loopahead-train', model_id,
                     features / 'manifest.json')
