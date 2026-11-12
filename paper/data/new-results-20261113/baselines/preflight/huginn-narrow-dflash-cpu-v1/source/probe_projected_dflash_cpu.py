"""CPU interface/loss/cache checks for the proposed narrow Huginn drafter."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'third_party/dflash'))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import Qwen3Config, DynamicCache
    from dflash_impl.projected_draft import TargetWidthProjections, projected_config, inference_draft_class
    from provenance import freeze_sources
    torch.set_num_threads(4)
    torch.manual_seed(42)
    # Load the exact official files without SpecForge's package-level optional
    # yunchang import, which queries CUDA even for a CPU/meta-only test.
    path = BASE/'third_party/SpecForge/specforge/modeling/draft/dflash.py'
    official = load_file('cpu_official_specforge_dflash', path)
    with patch.dict(sys.modules, {'specforge.modeling.draft.dflash': official}):
        core = load_file('cpu_official_online_dflash', BASE/'third_party/SpecForge/specforge/core/dflash.py')

    class ProjectedDFlashDraftModel(TargetWidthProjections, official.DFlashDraftModel):
        pass

    infer_class = inference_draft_class()
    base_cfg = Qwen3Config.from_pretrained(BASE/'checkpoints/dflash-huginn-3.5b-formal-v1/checkpoint-004000', local_files_only=True)
    counts = []
    for width, ffn, heads in [(2048,5632,16), (1024,3072,8)]:
        config = projected_config(base_cfg, width, ffn, heads)
        with torch.device('meta'):
            train_model = ProjectedDFlashDraftModel(config)
            infer_model = infer_class(config)
        train_shapes = {k:list(v.shape) for k,v in train_model.state_dict().items()}
        assert train_shapes == {k:list(v.shape) for k,v in infer_model.state_dict().items()}
        counts.append(dict(hidden_size=width, intermediate_size=ffn, attention_heads=heads,
            head_dim=config.head_dim, trainable_parameters=sum(p.numel() for p in train_model.parameters()),
            input_projection=train_shapes['input_projection.weight'],
            feature_fusion=train_shapes['fc.weight'], output_projection=train_shapes['output_projection.weight']))
    cfg = Qwen3Config(hidden_size=32, intermediate_size=80, num_attention_heads=4,
        num_key_value_heads=4, head_dim=8, num_hidden_layers=5, num_target_layers=8,
        vocab_size=67, max_position_embeddings=128, attention_bias=False, attention_dropout=0.,
        layer_types=['full_attention']*5, block_size=16, tie_word_embeddings=False,
        dflash_config=dict(mask_token_id=0, target_layer_ids=[1,2,3,4,5]))
    cfg = projected_config(cfg, 16, 48, 2)
    cfg._attn_implementation = 'sdpa'
    draft = ProjectedDFlashDraftModel(cfg)
    embedding = torch.nn.Embedding(67,32).requires_grad_(False)
    head = torch.nn.Linear(32,67,bias=False).requires_grad_(False)
    wrapper = core.OnlineDFlashModel(draft, head, embedding, 0, block_size=16,
        attention_backend='sdpa', num_anchors=4, loss_decay_gamma=7.)
    ids = torch.randint(1,67,(2,32))
    features = torch.randn(2,32,160)
    mask = torch.ones_like(ids,dtype=torch.float32);mask[:,:4]=0
    loss, accuracy = wrapper(ids, features, mask)
    assert torch.isfinite(loss) and torch.isfinite(accuracy)
    loss.backward()
    gradients = {}
    for name in ['fc.weight','input_projection.weight','output_projection.weight',
                 'layers.0.self_attn.q_proj.weight','layers.4.mlp.down_proj.weight']:
        grad = dict(draft.named_parameters())[name].grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().sum()>0), name
        gradients[name] = float(grad.norm())
    assert embedding.weight.grad is None and head.weight.grad is None
    draft.eval()
    infer = infer_class(cfg).eval();infer.load_state_dict(draft.state_dict(),strict=True)
    context, noise = torch.randn(1,8,160), torch.randn(1,16,32)
    kwargs = dict(position_ids=torch.arange(24).unsqueeze(0), target_hidden=context,
        noise_embedding=noise, is_causal=False)
    with torch.no_grad():
        expected, actual = draft(**kwargs), infer(**kwargs)
        assert actual.shape == (1,16,32)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        checkpoint = args.out/'tiny-checkpoint'
        draft.save_pretrained(checkpoint)
        restored, info = infer_class.from_pretrained(checkpoint, local_files_only=True,
            attn_implementation='sdpa', output_loading_info=True)
        assert all(not info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        restored.eval()
        torch.testing.assert_close(restored(**kwargs), actual, rtol=0, atol=0)
        cache = DynamicCache()
        restored(**kwargs, past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == 24
        cache.crop(8)
        extra = torch.randn(1,3,160)
        cached = restored(position_ids=torch.arange(8,27).unsqueeze(0), target_hidden=extra,
            noise_embedding=noise, past_key_values=cache, use_cache=True, is_causal=False)
        full = restored(position_ids=torch.arange(27).unsqueeze(0), target_hidden=torch.cat([context,extra],1),
            noise_embedding=noise, is_causal=False)
        torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)
        assert all(layer.get_seq_length()==27 for layer in cache.layers)
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), BASE/'dflash_impl/projected_draft.py', BASE/'provenance.py', path,
        BASE/'third_party/SpecForge/specforge/core/dflash.py', BASE/'third_party/dflash/dflash/model.py']
    freeze_sources(args.out, sources)
    report = dict(status='passed_cpu_interface_checks', configurations=counts,
        full_huginn_width_parameters=2116288320,
        reduction_fraction=1-counts[0]['trainable_parameters']/2116288320,
        tiny_official_training_loss=float(loss.detach()), tiny_accuracy=float(accuracy.detach()),
        projection_and_block_gradients=gradients, target_embedding_and_head_frozen=True,
        train_inference_max_abs=float((expected-actual).abs().max()),
        cache_crop_recompute_max_abs=float((cached-full).abs().max()),
        checkpoint_roundtrip_exact=True, cuda_initialized=False,
        torch=str(torch.__version__),
        limitations=['Parameter counts are exact for these proposed architectures, not speed or quality measurements.',
            'Forward/backward/cache checks use a small random model on CPU, not the full Huginn target.',
            'SpecForge files were loaded directly to avoid unrelated optional package CUDA initialization.',
            'GPU memory, real-target correctness, training and held-out speed/acceptance tests remain pending.'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
