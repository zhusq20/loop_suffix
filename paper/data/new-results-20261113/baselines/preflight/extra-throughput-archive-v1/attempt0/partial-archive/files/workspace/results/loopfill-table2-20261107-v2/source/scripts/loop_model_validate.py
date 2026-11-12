"""Untimed pinned-checkpoint forward oracle, followed by short execution checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_memory_editability.loop_model_registry import MODELS, checkpoint_manifest, tokenize_prompt
from llm_memory_editability.loop_raven_decode import RavenLoop, INIT_POLICY
from llm_memory_editability.loop_tree_decode import OuroLoop, DraftTree
from llm_memory_editability.loop_optimized_decode import prefill_chunk
from loop_batch_bench import enforce_precision


@torch.no_grad()
def validate(model_id, path, out, device='cuda:0', dtype='float32'):
    spec = MODELS[model_id]
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                                torch_dtype=getattr(torch, dtype)).to(device).eval()
    model = enforce_precision(model, dtype)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if spec.backend == 'ouro':
        model.model.total_ut_steps = spec.depth_total
    def backbone():
        if spec.backend == 'ouro':
            return OuroLoop(model, spec.depth_total, 256, 8)
        return RavenLoop(model, spec.depth_total, 256, 8, extra_depths=spec.loopspec_depths)
    b = backbone()
    results = []
    for text in ('The capital of France is', 'Compute 15 + 27.\nAnswer:'):
        prompt = tokenize_prompt(spec, tok, text)
        capture = []
        handle = model.lm_head.register_forward_hook(lambda _, inputs, value: capture.append(value.detach().float().clone()))
        prediction = b.prefill(prompt, spec.wavefront_depth)
        actual = capture[-1].reshape(-1, capture[-1].shape[-1])[-1]
        handle.remove()
        ids = torch.tensor([prompt], device=device)
        kwargs = dict(input_states=b.init[:len(prompt)][None], num_steps=spec.depth_total) if spec.backend == 'raven' else {}
        native = model(input_ids=ids, use_cache=False, **kwargs).logits[0, -1].float()
        error = (actual - native).abs()
        values = native.topk(2).values
        entry = dict(prompt=text, prompt_tokens=len(prompt), max_logit_error=float(error.max()),
                     mean_logit_error=float(error.mean()), native_top1=int(native.argmax()),
                     backend_top1=prediction, native_top1_margin=float(values[0]-values[1]))
        if dtype == 'float32':
            torch.testing.assert_close(actual, native, atol=2e-4, rtol=2e-4)
            if prediction != int(native.argmax()):
                raise AssertionError(f'FP32 native greedy mismatch: {entry}')
        # Appending a branched token tests recurrent state and KV prefix on a
        # different execution path from the full native forward.
        tree = DraftTree(8, 3)
        tree.reset(len(prompt))
        root = tree.add(prediction, None, 0, 'root')
        tree.add(15, root, 0, 'draft')
        tree.add(27, root, 0, 'rejected_sibling')
        b.embed(tree, tree.take_fresh())
        for _ in range(spec.depth_total):
            b.advance(tree, tree.nodes)
            for n in tree.nodes:
                n.step += 1
        capture = []
        handle = model.lm_head.register_forward_hook(lambda _, inputs, value: capture.append(value.detach().float().clone()))
        b.readout_top(tree, [], tree.nodes, 0)
        handle.remove()
        actual_rows = capture[-1]
        branch_errors = []
        for i, node in enumerate(tree.nodes):
            tokens = prompt + [prediction] + ([] if node is root else [node.token])
            kwargs = dict(input_states=b.init[:len(tokens)][None], num_steps=spec.depth_total) if spec.backend == 'raven' else {}
            native = model(input_ids=torch.tensor([tokens], device=device), use_cache=False, **kwargs).logits[0, -1].float()
            branch_errors.append(float((actual_rows[i]-native).abs().max()))
            if dtype == 'float32':
                torch.testing.assert_close(actual_rows[i], native, atol=2e-4, rtol=2e-4)
                assert int(actual_rows[i].argmax()) == int(native.argmax())
        entry['branch_max_errors'] = branch_errors
        c = backbone()
        start = 0
        while start < len(prompt):
            start, chunk_token = prefill_chunk(c, prompt, spec.wavefront_depth, start, 7)
        entry['chunk_top1'] = chunk_token
        entry['chunk_kv_max_error'] = float((b.kv.K[:, :len(prompt)].float()-c.kv.K[:, :len(prompt)].float()).abs().max())
        if dtype == 'float32':
            torch.testing.assert_close(b.kv.K[:, :len(prompt)], c.kv.K[:, :len(prompt)], atol=2e-4, rtol=2e-4)
            assert chunk_token == prediction
        results.append(entry)
        print(json.dumps(entry), flush=True)
    report = dict(model=model_id, dtype=dtype, checkpoint=checkpoint_manifest(spec, path),
                  loaded_class=f'{type(model).__module__}.{type(model).__name__}',
                  torch=torch.__version__, transformers=sys.modules['transformers'].__version__,
                  parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                  kv_dtype=str(b.kv.K.dtype), initialization_policy=INIT_POLICY if spec.backend == 'raven' else None,
                  native_input_states='explicit identical per-position states' if spec.backend == 'raven' else None,
                  fp32_tolerance=dict(atol=2e-4, rtol=2e-4), results=results, outcome='complete')
    root = Path(__file__).resolve().parents[1]
    report['validation_source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), root/'src/llm_memory_editability/loop_tree_decode.py',
                  root/'src/llm_memory_editability/loop_raven_decode.py',
                  root/'src/llm_memory_editability/loop_optimized_decode.py']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--model-path', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=('float32','bfloat16'), default='float32')
    a = p.parse_args()
    validate(a.model, a.model_path, a.out, a.device, a.dtype)
