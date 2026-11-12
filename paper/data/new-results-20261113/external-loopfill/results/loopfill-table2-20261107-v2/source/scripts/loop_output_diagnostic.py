"""Untimed common-prefix diagnostic; never modifies benchmark inputs or snapshots.

Runs real checkpoints on selected, explicitly recorded requests. Counterfactual
arithmetic modes are diagnostic only, not performance configurations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from loop_batch_bench import arm_config, enforce_precision, SPECS
from llm_memory_editability.loop_batch_decode import BatchRequest, FusedLoopExecutor, batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop
import llm_memory_editability.loop_batch_decode as batch_module


def canonical_attention(q, k, v, kv, rows, layer, fast=None):
    """Diagnostic only: one query at a time, one FP64 softmax over all visible keys."""
    kv.TK[rows.base + layer, rows.idx] = k
    kv.TV[rows.base + layer, rows.idx] = v
    outputs = []
    masks = rows.anc.reshape(-1, rows.anc.shape[-1])
    for i in range(len(q)):
        slot = int(rows.base[i]) + layer
        visible = torch.nonzero(masks[rows.pad[i]], as_tuple=False).flatten() + rows.lo
        keys = torch.cat([kv.K[slot,:kv.length], kv.TK[slot,visible]]).double()
        values = torch.cat([kv.V[slot,:kv.length], kv.TV[slot,visible]]).double()
        scores = torch.einsum('hd,lhd->hl', q[i].double(), keys) * q.shape[-1]**-0.5
        outputs.append(torch.einsum('hl,lhd->hd', scores.softmax(-1), values).to(q.dtype))
    return torch.stack(outputs)


def difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def compare(x, y, tokenizer):
    a, b = x['logits'], y['logits']
    ai, bi = int(a.argmax()), int(b.argmax())
    values, ids = a.topk(5)
    bv, bids = b.topk(5)
    return dict(argmax_equal=ai == bi, ar_token=ai, other_token=bi,
                ar_text=tokenizer.decode([ai]), other_text=tokenizer.decode([bi]),
                max_abs_logit_error=float((a-b).abs().max()),
                mean_abs_logit_error=float((a-b).abs().mean()),
                ar_top2_margin=float(values[0]-values[1]),
                other_top2_margin=float(bv[0]-bv[1]),
                ar_pair_margin=float(a[ai]-a[bi]), other_pair_margin=float(b[ai]-b[bi]),
                ar_top5=list(zip(ids.tolist(), values.tolist())),
                other_top5=list(zip(bids.tolist(), bv.tolist())),
                head_input_max_error=float((x['head_input']-y['head_input']).abs().max()),
                ar_prefix_length=x['prefix_length'], other_prefix_length=y['prefix_length'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=SPECS, required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--tokens', type=int, default=12)
    p.add_argument('--arms', default='ar,wavefront,loopspec-dual,budget-raw,budget-online')
    p.add_argument('--modes', default='baseline,no_flash_prefix,fp32_head,rowwise_linears,fp32')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-context', type=Path, help='saved B=8 AR summary supplying original request groups')
    p.add_argument('--max-seq', type=int, default=3328)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.model_path, trust_remote_code=True,
                local_files_only=True, torch_dtype=torch.bfloat16).eval().to('cuda')
    enforce_precision(model, 'bfloat16')
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    T, d, deep, _ = SPECS[a.model]
    ouro = a.model.startswith('ouro')
    if ouro:
        model.model.total_ut_steps = T
    inputs = {r['question_id']: r for r in map(json.loads, a.questions.read_text().splitlines())}
    selected = [inputs[int(i)] for i in a.ids.split(',')]
    original = {m: m.forward for m in model.modules() if isinstance(m, torch.nn.Linear)}
    head_original = model.lm_head.forward
    metadata = dict(args={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in [Path(__file__), Path('src/llm_memory_editability/loop_tree_decode.py'),
                                  Path('src/llm_memory_editability/loop_batch_decode.py')]},
                    input_sha256=hashlib.sha256(a.questions.read_bytes()).hexdigest(),
                    selection='explicit IDs from saved probe mismatches; diagnostic, not prevalence sample',
                    note='B=1, cold per-request suffix cache, fixed token count ignoring EOS; untimed instrumentation')
    (a.out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    summary = []
    original_attention = batch_module.tree_attention
    for mode in a.modes.split(','):
        batch_module.tree_attention = canonical_attention if mode == 'canonical_attention' else original_attention
        rowwise_enabled = [False]
        for m, forward in original.items():
            m.forward = forward
        model.lm_head.forward = head_original
        enforce_precision(model, 'float32' if mode == 'fp32' else 'bfloat16')
        if mode == 'fp32_head':
            weight = model.lm_head.weight.float()
            bias = model.lm_head.bias
            model.lm_head.forward = lambda x: F.linear(x.float(), weight, None if bias is None else bias.float())
        if mode == 'rowwise_linears':
            for m, forward in original.items():
                def rowwise(x, forward=forward):
                    if not rowwise_enabled[0]:
                        return forward(x)
                    flat = x.reshape(-1, x.shape[-1])
                    y = torch.cat([forward(row[None]) for row in flat], 0)
                    return y.reshape(*x.shape[:-1], y.shape[-1])
                m.forward = rowwise
        for question in selected:
            rid, prompt = question['question_id'], question['prompt_token_ids']
            reference = None
            reference_records = None
            for arm in a.arms.split(','):
                start = time.monotonic()
                cfg = arm_config('ar' if arm == 'batch-ar' else arm, T, d, deep, 64)
                nodes = 1 if arm in ('ar', 'batch-ar') else (65 if arm.startswith('budget-') else
                        cfg.physical_nodes if arm == 'loopspec-dual' else 1152)
                group = [rid]
                if a.batch_context and arm != 'ar':
                    order = json.loads(a.batch_context.read_text())['request_order']
                    at = order.index(rid)
                    group = order[at//8*8:at//8*8+8]
                slots = []
                for qid in group:
                    b = OuroLoop(model, T, a.max_seq, nodes) if ouro else HuginnLoop(
                        model, T, a.max_seq, nodes, 0, 3328, extra_depths=(deep,))
                    if mode == 'no_flash_prefix':
                        b.kv.fast = False
                    b.committed = list(inputs[qid]['prompt_token_ids'])
                    b.diagnostic_id = qid
                    commit = b.commit
                    def tracked_commit(tree, chain, b=b, commit=commit):
                        commit(tree, chain)
                        b.committed.extend(n.token for n in chain)
                    b.commit = tracked_commit
                    slots.append(b)
                records = {}
                mask_checks = [0]
                class Capture(FusedLoopExecutor):
                    def execute(self, name, jobs):
                        rowwise_enabled[0] = name != 'prefill'
                        chunks = []
                        handle = model.lm_head.register_forward_hook(lambda m, ins, out:
                            chunks.append((ins[0].detach(), out.detach())))
                        result = super().execute(name, jobs)
                        handle.remove()
                        if name == 'prefill' and jobs[0][0].diagnostic_id == rid:
                            inp, logits = chunks[-1]
                            records[0] = dict(context=tuple(prompt), logits=logits.reshape(-1, logits.shape[-1])[-1].float().cpu(),
                                head_input=inp.reshape(-1, inp.shape[-1])[-1].float().cpu(), prefix_length=len(prompt))
                        if name not in ('readout', 'readout_top') or not chunks:
                            return result
                        inp, logits = chunks[-1]
                        inp, logits = inp.reshape(-1, inp.shape[-1]), logits.reshape(-1, logits.shape[-1])
                        offset = 0
                        for bb, call in jobs:
                            offset += self.capture_job(bb, call, name, inp, logits, offset)
                        assert offset == len(logits)
                        return result

                    def capture_job(self, bb, call, name, inp, logits, offset):
                        tree = call.args[0]
                        if name == 'readout':
                            shallow, full, _, shared = call.args[1:]
                            ns = shallow + full
                            bases = ([bb.coda_full if shared else bb.coda_shallow] * len(shallow)
                                     + [bb.coda_full] * len(full)) if not ouro else [0] * len(ns)
                        else:
                            groups, full, _ = call.args[1:]
                            ns = [n for _, g in groups for n in g] + full
                            bases = ([bb.coda_extra.get(dep, bb.coda_shallow) for dep, g in groups for _ in g]
                                     + [bb.coda_full] * len(full)) if not ouro else [0] * len(ns)
                        ordered = [n for n, _ in sorted(zip(ns, bases), key=lambda z: z[1])]
                        if bb.diagnostic_id != rid:
                            return len(ordered)
                        for i, n in enumerate(ordered):
                            if n not in full:
                                continue
                            path, root = [], n
                            while True:
                                path.append(root)
                                if root.parent < 0:
                                    break
                                root = tree.nodes[root.parent]
                            # Flushed ancestors are in main KV and outside the active tree mask.
                            visible = {j for j in range(tree.lo, len(tree.nodes)) if tree.anc[n.idx, j]}
                            expected = {v.idx for v in path if v.idx >= tree.lo}
                            assert visible == expected, (visible, expected)
                            assert sorted(tree.position(v) for v in path if v.idx >= tree.lo) == list(range(bb.length, tree.position(n)+1))
                            mask_checks[0] += 1
                            context = tuple(bb.committed[:tree.base_pos]) + tuple(v.token for v in reversed(path))
                            pos = len(context) - len(prompt)
                            if reference is None or context == tuple(prompt) + tuple(reference[:pos]):
                                records[pos] = dict(context=context, logits=logits[offset+i].float().cpu(),
                                    head_input=inp[offset+i].float().cpu(), prefix_length=bb.length)
                        return len(ordered)
                cache = CompletionSuffixCache.arctic() if cfg.use_suffix else None
                with torch.no_grad():
                    rows, _ = batch_decode([BatchRequest(qid, inputs[qid]['prompt_token_ids'], a.tokens, set())
                        for qid in group], slots, cfg, cache=cache, max_nodes=nodes, executor=Capture(slots))
                tokens = next(r['tokens'] for r in rows if r['request_id'] == rid)
                for pos, record in records.items():
                    if pos < len(tokens) and record['context'] == tuple(prompt) + tuple(tokens[:pos]):
                        assert int(record['logits'].argmax()) == tokens[pos], (arm, pos)
                if arm == 'ar':
                    reference, reference_records = tokens, records
                comparisons = {str(pos): compare(reference_records[pos], record, tok)
                    for pos, record in records.items() if pos in reference_records and record['context'] == reference_records[pos]['context']}
                first = difference(reference, tokens)
                row = dict(model=a.model, mode=mode, arm=arm, request_id=rid,
                           batch_group=group,
                           prompt_tokens=len(prompt), tokens=tokens, first_mismatch=first,
                           mask_checks=mask_checks[0], comparisons=comparisons, wall_s=time.monotonic()-start)
                with (a.out/'runs.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                if first is not None:
                    torch.save({'ar':reference_records.get(first), 'other':records.get(first)},
                        a.out/f'{mode}-{rid}-{arm}-first.pt')
                small = {k:v for k,v in row.items() if k not in ('tokens','comparisons')}
                small['first_comparison'] = comparisons.get(str(first))
                summary.append(small)
                (a.out/'summary.json').write_text(json.dumps(summary, indent=2))
                print(json.dumps(small), flush=True)
                del b, slots, cache, tracked_commit, commit
                gc.collect()
                torch.cuda.empty_cache()
    (a.out/'complete.json').write_text(json.dumps({'runs':len(summary)}))


if __name__ == '__main__':
    main()
