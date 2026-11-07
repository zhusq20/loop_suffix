"""Locate the first arithmetic divergence for the first generated Ouro token.

Instrument only; keep the original production forward and scheduler untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from loop_batch_bench import arm_config, enforce_precision
import llm_memory_editability.loop_batch_decode as batch
import llm_memory_editability.loop_tree_decode as tree_module
from llm_memory_editability.loop_tree_decode import OuroLoop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--id', type=int, default=264)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,
                local_files_only=True, torch_dtype=torch.bfloat16).eval().to('cuda')
    enforce_precision(model, 'bfloat16')
    model.model.total_ut_steps = 4
    question = next(r for r in map(json.loads, args.questions.read_text().splitlines()) if r['question_id'] == args.id)
    prompt = question['prompt_token_ids']
    active = [None]
    events = {}
    original_attention = batch.tree_attention
    original_prefix = tree_module._prefix_attention
    handles = []
    def hook(name):
        def observe(module, inputs, output):
            if active[0] is None or not isinstance(output, torch.Tensor) or output.ndim < 2:
                return
            step, row = active[0]
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[row].float().cpu()
            y = output.detach().reshape(-1, output.shape[-1])[row].float().cpu()
            events[f'{step}:{name}'] = {'input':x, 'output':y}
        return observe
    for name, m in model.named_modules():
        if not list(m.children()):
            handles.append(m.register_forward_hook(hook(name)))
    def attention(q, k, v, kv, rows, layer, fast=None):
        out = original_attention(q, k, v, kv, rows, layer, fast)
        if active[0] is not None:
            step, _ = active[0]
            index = next((i for i, n in enumerate(rows.nodes) if int(rows.pos[i]) == len(prompt)), None)
            if index is not None:
                base = int(rows.base[index]) + layer
                G, M, count = rows.anc.shape
                kt = kv.TK[:, rows.lo:rows.lo+count][rows.gslot+layer].float()
                qp = q.new_zeros(G*M, *q.shape[1:])
                qp[rows.pad] = q
                scores = torch.einsum('gihd,gnhd->ghin', qp.view(G,M,*q.shape[1:]).float(), kt) * q.shape[-1]**-0.5
                scores = scores.masked_fill(~rows.anc[:,None], float('-inf'))
                tree_lse = torch.logsumexp(scores, dim=-1).transpose(1,2).reshape(G*M,-1)[rows.pad]
                def digest(t):
                    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                events[f'{step}:attention:{layer}'] = {
                    'q':q[index].float().cpu(), 'k':k[index].float().cpu(), 'v':v[index].float().cpu(),
                    'tree_lse':tree_lse[index].cpu(),
                    'output':out[index].float().cpu(),
                    'prefix_k_sha256':digest(kv.K[base,:kv.length]),
                    'prefix_v_sha256':digest(kv.V[base,:kv.length]),
                    'prefix_length':kv.length, 'query_rows':len(q),
                    'visible_tree_nodes':int(rows.anc.reshape(-1, rows.anc.shape[-1])[int(rows.pad[index])].sum()),
                    'visible_tree_positions':[int(rows.pos[index])],
                }
        return out
    batch.tree_attention = attention
    def prefix(q, kv, rows, layer, fast):
        out, lse = original_prefix(q, kv, rows, layer, fast)
        if active[0] is not None:
            step, _ = active[0]
            index = next((i for i, n in enumerate(rows.nodes) if int(rows.pos[i]) == len(prompt)), None)
            if index is not None:
                events[f'{step}:prefix_attention:{layer}'] = {
                    'output':out[index].float().cpu(), 'lse':lse[index].float().cpu()}
        return out, lse
    tree_module._prefix_attention = prefix
    class Trace(batch.FusedLoopExecutor):
        def execute(self, name, jobs):
            active[0] = None
            if name == 'advance':
                assert len(jobs) == 1
                _, call = jobs[0]
                tree, nodes = call.args
                ordered = sorted(nodes, key=lambda n:n.step)
                for i,n in enumerate(ordered):
                    if tree.position(n) == len(prompt):
                        active[0] = (n.step, i)
            result = super().execute(name, jobs)
            active[0] = None
            return result
    reference = {}
    summary = {}
    for mode in ['baseline', 'no_flash_prefix']:
        for arm in ['ar','wavefront']:
            b = OuroLoop(model,4,1024,128)
            if mode == 'no_flash_prefix':
                b.kv.fast = False
            events = {}
            cfg = arm_config(arm,4,1,2,64)
            rows,_ = batch.batch_decode([batch.BatchRequest(args.id,prompt,4,set())],[b],cfg,
                        max_nodes=128,executor=Trace([b]))
            torch.save(events,args.out/f'{mode}-{arm}.pt')
            if arm == 'ar':
                reference = events
            comparisons=[]
            for key,event in events.items():
                ref = reference[key]
                row={'operation':key}
                different=False
                for field,value in event.items():
                    if isinstance(value,torch.Tensor):
                        error=float((value-ref[field]).abs().max())
                        row[field+'_max_error']=error
                        different |= error != 0
                    else:
                        row[field+'_equal']=value==ref[field]
                if different:
                    comparisons.append(row)
            summary[f'{mode}-{arm}']={'tokens':rows[0]['tokens'],'events':len(events),
                'first_differing_operations':comparisons[:20],
                'differing_operations':len(comparisons)}
            (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps({k:v for k,v in summary[f'{mode}-{arm}'].items() if k!='first_differing_operations'}),flush=True)
    for handle in handles:
        handle.remove()
    batch.tree_attention=original_attention
    tree_module._prefix_attention=original_prefix


if __name__=='__main__':
    main()
