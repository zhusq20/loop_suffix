"""Pinned external-only AR/readout generation, full prompts, natural EOS."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from loop_suffix_spec_readouts import MODELS, OuroRunner, HuginnRunner


class CompatibleOuroRunner(OuroRunner):
    @torch.no_grad()
    def generate(self,prompt,max_new,stop):
        cache=DynamicCache()
        if hasattr(cache,'append_new_layers'):
            cache.append_new_layers(self.T*self.model.config.num_hidden_layers-1)
        output=[];current=prompt
        for _ in range(max_new):
            result,hs,_=self.model.model(input_ids=current,past_key_values=cache,use_cache=True)
            token=int(self.model.lm_head(hs[-1][:,-1]).argmax(-1));output.append(token)
            if token in stop:break
            cache=result.past_key_values;current=torch.tensor([[token]],device=prompt.device)
        return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--hf-root',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gate-root',type=Path,required=True)
    a=p.parse_args()
    for gpu in range(8):
        gate=json.loads((a.gate_root/f'official-fp32-gpu{gpu}'/'summary.json').read_text())
        if not gate['passed']:raise RuntimeError(f'official reference gate failed: GPU {gpu}')
    source=Path('results/loopfill-icml-20261103/external-calibration/calib128.jsonl')
    rows=list(map(json.loads,source.read_text().splitlines()))
    assert len(rows)==128 and [r['question_id'] for r in rows]==list(range(128))
    rows=rows[a.shard::4]
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'readouts').mkdir(exist_ok=True)
    index=a.out/f'index-shard{a.shard}.jsonl'
    if index.exists():raise FileExistsError(index)
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=MODELS[a.model];path=a.hf_root/spec['repo']/'snapshots'/spec['revision']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16).eval().to('cuda')
    ouro=a.model.startswith('ouro');T,d,deep=(4,1,2) if ouro else (32,4,8)
    if ouro:model.model.total_ut_steps=T
    runner=CompatibleOuroRunner(model,T) if ouro else HuginnRunner(model,T,0,4096,'cuda')
    eos=tok.eos_token_id if ouro else model.config.eos_token_id
    stop=set(eos if isinstance(eos,list) else [eos]);stop.add(tok.convert_tokens_to_ids(spec['end_turn']))
    depths=[d,deep,T]
    with torch.no_grad(),index.open('x') as log:
        for row in rows:
            prompt=tok.apply_chat_template([{'role':'user','content':row['turns'][0]}],add_generation_prompt=True)
            if len(prompt)!=row['prompt_tokens'][a.model] or len(prompt)+1024+T+64>4096:raise ValueError('tokenizer/budget mismatch')
            ids=torch.tensor([prompt],device='cuda');torch.cuda.synchronize();start=time.monotonic()
            output=runner.generate(ids,1024,stop);torch.cuda.synchronize();generated=time.monotonic()
            seq=torch.tensor([prompt+output],device='cuda')
            reads=runner.readouts(seq,len(prompt)-1,depths,8);torch.cuda.synchronize();end=time.monotonic()
            target=a.out/'readouts'/f"{row['question_id']}.npz"
            if target.exists():raise FileExistsError(target)
            np.savez_compressed(target,prompt=np.asarray(prompt,dtype=np.int32),output=np.asarray(output,dtype=np.int32),depths=np.asarray(depths,dtype=np.int32),ids=np.stack([r[0] for r in reads]),probs=np.stack([r[1] for r in reads]))
            record={'question_id':row['question_id'],'prompt_id':row['prompt_id'],'category':row['category'],'prompt_len':len(prompt),'output_len':len(output),'ended_eos':output[-1] in stop,'generate_s':generated-start,'readout_s':end-generated,'full_depth_teacher_forced_agreement':float(np.mean(reads[-1][0][:len(output),0]==output)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
    (a.out/f'shard{a.shard}-complete.json').write_text(json.dumps({'requests':len(rows),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'dtype':'bfloat16','max_new_tokens':1024,'depths':depths,'source':'external_ultrachat_only','wandb':'not_enabled_no_authenticated_tracking_configured'}))


if __name__=='__main__':main()
