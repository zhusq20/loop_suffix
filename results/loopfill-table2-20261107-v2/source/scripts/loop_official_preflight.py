"""Independent official-forward vs production AR check on external check12 only."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from loop_suffix_spec_readouts import HuginnRunner, MODELS
from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop, tree_decode
from llm_memory_editability.loop_suffix_spec import ArmConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--hf-root',type=Path,required=True)
    p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    spec=MODELS[a.model];path=a.hf_root/spec['repo']/'snapshots'/spec['revision']
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).eval().to('cuda')
    ouro=a.model.startswith('ouro');T,d=(4,1) if ouro else (32,4)
    if ouro:model.model.total_ut_steps=T
    runner=None if ouro else HuginnRunner(model,T,0,4096,'cuda')
    rows=list(map(json.loads,a.questions.read_text().splitlines()))
    rows.sort(key=lambda r:(r['prompt_tokens'][a.model],r['prompt_id']))
    rows=[rows[i*11//5] for i in range(6)][a.shard::4]
    captured=[]
    def hook(module,args,out):captured.append(out.detach().reshape(-1,out.shape[-1])[-1].float().cpu())
    handle=model.lm_head.register_forward_hook(hook)
    results=[]
    with torch.no_grad():
        for row in rows:
            ids=tok.apply_chat_template([{'role':'user','content':row['turns'][0]}],add_generation_prompt=True)
            prompt=torch.tensor([ids],device='cuda');captured.clear();start=time.monotonic()
            if ouro:
                # The pinned checkpoint explicitly supports a growable HF DynamicCache.
                # Supply it to avoid its legacy custom cache constructor on Transformers 4.55.
                cache=DynamicCache();cache.append_new_layers(T*model.config.num_hidden_layers-1);generated=[];current=prompt
                for i in range(16):
                    output,hs,_=model.model(input_ids=current,past_key_values=cache,use_cache=True)
                    logits=model.lm_head(hs[-1][:,-1]);token=int(logits.argmax(-1));generated.append(token)
                    cache=output.past_key_values;current=torch.tensor([[token]],device='cuda')
            else:generated=runner.generate(prompt,16,set())
            ref=captured.copy();captured.clear()
            max_seq=len(ids)+16+T+64
            backbone=OuroLoop(model,T,max_seq,64) if ouro else HuginnLoop(model,T,max_seq,64,0,4096)
            actual,_=tree_decode(backbone,ids,ArmConfig('ar',T,d,64,False,False),16,set())
            got=captured.copy();captured.clear()
            checks=[]
            for i,(x,y) in enumerate(zip(ref,got)):
                top=x.topk(2).values
                checks.append({'position':i,'same_prefix':generated[:i]==actual[:i],'argmax_equal':int(x.argmax())==int(y.argmax()),'max_error':float((x-y).abs().max()),'mean_error':float((x-y).abs().mean()),'reference_top2_margin':float(top[0]-top[1])})
            result={'question_id':row['question_id'],'prompt_id':row['prompt_id'],'prompt_tokens':len(ids),'official_tokens':generated,'engine_tokens':actual,'equal':actual==generated and len(ref)==len(got)==16,'logits':checks,'wall_s':time.monotonic()-start}
            results.append(result)
            with (a.out/'requests.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ['logits','official_tokens','engine_tokens']}),flush=True)
            del backbone
    handle.remove()
    passed=bool(results) and all(r['equal'] for r in results)
    (a.out/'summary.json').write_text(json.dumps({'passed':passed,'model':a.model,'dtype':'float32','T':T,'shard':a.shard,'requests':len(results),'tf32':False,'steps':16,'note':'16 fixed prefixes for official logits diagnostic; EOS deliberately ignored for this diagnostic only'},indent=2))
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
