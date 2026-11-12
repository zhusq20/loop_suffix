"""Compare offline Huginn LAD states with the frozen inference feature interface."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--cpus',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out,[Path(__file__),BASE/'dflash_impl/target.py',BASE/'loopahead/engine.py',BASE/'loopahead/lad.py'])
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,set(range(lo,hi+1)))
        result=dict(gpu_uuid=uuid,started_unix=time.time())
        try:
            run(result,args.dtype);result['status']='complete'
        except Exception:
            result.update(status='failed',error=traceback.format_exc());raise
        finally:
            result['finished_unix']=time.time()
            (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2),flush=True)


def run(report,dtype_name):
    import torch
    from transformers import AutoTokenizer
    from dflash_impl.target import LoopTarget
    sys.path[:0]=[str(ROOT/'results/loopfill-latency512-20261107/source/src'),'/tmp/loopfill-assets/ArcticInference']
    from llm_memory_editability.loop_tree_decode import HuginnLoop
    from llm_memory_editability.loop_optimized_decode import SharedKVPool
    from loopahead.lad import LADConfig,LookaheadDrafter
    from loopahead.engine import LoopAheadExecutor
    torch.set_num_threads(1);torch.manual_seed(42)
    protocol=json.loads((ROOT/'results/loopfill-latency512-20261107/protocol.json').read_text())
    path=protocol['models']['huginn-3.5b']['path']
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    prompt=tok.encode(('Write a Python function that returns the sum of a list of integers, and explain why it works. ')*5,add_special_tokens=False)
    assert 64<len(prompt)<480
    dtype=getattr(torch,dtype_name)
    target=LoopTarget.load(path,'huginn',512,dtype=dtype)
    ids=torch.tensor([prompt],device=target.device)
    with torch.inference_mode():
        offline=target.shallow_readout_states(ids,4)
        changed=ids.clone();changed[:,-8:]=(changed[:,-8:]+13)%target.model.config.vocab_size
        future=target.shallow_readout_states(changed,4)
        torch.testing.assert_close(offline[:,:-8],future[:,:-8],rtol=0,atol=0)
        padded=torch.zeros((2,len(prompt)+8),device=target.device,dtype=torch.long)
        padded[0,:len(prompt)]=ids[0];padded[1,:len(prompt)]=changed[0]
        batched=target.shallow_readout_states(padded,4)[0,:len(prompt)]
        pool=SharedKVPool(1,pad_decode_rows=False)
        b=HuginnLoop(target.model,32,512,32,0,512,extra_depths=(8,),kv_factory=pool)
        lad=LookaheadDrafter(LADConfig(target.hidden_size)).to(device='cuda',dtype=dtype)
        engine=LoopAheadExecutor(b,lad,list(range(8192)))
        engine.request(0,{0});engine._prefill(b,prompt,4)
        inference=engine.prefix_states
        difference=(offline[0,-64:].float()-inference.float()).abs()
        report.update(model='huginn-3.5b',dtype=dtype_name,prompt_tokens=len(prompt),depth=4,
            offline_vs_inference_max_abs=float(difference.max()),
            offline_vs_inference_exact=bool(torch.equal(offline[0,-64:],inference)),
            right_padding_max_abs=float((offline[0].float()-batched.float()).abs().max()),
            future_causality_exact=True)
        # Inference and offline use the same single-example prefill shape here;
        # any difference needs investigation before enabling Huginn training.
        torch.testing.assert_close(offline[0,-64:],inference,rtol=0,atol=0)
        if dtype_name=='float32':
            torch.testing.assert_close(offline[0],batched,rtol=1e-4,atol=1e-4)
    target.close()


if __name__=='__main__':main()
