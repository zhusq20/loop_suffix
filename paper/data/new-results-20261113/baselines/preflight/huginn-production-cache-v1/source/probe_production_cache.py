"""Real Huginn cache equivalence and memory/cost diagnostic on synthetic text."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--cpus',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out,[Path(__file__),BASE/'dflash_impl/target.py',BASE/'dflash_impl/production_cache.py',BASE/'provenance.py'])
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid={int(i.strip()):u.strip() for i,u in (l.split(',') for l in lines.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        assert uuid not in busy
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,set(range(lo,hi+1)))
        report=dict(stage='production_cache_diagnostic',gpu_uuid=uuid,started_unix=time.time())
        try:
            run(report)
            report['status']='complete'
        except Exception:
            report.update(status='failed',error=traceback.format_exc());raise
        finally:
            report['finished_unix']=time.time()
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2),flush=True)


def run(report):
    import torch
    from transformers import AutoTokenizer
    from dflash_impl.target import LoopTarget
    from dflash_impl.production_cache import HuginnProductionCache
    torch.set_num_threads(1);torch.manual_seed(0)
    protocol=json.loads((ROOT/'results/loopfill-latency512-20261107/protocol.json').read_text())
    path=protocol['models']['huginn-3.5b']['path']
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    target=LoopTarget.load(path,'huginn',512)
    prompt=tok.encode('Write a short Python function that adds two integers. Explain how to call this function.\n',add_special_tokens=False)
    inputs=torch.tensor([prompt]*16,device=target.device)
    gold_inputs,gold_logits=[],[]
    report.update(batch=16,prompt_tokens=len(prompt),decode_steps=128,noise_capacity=512,cache_capacity=len(prompt)+128)
    for mode in ['dictionary','preallocated']:
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        cache=target.new_cache() if mode=='dictionary' else HuginnProductionCache(len(prompt)+128)
        events=[];matches=0;max_difference=0.;top_matches=0
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            out=target.forward(inputs,cache,features=False)
            token=out.logits[:,-1].argmax(-1)
            del out
            for step in range(128):
                if mode=='dictionary':
                    gold_inputs.append(token.cpu())
                else:
                    token=gold_inputs[step].to(target.device)
                first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                first.record()
                out=target.forward(token[:,None],cache,features=False)
                last.record();events.append((first,last))
                logits=out.logits[:,0].cpu()
                if mode=='dictionary':
                    gold_logits.append(logits)
                    token=out.logits[:,0].argmax(-1)
                else:
                    matches+=int(torch.equal(logits,gold_logits[step]))
                    top_matches+=int((logits.argmax(-1)==gold_logits[step].argmax(-1)).sum())
                    max_difference=max(max_difference,float((logits.float()-gold_logits[step].float()).abs().max()))
                del out,logits
            torch.cuda.synchronize()
        report[mode]=dict(wall_s=time.perf_counter()-start,forward_gpu_s=sum(a.elapsed_time(b) for a,b in events)/1000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cache_slots=len(cache.key_cache),cache_length=cache.get_seq_length())
        assert len(cache.key_cache)==132 and cache.get_seq_length()==len(prompt)+128
        if mode=='preallocated':
            report.update(exact_logit_steps=matches,top1_matches=top_matches,comparisons=16*128,
                          max_abs_logit_difference=max_difference)
            assert matches==128 and top_matches==16*128,'Cache layout changed numeric results; investigate before adoption'
        del cache
        print(mode,json.dumps(report[mode]),flush=True)
    report['limitations']=['synthetic fixed-prefix compatibility test only','not a benchmark speedup',
                          'production optimization must not silently replace the timed DFlash inference cache']
    target.close()


if __name__=='__main__':main()
