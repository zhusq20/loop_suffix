"""Probe graph AR production on real non-test prompts before enabling it."""
import argparse
import fcntl
import gc
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
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--cpus',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out,[Path(__file__),BASE/'teacher_graph.py',BASE/'dflash_impl/target.py',BASE/'provenance.py'])
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,set(range(lo,hi+1)))
        report=dict(gpu_uuid=uuid,started_unix=time.time(),results=[])
        try:
            run(report);report['status']='complete'
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
    from dflash_impl.decode import generate
    from teacher_graph import GraphTeacher
    from llm_memory_editability.loop_model_registry import MODELS,tokenize_prompt
    torch.set_num_threads(1);torch.manual_seed(0)
    protocol=json.loads((ROOT/'results/loopfill-latency512-20261107/protocol.json').read_text())
    source=BASE/'data/manifests/loopahead-train.jsonl'
    rows=sorted([json.loads(l) for l in source.open()],key=lambda r:r['source_id'])
    # Fixed IDs chosen from training data, all five domains; no evaluation prompts.
    chosen=[]
    for pool in ['gsm8k','python','math','reasoning','ultrachat']:
        chosen.extend([r for r in rows if r['pool']==pool][:1 if pool in ('reasoning','ultrachat') else 2])
    assert len(chosen)==8
    report['source_ids']=[r['source_id'] for r in chosen]
    limits=[8,16,24,32,40,48,56,64]
    for name in ['huginn-3.5b','ouro-2.6b']:
        spec=MODELS[name];path=protocol['models'][name]['path']
        tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
        prompts=[tokenize_prompt(spec,tok,r['prompt'],'completion') for r in chosen]
        capacity=((max(map(len,prompts))+64+127)//128)*128
        for dtype in ['bfloat16','float32']:
            target=LoopTarget.load(path,spec.backend,3072,dtype=getattr(torch,dtype))
            results={};times={};details={}
            for graph in [False,True]:
                engine=GraphTeacher(target,8,capacity,graph=graph)
                warm=engine.warm(prompts)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                outputs,summary=engine.generate(prompts,limits,set())
                torch.cuda.synchronize();times[str(graph)]=time.perf_counter()-start
                results[str(graph)]=outputs
                details[str(graph)]=dict(warmup_s=warm,summary=summary,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                if graph:
                    repeated,_=engine.generate(prompts,limits,set())
                    assert repeated==outputs,'Graph replay changed identical requests'
                engine.close();del engine
            equal=sum(a==b for a,b in zip(results['False'],results['True']))
            entry=dict(model=name,dtype=dtype,capacity=capacity,requests=8,output_token_limits=limits,
                graph_off_on_exact_sequences=equal,wall_s=times,details=details)
            report['results'].append(entry)
            assert equal==8,'Captured graph changed its same-engine outputs'
            if dtype=='float32':
                native=[]
                for prompt,limit in zip(prompts,limits):
                    out=generate(target,torch.tensor([prompt],device='cuda'),limit,set(),seed=0)
                    native.append(out.tokens)
                entry['native_fp32_exact_sequences']=sum(a==b for a,b in zip(native,results['True']))
                entry['native_outputs']=native;entry['packed_outputs']=results['True']
                assert entry['native_fp32_exact_sequences']==8,'Packed full target differs from native FP32 target'
            print(json.dumps({k:v for k,v in entry.items() if k not in ['details','native_outputs','packed_outputs']}),flush=True)
            target.close();del target;gc.collect();torch.cuda.empty_cache()
    report['limitations']=['Non-test64-token probe only; no benchmark eta is changed',
        'Production backend must be explicitly recorded if adopted',
        'BF16 native-vs-packed arithmetic is not claimed bitwise equal']


if __name__=='__main__':main()
