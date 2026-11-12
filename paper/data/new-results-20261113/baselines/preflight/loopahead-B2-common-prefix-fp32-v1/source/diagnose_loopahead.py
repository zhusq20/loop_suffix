"""FP32 same-prefix replay for the observed random-LAD B2 BF16 disagreements."""
import argparse
from dataclasses import replace
import fcntl
import hashlib
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
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out,[Path(__file__),BASE/'loopahead/decode.py',BASE/'loopahead/engine.py',BASE/'loopahead/lad.py'])
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,set(range(lo,hi+1)))
        report=dict(gpu_uuid=uuid,started_unix=time.time(),dtype='float32',results=[])
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
    from transformers import AutoModelForCausalLM,AutoTokenizer
    suite=ROOT/'results/loopfill-latency512-20261107'
    sys.path[:0]=[str(suite/'source/src'),str(suite/'source/scripts'),'/tmp/loopfill-assets/ArcticInference']
    from loop_batch_bench import enforce_precision
    from loop_latency_worker import drive
    from llm_memory_editability.loop_model_registry import MODELS,tokenize_prompt
    from llm_memory_editability.loop_tree_decode import OuroLoop
    from llm_memory_editability.loop_optimized_decode import SharedKVPool
    from loopahead.decode import LoopAheadConfig,decode_steps
    from loopahead.engine import LoopAheadExecutor
    from loopahead.lad import LADConfig,LookaheadDrafter,RequestVocabulary
    torch.set_num_threads(1);torch.manual_seed(42)
    protocol=json.loads((suite/'protocol.json').read_text())
    path=protocol['models']['ouro-2.6b']['path']
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    source=suite/'inputs/gsm8k.jsonl'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['inputs']['gsm8k']['sha256']
    rows=[json.loads(l) for l in source.open()]
    prompts={r['source_id']:tokenize_prompt(MODELS['ouro-2.6b'],tok,r['prompt'],'chat') for r in rows}
    capacity=((max(map(len,prompts.values()))+512+8+255)//256)*256
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32).to('cuda').eval()
    model=enforce_precision(model,'float32')
    pool=SharedKVPool(1,pad_decode_rows=False)
    b=OuroLoop(model,4,capacity,32,kv_factory=pool);b.position_limit=capacity
    lad=LookaheadDrafter(LADConfig(model.config.hidden_size)).to(device='cuda',dtype=torch.float32).eval()

    class Reached(Exception):pass

    class Replay(LoopAheadExecutor):
        def _prefill(self,backbone,prompt,depth):
            super()._prefill(backbone,prompt,depth)
            self.vocab=RequestVocabulary(self.head,self.frequent_ids,prompt,self.stops)
            self.vocab.commit([self.forced[0]])
            return self.forced[0]

        def _readout(self,backbone,tree,groups,full):
            reads,finals=super()._readout(backbone,tree,groups,full)
            for node in full:
                if node.path==self.forced:
                    self.observed=finals[node.idx].detach().cpu()
                    raise Reached()
                if len(node.path)<len(self.forced) and node.path==self.forced[:len(node.path)]:
                    forced=torch.full_like(finals[node.idx],float('-inf'))
                    forced[self.forced[len(node.path)]]=0.
                    finals[node.idx]=forced
            return reads,finals

    engine=Replay(b,lad,list(range(8192)))
    bf_ar={r['source_id']:r for r in map(json.loads,(BASE/'preflight/loopahead-B2-ouro-2.6b-gsm8k-ar-bfloat16-n20-v1/requests.jsonl').open())}
    divergences=json.loads((BASE/'preflight/loopahead-B2-bfloat16-first-divergences.json').read_text())['first_divergences']
    with torch.inference_mode():
        for case in divergences:
            sid,pos=case['source_id'],case['position']
            assert pos>0
            engine.forced=tuple(bf_ar[sid]['tokens'][:pos])
            logits={}
            for name,cfg in [('serial',LoopAheadConfig(retain=1,execute=1,future=0)),('loopahead',LoopAheadConfig())]:
                engine.observed=None;engine.request(0,{tok.eos_token_id})
                try:
                    drive(b,engine,decode_steps(b,prompts[sid],cfg,512,{tok.eos_token_id},generator=engine.generator))
                except Reached:
                    pass
                assert engine.observed is not None
                logits[name]=engine.observed
            a,d=logits['serial'],logits['loopahead']
            entry=dict(source_id=sid,position=pos,common_prefix=list(engine.forced),
                max_abs_logit_difference=float((a-d).abs().max()),same_top1=int(a.argmax())==int(d.argmax()))
            for name,value in logits.items():
                values,ids=value.topk(3);probs=value.softmax(-1)[ids]
                entry[name]=dict(top_ids=ids.tolist(),top_logits=values.tolist(),top_probs=probs.tolist(),
                    probability_margin=float(probs[0]-probs[1]))
            report['results'].append(entry)
            assert entry['same_top1'],entry
    report['limitations']=['diagnostic forcing preserves only the observed BF16 common prefix; no forced token is reported as a generated benchmark output',
        'serial reference has one live node, four full loops and the same native readout interface',
        'original512-token proposal budget retained; stops immediately at the comparison position',
        'FP32 native proposals may change tree shapes relative to the original BF16 run']


if __name__=='__main__':main()
