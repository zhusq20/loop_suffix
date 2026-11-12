"""Experimental production-only AR batching around the frozen graph executor.

This module never participates in benchmark timing. Adoption requires a separate
real-model equivalence probe and explicit generation metadata.
"""
import gc
from pathlib import Path
import sys
import time

import torch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
SUITE=ROOT/'results/loopfill-latency512-20261107'
sys.path[:0]=[str(SUITE/'source/src'),str(SUITE/'source/scripts'),'/tmp/loopfill-assets/ArcticInference']
from loop_batch_bench import arm_config
from llm_memory_editability.loop_batch_decode import BatchRequest,batch_decode
from llm_memory_editability.loop_optimized_decode import SharedKVPool,PackedLoopExecutor
from llm_memory_editability.loop_tree_decode import OuroLoop,HuginnLoop


class GraphTeacher:
    def __init__(self,target,batch_size,max_seq,graph=True):
        self.target,self.batch_size,self.max_seq=target,batch_size,max_seq
        self.pool=SharedKVPool(batch_size,pad_decode_rows=True)
        self.slots=[]
        for _ in range(batch_size):
            if target.family=='ouro':
                b=OuroLoop(target.model,4,max_seq,1,kv_factory=self.pool)
            else:
                b=HuginnLoop(target.model,32,max_seq,1,0,max_seq,extra_depths=(),kv_factory=self.pool)
                # Reuse the already frozen full3072-position production noise.
                b.init=target.noise
            b.position_limit=max_seq
            self.slots.append(b)
        first,second=(1,2) if target.family=='ouro' else (4,8)
        self.config=arm_config('ar',target.depth,first,second,4)
        self.engine=PackedLoopExecutor(self.slots,cuda_graph=graph,max_graphs=64)

    @torch.inference_mode()
    def warm(self,prompts):
        start=time.perf_counter()
        requests=[BatchRequest(i,p[:4],2,set()) for i,p in enumerate(prompts)]
        batch_decode(requests,self.slots,self.config,max_nodes=1,executor=self.engine,schedule='fixed')
        self.engine.prime_verifier_shapes(1)
        self.engine.graphs.freeze()
        torch.cuda.synchronize()
        return time.perf_counter()-start

    @torch.inference_mode()
    def generate(self,prompts,limit,stops):
        assert len(prompts)==self.batch_size
        limits=[limit]*len(prompts) if isinstance(limit,int) else list(limit)
        assert len(limits)==len(prompts) and all(len(p)+n<=self.max_seq for p,n in zip(prompts,limits))
        requests=[BatchRequest(i,p,n,set(stops)) for i,(p,n) in enumerate(zip(prompts,limits))]
        rows,summary=batch_decode(requests,self.slots,self.config,max_nodes=1,executor=self.engine,schedule='fixed')
        rows.sort(key=lambda r:r['request_id'])
        assert [r['request_id'] for r in rows]==list(range(len(prompts)))
        assert all(r['stats']['calls']==self.target.depth*(len(r['tokens'])-1) for r in rows)
        return [r['tokens'] for r in rows],summary

    def close(self):
        del self.engine,self.slots,self.pool
        gc.collect()
        torch.cuda.empty_cache()
