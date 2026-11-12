"""Fit nested external calibrations with fresh completion-only history per size."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_prob_tree import Calibration, ProbConfig, ProbReplay, collect_calibration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['ouro-2.6b','huginn-3.5b'],required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--size',type=int,choices=[16,32,64,128],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    manifest=Path(f'results/loopfill-icml-20261103/external-calibration/calib{a.size}.jsonl')
    selected=list(map(json.loads,manifest.read_text().splitlines()))
    meta={}
    for file in sorted(a.runs.glob('index-shard*.jsonl')):
        for row in map(json.loads,file.read_text().splitlines()):
            if row['question_id'] in meta:raise ValueError('duplicate readout ID')
            meta[row['question_id']]=row
    T,d,deep=(4,1,2) if a.model.startswith('ouro') else (32,4,8)
    reqs=[]
    for expected in selected:
        qid=expected['question_id'];row=meta[qid]
        if row['prompt_id']!=expected['prompt_id']:raise ValueError('external identity mismatch')
        path=a.runs/'readouts'/f'{qid}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('readout hash mismatch')
        with np.load(path) as z:
            if list(z['depths'])!=[d,deep,T]:raise ValueError('readout depth mismatch')
            reqs.append((qid,z['prompt'].copy(),z['output'].copy(),{t:z['ids'][i].copy() for i,t in enumerate([d,deep])},{t:z['probs'][i].astype(np.float32) for i,t in enumerate([d,deep])}))
    start=time.monotonic();cal=Calibration();cache=CompletionSuffixCache.arctic()
    for qid,prompt,output,ids,probs in reqs:
        collect_calibration(qid,prompt,output,ids,probs,[d,deep],cache,cal,joint_pairs=[(d,deep)])
    offline_audit=cache.publications.copy()
    offline=cal.to_json();cache=CompletionSuffixCache.arctic();edges=[]
    config=ProbConfig('v3 A t0.01 offline-suffix',T,d,(d,deep),tau=.01)
    for qid,prompt,output,ids,probs in reqs:
        ProbReplay(qid,prompt,output,ids,probs,config,cache,cal,edge_log=edges).run()
    for q,m,rival,lead,hit in edges:cal.add_insitu(q,m,rival,lead,hit)
    expected_ids=[r['question_id'] for r in selected]
    for audit in [offline_audit,cache.publications]:
        if [r['request_id'] for r in audit]!=expected_ids:raise ValueError('history publication order mismatch')
    result={'model':a.model,'n_requests':{'dev':a.size},'source':'external_ultrachat_train_sft','dev':cal.to_json(),'offline':offline,'insitu_from':config.__dict__,'input_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'prompt_ids':[r['prompt_id'] for r in selected],'history_audit':{'offline':offline_audit,'insitu':cache.publications},'cpu_fit_s':time.monotonic()-start,'output_tokens':sum(len(r[2]) for r in reqs),'readout_gpu_s':sum(meta[q]['generate_s']+meta[q]['readout_s'] for q in expected_ids),'insitu_candidate_count':len(edges),'limitations':'Offline replay off-path approximation; validation/support/fallback diagnostics remain required before formal evaluation.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result)+'\n')
    print(json.dumps({'model':a.model,'size':a.size,'cpu_fit_s':result['cpu_fit_s'],'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}),flush=True)


if __name__=='__main__':main()
