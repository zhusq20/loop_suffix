"""LoopAhead latency runs on frozen inputs, plus explicitly separate dev diagnostics."""
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2): h.update(block)
    return h.hexdigest()


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['ouro-2.6b','ouro-1.4b','huginn-3.5b'],required=True)
    p.add_argument('--method',choices=['ar','loopahead'],required=True)
    p.add_argument('--dataset',choices=['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench'],default='gsm8k')
    p.add_argument('--temperature',type=float,choices=[0.,1.],default=0.)
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    p.add_argument('--checkpoint',type=Path,help='selected checkpoint/lad.pt')
    p.add_argument('--admission',type=Path)
    p.add_argument('--random-lad',action='store_true')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--trace',action='store_true')
    p.add_argument('--dev-calibration',action='store_true')
    p.add_argument('--paper-humaneval',action='store_true',help='separate B4 completion164/max1024 comparison')
    p.add_argument('--native-only',action='store_true',help='dev-only reference with LAD admission disabled')
    p.add_argument('--profile-dev',action='store_true',help='synchronized call timings, never benchmark latency')
    p.add_argument('--count',type=int,default=0)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--cpus',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    from gpu_allocation import allowed_gpus
    assert args.gpu in allowed_gpus(), 'GPU released by the user'
    if args.paper_humaneval and (args.dev_calibration or args.temperature!=0.):
        p.error('B4 HumanEval is a separate greedy paper-setting check')
    if (args.native_only or args.profile_dev) and not (args.dev_calibration and args.method=='loopahead'):
        p.error('Native-only and synchronized profiling are development calibration diagnostics only')
    if args.method=='loopahead' and args.checkpoint is None and not (args.random_lad and args.pilot):
        p.error('A trained LAD is required outside an explicit random pilot')
    if args.method=='loopahead' and args.temperature and args.model.startswith('ouro') and args.admission is None and not (args.pilot or args.dev_calibration):
        p.error('Formal Ouro sampling requires frozen dev calibration')
    args.out.mkdir(parents=True,exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out,[Path(__file__),BASE/'provenance.py',BASE/'gpu_startup.py',
        BASE/'gpu_allocation.py',*sorted((BASE/'loopahead').glob('*.py'))])
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid={int(i.strip()):u.strip() for i,u in (l.split(',') for l in lines.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,set(range(lo,hi+1)))
        started,status=time.time(),'failed'
        try:
            from gpu_startup import acquire_cuda_context
            lease,acquisition=acquire_cuda_context(args.gpu,uuid)
            save(args.out/'cuda-acquisition.json',acquisition)
            started=acquisition['acquired_unix']
            run(args,uuid);status='complete'
        except Exception:
            save(args.out/'failure.json',dict(error=traceback.format_exc()));raise
        finally:
            save(args.out/'reservation-cost.json',dict(gpu_uuid=uuid,started_unix=started,finished_unix=time.time(),
                reserved_gpu_seconds=time.time()-started,status=status))


def run(args,uuid):
    import torch
    import transformers
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    suite=ROOT/'results/loopfill-latency512-20261107'
    source_suite=ROOT/('results/loopfill-latency512-t1-20261108' if args.temperature else 'results/loopfill-latency512-20261107')
    sys.path[:0]=[str(source_suite/'source/src'),str(source_suite/'source/scripts'),'/tmp/loopfill-assets/ArcticInference']
    from loop_batch_bench import arm_config,enforce_precision
    from loop_latency_worker import drive
    from llm_memory_editability.loop_model_registry import MODELS,resolve_stop_ids,tokenize_prompt
    from llm_memory_editability.loop_tree_decode import OuroLoop,HuginnLoop,tree_decode_steps
    from llm_memory_editability.loop_optimized_decode import SharedKVPool,PackedLoopExecutor
    from loopahead.lad import LADConfig,LookaheadDrafter
    from loopahead.engine import LoopAheadExecutor
    from loopahead.decode import LoopAheadConfig,decode_steps
    from loopahead.admission import AdmissionPolicy
    protocol=json.loads((suite/'protocol.json').read_text())
    source_protocol=json.loads((source_suite/'protocol.json').read_text())
    for rel,expected in source_protocol['source_sha256'].items():
        assert digest(source_suite/'source'/rel)==expected
    spec=MODELS[args.model];path=Path(protocol['models'][args.model]['path'])
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    target_cfg=AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    generation=json.loads((path/'generation_config.json').read_text()) if (path/'generation_config.json').exists() else {}
    max_new_tokens=1024 if args.paper_humaneval else 512
    if args.paper_humaneval:
        input_path=BASE/'data/paper-inputs/humaneval.jsonl'
        manifest=json.loads(input_path.with_name('humaneval-source.json').read_text())
        audit=json.loads(input_path.with_name('humaneval-isolation-audit.json').read_text())
        assert digest(input_path)==manifest['input_sha256']==audit['input_sha256'] and audit['passed']
        rows=[json.loads(line) for line in input_path.open()]
        assert len(rows)==164 and len({r['source_id'] for r in rows})==164
        prompt_format='completion';dataset='paper-humaneval164'
    elif args.dev_calibration:
        input_path=BASE/'data/manifests/loopahead-dev.jsonl'
        manifest=json.loads((input_path.parent/'manifest.json').read_text())['files']['loopahead-dev']
        assert digest(input_path)==manifest['sha256']
        selection=json.loads((BASE/'configs/loopahead-sampling-dev16.json').read_text())
        selected={r['source_id'] for r in selection['prompts']}
        rows=[json.loads(l) for l in input_path.open()]
        rows=[r for r in rows if r['source_id'] in selected]
        assert len(rows)==16 and {r['source_id'] for r in rows}==selected
        prompt_format='completion';dataset='loopahead-dev16'
    else:
        input_path=suite/f'inputs/{args.dataset}.jsonl'
        assert digest(input_path)==protocol['inputs'][args.dataset]['sha256']
        rows=[json.loads(l) for l in input_path.open()]
        prompt_format='chat';dataset=args.dataset
    stops,stop_audit=resolve_stop_ids(spec,tok,target_cfg.eos_token_id,generation.get('eos_token_id'),target_cfg.vocab_size,prompt_format)
    prompts=[tokenize_prompt(spec,tok,r['prompt'],prompt_format) for r in rows]
    required=max(map(len,prompts))+max_new_tokens+spec.depth_total+4
    capacity=min(spec.context_limit,((required+255)//256)*256)
    assert all(len(p)+max_new_tokens+spec.depth_total+4<=capacity for p in prompts)
    full_count=len(rows);count=args.count or full_count
    assert 0<count<=full_count
    rows,prompts=rows[:count],prompts[:count]
    torch.set_num_threads(1);torch.manual_seed(42)
    dtype=getattr(torch,args.dtype)
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,torch_dtype=dtype).to('cuda').eval()
    model=enforce_precision(model,args.dtype)
    pool=SharedKVPool(1,pad_decode_rows=False)
    physical_nodes=1 if args.method=='ar' else 32
    if spec.backend=='ouro':
        backbone=OuroLoop(model,4,capacity,physical_nodes,kv_factory=pool)
        cfg=LoopAheadConfig(temperature=args.temperature)
        hidden=target_cfg.hidden_size
    else:
        backbone=HuginnLoop(model,32,capacity,physical_nodes,0,capacity,extra_depths=(8,),kv_factory=pool)
        cfg=LoopAheadConfig(total=32,first=4,second=8,future=2,guard=1,stride=2,family='huginn',temperature=args.temperature)
        hidden=target_cfg.n_embd
    backbone.position_limit=capacity
    if args.native_only:
        cfg=replace(cfg,future=0)
    policy=AdmissionPolicy.load(args.admission) if args.admission else None
    checkpoint_hash=None
    if args.method=='loopahead':
        if args.checkpoint:
            assert (args.checkpoint.parent/'complete.json').exists()
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            lad=LookaheadDrafter(LADConfig(**saved['config']))
            lad.load_state_dict(saved['model']);frequent=saved['frequent_ids'];checkpoint_hash=digest(args.checkpoint)
        else:
            lad=LookaheadDrafter(LADConfig(hidden));frequent=list(range(8192))
        if policy is not None:
            assert policy.record['model']==args.model, 'Admission calibration belongs to another model'
            assert policy.record['checkpoint_sha256']==checkpoint_hash, 'Admission calibration belongs to another LAD checkpoint'
        lad=lad.to(device='cuda',dtype=dtype).eval()
        engine=LoopAheadExecutor(backbone,lad,frequent,temperature=args.temperature)
        engine.capture_prefill_logits=args.trace
    else:
        engine=PackedLoopExecutor([backbone],cuda_graph=False)
    profile_events=[]
    profile_active=False
    if args.profile_dev:
        original_execute=engine.execute
        def profile_execute(name,jobs):
            if not profile_active or name in ('synchronize','prefill'):
                return original_execute(name,jobs)
            call=jobs[0][1]
            nodes=[]
            if name in ('advance','embed'):
                nodes=call.args[1]
            elif name=='lad_readout':
                _,groups,full=call.args
                nodes=[n for _,group in groups for n in group]+full
            torch.cuda.synchronize()
            start=time.perf_counter()
            result=original_execute(name,jobs)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            profile_events.append(dict(name=name,wall_s=elapsed,rows=len(nodes),
                prefix_length=backbone.length,
                node_depth_mean=sum(n.depth for n in nodes)/len(nodes) if nodes else 0.,
                produced_tokens=len(result[0][0]) if name=='lad_propose' else None))
            return result
        engine.execute=profile_execute
    ar_cfg=arm_config('ar',spec.depth_total,*spec.loopfill_readout_depths,4,draft_node_budget=16,model_top_k=2,
        child_suffix_caps=(4,4),suffix_c_initial=1.,suffix_eta=.1,loopspec_depths=spec.loopspec_depths,wavefront_depth=spec.wavefront_depth)
    metadata=dict(model=args.model,model_revision=spec.revision,method=args.method,dataset=dataset,prompt_format=prompt_format,
        input_sha256=digest(input_path),count=count,full_count=full_count,temperature=args.temperature,top_p=1.,top_k=0,seed=0,
        dtype=args.dtype,max_seq=capacity,max_new_tokens=max_new_tokens,stop_audit=stop_audit,gpu_uuid=uuid,
        cpu_affinity=sorted(os.sched_getaffinity(0)),physical_nodes=physical_nodes,
        config=asdict(cfg) if args.method=='loopahead' else asdict(ar_cfg),torch=torch.__version__,transformers=transformers.__version__,
        frozen_source_sha256=source_protocol['source_sha256'],checkpoint=str(args.checkpoint) if args.checkpoint else None,
        checkpoint_sha256=checkpoint_hash,admission_sha256=digest(args.admission) if args.admission else None,
        random_lad=args.random_lad,graph=False,compile=False,trace=args.trace,pilot=args.pilot,dev_calibration=args.dev_calibration,
        native_only=args.native_only,profile_dev=args.profile_dev,
        paper_check=args.paper_humaneval,
        paper_check_limitations=['Original164 completion prompts and max1024; EOS stopping uses the released checkpoint. Author-specific additional completion delimiters and exact checkpoint revisions were not published. Graph OFF on current hardware.'] if args.paper_humaneval else [],
        reportable=not(args.pilot or args.random_lad or args.trace or args.dev_calibration or args.paper_humaneval or args.dtype!='bfloat16' or count!=full_count))
    save(args.out/'metadata.json',metadata)

    def describe(logits):
        logits=logits.reshape(-1,logits.shape[-1])[-1].float()
        values,ids=logits.topk(3);probs=logits.softmax(-1)[ids]
        return dict(top_ids=ids.tolist(),top_logits=values.tolist(),top_probs=probs.tolist(),
                    top2_probability_margin=float(probs[0]-probs[1]),top1_logit_margin=float(values[0]-values[1]))

    def generate(row,prompt,limit,record=False):
        nonlocal profile_active
        profile_active=args.profile_dev and record
        profile_events.clear()
        key=f'0/{args.model}/{dataset}/{row["source_id"]}'.encode()
        seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1)
        if args.temperature:
            from llm_memory_editability.loop_sampling import TargetSampler
            backbone.target_sampler=TargetSampler(args.temperature,seed,capacity,backbone.hidden.device)
        candidates,trace=[],[]
        def observer(event,**data):
            if record and args.dev_calibration and event in ('native','lad'):
                # The committed prefix is known correct; store only its length
                # and the speculative suffix, avoiding quadratic token journals.
                offset=len(data.pop('root_path'))
                path=data.pop('path')
                candidates.append(dict(type=event,path_offset=offset,path_suffix=path[offset:],**data))
            if record and args.trace and event=='commit':
                trace.append(dict(position=data['position'],token=data['token'],**describe(data['logits'])))
        hook=None
        if args.method=='ar' and args.trace and record:
            head=model.lm_head
            hook=head.register_forward_hook(lambda mod,inp,out:trace.append(dict(position=len(trace),**describe(out))))
        try:
            with torch.inference_mode():
                if args.method=='loopahead':
                    engine.request(seed,stops)
                    steps=decode_steps(backbone,prompt,cfg,limit,stops,generator=engine.generator,
                        admission=policy,quality=policy.quality if policy else None,
                        observer=observer if record and (args.trace or args.dev_calibration) else None)
                else:
                    steps=tree_decode_steps(backbone,prompt,ar_cfg,limit,stops,max_nodes=physical_nodes)
                tokens,stats=drive(backbone,engine,steps)
        finally:
            if hook: hook.remove()
        if args.method=='loopahead' and args.trace and record:
            trace.insert(0,dict(position=0,token=tokens[0],**describe(engine.prefill_logits)))
        return tokens,stats,candidates,trace

    for row,prompt in list(zip(rows,prompts))[:2]: generate(row,prompt,32)
    engine.graphs.freeze();completed=[]
    with (args.out/'requests.jsonl').open('x') as f:
        for row,prompt in zip(rows,prompts):
            tokens,stats,candidates,trace=generate(row,prompt,max_new_tokens,True)
            assert stats['tokens']==len(tokens)-1
            record=dict(source_id=row['source_id'],prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
                tokens=tokens,stats=stats,stop_reason='eos' if tokens[-1] in stops else 'length',finished_unix=time.time())
            if args.trace:record['trace']=trace
            if args.dev_calibration:record['candidates']=candidates
            if args.profile_dev:record['profile']=list(profile_events)
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            completed.append(record)
            seconds=sum(r['stats']['decode_s'] for r in completed)
            n=sum(r['stats']['tokens'] for r in completed);calls=sum(r['stats']['calls'] for r in completed)
            summary=dict(requests=len(completed),decode_tokens=n,output_tokens=sum(len(r['tokens']) for r in completed),
                decode_s=seconds,tok_s=n/seconds,loop_steps=calls,gamma=spec.depth_total*n/calls if calls else None)
            save(args.out/'progress.json',summary)
            print(json.dumps(dict(request=len(completed),tokens=len(tokens),decode_s=stats['decode_s'],gamma=spec.depth_total*stats['tokens']/stats['calls'] if stats['calls'] else None)),flush=True)
    save(args.out/'summary.json',summary)
    save(args.out/'complete.json',dict(requests=count,requests_sha256=digest(args.out/'requests.jsonl'),finished_unix=time.time()))


if __name__=='__main__':
    main()
