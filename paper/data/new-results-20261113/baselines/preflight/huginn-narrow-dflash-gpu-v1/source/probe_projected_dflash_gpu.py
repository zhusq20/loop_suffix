"""One real Huginn optimizer update and narrow-drafter decoding compatibility.

This is a diagnostic, not a trained pilot or a benchmark result.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from gpu_allocation import allowed_gpus
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert args.gpu in allowed_gpus()
    uuids = {int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    uuid = uuids[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo,hi=map(int,args.cpus.split('-'));os.sched_setaffinity(0,set(range(lo,hi+1)))
        args.out.mkdir(parents=True,exist_ok=False)
        freeze_sources(args.out,[Path(__file__),BASE/'gpu_allocation.py',BASE/'configs/gpu-allocation.json',
            BASE/'configs/dflash-huginn-narrow-2048-v1.json',BASE/'dflash_impl/projected_draft.py',
            BASE/'dflash_impl/target.py',BASE/'dflash_impl/decode.py',BASE/'provenance.py'])
        report=dict(status='running',model='huginn-3.5b',gpu_uuid=uuid,started_unix=time.time())
        try:
            run(args,report)
            report['status']='passed_gpu_compatibility_not_trained_pilot'
        except Exception:
            report.update(status='failed',error=traceback.format_exc())
            raise
        finally:
            report['finished_unix']=time.time()
            save(args.out/'report.json',report)
            save(args.out/'reservation-cost.json',dict(gpu_uuid=uuid,started_unix=report['started_unix'],
                finished_unix=report['finished_unix'],reserved_gpu_seconds=report['finished_unix']-report['started_unix'],status=report['status']))
            print(json.dumps(report,indent=2),flush=True)


def run(args,report):
    import torch
    torch.cuda.set_device(0)  # Before SpecForge's optional CUDA-dependent imports.
    torch.set_num_threads(1);torch.manual_seed(42)
    sys.path.insert(0,str(BASE/'third_party/dflash'))
    from transformers import AutoTokenizer, Qwen3Config
    from specforge.core.dflash import OnlineDFlashModel
    from specforge.optimizer import BF16Optimizer
    from dflash_impl.target import LoopTarget
    from dflash_impl.decode import generate
    from dflash_impl.projected_draft import training_draft_class,inference_draft_class
    protocol=json.loads((BASE.parents[1]/'results/loopfill-latency512-20261107/protocol.json').read_text())
    path=protocol['models']['huginn-3.5b']['path']
    print('Loading full frozen Huginn target and projected drafter',flush=True)
    target=LoopTarget.load(path,'huginn',3072)
    cfg=Qwen3Config.from_pretrained(BASE/'configs/dflash-huginn-narrow-2048-v1.json',local_files_only=True)
    cfg._attn_implementation='flex_attention'
    draft=training_draft_class()(cfg).to(device=target.device,dtype=torch.bfloat16)
    draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert sum(p.numel() for p in draft.parameters())==332621056
    wrapper=OnlineDFlashModel(draft,target.head,target.embedding,65513,block_size=16,
        attention_backend='flex_attention',num_anchors=512,loss_decay_gamma=7.)
    teacher=BASE/'data/teachers/dflash-train/huginn-3.5b/shard0/teachers.jsonl'
    pool=[]
    with teacher.open() as stream:
        for _ in range(32):
            row=json.loads(next(stream))
            if len(row['output_tokens'])>=32:pool.append(row)
    row=min(pool,key=lambda r:len(r['prompt_tokens']))
    # Preserve the entire real prompt; shorten only the diagnostic answer budget.
    answer=row['output_tokens'][:32]
    ids=torch.tensor([row['prompt_tokens']+answer],device=target.device)
    loss_mask=torch.zeros_like(ids,dtype=torch.float32);loss_mask[:,len(row['prompt_tokens']):]=1
    save(args.out/'diagnostic-training-input.json',dict(source_id=row['source_id'],
        prompt_tokens=row['prompt_tokens'],output_tokens=answer,teacher_path=str(teacher),
        teacher_journal_sha256=json.loads(teacher.with_name('complete.json').read_text())['teachers_sha256']))
    optimizer=BF16Optimizer(draft,lr=6e-4,weight_decay=0.,max_grad_norm=1.,total_steps=100,warmup_ratio=.04)
    torch.cuda.synchronize();started=time.perf_counter()
    features=target.forward(ids,features=True,use_cache=False).features
    loss,accuracy=wrapper(ids,features,loss_mask)
    assert bool(torch.isfinite(loss));loss.backward()
    gradients={}
    for name in ['fc.weight','input_projection.weight','output_projection.weight']:
        grad=dict(draft.named_parameters())[name].grad
        assert grad is not None and bool(torch.isfinite(grad).all()) and bool(grad.abs().sum()>0)
        gradients[name]=float(grad.float().norm())
    assert all(not p.requires_grad and p.grad is None for p in target.model.parameters())
    grad_norm=optimizer.step();assert bool(torch.isfinite(grad_norm))
    torch.cuda.synchronize()
    report.update(draft_parameters=332621056,training_source_id=row['source_id'],
        training_prompt_tokens=len(row['prompt_tokens']),training_answer_tokens=len(answer),
        training_loss=float(loss.detach()),accuracy=float(accuracy.detach()),grad_norm=float(grad_norm),
        projection_gradients=gradients,update_wall_s=time.perf_counter()-started,
        training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        target_frozen=True,target_loop_depth=target.depth,target_features=target.expanded_ids,
        optimizer='Pinned official BF16Optimizer; one diagnostic update, microbatch1,512 anchors')
    checkpoint=args.out/'one-update-checkpoint';draft.save_pretrained(checkpoint)
    save(checkpoint/'complete.json',dict(step=1,diagnostic_only=True))
    del wrapper,optimizer,features,loss,accuracy,ids,loss_mask,draft
    torch.cuda.empty_cache()
    print('Optimizer update passed; checking deployment class and T0/T1 decoding',flush=True)
    infer,info=inference_draft_class().from_pretrained(checkpoint,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    assert all(not info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    save(args.out/'checkpoint-loading.json',info)
    infer=infer.to(target.device).eval()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=True)
    prompt='Write a short Python function that adds two integers.\n'
    ids=tok(prompt,add_special_tokens=False,return_tensors='pt').input_ids.to(target.device)
    outputs={}
    for name,model,temp in [('ar',None,0.),('projected',infer,0.),('repeat',infer,0.),('sampling',infer,1.)]:
        result=generate(target,ids,32,[tok.eos_token_id],draft=model,temperature=temp,seed=0)
        outputs[name]=dict(tokens=result.tokens,stats=result.stats)
        save(args.out/f'{name}.json',outputs[name])
    assert outputs['projected']['tokens']==outputs['repeat']['tokens']
    a,d=outputs['ar']['tokens'],outputs['projected']['tokens']
    report.update(repeat_exact=True,greedy_ar_equal=a==d,
        first_divergence=next((dict(position=i,ar=x,projected=y) for i,(x,y) in enumerate(zip(a,d)) if x!=y),
            None if len(a)==len(d) else dict(length_only=True,position=min(len(a),len(d)))),
        torch=str(torch.__version__),checkpoint_loading_clean=True,
        limitations=['One real-target optimizer update and one synthetic decode prompt only; not trained-pilot quality or speed.',
            'Microbatch1 on one short training example does not establish the longest-batch memory bound.',
            'AR/drafter finite-precision discrepancies, if any, require separate numerical diagnosis.',
            'Target/head remain full width; no claim of proportional end-to-end speed improvement.'])
    target.close()


if __name__=='__main__':
    main()
