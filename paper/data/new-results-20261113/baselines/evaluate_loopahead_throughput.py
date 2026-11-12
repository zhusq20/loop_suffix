"""LoopAhead Spec-Bench throughput, following frozen Table 2 input/timing policy."""
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
ROOT = BASE.parents[1]
TABLE2 = ROOT/'results/loopfill-table2-20261107-v2'


def ordered_requests(requests, mode, order_seed):
    """Match the historical finite stream; keep short correctness probes fixed."""
    ordered = list(requests)
    if mode == 'formal':
        random.Random(order_seed).shuffle(ordered)
    return ordered


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['huginn-3.5b', 'ouro-2.6b'], required=True)
    p.add_argument('--batch', type=int, choices=[1,2,4,8,16], required=True)
    p.add_argument('--mode', choices=['validate', 'smoke', 'formal'], required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert args.gpu in json.loads((BASE/'configs/gpu-allocation.json').read_text())['allowed_gpus']
    assert args.mode != 'validate' or args.batch == 2
    args.out.mkdir(parents=True, exist_ok=False)
    identities = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid = {int(i):u.strip() for i,u in (line.split(',') for line in identities.splitlines())}[args.gpu]
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    lo,hi = map(int,args.cpus.split('-'))
    os.sched_setaffinity(0,set(range(lo,hi+1)))
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status,started = 'failed',time.time()
        try:
            from gpu_startup import acquire_cuda_context
            lease, acquisition = acquire_cuda_context(args.gpu,uuid)
            save(args.out/'cuda-acquisition.json',acquisition)
            started = acquisition['acquired_unix']
            run(args,uuid)
            status = 'complete'
        except BaseException as exc:
            import torch
            status = 'oom' if isinstance(exc,torch.OutOfMemoryError) else 'failed'
            save(args.out/'failure.json',dict(status=status,error=traceback.format_exc(),finished_unix=time.time()))
            if status != 'oom':
                raise
        finally:
            save(args.out/'reservation-cost.json',dict(status=status,gpu_uuid=uuid,
                started_unix=started,finished_unix=time.time(),reserved_gpu_seconds=time.time()-started))


def run(args, uuid):
    sys.path[:0] = [str(TABLE2/'source/src'), str(TABLE2/'source/scripts'), '/tmp/loopfill-assets/ArcticInference']
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from loop_batch_bench import enforce_precision
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids
    from llm_memory_editability.loop_batch_decode import BatchRequest
    from llm_memory_editability.loop_optimized_decode import SharedKVPool
    from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop
    from loopahead.batch import BatchedLoopAheadExecutor, batch_decode
    from loopahead.decode import LoopAheadConfig, decode_steps
    from loopahead.engine import LoopAheadExecutor
    from loopahead.lad import LADConfig, LookaheadDrafter

    protocol = json.loads((TABLE2/'protocol.json').read_text())
    for rel,digest in protocol['source_sha256'].items():
        assert sha(TABLE2/'source'/rel) == digest, rel
    source_manifest = json.loads((Path(__file__).parent/'source-manifest.json').read_text())
    for rel,digest in source_manifest.items():
        assert sha(Path(__file__).parent/rel) == digest, rel
    spec = MODELS[args.model]
    target_path = Path(protocol['models'][args.model]['path'])
    input_path = TABLE2/'inputs'/f'{args.model}.jsonl'
    assert sha(input_path) == protocol['models'][args.model]['inputs_sha256']
    rows = [json.loads(line) for line in input_path.read_text().splitlines()]
    assert len(rows) == 480 and len({r['question_id'] for r in rows}) == 480
    tok = AutoTokenizer.from_pretrained(target_path,trust_remote_code=True,local_files_only=True)
    target_cfg = AutoConfig.from_pretrained(target_path,trust_remote_code=True,local_files_only=True)
    generation_path = target_path/'generation_config.json'
    generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    stops, stop_audit = resolve_stop_ids(spec,tok,target_cfg.eos_token_id,generation.get('eos_token_id'),target_cfg.vocab_size,'chat')
    for row in rows:
        assert row['model_id'] == args.model
        assert row['prompt_token_ids'] == tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True)
        assert len(row['prompt_token_ids'])+1024+spec.depth_total+4 <= 2688
    dtype_name = 'float32' if args.mode == 'validate' else 'bfloat16'
    dtype = getattr(torch,dtype_name)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    target = AutoModelForCausalLM.from_pretrained(target_path,trust_remote_code=True,
        local_files_only=True,torch_dtype=dtype).to('cuda').eval()
    target = enforce_precision(target,dtype_name)
    train_root = BASE/f'checkpoints/loopahead-{args.model}-formal-v1'
    training = json.loads((train_root/'complete.json').read_text())
    checkpoint = Path(training['selected_checkpoint'])/'lad.pt'
    assert (checkpoint.parent/'complete.json').exists()
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    lad = LookaheadDrafter(LADConfig(**saved['config']))
    lad.load_state_dict(saved['model'],strict=True)
    lad = lad.to(device='cuda',dtype=dtype).eval()
    cfg = (LoopAheadConfig() if spec.backend == 'ouro' else
        LoopAheadConfig(total=32,first=4,second=8,future=2,guard=1,stride=2,family='huginn'))
    pool = SharedKVPool(args.batch,pad_decode_rows=False)
    backbones = [(OuroLoop(target,4,2688,32,kv_factory=pool) if spec.backend == 'ouro' else
        HuginnLoop(target,32,2688,32,0,2688,extra_depths=(8,),kv_factory=pool)) for _ in range(args.batch)]
    for b in backbones:
        b.position_limit = 2688
    engine = BatchedLoopAheadExecutor(backbones,lad,saved['frequent_ids'])
    count = 480 if args.mode == 'formal' else (4 if args.mode == 'validate' else 2*args.batch+2)
    limit = 1024 if args.mode == 'formal' else (16 if args.mode == 'validate' else 32)
    reqs = [BatchRequest(row['question_id'],row['prompt_token_ids'],limit,stops) for row in rows[:count]]
    reqs = ordered_requests(reqs, args.mode, protocol['order_seed'])
    metadata = dict(model=args.model,model_revision=spec.revision,method='loopahead',dataset='specbench',
        mode=args.mode,reportable=args.mode=='formal',batch=args.batch,count=count,full_count=480,
        input_path=str(input_path),input_sha256=sha(input_path),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        config=asdict(cfg),dtype=dtype_name,temperature=0,seed=0,max_new_tokens=limit,max_seq=2688,
        schedule='continuous',prefill_chunk_size=128,graph=False,compile=False,pad_decode_rows=False,
        stop_audit=stop_audit,gpu_uuid=uuid,cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch=torch.__version__,transformers=transformers.__version__,target_source_sha256=protocol['source_sha256'],
        source_sha256=source_manifest,table2_protocol_sha256=sha(TABLE2/'protocol.json'),
        warmup_requests=min(2*args.batch,count),warmup_max_new_tokens=32,
        order_seed=protocol['order_seed'] if args.mode=='formal' else None,
        request_order=[req.request_id for req in reqs],
        timing='First admission through final CUDA synchronization, including prefill, decoding, scheduling and fsynced per-request journals; excluding model loading, warmup and final report serialization.',
        limitations=['Single timing, same host across dates; BF16 output differences are not task-quality validation.',
                    'LoopAhead paper reproduction using a new packed batching adapter; no DFlash code used.'])
    save(args.out/'metadata.json',metadata)
    if args.mode == 'validate':
        # Same full target and trained LAD, original single-request implementation.
        reference = LoopAheadExecutor(backbones[0],lad,saved['frequent_ids'])
        expected = []
        with torch.inference_mode():
            for req in reqs:
                reference.request(0,stops)
                steps = decode_steps(backbones[0],req.prompt,cfg,req.max_new_tokens,stops,generator=reference.generator)
                value = None
                while True:
                    try:
                        call = steps.send(value)
                    except StopIteration as done:
                        expected.append(done.value[0]);break
                    value = reference.execute(call.name,[(backbones[0],call)])[0]
        actual,summary = batch_decode(reqs,backbones,cfg,engine,chunk_size=128)
        pairs = [dict(request_id=r.request_id,reference=e,batched=a['tokens']) for r,e,a in zip(reqs,expected,actual,strict=True)]
        save(args.out/'comparison.json',dict(pairs=pairs,summary=summary))
        assert expected == [a['tokens'] for a in actual], 'FP32 batch/single token divergence; inspect comparison.json'
        assert summary['max_advance_requests'] == 2
        save(args.out/'complete.json',dict(status='passed_real_target_fp32_batch_parity',requests=count,
            checkpoint_sha256=sha(checkpoint),finished_unix=time.time()))
        return
    warmup = [replace(req,max_new_tokens=min(32,req.max_new_tokens)) for req in reqs[:2*args.batch]]
    batch_decode(warmup,backbones,cfg,engine,chunk_size=128)
    engine.graphs.freeze()
    torch.cuda.reset_peak_memory_stats()
    published = 0
    io_seconds = 0.
    with (args.out/'requests.jsonl').open('x') as stream:
        def progress(completed,wall):
            nonlocal published,io_seconds
            before = time.perf_counter()
            for item in completed[published:]:
                req = next(r for r in reqs if r.request_id == item['request_id'])
                assert 1 <= len(item['tokens']) <= req.max_new_tokens
                assert item['stats']['tokens'] == len(item['tokens'])-1
                record = dict(item,finished_unix=time.time(),stop_reason='eos' if item['tokens'][-1] in stops else 'length')
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
            published = len(completed)
            io_seconds += time.perf_counter()-before
            save(args.out/'progress.json',dict(requests=published,expected=count,wall_s=wall,
                output_tokens=sum(r['n_tokens'] for r in completed),updated_unix=time.time()))
            print(json.dumps(dict(requests=published,expected=count,wall_s=wall)),flush=True)
        completed,summary = batch_decode(reqs,backbones,cfg,engine,chunk_size=128,progress_callback=progress)
    assert len(completed) == count and [r['request_id'] for r in completed] == [r.request_id for r in reqs]
    assert summary['prefill_chunk_tokens'] == sum(len(r.prompt) for r in reqs)
    if args.batch > 1:
        assert summary['max_advance_requests'] > 1
        assert summary['refilled_requests'] > 0
    summary.update(model=args.model,batch=args.batch,reportable=args.mode=='formal',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),journal_write_s=io_seconds,
        cap_stopped=sum(len(r['tokens'])==limit and r['tokens'][-1] not in stops for r in completed))
    save(args.out/'summary.json',summary)
    save(args.out/'complete.json',dict(requests=count,requests_sha256=sha(args.out/'requests.jsonl'),finished_unix=time.time()))


if __name__ == '__main__':
    main()
