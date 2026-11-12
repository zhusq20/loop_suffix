from pathlib import Path
import sys,json,shutil,time,py_compile,hashlib
BASE=Path('/ossfs/workspace/sz/loopfill/experiments/baselines');sys.path.insert(0,str(BASE))
from packed_dflash_evidence import read,save,sha
ROOT=BASE/'logs/ouro14-packed-evaluation-v1';ROOT.mkdir(exist_ok=False)
for name in ['source','logs','checks','suite/source/scripts','suite/inputs','suite/logs','suite/runs']:(ROOT/name).mkdir(parents=True,exist_ok=True)
training=Path('/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3');trainplan=read(training/'plan.json')
for relative,expected in trainplan['source_sha256'].items():
 assert sha(training/relative)==expected,relative
 prefix='source/portable/'
 if relative.startswith(prefix):
  target=ROOT/'suite/source'/relative[len(prefix):];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(training/relative,target)
for name in ['resume_ouro14_dflash_evaluation.py','orchestrate_packed_dflash_supplements.py','supplement_packed_dflash.py','diagnose_packed_dflash.py','packed_dflash_evidence.py','native_cuda_env.py']:
 py_compile.compile(str(BASE/name),doraise=True);shutil.copy2(BASE/name,ROOT/'source'/name)
shutil.copy2('/ossfs/workspace/loop_suffix/scripts/loop_queue_recovery.py',ROOT/'source/loop_queue_recovery.py')
main=Path('/ossfs/workspace/loop_suffix/results/dflash-table12-packed-20261112-v2')
for name in ['loop_dflash_benchmark_worker.py','loop_dflash_benchmark_suite.py']:
 shutil.copy2(main/'source/scripts'/name,ROOT/'suite/source/scripts'/name)
# Only the denominator annotation changes. Target, decoder, warmup and timing are untouched.
p=ROOT/'suite/source/scripts/loop_dflash_benchmark_worker.py';original=p.read_text();old='original table AR, historical same-host measurement; no drift correction';new='paired fresh own-stack AR measured by the appendix supplemental worker'
assert original.count(old)==1;p.write_text(original.replace(old,new));assert p.read_text().replace(new,old)==original
sys.path[:0]=[str(ROOT/'suite/source'),str(ROOT/'suite/source/src')]
from baselines.infer import tokenizer_and_stops
from llm_memory_editability.loop_model_registry import MODELS,tokenize_prompt
model='ouro-1.4b';spec=MODELS[model];target=Path(trainplan['target_path'])
assert spec.revision=='574fa66cb8bf5abdc979642d01cf2b79b16bfab1' and spec.depth_total==4
for filename in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json']:
 assert sha(target/filename)==trainplan['input_sha256'][str(target/filename)]
tok,stops=tokenizer_and_stops(model,target,'chat',None)
old_suite=BASE.parents[1]/'results/loopfill-latency512-20261107';protocol=read(old_suite/'protocol.json')
jobs=[];source_inputs={}
for dataset in ['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench']:
 path=old_suite/f'inputs/{dataset}.jsonl';assert sha(path)==protocol['inputs'][dataset]['sha256'];source_inputs[str(path)]=sha(path)
 rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==protocol['inputs'][dataset]['count']<=500
 prepared=[dict(source_id=r['source_id'],prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),prompt_tokens=tokenize_prompt(spec,tok,r['prompt'],'chat')) for r in rows]
 assert all(p['prompt_sha256']==r['prompt_sha256'] for p,r in zip(prepared,rows,strict=True))
 required=max(len(r['prompt_tokens']) for r in prepared)+512+spec.depth_total+4;capacity=min(spec.context_limit,((required+255)//256)*256);assert required<=capacity
 dest=ROOT/f'suite/inputs/{model}-{dataset}.jsonl';dest.write_text(''.join(json.dumps(r)+'\n' for r in prepared))
 for temp in [0,1]:jobs.append(dict(id=f'table1-{model}-{dataset}-t{temp}',kind='latency',model=model,dataset=dataset,temperature=temp,batch=1,max_seq=capacity,max_new_tokens=512,count=len(rows),inputs=str(dest.relative_to(ROOT/'suite')),inputs_sha256=sha(dest),original_inputs_sha256=sha(path)))
validation=dict(id='validate-ou ro-1.4b'.replace(' ',''),kind='validate',model=model,dataset='synthetic-prefixes',temperature=0,batch=2,max_seq=128,max_new_tokens=33,count=6,inputs=jobs[0]['inputs'],inputs_sha256=jobs[0]['inputs_sha256'])
config=ROOT/'suite/source/baselines/configs/dflash-ouro-1.4b-163.6m.json'
import torch
from transformers import Qwen3Config
from baselines.dflash.projected_draft import inference_draft_class
with torch.device('meta'):draft=inference_draft_class()(Qwen3Config.from_pretrained(config,local_files_only=True))
parameters=sum(p.numel() for p in draft.parameters());assert parameters==163597568
files=[p for prefix in ['source','suite/source','suite/inputs'] for p in (ROOT/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
plan=dict(created_unix=time.time(),training_root=str(training),training_plan_sha256=sha(training/'plan.json'),python='/tmp/loopfill-dflash-venv/bin/python',target_path=str(target),revision=spec.revision,stop_ids=sorted(stops),draft_config=str(config),draft_parameters=parameters,jobs=jobs,validation_job=validation,gpu_policy=str(BASE/'configs/gpu-allocation.json'),balanced_queue='/ossfs/workspace/loop_suffix/results/gpu-balanced-queue-20261112-v1',frozen_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},original_inputs_sha256=source_inputs,source_adaptation=dict(file='scripts/loop_dflash_benchmark_worker.py',original_sha256=sha(main/'source/scripts/loop_dflash_benchmark_worker.py'),new_sha256=sha(ROOT/'suite/source/scripts/loop_dflash_benchmark_worker.py'),change='Only denominator annotation; no inference or timing change'),scope='Wait for existing full50k/six-epoch163.6M training and actual exit receipts; FP32 serial/batch validation, 12 timed+12 own-stack AR+12 profiles and six greedy numerical checks; no duplicate teachers or training.')
save(ROOT/'plan.json',plan)
from resume_ouro14_dflash_evaluation import verify
verify(ROOT/'plan.json')
save(ROOT/'checks/preparation.json',dict(status='passed',parameters=parameters,formal_cells=len(jobs),per_task_counts={j['dataset']:j['count'] for j in jobs},total_gpu_steps=43,preparation_sha256=sha(Path(__file__)),cuda_initialized=torch.cuda.is_initialized(),checked_unix=time.time()))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='passed',parameters=parameters,formal_cells=len(jobs),total_gpu_steps=43,preparation_cuda_initialized=False)))
