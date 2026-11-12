from pathlib import Path
import csv,json,sys,shutil,time,py_compile
BASE=Path('/ossfs/workspace/sz/loopfill/experiments/baselines');sys.path.insert(0,str(BASE))
from packed_dflash_evidence import read,save,sha,compare,rows
from orchestrate_greedy_diagnostics import paper_check_plan,development_check_plan,validate
ROOT=BASE/'logs/baseline-diagnostics-resume-v1'
ROOT.mkdir(exist_ok=False)
for name in ['source','logs','checks']:(ROOT/name).mkdir()
for name in ['orchestrate_baseline_diagnostics_resume.py','orchestrate_packed_dflash_supplements.py','diagnose_packed_dflash.py','packed_dflash_evidence.py','native_cuda_env.py']:
 py_compile.compile(str(BASE/name),doraise=True);shutil.copy2(BASE/name,ROOT/'source'/name)
shutil.copy2('/ossfs/workspace/loop_suffix/scripts/loop_queue_recovery.py',ROOT/'source/loop_queue_recovery.py')
shutil.copy2('/ossfs/workspace/loop_suffix/results/gpu-balanced-queue-20261112-v1/lock_handoff.py',ROOT/'source/lock_handoff.py')
agreements=read(BASE/'formal-agreement.json')
records=[r for r in csv.DictReader((BASE/'results.csv').open()) if r['temperature']=='0']
assert len(records)==21
paper,paper_agreement,paper_provenance=paper_check_plan()
records.append(paper)
groups=[];reused={}
for row in records:
 method,model,task=row['method'],row['model'],row['benchmark']
 identity=f'{method}-{model}-{task}-v1'
 if row.get('paper_check'):
  folder=BASE/'preflight/loopahead-B4-greedy-margins-v1';agreement=paper_agreement;scope=['--paper-check']
 else:
  folder=BASE/'preflight/formal-greedy-margins'/identity;agreement=agreements['/'.join([row['engine'],model,task,'0',method])];scope=[]
 actual_agreement=compare(rows(Path(row['method_run'])/'requests.jsonl'),rows(Path(row['ar_run'])/'requests.jsonl'))
 if row.get('paper_check'):
  assert all(actual_agreement[k]==agreement[k] for k in ['requests','exact_sequences','first_divergences'])
 else:
  assert actual_agreement==agreement
 python='/tmp/loopfill-venv/bin/python' if method=='loopahead' else '/tmp/loopfill-dflash-venv/bin/python'
 script='diagnose_formal_loopahead.py' if method=='loopahead' else 'diagnose_dflash.py'
 marker=folder/('report.json' if method=='loopahead' else 'complete.json')
 argv=[python,'-u',str(BASE/script),'--ar',row['ar_run'],'--'+method,row['method_run'],'--gpu','{GPU}','--cpus','{CPUS}','--out',str(folder),*scope]
 if method=='dflash':argv+=['--shard','0','--shards','1']
 steps=[dict(id=identity+'-observed',argv=argv,out=str(folder),marker=str(marker),legacy_lock=True)]
 if method=='loopahead' and agreement['first_divergences']:
  fp=folder/'fp32-common-prefix'
  steps.append(dict(id=identity+'-fp32',requires=[steps[0]['id']],out=str(fp),marker=str(fp/'report.json'),legacy_lock=True,
   argv=[python,'-u',str(BASE/'diagnose_trained_loopahead.py'),'--ar',row['ar_run'],'--loopahead',row['method_run'],'--observed-margin-report',str(marker),'--gpu','{GPU}','--cpus','{CPUS}','--out',str(fp),*scope]))
 g=dict(id=identity,kind='legacy',row=row,agreement=agreement,out=str(folder),steps=steps)
 if folder.exists():
  report=validate(row,folder,agreement)
  reused[identity]=dict(outcome='validated',report=report,origin='Existing completed original diagnostics; no GPU rerun')
 groups.append(g)
dev=development_check_plan()
for item in dev:
 label=item['label'];folder=BASE/f'preflight/huginn-dflash-dev-greedy-{label}-v1';identity='development-'+label
 assert not folder.exists()
 groups.append(dict(id=identity,kind='development',label=label,provenance=item,out=str(folder),steps=[dict(id=identity,out=str(folder),marker=str(folder/'complete.json'),legacy_lock=True,
  argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(BASE/'diagnose_huginn_dflash_dev.py'),'--label',label,'--gpu','{GPU}','--cpus','{CPUS}','--out',str(folder)])]))
supplement=BASE/'logs/packed-dflash-supplement-v1/plan.json'
for job in read(supplement)['jobs']:
 if job['temperature']!=0:continue
 identity=job['id']+'-greedy';folder=BASE/'preflight/packed-dflash-greedy-v1'/job['id']
 assert not folder.exists()
 ar=supplement.parent/'runs'/(job['id']+'-ar')
 groups.append(dict(id=identity,kind='packed',job=job['id'],out=str(folder),steps=[dict(id=identity,out=str(folder),marker=str(folder/'complete.json'),legacy_lock=False,
  argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(ROOT/'source/diagnose_packed_dflash.py'),'--plan',str(supplement),'--job',job['id'],'--ar',str(ar),'--out',str(folder),'--lock-fd','{LOCK_FD}'])]))
legacy_names=set(read(BASE/'logs/loopahead-ouro14-resume-v1/plan.json')['source_sha256'])
legacy_names.update(['orchestrate_greedy_diagnostics.py','diagnose_formal_loopahead.py','diagnose_dflash.py','diagnose_huginn_dflash_dev.py','compare_huginn_dflash_dev.py','loopahead_diagnostic_inputs.py','dflash_source_identity.py','evaluate_dflash.py','evaluate_projected_dflash.py','dflash_training_registry.py','provenance.py'])
legacy_names.update(str(p.relative_to(BASE)) for p in (BASE/'dflash_impl').glob('*.py'))
legacy_names.add('third_party/dflash/dflash/model.py')
plan=dict(created_unix=time.time(),base=str(BASE),balanced_queue='/ossfs/workspace/loop_suffix/results/gpu-balanced-queue-20261112-v1',supplement_plan=str(supplement),gpu_policy=str(BASE/'configs/gpu-allocation.json'),
 source_sha256={p.name:sha(p) for p in sorted((ROOT/'source').glob('*.py'))},legacy_source_sha256={n:sha(BASE/n) for n in sorted(legacy_names)},groups=groups,reused_groups=reused,paper_provenance=paper_provenance,
 limitation='Appendix Ouro1.4B diagnostic groups depend on its new training and evaluation and are not yet included. No completed measurement is rerun. Diagnostic traces keep original max512/max1024 or max128 development block budgets.')
assert len(groups)==27 and len(reused)==1
save(ROOT/'plan.json',plan)
from orchestrate_baseline_diagnostics_resume import verify_plan
verify_plan(ROOT/'plan.json')
save(ROOT/'checks/preparation.json',dict(status='passed',groups=len(groups),reused=list(reused),legacy_formal_cells=21,new_packed_formal_cells=3,paper_checks=1,development_checks=2,source_sha256=plan['source_sha256'],preparation_sha256=sha(Path(__file__)),checked_unix=time.time()))
print(json.dumps(dict(status='passed',groups=len(groups),reused_groups=list(reused),pending_steps=sum(len(g['steps']) for g in groups if g['id'] not in reused))))
