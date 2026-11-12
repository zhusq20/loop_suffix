taskset -c 80-83 python - <<'PY'
from pathlib import Path
import json,sys,time,subprocess,hashlib
b=Path('/ossfs/workspace/sz/loopfill/experiments/baselines');sys.path.insert(0,str(b))
from collect_results import read_run,sha
from dflash_source_identity import validate_evaluation_sources
out=b/'preflight/dflash-cuda-startup-live-check-v1';assert not out.exists()
name='dflash-stack-ouro-2.6b-gsm8k-t1';ar=b/'runs'/f'{name}-ar-v1';df=b/'runs'/f'{name}-dflash-v1'
a=read_run(ar,json.loads((b.parents[1]/'results/loopfill-latency512-20261107/protocol.json').read_text()))
m=json.loads((df/'metadata.json').read_text());am=validate_evaluation_sources(ar);dm=validate_evaluation_sources(df)
exit_receipt=json.loads((b/'logs'/f'{name}-ar-v1-finished.json').read_text());oldlaunch=json.loads((b/'logs'/f'{name}-ar-v1-launcher.json').read_text())
assert exit_receipt['exit_code']==0 and all(exit_receipt[k]==oldlaunch[k] for k in ['pid','argv','started_unix'])
keys=['model','model_revision','dataset','temperature','top_p','top_k','seed','count','full_count','max_new_tokens','max_seq','input_sha256','stop_audit','torch','transformers','cuda','dtype','graph','compile','gpu_uuid','cpu_affinity']
assert all(a['meta'][k]==m[k] for k in keys)
assert m['method']=='dflash' and m['reportable'] and not m['profiling'] and m['stage']=='formal'
auditpath=b/'training/dflash-ouro-2.6b-formal-audit-v1/report.json';audit=json.loads(auditpath.read_text());cp=Path(m['checkpoint'])
assert audit['status']=='complete_training_audit_not_benchmark' and audit['checkpoint']==str(cp)
assert m['checkpoint_sha256']=={'model.safetensors':audit['weights_sha256'],'config.json':audit['checkpoint_config_sha256']}
assert all(sha(cp/k)==v for k,v in m['checkpoint_sha256'].items())
loading=json.loads((df/'checkpoint-loading.json').read_text());assert all(not v for v in loading.values())
acq=json.loads((df/'cuda-acquisition.json').read_text());launch=json.loads((b/'logs'/f'{name}-dflash-v1-launcher.json').read_text())
assert acq['pid']==launch['pid']==181571 and acq['gpu']==2 and acq['gpu_uuid']==m['gpu_uuid']
assert acq['attempts']==1 and acq['cuda_busy_errors']==0 and acq['acquired_unix']>=acq['started_unix']
p=Path('/proc/181571');st=p.joinpath('stat').read_text().rsplit(')',1)[1].split();assert st[19]=='37697936'
assert [x.decode() for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x]==launch['argv']
contexts=[(u.strip(),int(pid)) for u,pid in (l.split(',') for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines())]
assert (acq['gpu_uuid'],181571) in contexts
allocation=json.loads((b/'configs/gpu-allocation.json').read_text());assert allocation['allowed_gpus']==[2,3,4,6]
lines=(df/'requests.jsonl').read_bytes().split(b'\n')[:-1][:20];assert lines
for row,reference in zip(map(json.loads,lines),a['rows']):
 assert row['source_id']==reference['source_id'] and row['prompt_sha256']==reference['prompt_sha256']
 assert 1<=len(row['tokens'])<=512 and row['stats']['decode_tokens']==len(row['tokens'])-1
assert all(am[k]==dm[k] for k in ['dflash_impl/target.py','dflash_impl/decode.py'])
out.mkdir();(out/'observed-request-prefix.jsonl').write_bytes(b'\n'.join(lines)+b'\n')
snapshots={'ar-complete.json':ar/'complete.json','ar-exit.json':b/'logs'/f'{name}-ar-v1-finished.json','ar-metadata.json':ar/'metadata.json','dflash-metadata.json':df/'metadata.json','cuda-acquisition.json':df/'cuda-acquisition.json','checkpoint-loading.json':df/'checkpoint-loading.json','dflash-launcher.json':b/'logs'/f'{name}-dflash-v1-launcher.json'}
for dest,src in snapshots.items():(out/dest).write_bytes(src.read_bytes())
report={'status':'actual_startup_verified_timed_evaluation_incomplete','checked_unix':time.time(),'model':m['model'],'dataset':m['dataset'],'temperature':m['temperature'],'pid':181571,'proc_start_ticks':st[19],'actual_state':st[0],'ar_requests':len(a['rows']),'ar_summary':a['summary'],'ar_exit_code':0,'ar_journal_sha256':sha(ar/'requests.jsonl'),'matched_configuration_keys':keys,'ar_source_manifest_sha256':sha(ar/'source-manifest.json'),'dflash_source_manifest_sha256':sha(df/'source-manifest.json'),'old_and_new_entrypoints_pass_reviewed_source_identity':True,'target_and_decode_sources_identical':True,'startup_helper_sha256':dm['gpu_startup.py'],'checkpoint':str(cp),'checkpoint_sha256':m['checkpoint_sha256'],'training_audit_sha256':sha(auditpath),'checkpoint_loading':loading,'cuda_acquisition':acq,'actual_gpu_context_verified':True,'allowed_gpus':allocation['allowed_gpus'],'observed_prefix_requests':len(lines),'observed_prefix_sha256':sha(out/'observed-request-prefix.jsonl'),'snapshot_sha256':{dest:sha(out/dest) for dest in snapshots},'limitations':['Actual GPU startup succeeded on first attempt; real busy-error retry was not exercised. Retry and fault cases retain prior CPU mock coverage.','Full timed DFlash evaluation and independent full-task profile remain incomplete; no formal matrix cell is added.','T1 sampled sequence equality is not a proof of accuracy or distribution preservation.'],'checker_note':'Initial read-only check used the generic audit status complete and rejected the report before writing artifacts. Corrected to its actual complete_training_audit_not_benchmark schema; no GPU process changed.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
PY
