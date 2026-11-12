import json, pathlib, subprocess, time
b=pathlib.Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
f=b/'logs/formal-controller-v1'
pause=f/'pause-new-jobs'
with pause.open('x') as stream:stream.write('Temporary lane reservation for first complete formal BF16 divergence replay.\n')
name='formal-greedy-margins-loopahead-huginn-3.5b-gsm8k-v1'
out=b/'preflight/formal-greedy-margins/loopahead-huginn-3.5b-gsm8k-v1'
cpus=['0-3','4-7','8-11','12-15','48-51','52-55','56-59','60-63']
started=time.time();child=None
try:
 while time.time()-started<2700:
  state=json.loads((f/'status.json').read_text())
  assert not state['failed'],state['failed']
  reserved={g for v in state['running'].values() for g in v['gpus']}
  uuids={int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
  busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
  free=[g for g in [1,3,5,6] if g not in reserved and uuids[g] not in busy]
  if free:break
  time.sleep(10)
 else:raise RuntimeError('No evaluation lane released within45min')
 gpu=free[0]
 argv=['/tmp/loopfill-venv/bin/python','-u',str(b/'diagnose_formal_loopahead.py'),'--ar',str(b/'runs/archived-ar-huginn-3.5b-gsm8k-t0'),'--loopahead',str(b/'runs/loopahead-huginn-3.5b-gsm8k-t0-v1'),'--gpu',str(gpu),'--cpus',cpus[gpu],'--out',str(out)]
 with (b/f'logs/{name}.log').open('x') as stream:child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 event=dict(pid=child.pid,argv=argv,gpu=gpu,started_unix=time.time())
 (b/f'logs/{name}-launcher.json').write_text(json.dumps(event,indent=2)+'\n');print(json.dumps(event),flush=True)
 loading_deadline=time.time()+300
 while child.poll() is None:
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
  if any(line.split(',')[0].strip()==str(child.pid) and uuids[gpu] in line for line in busy.splitlines()):break
  if time.time()>loading_deadline:
   child.terminate()
   raise RuntimeError('Diagnostic did not claim its GPU within5min; stopped its own child and released scheduling pause')
  time.sleep(1)
finally:
 pause.unlink(missing_ok=True)
if child is not None:
 code=child.wait()
 (b/f'logs/{name}-finished.json').write_text(json.dumps(dict(event,exit_code=code,finished_unix=time.time()),indent=2)+'\n')
 assert code==0,code
