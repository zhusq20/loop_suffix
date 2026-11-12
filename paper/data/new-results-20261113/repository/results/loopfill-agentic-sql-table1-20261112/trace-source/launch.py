from pathlib import Path
import subprocess,fcntl,json,os,time
root=Path(__file__).resolve().parents[1]
protocol=json.loads((root/'trace-protocol.json').read_text())
python='/tmp/loopfill-dflash-venv/bin/python'
source=Path(protocol['teacher_source'])
with (root/'trace-controller.lock').open('a') as control:
 fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
 devices={int(p[0]):p[1] for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines() if (p:=[s.strip() for s in row.split(',')])}
 occupied=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
 locks={}
 for gpu in [0,1,5,7]:
  assert devices[gpu] not in occupied, f'GPU {gpu} is occupied'
  lock=open('/tmp/loopfill-'+devices[gpu]+'.lock','a')
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  locks[gpu]=lock
 workers=[]
 for shard,gpu in enumerate([0,1,5,7]):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
  cpu={0:'0-3',1:'4-7',5:'52-55',7:'60-63'}[gpu]
  cmd=['taskset','-c',cpu,python,str(source/'runtime/native_cuda_env.py'),'--record',str(root/'logs'/f'trace-{shard}-driver.json'),'--',python,'-u',str(root/'trace-source/loop_agentic_sql.py'),'trace','--out',str(root),'--shard',str(shard),'--shards','4']
  log=(root/'logs'/f'trace-{shard}.log').open('x')
  process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(locks[gpu].fileno(),))
  workers.append((shard,gpu,process,log))
 (root/'trace-launch.json').write_text(json.dumps(dict(pid=os.getpid(),workers=[dict(shard=s,gpu=g,pid=p.pid) for s,g,p,l in workers]),indent=2))
 while any(p.poll() is None for s,g,p,l in workers):
  record=dict(stage='trace_generation',tasks_completed=len(list((root/'traces').glob('spider-dev-*.json'))),tasks_total=100,workers=[dict(shard=s,gpu=g,pid=p.pid,exit_code=p.poll()) for s,g,p,l in workers],updated_unix=time.time())
  temp=root/'trace-status.json.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(root/'trace-status.json')
  time.sleep(15)
 codes=[p.returncode for s,g,p,l in workers]
 for s,g,p,l in workers:
  l.close();locks[g].close()
 (root/'trace-status.json').write_text(json.dumps(dict(stage='trace_complete' if not any(codes) else 'trace_failed',tasks_completed=len(list((root/'traces').glob('spider-dev-*.json'))),exit_codes=codes),indent=2))
