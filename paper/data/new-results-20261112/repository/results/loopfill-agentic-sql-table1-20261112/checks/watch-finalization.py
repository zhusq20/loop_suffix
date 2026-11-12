from pathlib import Path
import json,subprocess,time,os,sys
s=Path(__file__).resolve().parents[1]
repo=Path(sys.argv[1])
controller=json.loads((s/'controller-launch.json').read_text())['pid']
while True:
 state=json.loads((s/'status.json').read_text())
 if state['status']=='complete':
  result=subprocess.run([sys.executable,str(s/'checks/finalize.py'),'--suite',str(s),'--repository',str(repo)])
  (s/'checks/finalization-status.json').write_text(json.dumps(dict(status='complete' if result.returncode==0 else 'audit_failed',exit_code=result.returncode),indent=2))
  sys.exit(result.returncode)
 if state['status'] in ('failed','stop_requested') or not Path(f'/proc/{controller}').exists():
  (s/'checks/finalization-status.json').write_text(json.dumps(dict(status='not_exported',reason=state['status']),indent=2))
  sys.exit(1)
 time.sleep(15)
