import json, subprocess, time
from pathlib import Path
out=Path(__file__).resolve().parent
args=['/tmp/loopfill-dflash-venv/bin/python', '/ossfs/workspace/sz/loopfill/experiments/baselines/native_cuda_env.py', '--record', str(out/'gpu-driver-environment.json'), '--', '/tmp/loopfill-dflash-venv/bin/python', '-u', '/ossfs/workspace/loop_suffix/scripts/loop_dflash_packed_probe.py', '--gpu', '2', '--cpus', '8-11', '--wait-seconds', '5400', '--out', str(out/'gpu-v1'), '--release-reservation']
with (out/'gpu.log').open('x') as log:
    process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd='/ossfs/workspace/loop_suffix')
    receipt=dict(argv=args,pid=process.pid,started_unix=time.time())
    (out/'gpu-launcher.json').write_text(json.dumps(receipt,indent=2)+'\n')
    receipt.update(exit_code=process.wait(),finished_unix=time.time())
    (out/'gpu-finished.json').write_text(json.dumps(receipt,indent=2)+'\n')
