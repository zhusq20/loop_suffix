"""Continue hardware telemetry after the original DFlash parent drains."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / 'dflash-table12-packed-20261112-v2'
while True:
    state = json.loads((HERE / 'status.json').read_text())
    if state['stage'] in ['complete', 'needs_attention'] or (HERE / 'failure.json').exists():
        break
    if not Path(f"/proc/{state['controller_pid']}/cmdline").exists():
        break
    csv = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,clocks.current.sm,clocks.current.memory,power.limit,temperature.gpu,utilization.gpu,memory.used',
        '--format=csv,noheader'], text=True)
    record = dict(unix=time.time(), gpu_csv=csv, controller_pid=state['controller_pid'])
    with (HERE / 'telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if state['original_controller_retired']:
        with (SUITE / 'telemetry.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
    time.sleep(20)
