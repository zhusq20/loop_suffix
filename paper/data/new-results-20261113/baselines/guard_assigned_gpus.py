"""Keep the user's assigned four GPUs available without touching released GPUs.

Only exact previously-authorized Weather workers with an actual CUDA context on
an assigned GPU are signaled. CPU launchers and workers on released GPUs are
never signaled. pidfds protect against PID reuse during the identity check.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from gpu_allocation import allowed_gpus

BASE=Path(__file__).resolve().parent
WEATHER=Path('/ossfs/workspace/weather/Weather_Global/artifacts')
FORECAST=str(WEATHER/'evaluations/res1_m1_s22_raw2k_extremes_2023_local/resume_forecast_shard.py')
TRAIN=str(WEATHER/'checkpoints/v24_Ilya/res1_m1_s22_raw3k_continuation_local/train_worker.py')


def known_worker(argv):
    if not argv or argv[0]!='/tmp/weather_seed_train_env/bin/python':return False
    args=argv[1:]
    if args and args[0]=='-u':args=args[1:]
    return (len(args)==2 and args[0]==FORECAST and args[1] in [str(i) for i in range(8)]) or args==[TRAIN]


def contexts():
    result={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
        pid,uuid=line.split(',');result.setdefault(int(pid),set()).add(uuid.strip())
    return result


def main():
    os.sched_setaffinity(0,{83})
    uuids={int(i):u.strip() for i,u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    sent=set()
    with open('/tmp/loopfill-assigned-gpu-guard.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (BASE/'logs/assigned-gpu-guard-stop').exists() and not (BASE/'logs/appendix-controller-v1/complete.json').exists():
            assigned={uuids[g] for g in allowed_gpus()}
            for pid,devices in contexts().items():
                # A multi-GPU job also using a released GPU is not touched.
                if not devices or not devices<=assigned:continue
                try:fd=os.pidfd_open(pid)
                except ProcessLookupError:continue
                try:
                    proc=Path(f'/proc/{pid}')
                    argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
                    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
                    key=(pid,stat[19])
                    if key in sent or stat[0]=='Z' or not known_worker(argv):continue
                    current=contexts().get(pid,set())
                    if current!=devices or not current<={uuids[g] for g in allowed_gpus()}:continue
                    signal.pidfd_send_signal(fd,signal.SIGTERM)
                    sent.add(key)
                    event=dict(pid=pid,start_ticks=stat[19],argv=argv,gpu_uuids=sorted(current),
                        signal='SIGTERM',sent_unix=time.time(),scope='Assigned GPUs only; released GPUs untouched')
                    with (BASE/'logs/assigned-gpu-guard-events.jsonl').open('a') as stream:
                        stream.write(json.dumps(event)+'\n')
                    print(json.dumps(event),flush=True)
                except (FileNotFoundError,ProcessLookupError):pass
                finally:os.close(fd)
            time.sleep(1)


if __name__=='__main__':main()
