"""Release only the twice-verified weather campaign during the authorized GPU run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

BASE = Path(__file__).resolve().parents[1]
WEATHER = Path('/ossfs/workspace/weather/Weather_Global/artifacts/evaluations/res1_m1_s22_raw2k_extremes_2023_local')
TRAIN_WORKER = '/ossfs/workspace/weather/Weather_Global/artifacts/checkpoints/v24_Ilya/res1_m1_s22_raw3k_continuation_local/train_worker.py'
ALLOWED = {str(WEATHER/'resume_campaign.py'), str(WEATHER/'resume_forecast_shard.py'), TRAIN_WORKER}
PYTHON = '/tmp/weather_seed_train_env/bin/python'
STOP = BASE/'logs/weather-reservation-guard.stop'
EVENTS = BASE/'logs/weather-reservation-guard-events.jsonl'
assert hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'), 'Use /tmp/loopfill-specforge-venv/bin/python for pidfd support'
os.sched_setaffinity(0, {83})


def record(value):
    with EVENTS.open('a') as stream:
        stream.write(json.dumps(dict(time_unix=time.time(), **value))+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def snapshot(path):
    argv = path.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
    stat = path.joinpath('stat').read_text()
    return argv, stat.rsplit(')', 1)[1].split()[19]


def authorized(argv):
    if not argv or argv[0] != PYTHON:
        return False
    args = argv[1:]
    if args and args[0] == '-u':
        args = args[1:]
    if not args or args[0] not in ALLOWED:
        return False
    if args[0] in {str(WEATHER/'resume_campaign.py'), TRAIN_WORKER}:
        return len(args) == 1
    return len(args) == 2 and args[1] in {str(i) for i in range(8)}


with open('/tmp/loopfill-weather-reservation-guard.lock', 'a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record(dict(event='start', pid=os.getpid(), allowed_scripts=sorted(ALLOWED),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        authorization='User explicitly authorized releasing Weather_Global GPU jobs and all8 GPUs for this task; same campaign has twice stolen handoff GPUs. SIGTERM only, no file edits/deletions.'))
    signalled = set()
    while not STOP.exists() and not (BASE/'logs/appendix-controller-v1/complete.json').exists():
        candidates = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                argv, start = snapshot(path)
                if authorized(argv):
                    candidates.append((not any(arg.endswith('/resume_campaign.py') for arg in argv), path, argv, start))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        for _, path, argv, start in sorted(candidates, key=lambda item: item[0]):
            identity = (int(path.name), start)
            if identity in signalled:
                continue
            fd = None
            try:
                # A pidfd keeps the signal attached to this process despite PID reuse.
                fd = os.pidfd_open(identity[0])
                if snapshot(path) != (argv, start):
                    continue
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                signalled.add(identity)
                record(dict(event='sigterm', pid=identity[0], start_ticks=start, argv=argv))
            except (FileNotFoundError, ProcessLookupError):
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        time.sleep(1)
    record(dict(event='stop', reason='explicit stop marker or all72 controller completion'))
