"""Durable process receipts and safe adoption for experiment queue recovery.

Only a process's parent can collect its exit status. A small independent parent
therefore records the real worker exit even if the queue controller goes away.
Legacy orphans without that evidence report UNKNOWN_EXIT, never an invented 0.
"""
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

UNKNOWN_EXIT = 'unknown_exit'


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Some Python 3.10 builds omit pidfd bindings although the Linux kernel
    # supports them. These architectures share the asm-generic syscall IDs.
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise NotImplementedError('pidfd bindings unavailable on this architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def pidfd_send_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(sig),
                          ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def process_identity(pid):
    try:
        proc = Path(f'/proc/{pid}')
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], start_ticks=fields[19],
                    argv=(proc / 'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError, ProcessLookupError):
        return None


def executable_argv(argv):
    # taskset execs the interpreter, so /proc no longer contains this prefix.
    if argv[:2] == ['taskset', '--cpu-list']:
        return argv[3:]
    return argv


def matching_process(expected):
    actual = process_identity(expected['pid'])
    if actual is None or actual['state'] == 'Z':
        return None
    if expected.get('start_ticks') not in (None, actual['start_ticks']):
        return None
    if executable_argv(actual['argv']) != executable_argv(expected['argv']):
        return None
    if 'start_ticks' not in expected and 'started_unix' in expected:
        boot = next(float(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime '))
        started = boot + int(actual['start_ticks']) / os.sysconf('SC_CLK_TCK')
        if abs(started - expected['started_unix']) > 5:
            return None
    return actual


class ObservedWorker:
    def __init__(self, launch, exit_path, supervisor=None):
        self.launch = dict(launch)
        self.pid = launch['pid']
        self.exit_path = Path(exit_path)
        self.supervisor = supervisor
        current = matching_process(self.launch)
        if current:
            self.launch['start_ticks'] = current['start_ticks']

    def poll(self):
        if self.supervisor is not None:
            self.supervisor.poll()  # Reap the receipt writer after it exits.
        if self.exit_path.exists():
            receipt = read(self.exit_path)
            for key in ('pid', 'argv', 'started_unix'):
                if receipt[key] != self.launch[key]:
                    raise ValueError(f'Exit receipt identity mismatch: {self.exit_path}')
            code = receipt.get('exit_code')
            return code if type(code) is int else UNKNOWN_EXIT
        if matching_process(self.launch):
            return None
        # The supervisor may be between wait() and its atomic receipt write.
        if self.supervisor is not None and self.supervisor.poll() is None:
            return None
        if self.launch.get('supervisor_identity') and matching_process(self.launch['supervisor_identity']):
            return None
        return UNKNOWN_EXIT


def spawn_supervised(argv, *, env, log, lock, launcher_path, exit_path,
                     metadata, cwd=None, extra_launcher=None, extra_exit=None):
    launcher_path, exit_path = Path(launcher_path), Path(exit_path)
    assert not launcher_path.exists() and not exit_path.exists()
    spec = dict(argv=argv, launcher_path=str(launcher_path), exit_path=str(exit_path),
                metadata=metadata, cwd=str(cwd) if cwd else None,
                extra_launcher=str(extra_launcher) if extra_launcher else None,
                extra_exit=str(extra_exit) if extra_exit else None)
    spec_path = launcher_path.with_name(launcher_path.name + '.supervisor.json')
    save(spec_path, spec)
    supervisor = subprocess.Popen(
        [sys.executable, '-u', str(Path(__file__).resolve()), '--spec', str(spec_path),
         '--lock-fd', str(lock.fileno())], env=env, stdout=log, stderr=subprocess.STDOUT,
        pass_fds=(lock.fileno(),), start_new_session=True)
    deadline = time.monotonic() + 20
    while not launcher_path.exists():
        if supervisor.poll() is not None:
            raise RuntimeError(f'Worker supervisor failed to launch; see {log.name}')
        if time.monotonic() > deadline:
            raise RuntimeError(f'Worker launch not acknowledged; inspect {spec_path}')
        time.sleep(.02)
    return ObservedWorker(read(launcher_path), exit_path, supervisor)


def supervise(spec_path, lock_fd):
    spec = read(spec_path)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = dict(spec['metadata'], argv=spec['argv'], started_unix=time.time())
    child = subprocess.Popen(spec['argv'], cwd=spec['cwd'], pass_fds=(lock_fd,))
    launch['pid'] = child.pid
    current = process_identity(child.pid)
    if current:
        launch['start_ticks'] = current['start_ticks']
    launch['supervisor_identity'] = process_identity(os.getpid())
    save(spec['launcher_path'], launch)
    if spec['extra_launcher']:
        save(spec['extra_launcher'], launch)
    code = child.wait()
    receipt = dict(launch, exit_code=code, finished_unix=time.time(),
                   exit_evidence='supervisor_waitpid')
    # External collectors must see the receipt before the queue does.
    if spec['extra_exit']:
        save(spec['extra_exit'], receipt)
    save(spec['exit_path'], receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    supervise(args.spec, args.lock_fd)
