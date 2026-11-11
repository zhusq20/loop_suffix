import fcntl
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import loop_queue_recovery as recovery  # noqa: E402


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('Timed out')


@pytest.mark.parametrize('code', [0, 7])
def test_supervisor_records_actual_exit_and_preserves_lock(tmp_path, code):
    gate = tmp_path / 'finish'
    lock = (tmp_path / 'gpu.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    argv = [sys.executable, '-c',
            'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
            '\nwhile not p.exists(): time.sleep(.02)\nsys.exit(int(sys.argv[2]))',
            str(gate), str(code)]
    with (tmp_path / 'log').open('w') as log:
        worker = recovery.spawn_supervised(argv, env=os.environ.copy(), log=log, lock=lock,
            launcher_path=tmp_path / 'launch.json', exit_path=tmp_path / 'exit.json',
            metadata=dict(gpu=0))
    lock.close()  # Model the controller disappearing; child retains the lease.
    try:
        with (tmp_path / 'gpu.lock').open('a+') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        adopted = recovery.ObservedWorker(recovery.read(tmp_path / 'launch.json'), tmp_path / 'exit.json')
        assert adopted.poll() is None
        gate.touch()
        wait_for(lambda: (tmp_path / 'exit.json').exists())
        assert adopted.poll() == code
        receipt = recovery.read(tmp_path / 'exit.json')
        assert receipt['pid'] == worker.pid and receipt['exit_evidence'] == 'supervisor_waitpid'
        worker.supervisor.wait(timeout=5)
        with (tmp_path / 'gpu.lock').open('a+') as contender:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        gate.touch()
        worker.supervisor.wait(timeout=5)


def test_real_controller_death_keeps_receipt_writer_alive(tmp_path):
    launcher = tmp_path / 'parent.py'
    launcher.write_text('''import fcntl,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from loop_queue_recovery import spawn_supervised
p=Path(sys.argv[2]); lock=(p/'lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (p/'log').open('w') as log:
 spawn_supervised([sys.executable,'-c',
  'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]);\\nwhile not p.exists(): time.sleep(.02)',
  str(p/'gate')],env=os.environ.copy(),log=log,lock=lock,
  launcher_path=p/'launch.json',exit_path=p/'exit.json',metadata={})
(p/'ready').touch()
time.sleep(60)
''')
    parent = subprocess.Popen([sys.executable, str(launcher), str(ROOT / 'scripts'), str(tmp_path)])
    try:
        wait_for(lambda: (tmp_path / 'ready').exists())
        parent.kill()
        parent.wait(timeout=5)
        adopted = recovery.ObservedWorker(recovery.read(tmp_path / 'launch.json'), tmp_path / 'exit.json')
        assert adopted.poll() is None
        (tmp_path / 'gate').touch()
        wait_for(lambda: (tmp_path / 'exit.json').exists())
        assert adopted.poll() == 0
    finally:
        (tmp_path / 'gate').touch()
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)


def test_unknown_exit_is_not_success_even_with_complete_artifact(tmp_path, monkeypatch):
    (tmp_path / 'complete.json').write_text(json.dumps(dict(requests=500)))
    monkeypatch.setattr(recovery, 'matching_process', lambda expected: None)
    worker = recovery.ObservedWorker(dict(pid=99999, argv=['python'], started_unix=1), tmp_path/'exit.json')
    assert worker.poll() == recovery.UNKNOWN_EXIT


def test_pid_reuse_and_taskset_exec_identity(monkeypatch):
    actual = dict(pid=10, start_ticks='20', state='R', argv=['python', 'worker.py'])
    monkeypatch.setattr(recovery, 'process_identity', lambda pid: actual)
    expected = dict(pid=10, start_ticks='20', argv=['taskset','--cpu-list','0-3','python','worker.py'])
    assert recovery.matching_process(expected) == actual
    assert recovery.matching_process(dict(expected, start_ticks='19')) is None
    assert recovery.matching_process(dict(expected, argv=['python','different.py'])) is None


def test_pidfd_fallback_on_python_without_native_bindings(monkeypatch):
    monkeypatch.delattr(os, 'pidfd_open', raising=False)
    monkeypatch.delattr(signal, 'pidfd_send_signal', raising=False)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    fd = recovery.pidfd_open(child.pid)
    try:
        recovery.pidfd_send_signal(fd, signal.SIGTERM)
        assert select.select([fd], [], [], 5)[0]
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        os.close(fd)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_mismatched_exit_receipt_rejected(tmp_path, monkeypatch):
    launch = dict(pid=10, argv=['python'], started_unix=1)
    monkeypatch.setattr(recovery, 'matching_process', lambda expected: None)
    recovery.save(tmp_path/'exit.json', dict(launch, pid=11, exit_code=0))
    worker = recovery.ObservedWorker(launch, tmp_path/'exit.json')
    with pytest.raises(ValueError, match='identity mismatch'):
        worker.poll()


def load_current_controller():
    path = ROOT / 'results/gpu-balanced-queue-20261112-v1/controller.py'
    if not path.exists():
        pytest.skip('Run-specific controller is not available in this checkout')
    spec = importlib.util.spec_from_file_location('balanced_queue_test', path)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    return controller


def test_stopped_legacy_controller_without_active_field(tmp_path, monkeypatch):
    controller = load_current_controller()
    recovery.save(tmp_path/'status.json', dict(status='stopped', counts=dict(complete=7)))
    recovery.save(tmp_path/'dispatch.json', dict(job=dict(outcome='complete')))
    monkeypatch.setattr(controller, 'SUITE', tmp_path)
    monkeypatch.setattr(controller, 'matching_process', lambda expected: None)
    controller.stop_idle_controller(dict(pid=99999))


def test_legacy_controller_with_running_workers_cannot_be_retired(tmp_path, monkeypatch):
    controller = load_current_controller()
    recovery.save(tmp_path/'status.json', dict(status='stopped'))
    recovery.save(tmp_path/'dispatch.json', dict(job=dict(outcome='running')))
    monkeypatch.setattr(controller, 'SUITE', tmp_path)
    with pytest.raises(AssertionError):
        controller.stop_idle_controller(dict(pid=99999))
