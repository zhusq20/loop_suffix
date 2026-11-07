import importlib.util
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('table2', Path(__file__).parents[1]/'scripts/loop_table2_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_full_matrix_never_prunes_large_batches():
    jobs = suite.cells()
    assert len(jobs) == len({j['key'] for j in jobs}) == 75
    assert {(j['model'],j['arm'],j['batch']) for j in jobs} == {
        (m,a,b) for m in suite.MODELS for a in suite.ARMS for b in suite.BATCHES}
    assert all(j['outcome']=='pending' for j in jobs)


@pytest.fixture
def finished(tmp_path):
    job = dict(arm='ar',batch=1,model='ouro-1.4b',cpus='0-3',gpu_uuid='GPU-test')
    inputs=tmp_path/'inputs.jsonl'
    inputs.write_text(json.dumps(dict(question_id=1,prompt_token_ids=[1,2]))+'\n')
    protocol=dict(models={'ouro-1.4b':dict(smoke_inputs=str(inputs),max_seq=2688)}, source_sha256={})
    record=dict(request_id=1,tokens=[3,4],n_tokens=2)
    key='ar-b2-seed0'
    for suffix in ['.jsonl','-completed.jsonl']:
        (tmp_path/(key+suffix)).write_text(json.dumps(record)+'\n')
    summary=dict(requests=1,output_tokens=2,output_tput=1.,wall_s=2.,concurrency=2,schedule='continuous')
    summary.update({k:0 for k in ['cuda_graph','pad_decode_rows','cuda_graph_shapes','cuda_graph_replays','core_padding_rows','padding_sink_bytes']})
    suite.save(tmp_path/(key+'-summary.json'),summary)
    suite.save(tmp_path/'complete.json',{})
    suite.save(tmp_path/'metadata.json',dict(cuda_graph=False,compile=False,args=dict(max_new_tokens=16,max_seq=2688),
        questions_sha256=suite.digest(inputs),cpu_affinity=[0,1,2,3],gpu_uuid='GPU-test',source_sha256={}))
    return tmp_path,job,protocol


def test_audit_requires_durable_records_and_consistent_timing(finished):
    path,job,protocol=finished
    assert suite.audit(path,job,protocol,True)['total_tput']==2.
    summary_path=path/'ar-b2-seed0-summary.json'
    summary=suite.read(summary_path);summary['output_tput']=7.
    suite.save(summary_path,summary)
    with pytest.raises(AssertionError):suite.audit(path,job,protocol,True)


def test_partial_run_is_never_a_table_result(finished):
    path,job,protocol=finished
    (path/'ar-b2-seed0-completed.jsonl').write_text('')
    with pytest.raises(AssertionError):suite.audit(path,job,protocol,True)


def test_graph_run_is_never_a_table_result(finished):
    path,job,protocol=finished
    s=suite.read(path/'ar-b2-seed0-summary.json');s['cuda_graph_replays']=1
    suite.save(path/'ar-b2-seed0-summary.json',s)
    with pytest.raises(AssertionError):suite.audit(path,job,protocol,True)


@pytest.fixture
def gpu_lane(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import loop_batch_sweep
    monkeypatch.setattr(loop_batch_sweep, 'check_resources', lambda lane: {})
    monkeypatch.setattr(suite, 'gpu_lock_path', lambda lane: tmp_path/'gpu.lock')
    return dict(gpu=0, uuid='test')


def test_busy_gpu_can_be_retried_without_consuming_a_job(gpu_lane):
    with suite.gpu_lock_path(gpu_lane).open('a') as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert suite.try_gpu_lock(gpu_lane) is None
    with suite.try_gpu_lock(gpu_lane) as available:
        assert available is not None


def test_unmanaged_gpu_process_is_retryable_but_settings_error_is_not(gpu_lane, monkeypatch):
    import loop_batch_sweep
    def occupied(lane):
        raise RuntimeError('GPU 0 occupied: 123')
    monkeypatch.setattr(loop_batch_sweep, 'check_resources', occupied)
    assert suite.try_gpu_lock(gpu_lane) is None
    def changed(lane):
        raise RuntimeError('fixed GPU resource settings changed')
    monkeypatch.setattr(loop_batch_sweep, 'check_resources', changed)
    with pytest.raises(RuntimeError, match='settings changed'):
        suite.try_gpu_lock(gpu_lane)
    monkeypatch.setattr(loop_batch_sweep, 'check_resources', lambda lane: {})
    with suite.try_gpu_lock(gpu_lane):
        pass


def test_gpu_lock_survives_handoff_until_worker_exits(gpu_lane):
    reservation = suite.try_gpu_lock(gpu_lane)
    child = subprocess.Popen([sys.executable, '-c',
        'import fcntl,sys; fcntl.flock(int(sys.argv[1]),fcntl.LOCK_EX|fcntl.LOCK_NB); '
        'print("ready",flush=True); sys.stdin.read(1)', str(reservation.fileno())],
        pass_fds=(reservation.fileno(),), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        reservation.close()
        assert child.stdout.readline().strip() == 'ready'
        assert suite.try_gpu_lock(gpu_lane) is None
        child.communicate('x', timeout=5)
        assert child.returncode == 0
        with suite.try_gpu_lock(gpu_lane):
            pass
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_requeue_only_lock_failures_and_preserve_history(tmp_path):
    lock_log=tmp_path/'lock.log'
    lock_log.write_text('fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
                        'BlockingIOError: [Errno 11] Resource temporarily unavailable\n')
    model_log=tmp_path/'model.log'
    model_log.write_text('RuntimeError: model error\n')
    for key, outcome, log in [('busy','worker_failed',lock_log), ('failed','worker_failed',model_log),
                              ('done','complete',lock_log), ('active','running',lock_log)]:
        suite.save(tmp_path/'jobs'/f'{key}.json', dict(key=key,outcome=outcome,worker_log=str(log),worker_returncode=1))
    originals = {name: (tmp_path/'jobs'/f'{name}.json').read_bytes() for name in ['failed','done','active']}
    assert suite.requeue_lock_conflicts(tmp_path) == ['busy']
    assert suite.requeue_lock_conflicts(tmp_path) == []
    job=suite.read(tmp_path/'jobs/busy.json')
    assert job['outcome']=='pending' and len(job['launch_failures'])==1
    for name, data in originals.items():
        assert (tmp_path/'jobs'/f'{name}.json').read_bytes()==data


def test_adopted_worker_does_not_mistake_recycled_pid_for_running(tmp_path, monkeypatch):
    path=tmp_path/'job.json'
    suite.save(path, dict(outcome='running'))
    task=suite.AdoptedWorker(123, 'first-start', path)
    monkeypatch.setattr(suite, 'process_identity', lambda pid: 'first-start')
    assert task.poll() is None
    monkeypatch.setattr(suite, 'process_identity', lambda pid: 'second-start')
    assert task.poll()==1
    suite.save(path, dict(outcome='complete'))
    assert task.poll()==0


def test_running_worker_can_be_found_without_restarting_it(tmp_path):
    job=tmp_path/'jobs/test.json'
    suite.save(job,dict(outcome='running'))
    child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read(1)',
        'loop_table2_suite.py','worker','--out',str(tmp_path),'--job',str(job),'--lane','2'],
        stdin=subprocess.PIPE,text=True)
    try:
        matches=suite.find_live_workers(tmp_path)
        assert len(matches)==1 and matches[0]['process'].pid==child.pid
        assert matches[0]['path']==job and matches[0]['lane']==2
        assert matches[0]['process'].poll() is None
    finally:
        child.communicate('x',timeout=5)
