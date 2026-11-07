"""Timing boundaries, deterministic subsets and crash recovery are experiment contracts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_latency_inputs import choose, rank
from loop_latency_suite import jobs, ARMS
from loop_latency_worker import drive, load_journal, summarize


def test_decode_excludes_prefill_and_includes_publication():
    now = [0.0]
    sync = []
    class Backbone:
        def synchronize(self):
            sync.append(now[0])
    class Engine:
        def execute(self, name, jobs):
            now[0] += 100 if name == 'prefill' else 2
            return [1]
    def steps():
        yield SimpleNamespace(name='prefill')
        now[0] += 3  # CPU suffix setup
        yield SimpleNamespace(name='advance')
        now[0] += 5  # completion publication
        return [1, 2], {'tokens': 1, 'calls': 4}
    tokens, stats = drive(Backbone(), Engine(), steps(), clock=lambda: now[0])
    assert tokens == [1, 2]
    assert stats['prefill_s'] == 100 and stats['decode_s'] == 10
    assert sync == [0, 100, 110]


def test_resume_truncates_only_torn_final_record(tmp_path):
    p = tmp_path / 'journal.jsonl'
    complete = json.dumps({'source_id': 'a', 'tokens': [1, 2]}) + '\n'
    p.write_text(complete + '{"source_id":')
    assert len(load_journal(p, ['a', 'b'])) == 1
    assert p.read_text() == complete
    with pytest.raises(ValueError, match='order'):
        load_journal(p, ['b', 'a'])
    p.write_text('{broken}\n')
    with pytest.raises(json.JSONDecodeError):
        load_journal(p, ['a'])


def test_subset_is_stable_independent_of_input_order():
    rows = [{'source_id': str(i)} for i in range(50)]
    assert choose(rows, 'gsm8k', 20) == choose(rows[::-1], 'gsm8k', 20)
    assert len({r['source_id'] for r in choose(rows, 'gsm8k', 20)}) == 20
    assert rank('gsm8k', '1', 'sample') != rank('gsm8k', '1', 'order')


def test_only_online_and_baselines():
    matrix = jobs()
    assert len(matrix) == len({j['key'] for j in matrix}) == 90
    assert set(ARMS) == {'ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-online'}
    assert all(j['batch_size'] == 1 for j in matrix)


def test_gamma_and_rate_are_pooled_not_mean_of_ratios():
    rows = [dict(tokens=[0, 1], stats=dict(decode_s=1, prefill_s=2, tokens=1, calls=4), stop_reason='eos'),
            dict(tokens=[0, 1, 2, 3], stats=dict(decode_s=3, prefill_s=2, tokens=3, calls=4), stop_reason='length')]
    s = summarize(rows, 4)
    assert s['gamma'] == 2 and s['decode_token_s'] == 1
    assert s['cap_stopped'] == 1 and s['decode_tokens'] == 4 and s['output_tokens'] == 6


def test_latency_queue_shares_table2_locks_and_refills_after_release(tmp_path, monkeypatch):
    """Exercise the real queue loop with two GPUs and a competing Table 2 lock."""
    import copy
    import fcntl
    import uuid
    import loop_latency_suite as latency
    import loop_table2_suite as table2
    import loop_batch_sweep

    lanes=[dict(gpu=i, uuid=f'queue-test-{uuid.uuid4()}', cpus='0-3') for i in range(2)]
    matrix=copy.deepcopy(latency.jobs()[:2])
    monkeypatch.setattr(latency, 'jobs', lambda: copy.deepcopy(matrix))
    monkeypatch.setattr(loop_batch_sweep, 'check_resources', lambda lane: {})
    (tmp_path/'logs').mkdir()
    key=tmp_path/'test-key'
    key.write_text('test-only')
    protocol=dict(source=str(tmp_path/'source'),assets=str(tmp_path/'assets'),
                  wandb_key_file=str(key),wandb_entity='test',wandb_project='test',
                  name='test',lanes=lanes,python=sys.executable)
    latency.save(tmp_path/'protocol.json',protocol)
    for job in matrix:
        latency.save(tmp_path/'jobs'/f"{job['key']}.json",job)
    tracking=SimpleNamespace(url='test',log=lambda *a,**k: None,summary={},
                             log_artifact=lambda *a: None,finish=lambda: None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(
        init=lambda **k: tracking, Settings=lambda **k: None,
        Artifact=lambda *a,**k: SimpleNamespace(add_file=lambda *a: None)))
    monkeypatch.setenv('WANDB_API_KEY','test-only')
    states={lane['uuid']:dict(busy=False, **{'memory.used':0,'utilization.gpu':0}) for lane in lanes}
    monkeypatch.setattr(latency,'gpu_state',lambda: states)
    holder=[table2.try_gpu_lock(lanes[0])]
    launches=[]

    def launch(command, **kwargs):
        gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']
        lane=next(lane for lane in lanes if lane['uuid']==gpu)
        # Even between queue selection and model startup, Table 2 cannot enter.
        assert table2.try_gpu_lock(lane) is None
        job_key=command[command.index('--job')+1]
        smoke='--smoke' in command
        latency.save(tmp_path/('validation' if smoke else 'runs')/job_key/'complete.json',{})
        launches.append((job_key,gpu,smoke))
        return SimpleNamespace(pid=10000+len(launches),poll=lambda: 0)

    monkeypatch.setattr(latency.subprocess,'Popen',launch)
    cycles=[0]
    def tick(seconds):
        cycles[0]+=1
        assert cycles[0]<10, 'queue failed to drain'
        if holder[0] is not None:
            # Busy GPU 0 did not consume or fail the second job; GPU 1 kept working.
            assert len(launches)==1 and launches[0][1]==lanes[1]['uuid']
            pending=latency.read(tmp_path/'jobs'/f"{matrix[1]['key']}.json")
            assert pending['outcome']=='pending' and pending['failures']==0
            holder[0].close()
            holder[0]=None
    monkeypatch.setattr(latency.time,'sleep',tick)
    try:
        latency.run(tmp_path)
        for job in matrix:
            result=latency.read(tmp_path/'jobs'/f"{job['key']}.json")
            assert result['outcome']=='complete' and result['failures']==0
            assert sum(k==job['key'] and not smoke for k,gpu,smoke in launches)==1
        assert any(gpu==lanes[0]['uuid'] for _,gpu,_ in launches)
        for lane in lanes:
            with table2.try_gpu_lock(lane):
                pass
    finally:
        if holder[0] is not None:
            holder[0].close()
        for lane in lanes:
            table2.gpu_lock_path(lane).unlink(missing_ok=True)
