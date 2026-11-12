"""An amended queue can fill free cards without changing frozen inference inputs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import loop_agentic_suite as suite_module
import loop_queue_recovery as recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def queue_fixture(root):
    jobs = [dict(id=f'ouro-1.4b-{arm}-t1', model='ouro-1.4b', temperature=1,
                 arm=arm, stage='main', q=4, n=16, cache_mode='stream')
            for arm in ['ar', 'wavefront']]
    write(root/'protocol.json', dict(allowed_gpus=[7], gpu_bindings={'ouro-1.4b/t1':7},
                                   python=sys.executable, require_exit_receipts=True))
    write(root/'jobs.json', jobs)
    for job in jobs:
        write(root/'validation'/job['id']/'summary.json', {'status':'complete'})
    return dict(placement='first_idle_per_cell', allowed_gpus=list(range(8)),
                protocol_sha256=suite_module.digest(root/'protocol.json'),
                jobs_sha256=suite_module.digest(root/'jobs.json'),
                scheduler_root=str(root), scheduler_sha256={})


@pytest.mark.parametrize('amended,expected', [(True, [0, 2]), (False, [])])
def test_two_methods_use_two_free_cards_while_legacy_binding_is_busy(
        tmp_path, monkeypatch, amended, expected):
    scheduling = queue_fixture(tmp_path)
    launched = []

    def smi(argv, **kwargs):
        if '--query-gpu=index,uuid,compute_mode' in argv:
            return '0, TEST-SQL-GPU0, Exclusive_Process\n2, TEST-SQL-GPU2, Exclusive_Process\n7, TEST-SQL-GPU7, Exclusive_Process\n'
        return 'TEST-SQL-GPU7\n'

    def spawn(argv, **kwargs):
        launched.append(kwargs['metadata']['gpu'])
        return SimpleNamespace(pid=9000+len(launched), launch={'pid':9000+len(launched)})

    def finish_iteration(seconds):
        raise RuntimeError('one scheduler iteration')

    monkeypatch.setattr(suite_module.subprocess, 'check_output', smi)
    monkeypatch.setattr(recovery, 'spawn_supervised', spawn)
    monkeypatch.setattr(suite_module, 'summarize', lambda path: [])
    monkeypatch.setattr(suite_module.time, 'sleep', finish_iteration)
    with pytest.raises(RuntimeError, match='one scheduler iteration'):
        suite_module.run_queue(tmp_path, scheduling=scheduling if amended else None)
    assert launched == expected
    status = json.loads((tmp_path/'status.json').read_text())
    assert len(status['running']) == len(expected)
    if amended:
        assert status['bindings'] == {}
        assert set(status['cell_gpus'].values()) == {0, 2}
        assert status['allowed_gpus'] == list(range(8))


def test_amendment_rejects_changed_frozen_protocol(tmp_path):
    scheduling = queue_fixture(tmp_path)
    write(tmp_path/'protocol.json', {'changed':True})
    with pytest.raises(AssertionError):
        suite_module.validate_scheduling(tmp_path, scheduling)


def test_cross_device_export_records_actual_devices_and_checks_receipts(tmp_path, monkeypatch):
    import csv
    import loop_agentic_finalize as finalizer

    scheduling = queue_fixture(tmp_path)
    jobs = json.loads((tmp_path/'jobs.json').read_text())
    model = 'ouro-1.4b'
    protocol = json.loads((tmp_path/'protocol.json').read_text())
    protocol.update(models={model:dict(input_sha256='frozen-input', checkpoint_sha256={})},
                    temperatures=[1], main_arms=['ar','wavefront'],
                    input_files_sha256={}, source_sha256={})
    write(tmp_path/'protocol.json', protocol)
    scheduling['protocol_sha256'] = suite_module.digest(tmp_path/'protocol.json')
    write(tmp_path/'status.json', {'status':'complete'})
    write(tmp_path/'inputs/manifest.json', dict(dataset='SQL', semantics='fixed replay',
        input_sha256='frozen-input', phase_counts={'early':100}))
    inputs = [dict(source_id=str(i), question_id=str(i), phase='early', instance_id=str(i),
                   prompt_tokens=3, prompt_sha256=f'prompt-{i}') for i in range(100)]
    (tmp_path/f'inputs/{model}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    dispatch = dict(cell_gpus={}, cell_gpu_uuids={})
    measurements = []
    for job, gpu in zip(jobs, [0, 2]):
        directory = tmp_path/'runs'/job['id']
        write(directory/'metadata.json', dict(smoke=False, diagnostic=False, max_new_tokens=512,
            batch=1, source_sha256={}, input_sha256='frozen-input', checkpoint_sha256={},
            gpu_uuid=f'TEST-{gpu}'))
        records = [dict(r, tokens=[1,2], stop_reason='eos',
                        stats=dict(tokens=1, decode_s=1., prefill_s=.1, calls=4)) for r in inputs]
        journal = directory/'requests.jsonl'
        journal.write_text(''.join(json.dumps(r)+'\n' for r in records))
        summary = dict(status='complete', reportable=True,
            requests_sha256=suite_module.digest(journal), decode_tokens=100, decode_s=100.,
            decode_token_s=1., gamma=1., mean_decode_s=1., prefill_s=10., output_tokens=200,
            cap_stopped=0)
        write(directory/'summary.json', summary)
        write(tmp_path/'logs'/(job['id']+'-exit.json'),
              dict(exit_code=0, job_id=job['id'], smoke=False, gpu=gpu))
        dispatch['cell_gpus'][job['id']] = gpu
        dispatch['cell_gpu_uuids'][job['id']] = f'TEST-{gpu}'
        measurements.append(dict(id=job['id'], **summary))
    write(tmp_path/'dispatch.json', dispatch)
    with (tmp_path/'measurements.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    (tmp_path/'table1-agentic.tex').write_text('test table')
    monkeypatch.setattr(finalizer.subprocess, 'run', lambda *args, **kwargs: None)
    finalizer.finalize(tmp_path, tmp_path/'export', scheduling=scheduling)
    audit = json.loads((tmp_path/'checks/final-audit.json').read_text())
    assert audit['passed'] and not audit['same_device_per_model_temperature']
    assert audit['actual_gpu_uuids'][f'{model}/t1'] == ['TEST-0','TEST-2']
    with pytest.raises(AssertionError):
        finalizer.finalize(tmp_path, tmp_path/'legacy-export')
    receipt = tmp_path/'logs'/(jobs[1]['id']+'-exit.json')
    bad = json.loads(receipt.read_text())
    bad['gpu'] = 0
    write(receipt, bad)
    with pytest.raises(AssertionError):
        finalizer.finalize(tmp_path, tmp_path/'bad-export', scheduling=scheduling)
