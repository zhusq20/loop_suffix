"""Benchmark provenance gates and cross-process GPU lock ownership."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
spec = importlib.util.spec_from_file_location('loop_dflash_benchmark_worker', SCRIPTS/'loop_dflash_benchmark_worker.py')
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)
spec = importlib.util.spec_from_file_location('dflash_suite', SCRIPTS/'loop_dflash_benchmark_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_uuid_lock_excludes_other_queue_and_survives_parent_release():
    identity = f'dflash-test-{os.getpid()}'
    lock = suite.try_gpu_lock(identity)
    assert lock is not None
    assert suite.try_gpu_lock(identity) is None
    child = subprocess.Popen([sys.executable, '-c',
        'import os,sys; os.fstat(int(sys.argv[1])); print("ready",flush=True); sys.stdin.readline()',
        str(lock.fileno())], pass_fds=(lock.fileno(),), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        lock.close()
        assert suite.try_gpu_lock(identity) is None
        child.communicate('\n', timeout=10)
        again = suite.try_gpu_lock(identity)
        assert again is not None
        again.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        Path(f'/tmp/loopfill-{identity}.lock').unlink(missing_ok=True)


@pytest.fixture
def finished(tmp_path):
    job = dict(kind='throughput', inputs='inputs.jsonl', count=2, max_new_tokens=4, batch=2)
    inputs = [dict(source_id=str(i), prompt_tokens=[1,2,3]) for i in range(2)]
    (tmp_path/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    suite.save(tmp_path/'protocol.json', dict(source_sha256={}))
    out = tmp_path/'run'
    out.mkdir()
    suite.save(out/'metadata.json', dict(job=job, source_sha256={}))
    records = [dict(request_id=str(i), tokens=[5,6,7], stats=dict(decode_tokens=2,
        target_verifications=1, emitted_per_verification=[2])) for i in range(2)]
    (out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary = dict(requests=2, output_tokens=6, input_tokens=6, prefill_chunk_tokens=6,
        wall_s=3., output_tput=2., gamma=2., concurrency=2, core_padding_rows=0,
        cuda_graph=False, pad_decode_rows=False)
    suite.save(out/'summary.json', summary)
    def sign():
        suite.save(out/'complete.json', dict(requests=2, requests_sha256=suite.digest(out/'requests.jsonl'),
                                            summary_sha256=suite.digest(out/'summary.json')))
    sign()
    return tmp_path, job, out, records, summary, sign


def test_audit_recomputes_throughput_and_validates_request_coverage(finished):
    root, job, out, records, summary, sign = finished
    assert suite.audit(root, job, out)['output_tput'] == 2.
    summary['output_tput'] = 3.
    suite.save(out/'summary.json', summary)
    sign()
    with pytest.raises(AssertionError):
        suite.audit(root, job, out)


@pytest.mark.parametrize('change', ['duplicate', 'budget', 'wrong_input', 'truncated'])
def test_audit_rejects_invalid_journals(finished, change):
    root, job, out, records, summary, sign = finished
    if change == 'duplicate':
        records[1]['request_id'] = records[0]['request_id']
    elif change == 'budget':
        records[0]['tokens'] *= 2
    elif change == 'wrong_input':
        records[0]['request_id'] = 'not-selected'
    text = ''.join(json.dumps(r)+'\n' for r in records)
    if change == 'truncated':
        text = text[:-10]
    (out/'requests.jsonl').write_text(text)
    sign()
    with pytest.raises((AssertionError, json.JSONDecodeError)):
        suite.audit(root, job, out)


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
def test_worker_validation_writes_auditable_real_decode_artifacts(tmp_path, monkeypatch, family):
    import torch
    from baselines.dflash.target import LoopTarget
    from baselines.dflash.projected_draft import inference_draft_class
    from baselines.vendor.dflash_inference import DFlashDraftModel
    from test_baselines.test_portable_training import TinyTarget, tiny_config

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        native = TinyTarget(family)
        cfg = tiny_config(native, family == 'huginn')
        cfg._attn_implementation = 'sdpa'
        draft = (inference_draft_class() if family == 'huginn' else DFlashDraftModel)(cfg).eval()
        checkpoint = tmp_path/'checkpoint'
        from safetensors.torch import save_file
        cfg.save_pretrained(checkpoint)
        save_file(draft.state_dict(), checkpoint/'model.safetensors')
        model = dict(path=str(tmp_path/'unused'), family=family, checkpoint=str(checkpoint), stop_ids=[],
            checkpoint_sha256={name:worker.digest(checkpoint/name) for name in ['config.json','model.safetensors']})
        monkeypatch.setattr(LoopTarget, 'load', lambda *args, **kwargs: native)
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'cpu-test')
        inputs = tmp_path/'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(dict(source_id=str(i), prompt_tokens=[1,2,3,4]))+'\n'
                                 for i in range(6)))
        job = dict(id='validate-tiny', kind='validate', model='tiny', dataset='synthetic-prefixes',
            temperature=0, batch=2, max_seq=128, max_new_tokens=33, count=6,
            inputs='inputs.jsonl', inputs_sha256=worker.digest(inputs))
        worker.save(tmp_path/'protocol.json', dict(models={'tiny':model}, source_sha256={}))
        out = tmp_path/'run'
        out.mkdir()
        worker.run(tmp_path, job, out)
        assert suite.audit(tmp_path, job, out)['status'] == 'passed_full_weight_fp32'
    finally:
        torch.set_num_threads(previous)
