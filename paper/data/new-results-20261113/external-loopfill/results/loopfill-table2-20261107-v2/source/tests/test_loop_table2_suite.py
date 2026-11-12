import importlib.util
import json
from pathlib import Path

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
