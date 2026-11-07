"""Finalize a fully measured cell without regenerating or changing measurements."""
import os, sys, json, time, math, shutil
from pathlib import Path
suite = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(suite/'source/scripts'))
from loop_latency_worker import read, save, digest, summarize
from loop_latency_suite import aggregate
key = 'ouro-2.6b-specbench-budget-online'
target = suite/'runs'/key
recovery = Path(__file__).parent
protocol = read(suite/'protocol.json')
jobpath = suite/'jobs'/f'{key}.json'
job = read(jobpath)
for rel, sha in protocol['source_sha256'].items():
    assert digest(Path(protocol['source'])/rel) == sha, rel
inp = suite/'inputs/specbench.jsonl'
assert digest(inp) == protocol['inputs']['specbench']['sha256']
inputs = [json.loads(x) for x in inp.read_text().splitlines()]
raw = (target/'requests.jsonl').read_bytes()
assert raw.endswith(b'\n')
rows = [json.loads(x) for x in raw.splitlines()]
assert len(rows) == len(inputs) == 480
for i, (r, p) in enumerate(zip(rows, inputs)):
    assert r['source_id'] == p['source_id'] and r['question_id'] == i
    assert r['prompt_sha256'] == p['prompt_sha256']
    assert r['stats']['tokens'] == len(r['tokens'])-1
summary = read(target/'summary.json')
for k,v in summarize(rows,4).items():
    assert math.isclose(v,summary[k],rel_tol=1e-12,abs_tol=1e-12), k
assert summary['total_requests'] == 480 and not summary['cuda_graph']
hashes = {n:digest(target/n) for n in ['metadata.json','requests.jsonl','summary.json']}
if not (recovery/'original-job.json').exists():
    shutil.copy2(jobpath,recovery/'original-job.json')
    shutil.copy2(suite/'status.json',recovery/'original-status.json')
save(recovery/'validation.json',dict(validated_unix=time.time(),requests=480,sha256=hashes,source_hashes_verified=len(protocol['source_sha256']),input_sha256=digest(inp)))
print('Validated all 480 records, frozen source/input hashes and recomputed summary.',flush=True)
os.environ['WANDB_API_KEY'] = Path(protocol['wandb_key_file']).read_text().strip()
import wandb
run = wandb.init(entity=protocol['wandb_entity'],project=protocol['wandb_project'],id=job['wandb_id'],resume='must',dir=str(target),settings=wandb.Settings(init_timeout=90))
run.summary.update(summary)
run.summary['upload_recovered'] = True
for n in hashes:
    run.save(str(target/n),base_path=str(target),policy='now')
a = wandb.Artifact(key,type='latency-results',metadata={'complete':True})
for n in hashes: a.add_file(str(target/n))
logged = run.log_artifact(a)
logged.wait(timeout=180)
artifact_name = logged.qualified_name
run.finish()
api = wandb.Api(timeout=60)
remote = api.run(f"{protocol['wandb_entity']}/{protocol['wandb_project']}/{job['wandb_id']}")
assert remote.state == 'finished', remote.state
assert remote.summary['completed_requests'] == 480
artifact = api.artifact(artifact_name)
download = Path(artifact.download(root=str(recovery/'verified-artifact')))
for n,sha in hashes.items(): assert digest(download/n) == sha,n
save(recovery/'wandb-verification.json',dict(verified_unix=time.time(),run_url=remote.url,cloud_state=remote.state,artifact=artifact_name,sha256=hashes,download_matches=True))
save(target/'complete.json',dict(requests=480,finished_unix=time.time(),requests_sha256=hashes['requests.jsonl'],recovery='upload-finalization'))
job.update(outcome='complete',recovered_unix=time.time(),recovery='recovery/upload-finalization/wandb-verification.json')
# Preserve original worker exit_code/failures as historical evidence.
save(jobpath,job)
status = aggregate(suite)
queue_id = read(suite/'launcher.json')['wandb_url'].rsplit('/',1)[-1]
queue = wandb.init(entity=protocol['wandb_entity'],project=protocol['wandb_project'],id=queue_id,resume='must',dir=str(suite))
queue.summary.update(status)
queue.summary.update({'cells/complete':90,'cells/failed':0,'cells/pending':0,'cells/running':0,'gpus/active':0})
for n in ['status.json','measurements.csv','measurements.json']: queue.save(str(suite/n),base_path=str(suite),policy='now')
queue.finish()
print(json.dumps(status),flush=True)
