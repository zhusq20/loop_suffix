from pathlib import Path
from collections import defaultdict
import hashlib,json,time
b=Path(__file__).resolve().parents[2];root=b/'data/teachers/dflash-train/ouro-2.6b'
raw=(b/'data/manifests/dflash-train.jsonl').read_bytes();manifest={r['source_id']:r for r in map(json.loads,raw.splitlines())};mh=hashlib.sha256(raw).hexdigest()
assert all((root/f'shard{s}/complete.json').exists() for s in range(8)), 'Full teacher generation is required'
lengths=defaultdict(list)
for key,row in manifest.items():lengths[row['prompt_lengths']['ouro-2.6b']].append(key)
protocol=json.loads((b.parents[1]/'results/loopfill-latency512-20261107/protocol.json').read_text());cfg=json.loads((Path(protocol['models']['ouro-2.6b']['path'])/'config.json').read_text());bytes_per_position=2*192*cfg['num_key_value_heads']*128*2
batches=[];batch=[]
for length,ids in sorted(lengths.items()):
 for key in ids:
  limit=min(32,max(1,(70*1024**3)//(bytes_per_position*(length+512))))
  if len(batch)>=limit:batches.append(batch);batch=[]
  batch.append(key)
if batch:batches.append(batch)
ph=hashlib.sha256(json.dumps(batches).encode()).hexdigest()
report={'stage':'partial_teacher_audit_not_benchmark','observed_unix':time.time(),'model':'ouro-2.6b','manifest_sha256':mh,'completed_shards':[],'running_shards':[]}
allids=set();counts=0
for s in range(8):
 d=root/f'shard{s}';md=json.loads((d/'metadata.json').read_text());ident=md['identity'];assert ident['shard']==s and ident['shards']==8 and ident['manifest_sha256']==mh and ident['batch_plan_sha256']==ph
 expected=[[i,ids] for i,ids in enumerate(batches) if i%8==s];assigned=[k for _,ids in expected for k in ids];assert len(assigned)==len(set(assigned))==md['planned_requests']
 bp=d/'batch-plans.json'
 if bp.exists():
  plans=json.loads(bp.read_text());assert plans['assignment']==expected;executed=[k for _,ids in plans['execution'] for k in ids];assert len(executed)==len(set(executed))==len(assigned) and set(executed)==set(assigned)
 assert not allids.intersection(assigned);allids.update(assigned)
 if (d/'complete.json').exists():
  done=json.loads((d/'complete.json').read_text());data=(d/'teachers.jsonl').read_bytes();h=hashlib.sha256(data).hexdigest();assert done['teachers_sha256']==h
  rows=[json.loads(line) for line in data.splitlines()];ids=[r['source_id'] for r in rows];assert len(ids)==len(set(ids))==md['planned_requests']==done['requests'];assert set(ids)==set(assigned)
  for r in rows:
   m=manifest[r['source_id']];assert r['prompt_sha256']==hashlib.sha256(m['prompt'].encode()).hexdigest() and r['question_sha256']==m['question_sha256'];assert len(r['prompt_tokens'])==m['prompt_lengths']['ouro-2.6b'];assert 1<=len(r['output_tokens'])<=512;assert all(isinstance(t,int) and 0<=t<49152 for t in r['prompt_tokens']+r['output_tokens'])
  assert sum(len(r['output_tokens']) for r in rows)==done['output_tokens']
  counts+=len(rows);report['completed_shards'].append({'shard':s,**done,'exact_assigned_ids':True,'unique_ids':True,'manifest_fields_and_token_bounds_valid':True})
 else:
  graph=[json.loads(l) for l in (d/'graph-batches.jsonl').open()];assert graph and all(r['summary']['cuda_graph'] and r['summary']['cuda_graph_replays']>0 for r in graph)
  report['running_shards'].append({'shard':s,'planned_requests':len(assigned),'exact_assignment_execution_id_set':True,'observed_graph_batches':len(graph),'observed_graph_replays':sum(r['summary']['cuda_graph_replays'] for r in graph)})
assert allids==set(manifest) and len(allids)==50000
report.update(completed_requests=counts,total_frozen_requests=len(allids),all_eight_shard_assignments_disjoint_and_exhaustive=True,global_assignment_sha256=ph)
p=b/'preflight/dflash-ouro-2.6b-full-teachers-audit.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
# Full-generation audit also exercises the exact production training input gate.
from collections import Counter
from types import SimpleNamespace
import sys
sys.path.insert(0,str(b))
from train_dflash import read_teachers
assert counts==50000 and len(report['completed_shards'])==8 and not report['running_shards']
graph_checks=[];costs=[]
for shard in range(8):
 d=root/f'shard{shard}';meta=json.loads((d/'metadata.json').read_text());done=json.loads((d/'complete.json').read_text())
 reservations=[json.loads(l) for l in (d/'reservation-cost.jsonl').open()]
 assert len(reservations)==1 and reservations[0]['status']=='complete'
 assert reservations[0]['gpu_uuid']==meta['gpu_uuid']
 costs.extend(reservations)
 if (d/'graph-batches.jsonl').exists():
  graphs=[json.loads(l) for l in (d/'graph-batches.jsonl').open()];plans=json.loads((d/'batch-plans.json').read_text())
  assert len(graphs)==len(plans['execution'])==meta['planned_batches']
  for graph,(_,ids) in zip(graphs,plans['execution']):
   summary=graph['summary'];assert graph['source_ids']==ids and summary['requests']==len(ids)
   assert summary['cuda_graph'] and summary['cuda_graph_replays']>0
   assert summary['core_rows']==4*(summary['output_tokens']-summary['requests'])
  assert sum(g['summary']['output_tokens'] for g in graphs)==done['output_tokens']
  graph_checks.append(dict(shard=shard,batches=len(graphs),replays=sum(g['summary']['cuda_graph_replays'] for g in graphs),full_depth_all_batches=True))
rows,sources,excluded=read_teachers(SimpleNamespace(teachers=root,model='ouro-2.6b',smoke=False))
assert len(rows)+len(excluded)==50000 and len(sources)==8
report.update(stage='full_teacher_audit_not_benchmark',status='complete',graph_checks=graph_checks,
    training_input_gate_passed=True,eligible_training_examples=len(rows),excluded_under_32_tokens=len(excluded),
    eligible_source_counts=dict(Counter(r['pool'] for r in rows)),
    total_output_tokens=sum(r['output_tokens'] for r in report['completed_shards']),
    actual_generation_gpu_hours=sum(r['reserved_gpu_seconds'] for r in costs)/3600,
    training_reader_sha256=hashlib.sha256((b/'train_dflash.py').read_bytes()).hexdigest(),
    audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),finished_unix=time.time())
p.write_text(json.dumps(report,indent=2)+'\n')
Path(__file__).with_name('teacher-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['completed_shards','running_shards','graph_checks']},indent=2))
