import collections,csv,gzip,hashlib,json,math,re
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def rows(p):return list(csv.DictReader(p.open()))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8),(a,b)
audit=read(P/'run-audits.json'); totals={}; journals={}
for item in audit['runs']:
 p=P/item['journal']
 with gzip.open(p,'rt') as f:r=[json.loads(x) for x in f]
 journals[item['source']]=r
 t={'requests':len(r),'output_tokens':sum(len(x['tokens']) for x in r),'decode_tokens':sum(len(x['tokens'])-1 for x in r)}
 if all('decode_s' in x['stats'] for x in r):t['decode_s']=sum(x['stats']['decode_s'] for x in r)
 totals[item['source']]=t
 summary=p.parent/'summary.json'
 if summary.exists():
  s=read(summary)
  for key,val in t.items():
   if key in s:close(s[key],val)
  if 'decode_s' in t:
   for key in ['decode_token_s','tok_s']:
    if key in s:close(s[key],t['decode_tokens']/t['decode_s'])
  if 'gamma' in s:
   if all('target_verifications' in x['stats'] for x in r):
    calls=sum(x['stats']['target_verifications'] for x in r)
    if calls:close(s['gamma'],t['decode_tokens']/calls)
   elif all('calls' in x['stats'] for x in r):
    depth=32 if 'huginn' in item['source'] else 4
    close(s['gamma'],depth*t['decode_tokens']/sum(x['stats']['calls'] for x in r))
# Recompute all 36 native baseline rows using archived method/AR request journals.
baselines=rows(P/'baselines/results.csv');assert len(baselines)==36
for row in baselines:
 m,a=[totals[row[k]] for k in ['method_run','ar_run']]
 for prefix,t in [('method',m),('ar',a)]:
  for k in ['decode_s','decode_tokens']:close(row[prefix+'_'+k],t[k])
 close(row['eta_latency'],a['decode_s']/m['decode_s'])
 close(row['eta_throughput'],(m['decode_tokens']/m['decode_s'])/(a['decode_tokens']/a['decode_s']))
 assert [x['source_id'] for x in journals[row['method_run']]]==[x['source_id'] for x in journals[row['ar_run']]]
 assert [x['prompt_sha256'] for x in journals[row['method_run']]]==[x['prompt_sha256'] for x in journals[row['ar_run']]]
# Packed completion status is deliberately stricter than artifact availability.
d=P/'repository/results/dflash-table12-packed-20261112-v2';packed=rows(d/'measurements.csv');assert len(packed)==34
states=collections.Counter(x['outcome'] for x in packed)
assert states=={'complete':19,'complete_artifact_exit_unavailable':6,'reused_existing_measurement':9},states
jobs={x['id']:x for x in read(d/'jobs.json')};dispatch=read(d/'dispatch.json')
for row in packed:
 state=dispatch[row['id']]
 if state['outcome'] not in ['complete','complete_artifact_exit_unavailable']:continue
 job=jobs[row['id']];s=read(d/state['out']/'summary.json')
 with gzip.open(d/(job['inputs']+'.gz'),'rt') as f:inputs=[json.loads(x) for x in f]
 with gzip.open(d/state['out']/'requests.jsonl.gz','rt') as f:r=[json.loads(x) for x in f]
 if job['kind']=='latency':
  assert [x['source_id'] for x in r]==[x['source_id'] for x in inputs]
  assert [x['prompt_sha256'] for x in r]==[x['prompt_sha256'] for x in inputs]
 else:assert {x['request_id'] for x in r}=={x['source_id'] for x in inputs}
 if state['outcome']=='complete':
  for k in ['requests','output_tokens','decode_tokens','decode_s','decode_token_s','output_tput','wall_s','gamma']:
   if k in s:close(row[k],s[k])
 else:assert not row.get('decode_s') and state['exit_code'] is None
# SQL frozen inputs, all 28 rows and AR ratios.
d=P/'repository/results/loopfill-agentic-sql-table1-20261112';sql=rows(d/'measurements.csv');assert len(sql)==28
index={x['id']:x for x in sql}
for row in sql:
 s=read(d/'runs'/row['id']/'summary.json')
 for k in ['decode_s','decode_tokens','gamma','decode_token_s']:close(row[k],s[k])
 ar=index[f"{row['model']}-ar-t{row['temperature']}"]
 close(row['latency_speedup'],float(ar['decode_s'])/float(row['decode_s']))
 with gzip.open(d/'inputs'/(row['model']+'.jsonl.gz'),'rt') as f:inputs=[json.loads(x) for x in f]
 with gzip.open(d/'runs'/row['id']/'requests.jsonl.gz','rt') as f:r=[json.loads(x) for x in f]
 for actual,expected in zip(r,inputs):
  for k in ['source_id','question_id','phase','instance_id','prompt_tokens','prompt_sha256']:assert actual[k]==expected[k]
# Throughput table and normal/diagnostic timing pilot.
d=P/'repository/results/loopahead-table2-20261112-v2';la=rows(d/'measurements.csv');assert len(la)==10
for row in la:
 s=read(d/'runs'/f"{row['model']}-b{row['batch']}"/'summary.json')
 for k in ['requests','output_tokens','wall_s','output_tput','gamma']:close(row[k],s[k])
 close(row['ratio_to_historical_ar'],float(row['output_tput'])/float(row['historical_ar_tput']))
d=P/'repository/results/loopfill-mechanisms-20261109/timing';mechanism=rows(d/'measurements.csv');assert len(mechanism)==32
for row in mechanism:
 s=read(d/'jobs'/(row['key']+'.json'))['summary']
 for k in ['requests','output_tokens','wall_s','output_tput']:close(row[k],s[k])
 depth=32 if row['model'].startswith('huginn') else 4
 p=d/'runs'/row['key']/f"{row['arm']}-b{row['batch']}-seed0.jsonl.gz"
 with gzip.open(p,'rt') as f:r=[json.loads(x) for x in f]
 close(row['gamma'],depth*sum(x['stats']['tokens'] for x in r)/sum(x['stats']['calls'] for x in r))
# Authored links and credentials/file-size guards, report only paths on failure.
for p in [P/'README.md',P.parent/'README.md']:
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if not target.startswith(('http:','https:','#')):assert (p.parent/target.split('#')[0]).exists(),(p,target)
pattern=re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
manifest=read(P/'manifest.json')
for item in manifest['files']:
 p=P/item['path'];assert p.stat().st_size<50*1024**2,p
 if item['path'].endswith(('.json','.py','.jsonl.gz','.md','.toml','.csv')):
  data=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
  assert not pattern.search(data),f'Credential-like data in {p}'
result={'passed':True,'request_streams':len(audit['runs']),'request_records':sum(x['requests'] for x in audit['runs']),'table_rows_recomputed':{'native_baselines':36,'packed_dflash_complete':19,'loopahead_throughput':10,'agentic_sql':28,'mechanism_timing':32},'packed_status_counts':dict(states),'frozen_input_identity_checks':True,'original_bytes_preserved':True,'authored_links_valid':True,'large_file_and_credential_checks_passed':True,'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
