"""Materialize preregistered external prompts; never consume assistant answers."""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', s).lower().split())


def features(s):
    words = norm(s).split()
    return set(zip(*(words[i:] for i in range(5)))) if len(words) >= 20 else set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--benchmarks', type=Path, required=True)
    p.add_argument('--hf-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    policy_path = Path('results/loopfill-icml-20261103/calibration-policy.json')
    policy = json.loads(policy_path.read_text())
    refs = []
    spec = Path('results/loop-spec-reuse-20261030/raw/data/spec_bench_question.jsonl')
    for row in map(json.loads, spec.read_text().splitlines()):
        refs.extend((f'spec:{row["question_id"]}:{i}', t) for i, t in enumerate(row['turns']))
    for row in map(json.loads, (a.benchmarks/'test.jsonl').read_text().splitlines()):
        refs.append((f'math:{row["unique_id"]}', row['problem']))
    for row in pq.read_table(a.benchmarks/'test-00000-of-00001.parquet').to_pylist():
        for key in ['before', 'instruction_descriptive']:
            refs.append((f'edit:{row["id"]}:{key}', row[key]))
    seen = {norm(t): rid for rid, t in refs}
    fuzzy = [(rid, features(t)) for rid, t in refs]
    tok = {}
    for model, repo in [('ouro-2.6b','ByteDance--Ouro-2.6B'), ('huginn-3.5b','tomg-group-umd--huginn-0125')]:
        path = a.hf_root/('models--'+repo)/'snapshots'/policy['reference']['model_revisions'][model]
        tok[model] = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    rows = []
    source_files = []
    for name in policy['source']['files']:
        path = a.source/Path(name).name
        source_files.append({'name':name,'sha256':digest(path)})
        # The source prompt column is the first user turn; assistant messages are never read.
        rows.extend(pq.read_table(path, columns=['prompt_id','prompt']).to_pylist())
    rows.sort(key=lambda r:(hashlib.sha256(('20261103:'+r['prompt_id']).encode()).hexdigest(),r['prompt_id']))
    chosen, exclusions, seen_ids = [], [], set()
    for r in rows:
        rid, text = r['prompt_id'], r['prompt']
        n = norm(text)
        if rid in seen_ids:
            exclusions.append({'id':rid,'reason':'duplicate_id'}); continue
        seen_ids.add(rid)
        match = seen.get(n)
        grams = features(text)
        if match is None and grams:
            for other, g in fuzzy:
                if not g: continue
                shared = len(grams & g)
                if shared / len(grams | g) >= .8 or shared / min(len(grams),len(g)) >= .9:
                    match = other; break
        if match is not None:
            exclusions.append({'id':rid,'reason':'exact_or_near_duplicate','match':match}); continue
        lengths = {m:len(t.apply_chat_template([{'role':'user','content':text}],add_generation_prompt=True)) for m,t in tok.items()}
        if not n or any(lengths[m]+1024+(68 if m.startswith('ouro') else 96)>4096 for m in tok):
            exclusions.append({'id':rid,'reason':'full_budget_context','lengths':lengths}); continue
        seen[n] = rid; fuzzy.append((rid,grams))
        chosen.append({'question_id':len(chosen),'prompt_id':rid,'category':'external_ultrachat','turns':[text], 'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_tokens':lengths})
        if len(chosen)==320: break
    if len(chosen)!=320: raise RuntimeError('insufficient eligible disjoint prompts')
    calibration, dev = chosen[:128], chosen[128:]
    by_length = sorted(dev,key=lambda r:(len(r['turns'][0]),r['prompt_id']))
    pilot = [by_length[i*191//23] for i in range(24)]
    outputs = {f'calib{n}':calibration[:n] for n in [16,32,64,128]}
    outputs.update(load192=dev,validation32=dev[:32],pilot24=pilot,check12=[pilot[i*23//11] for i in range(12)])
    a.out.mkdir(parents=True)
    files = {}
    for name, items in outputs.items():
        path=a.out/(name+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
        files[name]={'count':len(items),'sha256':digest(path),'prompt_ids':[r['prompt_id'] for r in items]}
    manifest={'policy_sha256':digest(policy_path),'source_files':source_files,'selection':'SHA256(20261103:prompt_id), then prompt_id; greedy retain eligible unique prompts', 'dedup':'NFKC lowercase whitespace exact; >=20 words: token 5-gram Jaccard >=0.8 or containment >=0.9; compare evaluation prompts and earlier retained external prompts', 'limitations':'Lexical near-duplicate screening does not rule out semantic paraphrases or pretraining contamination.', 'context':'both pinned tokenizers; prompt+1024+T+64 <=4096', 'exclusions':exclusions,'files':files,'benchmark_hashes':{str(p):digest(p) for p in [spec,a.benchmarks/'test.jsonl',a.benchmarks/'test-00000-of-00001.parquet']},'uses_model_outputs':False}
    (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['count'] for k,v in files.items()}))


if __name__=='__main__':main()
