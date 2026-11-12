"""Freeze mixed-domain prompt pools with complete test-source lexical isolation."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from data_isolation import IsolationIndex, normalize, write_protection

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def python_example(instruction, answer):
    other = r'\b(?:javascript|typescript|java|ruby|php|rust|swift|kotlin|sql|html|css)\b|C\+\+|C#|\b(?:in|using)\s+(?:C|Go|R)\b'
    if re.search(other, instruction, flags=re.I):
        return False
    if re.search(r'\bpython\b', instruction, flags=re.I):
        return True
    candidates = re.findall(r'```(?:python|py)?\s*\n(.*?)```', answer, flags=re.S) or [answer]
    for code in candidates:
        try:
            tree = ast.parse(code)
            if any(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.For, ast.ListComp))
                   for n in ast.walk(tree)):
                return True
        except SyntaxError:
            pass
    return False


def load_pools():
    import pyarrow.parquet as pq
    pools, stats = defaultdict(list), Counter()
    seen_files = set()
    source_rows = [json.loads(line) for line in (BASE / 'data/sources.jsonl').read_text().splitlines()]
    taco_sources = {row['filename']: row for row in source_rows if row.get('repo') == 'BAAI/TACO'}
    if len(taco_sources) != 9:
        raise RuntimeError('Need all9 pinned TACO train shards for index restoration')
    taco = []
    for name, info in sorted(taco_sources.items()):
        taco.extend(pq.read_table(info['path'], columns=['question', 'source']).to_pylist())
    assert len(taco) == 25443

    def add(pool, identity, question, source, **metadata):
        if not isinstance(question, str) or not question.strip():
            stats[pool + ':empty'] += 1
            return
        question = question.strip()
        if len(question) > 12000:
            stats[pool + ':over_12000_chars'] += 1
            return
        pools[pool].append(dict(source_id=identity, pool=pool, question=question,
                                source=source, question_sha256=sha_text(question), **metadata))

    for info in source_rows:
        if info['status'] != 'downloaded':
            continue
        path = Path(info['path'])
        if str(path) in seen_files:
            continue
        seen_files.add(str(path))
        source = {k: info[k] for k in ['repo', 'revision', 'filename', 'sha256'] if k in info}
        if path.suffix == '.jsonl' and info['filename'] == 'gsm8k-train.jsonl':
            for i, raw in enumerate(path.read_text().splitlines()):
                row = json.loads(raw)
                add('gsm8k', f'gsm8k/train/{i}', row['question'], source)
        elif info['filename'] == 'code_alpaca_20k.json':
            for i, row in enumerate(json.loads(path.read_text())):
                instruction, context = row['instruction'].strip(), row['input'].strip()
                question = instruction + ('\n\n' + context if context and context.lower() not in ('<noinput>', '< noinput >') else '')
                add('codealpaca', f'codealpaca/{i}', question, source)
                if python_example(question, row['output']):
                    add('python', f'codealpaca/{i}', question, source)
        elif info.get('repo') == 'EleutherAI/hendrycks_math':
            for i, row in enumerate(pq.read_table(path).to_pylist()):
                add('math', f'math/{info["filename"]}/{i}', row['problem'], source)
        elif info.get('repo') == 'HuggingFaceH4/ultrachat_200k':
            file = pq.ParquetFile(path)
            for batch in file.iter_batches(batch_size=1024, columns=['prompt', 'prompt_id']):
                for row in batch.to_pylist():
                    add('ultrachat', f'ultrachat/{row["prompt_id"]}', row['prompt'], source)
        elif info.get('repo') == 'nvidia/Nemotron-Post-Training-Dataset-v1':
            file = pq.ParquetFile(path)
            domain = info['domain']
            pool = 'reasoning' if domain == 'stem' else 'nemotron_' + domain
            for batch in file.iter_batches(batch_size=512, columns=['uuid', 'messages', 'license', 'metadata']):
                for row in batch.to_pylist():
                    messages = row['messages']
                    user = next((m.get('content', '') for m in messages if m.get('role') == 'user'), '')
                    meta = str(row.get('metadata', ''))
                    if re.search(r'gsm8k|math[-_ ]?500|human[-_ ]?eval|mbpp|big[-_ ]?bench|mt[-_ ]?bench|specbench', meta, re.I):
                        stats[pool + ':benchmark_metadata'] += 1
                        continue
                    original = {}
                    if domain == 'code' and (not user.strip() or user.strip() == '-'):
                        metadata = json.loads(row['metadata'])
                        if metadata.get('dataset') != 'taco' or metadata.get('split') != 'train':
                            stats[pool + ':redacted_other_source_or_nontrain'] += 1
                            continue
                        index = metadata['index']
                        assert 0 <= index < len(taco) and taco[index]['source'] == metadata['source']
                        user = taco[index]['question']
                        original = dict(original_taco_train_index=index,
                                        original_taco_revision=next(iter(taco_sources.values()))['revision'])
                        stats[pool + ':restored_taco_train_source_checked'] += 1
                    add(pool, 'nemotron-v1/' + row['uuid'], user, source,
                        license=row['license'], metadata_sha256=sha_text(meta), **original)
    for name, rows in pools.items():
        rows.sort(key=lambda r: (sha_text('42:' + r['source_id']), r['source_id']))
        stats[name + ':candidate_count'] = len(rows)
    return pools, dict(stats)


def training_prompt(row, method):
    if method == 'dflash':
        return ('Write a Python solution for the following programming problem:\n\n' if 'original_taco_train_index' in row else '') + row['question']
    if row['pool'] == 'gsm8k':
        return 'Q: ' + row['question'] + '\nA:'
    if row['pool'] == 'math':
        return 'Problem:\n' + row['question'] + '\n\nSolution:'
    return '### Instruction:\n' + row['question'] + '\n\n### Response:\n'


def main():
    os.sched_setaffinity(0, set(range(128, 144)))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT / 'results/loopfill-latency512-20261107/source/src'))
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt
    start = time.time()
    out = BASE / 'data/manifests'
    out.mkdir(parents=True, exist_ok=False)
    protected = write_protection(ROOT, BASE / 'data')
    test_index = IsolationIndex(protected, bbh_boilerplate=True)
    heldout_index = IsolationIndex()
    pools, source_stats = load_pools()
    print(json.dumps(source_stats, indent=2), flush=True)
    protocol = json.loads((ROOT / 'results/loopfill-latency512-20261107/protocol.json').read_text())
    tokenizers = {m: AutoTokenizer.from_pretrained(protocol['models'][m]['path'], trust_remote_code=True,
                   local_files_only=True) for m in ['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b']}
    test_cache, length_cache, rejected = {}, {}, Counter()
    heldout_ids, train_seen = set(), defaultdict(set)
    files, rejection_log = {}, (BASE / 'data/rejections.jsonl').open('w')

    def select(pool, count, method, split):
        result = []
        phase = f'{method}/{split}/{pool}'
        local_seen = set()
        for row in pools[pool]:
            identity = row['source_id']
            norm = normalize(row['question'])
            key = sha_text(norm)
            if identity in heldout_ids or key in local_seen or (split == 'train' and key in train_seen[method]):
                continue
            if key not in test_cache:
                test_cache[key] = test_index.match(row['question'])
            reason = test_cache[key] or heldout_index.match(row['question'])
            if reason:
                rejected[phase + ':' + reason['reason']] += 1
                rejection_log.write(json.dumps(dict(phase=phase, source_id=identity, **reason)) + '\n')
                continue
            prompt = training_prompt(row, method)
            length_key = (method, sha_text(prompt))
            if length_key not in length_cache:
                fmt = 'chat' if method == 'dflash' else 'raw'
                length_cache[length_key] = {m: len(tokenize_prompt(MODELS[m], t, prompt, fmt)) for m, t in tokenizers.items()}
            lengths = length_cache[length_key]
            if min(lengths.values()) < 1 or max(lengths.values()) > 2560:
                rejected[phase + ':prompt_over2560_tokens'] += 1
                continue
            record = dict(row, method=method, split=split, prompt=prompt, prompt_lengths=lengths,
                          prompt_format='chat' if method == 'dflash' else 'raw',
                          max_new_tokens=512, normalized_question_sha256=key)
            result.append(record)
            local_seen.add(key)
            if split == 'train':
                train_seen[method].add(key)
            else:
                heldout_ids.add(identity)
                heldout_index.add(dict(id=identity, bucket=f'{method}/{split}', text=row['question']))
            if len(result) == count:
                break
        if len(result) != count:
            raise RuntimeError(f'{phase}: need{count}, only{len(result)} eligible; no silent shrink')
        print(phase, len(result), flush=True)
        return result

    def save(name, rows):
        rows.sort(key=lambda r: (sha_text('order42:' + r['source_id']), r['source_id']))
        path = out / (name + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        files[name] = dict(count=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           pools=dict(Counter(r['pool'] for r in rows)), source_ids=[r['source_id'] for r in rows])

    la = [('gsm8k', 2000, 128), ('python', 2000, 128), ('math', 2000, 128),
          ('reasoning', 2025, 270), ('ultrachat', 1000, 128)]
    # Hold out every method's dev/confirmation before selecting either method's training data.
    for split in ['dev', 'confirmation']:
        rows = [r for pool, _, count in la for r in select(pool, count, 'loopahead', split)]
        save('loopahead-' + split, rows)
    df_dev = [('nemotron_math', 400), ('nemotron_code', 200), ('codealpaca', 100), ('ultrachat', 300)]
    save('dflash-dev', [r for pool, count in df_dev for r in select(pool, count, 'dflash', 'dev')])
    save('loopahead-train', [r for pool, count, _ in la for r in select(pool, count, 'loopahead', 'train')])
    df = [('nemotron_math', 20000), ('nemotron_code', 10000), ('codealpaca', 5000), ('ultrachat', 15000)]
    training = [r for pool, count in df for r in select(pool, count, 'dflash', 'train')]
    save('dflash-train', training)
    pilot = []
    for pool, count in [('nemotron_math', 800), ('nemotron_code', 400), ('codealpaca', 200), ('ultrachat', 600)]:
        pilot.extend([r for r in training if r['pool'] == pool][:count])
    assert len(pilot) == 2000
    save('dflash-pilot', pilot)
    rejection_log.close()
    manifest = dict(started_unix=start, finished_unix=time.time(), seed=42, files=files,
                    sources=source_stats, rejections=dict(rejected), test_protection='../protection-report.json',
                    training_label_policy='Only corresponding frozen target regenerations, no source answers used as labels',
                    deviations=['DFlash public NemotronV1 replaces gatedV2; fixed hash-selected source shards',
                        'NemotronV1 code prompts restored only for TACO train indices with exact source-field agreement; other redacted sources excluded',
                        'NemotronV1 chat prompts absent; replace15000 chat prompts with public UltraChat train_sft first requests',
                        'LoopAhead BBH training share replaced by public NemotronV1 STEM reasoning',
                        'Teacher prompt formats/512 cap specified locally because original details incomplete'],
                    isolation='all test sources plus all methods dev/confirmation excluded from either training set; '
                              'training sets may share training-only prompts with each other; pilot is subset of DFlash train')
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('MANIFEST COMPLETE', {k: v['count'] for k, v in files.items()}, flush=True)


if __name__ == '__main__':
    main()
