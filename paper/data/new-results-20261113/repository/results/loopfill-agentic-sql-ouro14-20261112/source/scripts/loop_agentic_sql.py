"""Build causal Spider SQL execution/repair traces, then freeze shared replay calls.

The trace teacher sees schema, question and real SQLite errors, never gold SQL.
Successful execution ends a trajectory; at most two error-triggered repairs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

SEED = 20261112
MODELS = ('huginn-3.5b', 'ouro-2.6b')
SYSTEM = ('You are a SQL agent. Write one read-only SQLite query that answers the user question. '
          'Return only the SQL query, optionally inside a single sql code fence. '
          'An execution tool will run your query. If it reports an error, correct the query using '
          'the database schema and error. Do not change the database.')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank(key):
    return sha(f'{SEED}/{key}'.encode())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def tokenizations(messages, tokenizers):
    result = {}
    for name, tokenizer in tokenizers.items():
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        result[name] = dict(prompt_token_ids=ids, prompt_tokens=len(ids),
                           prompt_sha256=sha(json.dumps(ids, separators=(',', ':')).encode()))
    return result


def extract_sql(text):
    fences = re.findall(r'```(?:sql|sqlite)?\s*\n?(.*?)```', text, flags=re.I | re.S)
    if len(fences) > 1:
        raise ValueError('Return exactly one SQL query, not multiple code blocks.')
    sql = (fences[0] if fences else text).strip()
    # execute() also rejects multiple statements. SELECT/WITH plus query_only
    # allow genuine SQLite execution diagnostics without writes or extensions.
    clean = re.sub(r'/\*.*?\*/|--[^\n]*', '', sql, flags=re.S).lstrip()
    if not re.match(r'(?i)(select|with)\b', clean):
        raise ValueError('Expected one read-only SQLite SELECT or WITH query.')
    return sql


def execute_response(text, database, timeout=2.):
    try:
        sql = extract_sql(text)
    except ValueError as exc:
        return dict(ok=False, kind='response_format', error=str(exc))
    begin = time.monotonic()
    try:
        with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as connection:
            connection.execute('PRAGMA query_only=ON')
            connection.set_progress_handler(lambda: int(time.monotonic() - begin > timeout), 1000)
            cursor = connection.execute(sql)
            rows = cursor.fetchmany(6)
            return dict(ok=True, kind='sqlite', sql=sql, columns=[c[0] for c in cursor.description],
                        preview_rows=[[v.hex() if isinstance(v,bytes) else v for v in row] for row in rows[:5]],
                        preview_truncated=len(rows) > 5)
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return dict(ok=False, kind='sqlite', sql=sql, error=str(exc)[:800])


def repair_message(observation):
    if observation['ok']:
        raise ValueError('Successful execution must not be turned into a fabricated error')
    feedback = {k: observation[k] for k in ('kind', 'error')}
    return dict(role='user', content='SQL execution tool reported:\n' + json.dumps(feedback) +
                '\nCorrect the query to answer the original question. Return only one SQLite query.')


def load_tokenizers(paths):
    from transformers import AutoTokenizer
    return {name: AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
            for name, path in paths.items()}


def prepare(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'tasks.json').exists():
        raise FileExistsError('Task selection is already frozen')
    paths = dict(zip(MODELS, map(str, args.model_paths)))
    tokenizers = load_tokenizers(paths)
    data = args.data.resolve()
    original = json.loads((data / 'dev.json').read_text())
    groups = defaultdict(list)
    excluded = []
    databases = {}
    for index, item in enumerate(original):
        db = item['db_id']
        path = data / 'database' / db / (db + '.sqlite')
        if db not in databases:
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
                ddl = [r[0] for r in connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            databases[db] = dict(path=str(path), sha256=sha(path.read_bytes()), schema=';\n'.join(ddl) + ';')
        messages = [dict(role='system', content=SYSTEM), dict(role='user', content=
            'Database schema:\n' + databases[db]['schema'] + '\n\nQuestion: ' + item['question'])]
        tokens = tokenizations(messages, tokenizers)
        if max(t['prompt_tokens'] for t in tokens.values()) > 1900:
            excluded.append(index)
            continue
        # Do not copy the gold SQL or parsed answer into any teacher/task input.
        groups[db].append(dict(instance_id=f'spider-dev-{index:04d}', original_index=index,
            db_id=db, question=item['question'], messages=messages))
    for rows in groups.values():
        rows.sort(key=lambda row: rank(row['instance_id']))
    tasks = []
    while len(tasks) < 100 and any(groups.values()):
        for db in sorted(groups, key=rank):
            if groups[db] and len(tasks) < 100:
                tasks.append(groups[db].pop(0))
    if len(tasks) != 100:
        raise ValueError('Fewer than 100 context-eligible development questions')
    tasks.sort(key=lambda row: rank(row['instance_id']))
    write_json(root / 'tasks.json', tasks)
    write_json(root / 'trace-protocol.json', dict(
        dataset='Spider development SQL execution/repair request replay', dataset_id='spider-repair100',
        source='https://yale-lily.github.io/spider', release=json.loads((data.parent/'source.json').read_text()),
        dev_sha256=sha((data/'dev.json').read_bytes()), databases=databases,
        tasks_sha256=sha((root/'tasks.json').read_bytes()), seed=SEED,
        model_paths=paths, teacher='ouro-2.6b', teacher_temperature=0, teacher_max_new_tokens=512,
        teacher_source=str(args.code_snapshot.resolve()),
        max_calls_per_task=3, stop='first successful SQLite execution or three calls',
        max_prompt_tokens=3548, task_count=100, task_selection='Database round-robin; SHA256 ranking; '
        'initial prompt <=1900 tokens in both models. No answer-quality or repetition filtering.',
        excluded_long_questions=excluded,
        script_sha256=sha(Path(__file__).read_bytes()), sqlite_version=sqlite3.sqlite_version))
    print(json.dumps(dict(tasks=len(tasks), databases=len({r['db_id'] for r in tasks}),
                          excluded_long_questions=len(excluded))))


def trace(args):
    root = args.out.resolve()
    protocol = json.loads((root/'trace-protocol.json').read_text())
    if sha((root/'tasks.json').read_bytes()) != protocol['tasks_sha256']:
        raise ValueError('Task selection changed')
    if sha(Path(__file__).read_bytes()) != protocol['script_sha256']:
        raise ValueError('Trace builder changed')
    source = Path(protocol['teacher_source'])
    for rel, expected in json.loads((source.parent/'protocol.json').read_text())['source_sha256'].items():
        if sha((source/rel).read_bytes()) != expected:
            raise ValueError('Teacher source changed: ' + rel)
    sys.path[:0] = [str(source), str(source/'src'), str(source/'scripts')]
    import torch
    from baselines.dflash.target import LoopTarget
    from baselines.dflash.packed_target import PackedDFlashTarget
    from baselines.dflash.decode import generate
    from llm_memory_editability.loop_model_registry import MODELS as specs, resolve_stop_ids
    torch.set_num_threads(1)
    torch.manual_seed(0)
    tokenizers = load_tokenizers(protocol['model_paths'])
    name = protocol['teacher']
    native = LoopTarget.load(protocol['model_paths'][name], 'ouro', 4096)
    native.close()
    target = PackedDFlashTarget(native, block_size=1)
    stops, audit = resolve_stop_ids(specs[name], tokenizers[name], native.model.config.eos_token_id,
                                   None, native.model.config.vocab_size, 'chat')
    tasks = json.loads((root/'tasks.json').read_text())[args.shard::args.shards]
    directory = root/'traces'
    directory.mkdir(exist_ok=True)
    write_json(directory/f'worker-{args.shard}.json', dict(pid=os.getpid(), shard=args.shard,
        shards=args.shards, gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), stop_audit=audit))
    for task in tasks:
        path = directory/(task['instance_id']+'.json')
        if path.exists():
            continue
        db = protocol['databases'][task['db_id']]
        if sha(Path(db['path']).read_bytes()) != db['sha256']:
            raise ValueError('Database changed')
        history = task['messages'][:]
        calls, exclusions = [], []
        for turn in range(3):
            tokens = tokenizations(history, tokenizers)
            if max(t['prompt_tokens'] for t in tokens.values()) > protocol['max_prompt_tokens']:
                exclusions.append(dict(turn=turn, reason='context_limit; no truncation'))
                break
            prompt = tokens[name]['prompt_token_ids']
            result = generate(target, torch.tensor([prompt], device='cuda'), 512, stops)
            response = tokenizers[name].decode(result.tokens, skip_special_tokens=True)
            observation = execute_response(response, db['path'])
            # Copy the prefix now: the current response is trace output, not input.
            calls.append(dict(source_id=f"{task['instance_id']}/call-{turn}",
                phase=('early','middle','late')[turn], assistant_turn=turn,
                messages=history[:], tokenizations=tokens, response=response,
                response_token_ids=result.tokens, observation=observation))
            if observation['ok']:
                break
            history.extend([dict(role='assistant', content=response), repair_message(observation)])
        write_json(path, dict(task_id=task['instance_id'], db_id=task['db_id'],
                             calls=calls, exclusions=exclusions))
        print(json.dumps(dict(task=task['instance_id'], calls=len(calls),
                              executed=bool(calls and calls[-1]['observation']['ok']))), flush=True)


def freeze_inputs(args):
    root = args.out.resolve()
    protocol = json.loads((root/'trace-protocol.json').read_text())
    tasks = json.loads((root/'tasks.json').read_text())
    candidates, sources, outcomes = [], [], Counter()
    for task in tasks:
        path = root/'traces'/(task['instance_id']+'.json')
        trace_data = json.loads(path.read_text())
        source_hash = sha(path.read_bytes())
        sources.append(dict(path=str(path.relative_to(root)), sha256=source_hash))
        outcomes.update(execution_success=int(trace_data['calls'][-1]['observation']['ok']),
                        context_excluded=len(trace_data['exclusions']))
        for call in trace_data['calls']:
            candidates.append(dict(**{k:v for k,v in call.items() if k not in
                ('response','response_token_ids','observation')}, instance_id=task['instance_id'],
                repo=task['db_id'], source_sha256=source_hash, max_new_tokens=512))
    requested = dict(early=34, middle=33, late=33)
    available = Counter(r['phase'] for r in candidates)
    counts = {p:min(n, available[p]) for p,n in requested.items()}
    while sum(counts.values()) < 100:
        for p in counts:
            if counts[p] < available[p] and sum(counts.values()) < 100:
                counts[p] += 1
    selected = []
    for phase, count in counts.items():
        pool = sorted((r for r in candidates if r['phase']==phase), key=lambda r:rank(r['source_id']))
        selected.extend(pool[:count])
    selected.sort(key=lambda r:(rank(r['instance_id']), r['assistant_turn']))
    inputs = root/'inputs'
    inputs.mkdir(exist_ok=False)
    for index, row in enumerate(selected):
        row.update(question_id=index, dataset=protocol['dataset_id'], category=row['phase'])
    master = inputs/'spider-repair100.jsonl'
    master.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in selected))
    model_inputs = {}
    for model in MODELS:
        rows = [dict(**{k:v for k,v in r.items() if k!='tokenizations'},
                     **r['tokenizations'][model], model=model) for r in selected]
        path = inputs/(model+'.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
        lengths = [r['prompt_tokens'] for r in rows]
        model_inputs[model] = dict(path=path.name, sha256=sha(path.read_bytes()),
            min_prompt_tokens=min(lengths), max_prompt_tokens=max(lengths), mean_prompt_tokens=sum(lengths)/len(lengths))
    manifest = dict(dataset=protocol['dataset'], dataset_id=protocol['dataset_id'], source_metadata=protocol['source'],
        model_paths=protocol['model_paths'], model_inputs=model_inputs, count=100,
        instances=len({r['instance_id'] for r in selected}), phase_counts=dict(Counter(r['phase'] for r in selected)),
        requested_phase_counts=requested, eligible_by_phase=dict(available), trace_outcomes=dict(outcomes),
        max_prompt_tokens=3548, input_sha256=sha(master.read_bytes()), sources=sources,
        selection=protocol['task_selection']+' Call quotas 34 initial / 33 first repair / 33 second repair; '
            'capacity-limited deficits redistributed; SHA256 ranking within phase; chronological order within task.',
        semantics='Fixed causal SQL request replay. Ouro-2.6B greedy teacher, actual read-only SQLite execution, '
            'at most two error-triggered repairs. No gold SQL feedback. Successful execution ends a trace. '
            'Current teacher response, tool result and future calls removed from replay inputs. '
            'No live-agent or official Spider accuracy claim. Distinct from the unpublished SuffixDecoding AgenticSQL dataset.')
    write_json(inputs/'manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','trace','freeze-inputs'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/tmp/loopfill-assets/spider/spider_data'))
    parser.add_argument('--model-paths', type=Path, nargs=2)
    parser.add_argument('--code-snapshot', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    {'prepare':prepare, 'trace':trace, 'freeze-inputs':freeze_inputs}[args.action](args)
