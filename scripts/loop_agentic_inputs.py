"""Freeze causal OpenHands request prefixes; never use the held-out next response."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests

BUCKET = 'https://swe-bench-submissions.s3.amazonaws.com/'
PREFIX = 'verified/20241029_OpenHands-CodeAct-2.1-sonnet-20241022/trajs/'
SEED = 20261112


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank(value):
    return sha(f'{SEED}/{value}'.encode())


def normalize_messages(raw):
    """Serialize API tool fields explicitly for models without a tool chat template."""
    result=[]
    for message in raw:
        content=message.get('content')
        if isinstance(content,list):
            if any(p.get('type')!='text' for p in content):
                raise ValueError('Non-text content cannot be dropped')
            content='\n'.join(p['text'] for p in content)
        if content is None:
            content=''
        if not isinstance(content,str):
            raise ValueError('Unsupported content')
        role=message['role']
        if message.get('tool_calls'):
            content+='\n<tool_calls>\n'+json.dumps(message['tool_calls'],ensure_ascii=False,sort_keys=True)+'\n</tool_calls>'
        if role=='tool':
            content='Tool result '+json.dumps({k:message[k] for k in ('name','tool_call_id') if k in message},sort_keys=True)+':\n'+content
            role='user'
        unexpected=set(message)-{'role','content','tool_calls','tool_call_id','name'}
        if unexpected:
            raise ValueError(f'Unknown message fields: {unexpected}')
        result.append(dict(role=role,content=content))
    return result


def causal_candidates(messages):
    """Label all calls by original trajectory thirds, before any length filtering."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected a nonempty message list')
    for message in messages:
        if message['role'] not in ('system', 'user', 'assistant') or not isinstance(message['content'], str):
            raise ValueError('Unsupported trajectory serialization; do not silently lose tool calls')
        if set(message) != {'role', 'content'}:
            raise ValueError('Unexpected message fields')
    turns = [i for i, message in enumerate(messages) if message['role'] == 'assistant']
    if len(turns) < 3:
        return []
    return [dict(phase=('early','middle','late')[min(2,3*ordinal//len(turns))],
                 assistant_turn=ordinal, message_index=turns[ordinal],
                 trajectory_calls=len(turns), messages=messages[:turns[ordinal]])
            for ordinal in range(len(turns))]


def fetch(url):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def prepare(args):
    from transformers import AutoTokenizer
    root = args.out.resolve()
    raw = root/'sources'/'openhands'
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'sources'/'objects.json'
    if manifest_path.exists():
        objects = json.loads(manifest_path.read_text())
    else:
        listing = fetch(BUCKET+'?list-type=2&prefix='+PREFIX+'&max-keys=1000')
        (root/'sources'/'listing.xml').write_bytes(listing)
        tree = ET.fromstring(listing)
        ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        if tree.findtext('s:IsTruncated', namespaces=ns) != 'false':
            raise ValueError('Unexpected paginated archive; freeze the complete listing first')
        objects = [dict(key=n.findtext('s:Key', namespaces=ns),
                        etag=n.findtext('s:ETag', namespaces=ns),
                        size=int(n.findtext('s:Size', namespaces=ns)))
                   for n in tree.findall('s:Contents', ns)]
        objects = [o for o in objects if o['key'].endswith('.json')]
        manifest_path.write_text(json.dumps(objects, indent=2)+'\n')

    def download(obj):
        path = raw/Path(obj['key']).name
        if not path.exists():
            data = fetch(BUCKET+obj['key'])
            if len(data) != obj['size']:
                raise ValueError('Archive object size changed')
            path.write_bytes(data)
        data = path.read_bytes()
        return dict(**obj, sha256=sha(data), path=str(path.relative_to(root)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        sources = list(pool.map(download, objects))
    print(json.dumps(dict(downloaded=len(sources), bytes=sum(o['size'] for o in sources))), flush=True)
    paths=dict(zip(('huginn-3.5b','ouro-2.6b'),args.model_paths))
    tokenizers={name:AutoTokenizer.from_pretrained(path,trust_remote_code=True,local_files_only=True)
                for name,path in paths.items()}
    candidates, excluded = [], []
    for obj in sorted(sources, key=lambda o: rank(o['key'])):
        messages = normalize_messages(json.loads((root/obj['path']).read_bytes()))
        instance = Path(obj['key']).stem
        repo = instance.rsplit('-', 1)[0]
        exceeded=False
        for row in causal_candidates(messages):
            ident = f"openhands/{instance}/turn-{row['assistant_turn']}"
            if exceeded:
                excluded.append(dict(source_id=ident,phase=row['phase'],reason='later cumulative prefix after context overflow'))
                continue
            tokens={name:tok.apply_chat_template(row['messages'],add_generation_prompt=True)
                    for name,tok in tokenizers.items()}
            if any(len(ids)>args.max_prompt_tokens for ids in tokens.values()):
                exceeded=True
                excluded.append(dict(source_id=ident, phase=row['phase'], prompt_tokens={k:len(v) for k,v in tokens.items()},
                                     reason='context_limit; no truncation'))
                continue
            candidates.append(dict(**row, source_id=ident, instance_id=instance, repo=repo,
                tokenizations={name:dict(prompt_token_ids=ids,prompt_tokens=len(ids),
                    prompt_sha256=sha(json.dumps(ids,separators=(',',':')).encode())) for name,ids in tokens.items()},
                source_sha256=obj['sha256'],source_url=BUCKET+obj['key'],max_new_tokens=512))
    requested_counts = dict(early=34, middle=33, late=33)
    available = Counter(r['phase'] for r in candidates)
    counts = {phase:min(n,available[phase]) for phase,n in requested_counts.items()}
    if sum(available.values()) < 100:
        raise ValueError('Fewer than 100 eligible calls')
    # A phase with too few complete contexts contributes every eligible call.
    # Redistribute its quota before inference, never truncate or duplicate requests.
    while sum(counts.values()) < 100:
        for phase in counts:
            if counts[phase] < available[phase] and sum(counts.values()) < 100:
                counts[phase] += 1
    selected = []
    for phase, count in counts.items():
        # Round-robin repositories; hash ranking within each. No output-quality filtering.
        groups = defaultdict(list)
        for row in candidates:
            if row['phase'] == phase:
                groups[row['repo']].append(row)
        for rows in groups.values():
            rows.sort(key=lambda r: rank(r['source_id']))
        order = sorted(groups, key=rank)
        pool = []
        while any(groups.values()):
            for repo in order:
                if groups[repo]:
                    pool.append(groups[repo].pop(0))
        if len(pool) < count:
            raise ValueError(f'{phase}: only {len(pool)} eligible calls, need {count}')
        selected.extend(pool[:count])
    selected.sort(key=lambda r: (rank(r['instance_id']), r['assistant_turn']))
    inputs = root/'inputs'
    inputs.mkdir(exist_ok=True)
    for i, row in enumerate(selected):
        row.update(question_id=i, dataset='openhands100', category=row['phase'])
    path = inputs/'openhands100.jsonl'
    if path.exists():
        raise FileExistsError('Frozen inputs already exist')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
    model_inputs={}
    for name in tokenizers:
        model_rows=[dict(**{k:v for k,v in row.items() if k!='tokenizations'},**row['tokenizations'][name],model=name) for row in selected]
        filename=inputs/(name+'.jsonl')
        filename.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in model_rows))
        lengths=[r['prompt_tokens'] for r in model_rows]
        model_inputs[name]=dict(path=filename.name,sha256=sha(filename.read_bytes()),
                               min_prompt_tokens=min(lengths),max_prompt_tokens=max(lengths),
                               mean_prompt_tokens=sum(lengths)/len(lengths))
    manifest = dict(dataset='SWE-bench Verified / OpenHands CodeAct 2.1 Sonnet 20241022',
        source_metadata='https://github.com/SWE-bench/experiments/blob/main/evaluation/verified/20241029_OpenHands-CodeAct-2.1-sonnet-20241022/metadata.yaml',
        seed=SEED, count=len(selected), instances=len({r['instance_id'] for r in selected}),
        phase_counts=dict(Counter(r['phase'] for r in selected)),
        requested_phase_counts=requested_counts,
        repo_counts=dict(Counter(r['repo'] for r in selected)),
        model_inputs=model_inputs,
        max_prompt_tokens=args.max_prompt_tokens, sources=sources, exclusions=excluded,
        eligible_by_phase=dict(Counter(r['phase'] for r in candidates)),
        input_sha256=sha(path.read_bytes()), model_paths={k:str(v) for k,v in paths.items()},
        tokenizer_chat_template_sha256={k:sha(tok.chat_template.encode()) for k,tok in tokenizers.items()},
        selection='All assistant calls labelled by original trajectory thirds; context eligibility; '
                  '34/33/33 requested phase quotas, capacity-limited deficits redistributed equally '
                  'across available phases; repository round-robin and SHA256 ordering.',
        semantics='Fixed historical request prefixes, target free generation. API tool calls serialized '
                  'as explicit JSON blocks; tool observations preserved as user messages; API cache hints removed. '
                  'Held-out next responses never enter inputs or suffix cache. '
                  'This is decoding on agent requests, not a live agent or task-success evaluation.')
    (inputs/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('sources','exclusions')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-paths', type=Path, nargs=2, required=True,help='Huginn then Ouro')
    parser.add_argument('--max-prompt-tokens', type=int, default=3548)
    prepare(parser.parse_args())
