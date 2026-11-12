"""Fetch pinned public training sources. No teacher generation, no training."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time

BASE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    import requests
    os.environ['HF_XET_CACHE'] = str(BASE / 'data/raw/xet-cache')
    raw = BASE / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    discoveries = json.loads((BASE / 'configs/dataset-discovery.json').read_text())
    nemotron = json.loads((BASE / 'configs/nemotron-v1-discovery.json').read_text())
    tasks = []
    for info in discoveries:
        repo = info['repo']
        if 'Nemotron' in repo:
            continue
        for f in info['files']:
            name = f['name']
            wanted = name == 'code_alpaca_20k.json' or '/train-' in name or '/train_sft-' in name
            if wanted:
                tasks.append(dict(repo=repo, revision=info['revision'], filename=name,
                                  expected_bytes=f['size']))
    # Fixed shard sampling, declared before looking at content. Different from global row sampling.
    for domain, count in [('math', 4), ('code', 4), ('chat', 2), ('stem', 1)]:
        files = [f for f in nemotron['files'] if f['name'].startswith('data/' + domain + '-')]
        files.sort(key=lambda f: hashlib.sha256(('loopfill-baselines-42:' + f['name']).encode()).hexdigest())
        for f in files[:count]:
            tasks.append(dict(repo=nemotron['repo'], revision=nemotron['revision'], filename=f['name'],
                              expected_bytes=f['size'], domain=domain))
    selection = dict(nemotron_fallback='V2 access returned401/GatedRepoError; use public official V1, not a mirror',
                     shard_selection='lowest SHA256(loopfill-baselines-42:filename), per domain', tasks=tasks)
    (BASE / 'configs/data-download-plan.json').write_text(json.dumps(selection, indent=2) + '\n')
    report = BASE / 'data/sources.jsonl'
    lock = threading.Lock()

    def save(row):
        with lock, report.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        print(row.get('repo', row.get('url')), row.get('filename'), row['status'], flush=True)

    def fetch(item):
        item = dict(item, started_unix=time.time())
        try:
            try:
                path = hf_hub_download(item['repo'], item['filename'], repo_type='dataset',
                    revision=item['revision'], cache_dir='/tmp/loopfill-assets/datasets', local_files_only=True)
            except Exception:
                path = hf_hub_download(item['repo'], item['filename'], repo_type='dataset',
                    revision=item['revision'], cache_dir=str(raw / 'hf'))
            path = Path(path)
            assert path.stat().st_size == item['expected_bytes']
            item.update(status='downloaded', path=str(path.resolve()), sha256=sha(path))
        except Exception as exc:
            item.update(status='failed', error_type=type(exc).__name__,
                        http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
        item['finished_unix'] = time.time()
        save(item)
        return item

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, tasks))
    urls = [
        ('gsm8k-train.jsonl', 'https://raw.githubusercontent.com/openai/grade-school-math/'
         '3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl'),
        ('mbpp-all.jsonl', 'https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl'),
    ]
    for name, url in urls:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path = raw / name
        path.write_bytes(response.content)
        row = dict(url=url, filename=name, path=str(path.resolve()), sha256=sha(path),
                   bytes=path.stat().st_size, status='downloaded')
        save(row)
    if any(r['status'] != 'downloaded' for r in rows):
        raise RuntimeError('Some sources failed; inspect data/sources.jsonl before proceeding')


if __name__ == '__main__':
    main()
