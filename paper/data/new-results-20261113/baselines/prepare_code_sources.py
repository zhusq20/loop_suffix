"""Public original TACO train questions referenced by Nemotron's redacted code rows."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading

from huggingface_hub import hf_hub_download

from prepare_data_sources import sha


def main():
    base = Path(__file__).resolve().parent
    source = json.loads((base / 'configs/taco-discovery.json').read_text())
    files = [f for f in source['files'] if f['name'].startswith('ALL/train-')]
    lock = threading.Lock()

    def fetch(file):
        path = Path(hf_hub_download(source['repo'], file['name'], repo_type='dataset',
                     revision=source['revision'], cache_dir=str(base / 'data/raw/hf')))
        assert path.stat().st_size == file['size']
        row = dict(repo=source['repo'], revision=source['revision'], filename=file['name'],
                   path=str(path.resolve()), sha256=sha(path), status='downloaded')
        with lock, (base / 'data/sources.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
            f.flush()
            os.fsync(f.fileno())
        print(file['name'], 'downloaded', flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))


if __name__ == '__main__':
    main()
