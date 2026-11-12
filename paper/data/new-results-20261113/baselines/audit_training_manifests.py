"""Independently check frozen manifest hashes, counts, exact isolation and raw test-source hashes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from data_isolation import normalize


def rows(path):
    # Iterating lines splits only on JSONL newlines, not Unicode U+2028 inside valid strings.
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    base = Path(__file__).resolve().parent
    folder = base / 'data/manifests'
    manifest = json.loads((folder / 'manifest.json').read_text())
    protected = {normalize(x['text']) for x in rows(base / 'data/protected.jsonl')}
    sets, checks = {}, []
    for name, meta in manifest['files'].items():
        path = folder / (name + '.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
        records = rows(path)
        assert len(records) == meta['count'] == len({r['source_id'] for r in records})
        texts = {normalize(r['question']) for r in records}
        assert len(texts) == len(records) and not texts & protected
        assert dict(Counter(r['pool'] for r in records)) == meta['pools']
        assert all(max(r['prompt_lengths'].values()) <= 2560 for r in records)
        sets[name] = texts
        checks.append(dict(name=name, count=len(records), sha256=meta['sha256']))
    heldout = set()
    for name in ['loopahead-dev', 'loopahead-confirmation', 'dflash-dev']:
        assert not heldout & sets[name]
        heldout.update(sets[name])
    for name in ['dflash-train', 'loopahead-train']:
        assert not heldout & sets[name]
    assert sets['dflash-pilot'] <= sets['dflash-train']
    source = base.parents[1] / 'results/loopfill-latency512-20261107/inputs'
    old = json.loads((source / 'manifest.json').read_text())
    for name, meta in old['sources'].items():
        assert hashlib.sha256((source / 'sources' / name).read_bytes()).hexdigest() == meta['sha256']
    report = dict(passed=True, checked_unix=time.time(), files=checks, protected_unique_questions=len(protected),
                  exact_test_and_cross_method_holdout_overlap=0, source_hash_checks=len(old['sources']),
                  near_duplicate_check='Applied by prepare_training_manifests.py during selection; '
                                       'see manifest rejections and protection policy')
    (base / 'data/manifest-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
