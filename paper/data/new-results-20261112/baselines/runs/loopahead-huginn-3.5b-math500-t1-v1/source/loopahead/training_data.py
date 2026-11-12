"""Memory-mapped, hash-verified shallow-state datasets and deterministic anchors."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


class FeatureStore:
    def __init__(self, directory, stage, model):
        self.directory = Path(directory)
        self.rows, self.counts, self.provenance = [], Counter(), []
        manifest = json.loads((BASE/'data/manifests/manifest.json').read_text())['files'][stage]
        component_hashes = set()
        self.components = None
        for index_path in sorted(self.directory.glob('shard*/index.json')):
            complete = json.loads(index_path.with_name('complete.json').read_text())
            assert digest(index_path) == complete['index_sha256']
            index = json.loads(index_path.read_text())
            identity = index['identity']
            assert identity['model'] == model and identity['teacher_identity']['manifest_sha256'] == manifest['sha256']
            self.provenance.append(dict(path=str(index_path.resolve()), sha256=complete['index_sha256']))
            for chunk in index['chunks']:
                path = index_path.parent/chunk['file']
                assert digest(path) == chunk['sha256']
                rows = torch.load(path, mmap=True, map_location='cpu', weights_only=True)
                assert [r['source_id'] for r in rows] == chunk['source_ids']
                for row in rows:
                    row['tokens'] = row['prompt_tokens'] + row['output_tokens']
                    assert row['states'].shape[0] == len(row['tokens']) and row['states'].dtype == torch.bfloat16
                    assert row['anchor_start'] == max(64, len(row['prompt_tokens']))
                    assert row['anchor_stop'] == max(row['anchor_start'], len(row['tokens'])-1)
                self.rows.extend(rows)
            self.counts.update({int(k): v for k, v in index['output_token_counts'].items()})
            components = index_path.parent/'frozen-components.safetensors'
            assert digest(components) == index['frozen_components_sha256']
            component_hashes.add(index['frozen_components_sha256'])
            self.components = components
        assert len(component_hashes) == 1 and len(self.rows) == manifest['count']
        ids = [r['source_id'] for r in self.rows]
        assert len(set(ids)) == len(ids) and set(ids) == set(manifest['source_ids'])
        self.rows.sort(key=lambda r: r['source_id'])
        self.component_hash = next(iter(component_hashes))
        self.eligible = [i for i,r in enumerate(self.rows) if r['anchor_stop'] > r['anchor_start']]
        self.ineligible = [r['source_id'] for r in self.rows if r['anchor_stop'] == r['anchor_start']]

    def anchors(self, count, epoch, shuffle):
        windows = []
        for i in self.eligible:
            row = self.rows[i]
            key = f'42/{epoch}/{row["source_id"]}'.encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
            rng = torch.Generator().manual_seed(seed)
            n = row['anchor_stop'] - row['anchor_start']
            positions = (torch.randperm(n, generator=rng)[:count] if n >= count else
                         torch.randint(n, (count,), generator=rng))
            windows.extend((i, row['anchor_start'] + int(t)) for t in positions)
        if shuffle:
            order = torch.randperm(len(windows), generator=torch.Generator().manual_seed(42+epoch)).tolist()
            windows = [windows[i] for i in order]
        return windows

    def batch(self, windows, device):
        contexts, anchors, labels, rows = [], [], [], []
        for i, t in windows:
            row = self.rows[i]
            contexts.append(row['states'][t-64:t])
            anchors.append(row['tokens'][t])
            future = row['tokens'][t+1:t+8]
            labels.append(future + [-100]*(7-len(future)))
            rows.append(row)
        return (torch.stack(contexts).to(device), torch.tensor(anchors, device=device),
                torch.tensor(labels, device=device), rows)
