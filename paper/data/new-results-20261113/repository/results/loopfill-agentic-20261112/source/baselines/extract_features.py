"""Extract real d1 head-input states for standalone LoopAhead training."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .common import (digest, environment, load_target, read_teachers, save,
                     target_arguments, token_ids)


def extract(target, rows, out, identity, batch_size=8, chunk_size=64):
    """Write the same shard format consumed by FeatureStore and existing LAD runs."""
    if min(batch_size, chunk_size) < 1:
        raise ValueError('Batch/chunk sizes must be positive')
    out.mkdir(parents=True, exist_ok=False)
    components = {'embedding': target.embedding.weight.detach().cpu().contiguous().clone(),
                  'head': target.head.weight.detach().cpu().contiguous().clone()}
    if getattr(target.head, 'bias', None) is not None:
        components['head_bias'] = target.head.bias.detach().cpu().contiguous().clone()
    save_file(components, out / 'frozen-components.safetensors')
    save(out / 'metadata.json', identity)
    rows = sorted(rows, key=lambda r: (len(r['prompt_tokens']) + len(r['output_tokens']), r['source_id']))
    chunks, counts = [], Counter()
    for offset in range(0, len(rows), chunk_size):
        group, chunk = rows[offset:offset + chunk_size], []
        for start in range(0, len(group), batch_size):
            micro = group[start:start + batch_size]
            tokens = [token_ids(r['prompt_tokens'] + r['output_tokens'],
                                vocab_size=target.model.config.vocab_size) for r in micro]
            ids = torch.zeros((len(micro), max(map(len, tokens))), device=target.device, dtype=torch.long)
            for i, seq in enumerate(tokens):
                ids[i, :len(seq)] = torch.tensor(seq, device=target.device)
            states = target.shallow_readout_states(ids, identity['depth'])
            for i, row in enumerate(micro):
                first = max(64, len(row['prompt_tokens']))
                chunk.append(dict(source_id=row['source_id'], pool=row.get('pool', 'default'),
                                  prompt_tokens=row['prompt_tokens'], output_tokens=row['output_tokens'],
                                  states=states[i, :len(tokens[i])].to(device='cpu', dtype=torch.bfloat16).contiguous().clone(),
                                  anchor_start=first, anchor_stop=max(first, len(tokens[i]) - 1)))
        dest = out / f'chunk-{len(chunks):05d}.pt'
        torch.save(chunk, dest)
        chunks.append(dict(file=dest.name, sha256=digest(dest), requests=len(chunk),
                           source_ids=[r['source_id'] for r in chunk]))
        for row in group:
            counts.update(row['output_tokens'])
        print(f'features {offset + len(group)}/{len(rows)}', flush=True)
    save(out / 'index.json', dict(identity=identity, chunks=chunks, output_token_counts=dict(counts),
                                  frozen_components_sha256=digest(out / 'frozen-components.safetensors')))
    save(out / 'complete.json', dict(requests=len(rows), index_sha256=digest(out / 'index.json')))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    target_arguments(p)
    p.add_argument('--teachers', type=Path, required=True)
    p.add_argument('--manifest', type=Path, help='Optional source manifest to validate full split coverage')
    p.add_argument('--features', type=Path, required=True, help='Destination feature root')
    p.add_argument('--stage', choices=('loopahead-train', 'loopahead-dev', 'loopahead-confirmation'), required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=64)
    args = p.parse_args(argv)
    rows, sources = read_teachers(args.teachers, args.model, args.max_seq, args.manifest, args.stage)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())['files'][args.stage]
    else:
        manifest = dict(sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                        count=len(rows), source_ids=[r['source_id'] for r in rows])
    manifest_path = args.features / 'manifest.json'
    combined = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(files={})
    if args.stage in combined['files'] and combined['files'][args.stage] != manifest:
        raise ValueError('Feature root already contains a different dataset for this stage')
    torch.set_num_threads(1)
    target = load_target(args)
    try:
        identity = dict(model=args.model, target_path=str(args.target_path.resolve()),
                        depth=1 if target.family == 'ouro' else 4,
                        teacher_identity=dict(manifest_sha256=manifest['sha256']),
                        sources=sources, target_dtype=args.dtype, stored_dtype='bfloat16',
                        environment=environment())
        extract(target, rows, args.features / args.stage / args.model / 'shard000', identity,
                args.batch_size, args.chunk_size)
        combined['files'][args.stage] = manifest
        save(manifest_path, combined)
    finally:
        target.close()


if __name__ == '__main__':
    main()
