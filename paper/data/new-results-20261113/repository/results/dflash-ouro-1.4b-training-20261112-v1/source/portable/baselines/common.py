"""Explicit artifacts and environment records shared by portable baseline commands."""
from __future__ import annotations

import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path

import torch

from llm_memory_editability.loop_model_registry import MODELS

MODEL_IDS = ('ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def environment():
    import transformers
    root = Path(__file__).parent
    return dict(torch=str(torch.__version__), transformers=transformers.__version__,
                cuda=torch.version.cuda, package=str(root.resolve()),
                source_sha256={str(p.relative_to(root)): digest(p)
                               for p in sorted(root.rglob('*.py'))},
                executor_sha256={name: digest(find_spec(f'llm_memory_editability.{name}').origin)
                                 for name in ('loop_optimized_decode', 'loop_tree_decode',
                                              'loop_budget_decode')})


def target_arguments(parser):
    parser.add_argument('--model', choices=MODEL_IDS, required=True)
    parser.add_argument('--target-path', type=Path, required=True,
                        help='Local HF snapshot, including its model implementation and tokenizer')
    parser.add_argument('--device', default='cuda', help='cpu, cuda, or cuda:N')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--max-seq', type=int, default=3072)


def load_target(args):
    from .dflash.target import LoopTarget
    return LoopTarget.load(args.target_path, MODELS[args.model].backend, args.max_seq,
                           device=args.device, dtype=getattr(torch, args.dtype))


def token_ids(values, *, vocab_size=None):
    if not isinstance(values, list) or not values or any(
            type(t) is not int or t < 0 or (vocab_size is not None and t >= vocab_size)
            for t in values):
        raise ValueError('Expected a nonempty list of valid token IDs')
    return values


def read_teachers(path, model, max_seq, manifest_path=None, stage=None):
    """Read raw JSONL or completed shard*/teachers.jsonl, checking available receipts.

    A supplied manifest requires the full split, never silently accepts a subset.
    Without one, the supplied JSONL files define the dataset and their hashes are recorded.
    """
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.glob('shard*/teachers.jsonl'))
    if not files and (path / 'teachers.jsonl').is_file():
        files = [path / 'teachers.jsonl']
    if not files:
        raise ValueError(f'No teacher JSONL files found: {path}')
    rows, sources = {}, []
    for file in files:
        sha = digest(file)
        metadata = file.with_name('metadata.json')
        identity = {}
        if metadata.exists():
            identity = json.loads(metadata.read_text())['identity']
            complete = json.loads(file.with_name('complete.json').read_text())
            if identity['model'] != model or complete['teachers_sha256'] != sha:
                raise ValueError(f'Teacher identity/hash mismatch: {file}')
        records = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        if identity and len(records) != complete['requests']:
            raise ValueError(f'Incomplete teacher shard: {file}')
        for row in records:
            sid = row['source_id']
            if not isinstance(sid, str) or sid in rows:
                raise ValueError(f'Duplicate or invalid teacher source_id: {sid}')
            prompt, output = token_ids(row['prompt_tokens']), token_ids(row['output_tokens'])
            if len(prompt) + len(output) > max_seq:
                raise ValueError(f'Teacher exceeds --max-seq: {sid}')
            rows[sid] = row
        sources.append(dict(path=str(file.resolve()), sha256=sha, metadata=identity))
    if manifest_path:
        manifests = json.loads(Path(manifest_path).read_text())['files']
        if stage is None:
            hashes = {s['metadata'].get('manifest_sha256') for s in sources}
            matches = [v for v in manifests.values() if hashes == {v['sha256']}]
            if len(matches) != 1:
                raise ValueError('Pass --stage to select a unique manifest split')
            expected = matches[0]
        else:
            expected = manifests[stage]
        if set(rows) != set(expected['source_ids']) or len(rows) != expected['count']:
            raise ValueError('Teachers do not exactly cover the requested manifest split')
        if any(s['metadata'] and s['metadata'].get('manifest_sha256') != expected['sha256']
               for s in sources):
            raise ValueError('Teacher provenance belongs to another manifest split')
    if not rows:
        raise ValueError('Empty teacher dataset')
    return [rows[k] for k in sorted(rows)], sources


def check_draft_config(config, target):
    details = config.dflash_config
    width = details.get('target_hidden_size', config.hidden_size)
    if width != target.hidden_size or details['target_layer_ids'] != target.expanded_ids:
        raise ValueError('Drafter feature widths/layers do not match this target')
    if config.vocab_size != target.model.config.vocab_size:
        raise ValueError('Drafter vocabulary does not match this target')
    if config.block_size != 16 or config.num_hidden_layers != 5:
        raise ValueError('Expected a five-layer block16 DFlash checkpoint')
    token_ids([details['mask_token_id']], vocab_size=config.vocab_size)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
