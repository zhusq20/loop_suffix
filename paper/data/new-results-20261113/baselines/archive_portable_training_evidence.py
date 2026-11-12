"""Archive immutable portable training evidence without copying live journals.

The original controller remains authoritative. Each invocation adds byte-identical
files and a new manifest; conflicting copies fail instead of being overwritten.
Large teacher and input JSONL files stay in the workspace, outside Git.
"""
import argparse
import fcntl
from pathlib import Path
import shutil
import time

from external_dflash_costs import collect
from packed_dflash_evidence import read, save, sha

BASE = Path(__file__).resolve().parent


def archive(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'archive.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        accounting = collect()
        root = Path(accounting['root'])
        plan = read(root/'plan.json')
        files = {}

        def copy(path, expected=None):
            path = Path(path)
            relative = path.relative_to(root)
            assert path.is_file() and not path.is_symlink() and '..' not in relative.parts
            digest = sha(path)
            assert expected is None or digest == expected, path
            dest = out/'files'/relative
            if dest.exists():
                assert sha(dest) == digest, f'Preserve conflicting copy: {dest}'
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as source, dest.open('xb') as target:
                    shutil.copyfileobj(source, target, length=8*1024*1024)
                assert sha(dest) == digest, path
            assert sha(path) == digest, f'Source changed while copying: {path}'
            files[str(path)] = dict(archived=str(dest.resolve()), sha256=digest,
                                   bytes=dest.stat().st_size)

        for name in ['plan.json', 'launcher.json', 'selection.json', 'native-driver.json']:
            copy(root/name)
        for name, digest in plan['source_sha256'].items():
            copy(root/name, digest)
        for name, digest in plan['input_sha256'].items():
            path = Path(name)
            if path.is_relative_to(root):
                copy(path, digest)
        closed = []
        binary_suffixes = {'.pt', '.pth', '.safetensors', '.pyc'}
        for stage, folder in [('teachers','teachers/shard0'),
                              ('smoke','checks/training-smoke'), ('train','training')]:
            launcher, receipt = root/f'{stage}-launcher.json', root/f'{stage}-exit.json'
            if launcher.exists():
                copy(launcher)
            if not receipt.exists():
                continue
            assert read(receipt)['exit_code'] == 0, f'Failed stage needs a separate failure archive: {stage}'
            closed.append(stage)
            copy(receipt)
            copy(root/'logs'/f'{stage}.log')
            for path in sorted((root/folder).rglob('*')):
                if path.is_file() and path.suffix not in binary_suffixes and '__pycache__' not in path.parts:
                    copy(path)
        # The one-time metadata and training launch are stable while metrics
        # and checkpoints are still being written; never archive live metrics.
        if (root/'training/metadata.json').exists():
            copy(root/'training/metadata.json')
        if 'teachers' in closed:
            copy(root/'checks/teacher-audit.json')
            copy(root/'checks/longest32-teachers.jsonl')
        if accounting['status'] == 'complete':
            for name in ['complete.json','audit-launcher.json','audit-exit.json',
                         'checks/training-audit.json','logs/audit.log']:
                copy(root/name)
        ignored = ('/archive.lock\n/files/teachers/shard0/teachers.jsonl\n'
                   '/files/inputs/dflash-train.jsonl\n')
        ignore = out/'.gitignore'
        if ignore.exists():
            assert ignore.read_text() == ignored
        else:
            ignore.write_text(ignored)
        manifest = dict(status='closed_portable_training_evidence_archived',
            original_root=str(root), pipeline_status=accounting['status'], closed_stages=closed,
            files=files, bytes=sum(v['bytes'] for v in files.values()),
            training_plan_sha256=accounting['training_plan_sha256'],
            closed_pipeline_gpu_hours=accounting['total_recorded_gpu_hours'],
            external_model_inputs={p:h for p,h in plan['input_sha256'].items()
                                   if not Path(p).is_relative_to(root)},
            archiver_sha256=sha(Path(__file__)), checked_unix=time.time(), new_gpu_measurements=0,
            limitations=['Only exited stages and immutable plan/metadata are copied; live journals are excluded.',
                         'Absolute paths inside copies are unchanged; use this mapping to find local evidence.',
                         'Teacher and training-input JSONL files are archived locally and excluded from Git.',
                         'Target assets, trained weights and optimizer states retain their recorded locations.',
                         'Archiving is not a new training, numerical-equivalence or benchmark validation.'])
        prior_path = out/'latest.json'
        if prior_path.exists():
            prior = read(read(prior_path)['manifest'])
            assert all(files.get(k) == v for k,v in prior['files'].items())
        index = out/f'manifest-{time.time_ns()}.json'
        save(index, manifest)
        save(prior_path, dict(manifest=str(index.resolve()), sha256=sha(index)))
        print(dict(pipeline_status=accounting['status'], closed_stages=closed,
                   files=len(files), bytes=manifest['bytes'], manifest=str(index)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=BASE/'external-evidence/portable-dflash-ouro14-v3')
    archive(parser.parse_args().out.resolve())
