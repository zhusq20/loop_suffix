"""Archive completed external canonical measurements under the baseline directory.

Copies are byte-identical, append-only evidence. Original locations, unavailable
exit receipts and absolute paths inside the artifacts remain explicit. No
measurement is launched, moved, relabelled or substituted by this command.
"""
import argparse
import csv
import fcntl
import io
import json
from pathlib import Path
import time

from packed_dflash_evidence import read, save, sha

BASE = Path(__file__).resolve().parent
EXTERNAL = Path('/ossfs/workspace/loop_suffix/results')


def archive(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'archive.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        csv_path, evidence_path = [BASE/'combined-results'/n for n in ['results.csv','evidence.json']]
        csv_hash, evidence_hash = sha(csv_path), sha(evidence_path)
        results = list(csv.DictReader(io.StringIO(csv_path.read_text())))
        evidence = read(evidence_path)
        assert read(BASE/'combined-results/status.json')['results_sha256'] == csv_hash
        copied, cells, suites = {}, {}, {}

        def copy(path, expected=None):
            path = Path(path)
            relative = path.relative_to(EXTERNAL)
            assert '..' not in relative.parts and path.is_file() and not path.is_symlink(), path
            digest = sha(path)
            assert expected is None or digest == expected, path
            destination = out/'files'/relative
            if destination.exists():
                assert sha(destination) == digest, f'Preserve conflicting archive: {destination}'
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = path.read_bytes()
                with destination.open('xb') as stream:
                    stream.write(data)
                assert sha(destination) == digest, path
            assert sha(path) == digest, f'Source changed while copying: {path}'
            copied[str(path)] = dict(archived=str(destination.resolve()), sha256=digest,
                                     bytes=destination.stat().st_size)
            return str(destination.resolve())

        for row in results:
            method = Path(row['method_run'])
            if not method.is_relative_to(EXTERNAL):
                continue
            assert row['method'] == 'dflash' and row['target_backend'] == 'packed_full_loop'
            identity = '/'.join(row[k] for k in ['method','model','benchmark','temperature'])
            proof = evidence['cells'][identity]
            assert proof['origin'] == 'verified_packed_supplement'
            source = proof['sources']['method']
            assert Path(source['path']) == method
            # The supplement plan is already local; it identifies the frozen
            # external suite used by this particular timed measurement.
            profile_meta = read(Path(row['profile_run'])/'metadata.json')
            supplement = read(profile_meta['plan']) if 'plan' in profile_meta else None
            if supplement is None:
                supplement_root = Path(row['profile_run']).parents[1]
                supplement = read(supplement_root/'plan.json')
            suite = Path(supplement['suite'])
            assert method.is_relative_to(suite/'runs')
            protocol = read(suite/'protocol.json')
            metadata = read(method/'metadata.json')
            job = metadata['job']
            assert job in read(suite/'jobs.json') and job['kind'] == 'latency'
            assert job['dataset'] == row['benchmark'] and job['temperature'] == int(row['temperature'])
            if str(suite) not in suites:
                suites[str(suite)] = dict(protocol=copy(suite/'protocol.json'), jobs=copy(suite/'jobs.json'))
                for name, digest in protocol['source_sha256'].items():
                    copy(suite/'source'/name, digest)
                for name, digest in protocol['reference_sha256'].items():
                    copy(suite/name, digest)
            inputs = copy(suite/job['inputs'], job['inputs_sha256'])
            archived_run = {}
            for name, digest in source['sha256'].items():
                archived_run[name] = copy(method/name, digest)
            for name in ['preflight.json','progress.json']:
                if (method/name).exists():
                    archived_run[name] = copy(method/name)
            receipt = proof['original_timed_exit']
            archived_exit = None
            if receipt['actual_exit_code'] == 0:
                path = Path(receipt['path'])
                archived_exit = copy(path, receipt['sha256'])
                finish = read(path)
                assert finish['exit_code'] == 0
                launch_path = path.with_name(path.name.replace('-exit.json','-launcher.json'))
                assert launch_path != path and launch_path.exists()
                launch = read(launch_path)
                assert all(finish[k] == v for k,v in launch.items())
                copy(launch_path)
            else:
                assert receipt['actual_exit_code'] is None and receipt['limitation']
            cells[identity] = dict(original_run=str(method), artifacts=archived_run,
                inputs=inputs, original_exit_evidence=receipt, archived_exit=archived_exit,
                same_stack_ar=row['ar_run'], independent_profile=row['profile_run'])
        assert sha(csv_path) == csv_hash and sha(evidence_path) == evidence_hash
        manifest = dict(status='completed_external_measurements_archived', cells=cells, files=copied,
            suites=suites, canonical_cells=len(results), external_canonical_cells=len(cells),
            bytes=sum(v['bytes'] for v in copied.values()),
            source_results_sha256=csv_hash, source_evidence_sha256=evidence_hash,
            archiver_sha256=sha(Path(__file__)), checked_unix=time.time(), new_gpu_measurements=0,
            limitations=['Copies retain original absolute paths; use this mapping to locate archived artifacts.',
                         'Unknown original exit status remains unknown; copies do not recreate process receipts.',
                         'Only completed external canonical latency evidence is covered here. Running training '
                         'and additional throughput artifacts are outside this archive scope.',
                         'Target model and trained weights retain their separately recorded checkpoint locations.'])
        index = out/f'manifest-{time.time_ns()}.json'
        save(index, manifest)
        save(out/'latest.json', dict(manifest=str(index.resolve()), sha256=sha(index)))
        print(json.dumps(dict(manifest=str(index), canonical_cells=len(results),
                              external_cells=len(cells), files=len(copied), bytes=manifest['bytes'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE/'external-evidence/packed-measurements')
    archive(parser.parse_args().out.resolve())
