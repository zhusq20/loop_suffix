"""Validate source and execution evidence before a formal run enters the table.

This is a CPU-only artifact check. It does not establish numerical equivalence
or replace the separate sampling and first-divergence diagnostics.
"""
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


class ExitReceiptPending(RuntimeError):
    """A complete journal is present, but the launcher has not recorded exit."""


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_run_artifacts(run):
    path, meta, rows = run['path'], run['meta'], run['rows']
    protocol_path = ROOT/'results/loopfill-latency512-20261107/protocol.json'
    protocol = read(protocol_path)
    assert meta['model_revision'] == protocol['models'][meta['model']]['revision']
    token_key = 'decode_tokens' if run['stack'] == 'official' else 'tokens'
    for row in rows:
        assert row['stats'][token_key] == len(row['tokens']) - 1
        assert math.isfinite(row['stats']['decode_s']) and row['stats']['decode_s'] >= 0

    provenance = meta.get('measurement_provenance')
    if provenance is not None:
        assert meta['method'] == 'ar' and run['stack'] == 'ours'
        assert provenance['kind'] == 'archived_same_host_AR' and provenance['new_measurement'] is False
        suite = ROOT/'results'/('loopfill-latency512-t1-20261108' if meta['temperature'] else 'loopfill-latency512-20261107')
        source = suite/f'runs/{meta["model"]}-{meta["dataset"]}-ar'
        assert Path(provenance['source']).resolve() == source.resolve()
        assert set(provenance['source_sha256']) == {'metadata.json', 'requests.jsonl', 'summary.json', 'complete.json'}
        for name, expected in provenance['source_sha256'].items():
            imported = path/('original-metadata.json' if name == 'metadata.json' else name)
            assert sha(source/name) == sha(imported) == expected
        original = read(source/'metadata.json')
        original_protocol = read(suite/'protocol.json')
        assert original['source_sha256'] == meta['frozen_source_sha256'] == original_protocol['source_sha256']
        for relative, expected in original_protocol['source_sha256'].items():
            assert sha(suite/'source'/relative) == expected
        assert original['model']['revision'] == meta['model_revision']
        assert original['arm'] == 'ar' and original['batch_size'] == original['physical_nodes'] == 1
        assert not original['pad_decode_rows']
        assert provenance['first_request_completed_unix'] == rows[0]['completed_unix']
        assert provenance['last_request_completed_unix'] == rows[-1]['completed_unix']
        return dict(kind='verified_archived_AR', source=str(source),
                    requests_sha256=provenance['source_sha256']['requests.jsonl'],
                    limitation='Historical measurement, not a new process exit or same-date timing.')

    if run['stack'] == 'official':
        from dflash_source_identity import validate_evaluation_sources
        validate_evaluation_sources(path)
        if meta['method'] == 'dflash':
            loading = read(path/'checkpoint-loading.json')
            assert not any(loading.values()), 'Drafter checkpoint loading was not clean'
    else:
        from loopahead_diagnostic_inputs import validate_replay_sources
        validate_replay_sources(path)
        assert meta['method'] == 'loopahead' and meta['prompt_format'] == 'chat'
        source_suite = ROOT/'results'/('loopfill-latency512-t1-20261108' if meta['temperature'] else 'loopfill-latency512-20261107')
        source_protocol = read(source_suite/'protocol.json')
        assert meta['frozen_source_sha256'] == source_protocol['source_sha256']
        for relative, expected in meta['frozen_source_sha256'].items():
            assert sha(source_suite/'source'/relative) == expected

    launch_path = BASE/f'logs/{path.name}-launcher.json'
    finish_path = BASE/f'logs/{path.name}-finished.json'
    launch = read(launch_path)
    if not finish_path.exists():
        raise ExitReceiptPending(f'No actual exit receipt yet: {finish_path}')
    finish = read(finish_path)
    assert finish['exit_code'] == 0, f'Formal process failed: {finish_path}'
    assert all(finish[k] == v for k, v in launch.items()), 'Exit receipt differs from launch identity'
    assert launch['name'] == path.name and type(launch['pid']) is int and launch['pid'] > 0
    argv = launch['argv']

    def value(flag):
        assert argv.count(flag) == 1, f'Expected exactly one {flag}'
        return argv[argv.index(flag)+1]

    assert Path(value('--out')).resolve() == path.resolve()
    for flag, key in [('--model', 'model'), ('--method', 'method'), ('--dataset', 'dataset')]:
        assert value(flag) == meta[key]
    assert float(value('--temperature')) == meta['temperature']
    lo, hi = map(int, value('--cpus').split('-'))
    assert meta['cpu_affinity'] == list(range(lo, hi+1))
    assert ('--profile' in argv) == bool(meta.get('profiling'))
    assert not any(flag in argv for flag in ['--pilot', '--trace', '--paper-check', '--dev-calibration', '--random-lad'])
    if meta['method'] != 'ar':
        assert Path(value('--checkpoint')).resolve() == Path(meta['checkpoint']).resolve()
    times = [row['finished_unix'] for row in rows]
    assert all(math.isfinite(t) for t in times) and times == sorted(times)
    assert launch['started_unix'] <= times[0] <= times[-1]
    assert times[-1] <= read(path/'complete.json')['finished_unix'] <= finish['finished_unix']
    return dict(kind='actual_successful_process_exit', pid=launch['pid'],
                launch_sha256=sha(launch_path), exit_receipt_sha256=sha(finish_path),
                source_manifest_sha256=sha(path/'source-manifest.json'))
