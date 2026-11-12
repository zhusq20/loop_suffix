"""Audit saved benchmark evidence without running inference or changing old results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def command(args):
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return dict(args=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def audit_suite(root, relative):
    suite = root / relative
    p = read(suite / 'protocol.json')
    checks = []
    for rel, expected in p['source_sha256'].items():
        # Paths in the old protocol refer to the original checkout. Audit this copy.
        f = suite / 'source' / rel
        observed = sha(f) if f.exists() else None
        checks.append(dict(path=str(f.relative_to(root)), expected=expected,
                           observed=observed, ok=observed == expected))
    inputs = {}
    for dataset, value in p.get('inputs', {}).items():
        f = suite / 'inputs' / (dataset + '.jsonl')
        rows = [json.loads(line) for line in f.read_text().split('\n') if line]
        observed = sha(f)
        assert [r['source_id'] for r in rows] == value['source_ids']
        assert all(hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256']
                   for r in rows)
        checks.append(dict(path=str(f.relative_to(root)), expected=value['sha256'],
                           observed=observed, ok=observed == value['sha256']))
        inputs[dataset] = dict(count=len(rows), sha256=observed,
                               path=str(f.relative_to(root)))
    assert all(c['ok'] for c in checks), 'Frozen source/input mismatch'
    return dict(suite=relative, checks=checks, inputs=inputs,
                settings={k: p[k] for k in ['max_new_tokens', 'cuda_graph', 'batches',
                    'metric', 'timing', 'temperature', 'top_p', 'top_k', 'sampling_seed']
                    if k in p}, protocol_sha256=sha(suite / 'protocol.json'))


def historical_gsm(root):
    suite = root / 'results/loopfill-latency512-20261107'
    reports = []
    for model, depth, eta_ref, gamma_ref in [('huginn-3.5b', 32, 3.47, 4.88),
                                            ('ouro-2.6b', 4, 3.23, 3.59)]:
        data = {}
        for method in ['ar', 'loopspec-dual']:
            run = suite / 'runs' / f'{model}-gsm8k-{method}'
            rows = [json.loads(line) for line in (run / 'requests.jsonl').read_text().split('\n') if line]
            summary = read(run / 'summary.json')
            seconds = sum(r['stats']['decode_s'] for r in rows)
            tokens = sum(r['stats']['tokens'] for r in rows)
            calls = sum(r['stats']['calls'] for r in rows)
            assert len(rows) == summary['completed_requests'] == 300
            assert abs(seconds - summary['decode_s']) < 1e-7
            assert tokens == summary['decode_tokens'] and calls == summary['loop_steps']
            assert abs(depth * tokens / calls - summary['gamma']) < 1e-12
            if method == 'ar':
                assert depth * tokens == calls
            data[method] = dict(rows=rows, decode_s=seconds, tokens=tokens,
                                tok_s=tokens / seconds, gamma=depth * tokens / calls)
        ar, ls = data['ar'], data['loopspec-dual']
        assert [r['source_id'] for r in ar['rows']] == [r['source_id'] for r in ls['rows']]
        exact = sum(a['tokens'] == b['tokens'] for a, b in zip(ar['rows'], ls['rows']))
        eta_latency = ar['decode_s'] / ls['decode_s']
        eta_throughput = ls['tok_s'] / ar['tok_s']
        assert round(eta_latency, 2) == eta_ref and round(ls['gamma'], 2) == gamma_ref
        reports.append(dict(model=model, benchmark='gsm8k', temperature=0, requests=300,
            evidence_kind='recomputed_historical_logs_NOT_new_inference',
            ar_tok_s=ar['tok_s'], loopspec_tok_s=ls['tok_s'],
            eta_latency=eta_latency, eta_throughput=eta_throughput, gamma=ls['gamma'],
            paper_eta=eta_ref, paper_gamma=gamma_ref,
            ar_decode_s=ar['decode_s'], loopspec_decode_s=ls['decode_s'],
            exact_sequence_matches=exact, exact_sequence_match_rate=exact / 300))
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    root = args.repo.resolve()
    out = Path(__file__).resolve().parent / 'preflight'
    suites = [audit_suite(root, 'results/' + s) for s in
              ['loopfill-latency512-20261107', 'loopfill-latency512-t1-20261108',
               'loopfill-table2-20261107-v2']]
    write(out / 'frozen-evidence.json', suites)
    historical = historical_gsm(root)
    write(out / 'historical-gsm8k.json', historical)
    with (out / 'historical-gsm8k.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(historical[0]))
        w.writeheader()
        w.writerows(historical)
    import importlib.metadata
    versions = {}
    for package in ['torch', 'transformers', 'triton', 'flash_attn', 'numpy']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), versions=versions,
        gpu=command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,compute_mode,driver_version,clocks.current.graphics,clocks.current.memory,power.limit', '--format=csv']),
        occupants=command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv']),
        disk=command(['df', '-h', str(root), '/tmp']),
        source_commit=command(['git', '-C', str(root), 'rev-parse', 'HEAD']),
        policy='read-only collection; no GPU clock/power/compute-mode changes')
    write(out / 'environment.json', environment)
    print(json.dumps(dict(source_input_checks=sum(len(s['checks']) for s in suites),
                          all_pass=True, historical=historical), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
