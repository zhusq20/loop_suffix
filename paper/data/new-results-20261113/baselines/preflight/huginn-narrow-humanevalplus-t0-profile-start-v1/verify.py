"""Verify full HumanEval+ final narrow timing and its live profile successor."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from collect_results import read_run, compare, sha
from result_evidence import validate_run_artifacts
from dflash_source_identity import validate_evaluation_sources
from dflash_training_registry import validate_projected_completion


def read(path):
    return json.loads(path.read_text())


out = Path(__file__).parent
assert not (out/'report.json').exists()
protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
stem = 'dflash-stack-huginn-3.5b-humanevalplus-t0-'
ar = read_run(BASE/f'runs/{stem}ar-v1', protocol)
timed = read_run(BASE/f'runs/{stem}dflash-v1', protocol)
evidence = {label: validate_run_artifacts(run) for label, run in [('ar', ar), ('timed', timed)]}
assert len(ar['rows']) == len(timed['rows']) == 164
agreement = compare(timed, ar)
profile_path = BASE/f'runs/{stem}dflash-profile-v1'
meta = read(profile_path/'metadata.json')
validate_evaluation_sources(profile_path)
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity',
          'checkpoint', 'checkpoint_sha256', 'draft_architecture', 'draft_parameters',
          'training_completion_evidence']
assert all(meta[key] == timed['meta'][key] for key in common)
assert meta['model'] == 'huginn-3.5b' and meta['dataset'] == 'humanevalplus' and meta['temperature'] == 0
assert meta['method'] == 'dflash' and meta['stage'] == 'formal'
assert meta['profiling'] and not meta['reportable'] and not meta['trace']
assert meta['draft_parameters'] == 332621056 and meta['draft_architecture'] == 'target-width-projections-v1'
binding = validate_projected_completion()
assert meta['training_completion_evidence'] == binding
assert meta['checkpoint'] == binding['checkpoint']
assert meta['checkpoint_sha256'] == {'model.safetensors': binding['weights_sha256'],
                                    'config.json': sha(Path(binding['checkpoint'])/'config.json')}
loading = read(profile_path/'checkpoint-loading.json')
assert not any(loading.values()), loading
finish = read(BASE/f'logs/{timed["path"].name}-finished.json')
launch = read(BASE/f'logs/{profile_path.name}-launcher.json')
assert finish['exit_code'] == 0 and finish['pid'] == 330907
assert finish['finished_unix'] <= launch['started_unix'] and launch['pid'] == 336954
pid = launch['pid']
proc = Path('/proc')/str(pid)
stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
assert stat[0] in ['R', 'S', 'D'] and stat[1] == '299577'
assert stat[19] == '42788686'
argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert argv == launch['argv'] and '--profile' in argv
assert argv[argv.index('--gpu')+1] == '4' and argv[argv.index('--cpus')+1] == '48-51'
assert not {'--pilot', '--trace', '--count'} & set(argv)
assert os.sched_getaffinity(pid) == {48, 49, 50, 51}
boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
assert abs(boot+int(stat[19])/os.sysconf('SC_CLK_TCK')-launch['started_unix']) < 3
assert '/usr/lib64/libcuda.so.570.148.08' in proc.joinpath('maps').read_text()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert meta['gpu_uuid'] == 'GPU-a688c63c-924d-a0b0-49f3-d8fbe7df3dda'
assert (meta['gpu_uuid'], pid) in {(uuid.strip(), int(p)) for uuid, p in (line.split(',') for line in apps.splitlines())}
lines = (profile_path/'requests.jsonl').read_bytes().splitlines(keepends=True)
prefix = [json.loads(line) for line in lines[:5] if line.endswith(b'\n')]
assert len(prefix) >= 3, 'Wait for at least three complete profile requests'
for row, reference in zip(prefix, timed['rows']):
    assert row['source_id'] == reference['source_id'] and row['prompt_sha256'] == reference['prompt_sha256']
    assert row['tokens'] == reference['tokens']
    assert row['stats']['target_verifications'] == reference['stats']['target_verifications']
    assert row['stats']['decode_tokens'] == len(row['tokens'])-1
    assert math.isfinite(row['stats']['draft_gpu_s']) and row['stats']['draft_gpu_s'] >= 0
(out/'observed-prefix.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in prefix))
for name in ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']:
    assert (BASE/name).read_bytes() == subprocess.check_output(['git', 'show', f'HEAD:experiments/baselines/{name}'], cwd=ROOT)
assert read(BASE/'formal-matrix-status.json')['completed'] == 29
report = dict(status='complete_final_narrow_timing_live_profile_formal_cell_pending', checked_unix=time.time(),
              timed_summary=timed['summary'], ar_summary=ar['summary'], requests_per_completed_stage=164,
              journal_sha256={label: sha(run['path']/'requests.jsonl') for label, run in [('ar', ar), ('timed', timed)]},
              execution_evidence=evidence, timed_actual_exit=finish, training_binding=binding,
              agreement={k: v for k, v in agreement.items() if k != 'first_divergences'},
              first_divergences=len(agreement['first_divergences']),
              checkpoint_loading=loading, common_config_keys=common,
              profile_metadata_sha256=sha(profile_path/'metadata.json'), profile_launcher=launch,
              actual_parent=299577, actual_start_ticks=stat[19],
              actual_native_driver_map='/usr/lib64/libcuda.so.570.148.08', observed_requests=len(prefix),
              observed_prefix_sha256=sha(out/'observed-prefix.jsonl'),
              observed_prefix_tokens_and_verification_counts_match=True, formal_outputs_unchanged=True,
              verification_source_sha256=sha(Path(__file__)),
              limitations=['Complete 164-request final narrow timing, but the full profile and actual worker exit remain pending.',
                           'Only the observed profile prefix is compared here; no full-task draft fraction is inferred.',
                           'BF16 first divergences require later actual-prefix BF16 and FP32 diagnostics.',
                           'Shared-host timings remain uncorrected; this is not a trained wide-versus-narrow comparison.',
                           'Formal table remains at 29/72 cells.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['status', 'timed_summary', 'ar_summary', 'agreement', 'observed_requests']}))
