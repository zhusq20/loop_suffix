"""Verify complete MBPP+ AR and the live final narrow DFlash timing successor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from collect_results import read_run, sha
from result_evidence import validate_run_artifacts
from dflash_source_identity import validate_evaluation_sources
from dflash_training_registry import validate_projected_completion


def read(path):
    return json.loads(path.read_text())


out = Path(__file__).parent
assert not (out/'report.json').exists()
protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
ar_path = BASE/'runs/dflash-stack-huginn-3.5b-mbppplus-t0-ar-v1'
path = BASE/'runs/dflash-stack-huginn-3.5b-mbppplus-t0-dflash-v1'
ar = read_run(ar_path, protocol)
ar_evidence = validate_run_artifacts(ar)
assert len(ar['rows']) == 378
meta = read(path/'metadata.json')
validate_evaluation_sources(path)
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity']
assert all(meta[key] == ar['meta'][key] for key in common)
assert meta['model'] == 'huginn-3.5b' and meta['dataset'] == 'mbppplus' and meta['temperature'] == 0
assert meta['method'] == 'dflash' and meta['stage'] == 'formal' and meta['reportable']
assert not meta['profiling'] and not meta['trace']
assert meta['draft_parameters'] == 332621056 and meta['draft_architecture'] == 'target-width-projections-v1'
binding = validate_projected_completion()
assert meta['training_completion_evidence'] == binding
assert meta['checkpoint'] == binding['checkpoint']
checkpoint = Path(binding['checkpoint'])
assert meta['checkpoint_sha256'] == {'model.safetensors': binding['weights_sha256'], 'config.json': sha(checkpoint/'config.json')}
loading = read(path/'checkpoint-loading.json')
assert not any(loading.values()), loading
ar_exit = read(BASE/f'logs/{ar_path.name}-finished.json')
launch = read(BASE/f'logs/{path.name}-launcher.json')
assert ar_exit['exit_code'] == 0 and ar_exit['pid'] == 343065
assert ar_exit['finished_unix'] <= launch['started_unix']
pid = launch['pid']
proc = Path('/proc')/str(pid)
stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
assert stat[0] in ['R', 'S', 'D'] and stat[1] == '343056'
argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert argv == launch['argv']
assert argv[argv.index('--gpu')+1] == '4' and argv[argv.index('--cpus')+1] == '48-51'
assert argv[argv.index('--checkpoint')+1] == str(checkpoint)
assert not {'--pilot', '--trace', '--profile', '--count'} & set(argv)
assert os.sched_getaffinity(pid) == {48, 49, 50, 51}
boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
assert abs(boot+int(stat[19])/os.sysconf('SC_CLK_TCK')-launch['started_unix']) < 3
assert '/usr/lib64/libcuda.so.570.148.08' in proc.joinpath('maps').read_text()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert meta['gpu_uuid'] == 'GPU-a688c63c-924d-a0b0-49f3-d8fbe7df3dda'
assert (meta['gpu_uuid'], pid) in {(uuid.strip(), int(pid)) for uuid, pid in (line.split(',') for line in apps.splitlines())}
lines = (path/'requests.jsonl').read_bytes().splitlines(keepends=True)
prefix = [json.loads(line) for line in lines[:5] if line.endswith(b'\n')]
assert len(prefix) >= 3, 'Wait for three complete requests'
for row, reference in zip(prefix, ar['rows']):
    assert row['source_id'] == reference['source_id'] and row['prompt_sha256'] == reference['prompt_sha256']
    assert 1 <= len(row['tokens']) <= 512 and row['stats']['decode_tokens'] == len(row['tokens'])-1
(out/'observed-prefix.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in prefix))
names = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
for name in names:
    assert (BASE/name).read_bytes() == subprocess.check_output(['git', 'show', f'HEAD:experiments/baselines/{name}'], cwd=ROOT)
assert read(BASE/'formal-matrix-status.json')['completed'] == 30
policy = read(BASE/'configs/gpu-allocation.json')
assert policy['allowed_gpus'] == [2, 3, 4, 6] and policy['released_gpus'] == [0, 1, 5, 7]
report = dict(status='complete_AR_actual_final_narrow_GPU_decode_started_formal_cell_pending', checked_unix=time.time(),
              ar_summary=ar['summary'], ar_requests=378, ar_journal_sha256=sha(ar_path/'requests.jsonl'),
              ar_execution_evidence=ar_evidence, ar_actual_exit=ar_exit, training_binding=binding,
              checkpoint_loading=loading, common_config_keys=common, method_metadata_sha256=sha(path/'metadata.json'),
              method_launcher=launch, actual_parent=343056, actual_start_ticks=stat[19],
              actual_native_driver_map='/usr/lib64/libcuda.so.570.148.08',
              observed_requests=len(prefix), observed_prefix_sha256=sha(out/'observed-prefix.jsonl'),
              formal_outputs_unchanged=True, gpu_policy=policy, verification_source_sha256=sha(Path(__file__)),
              limitations=['Live MBPP+ timing of the full 8886-step narrow model; not a completed benchmark cell.',
                           'Only the observed prefix IDs, prompts, counts and actual runtime were checked here.',
                           'Full 378-request timed run, separate profile and greedy diagnostics remain pending.',
                           'No final speedup or GPU-startup retry count is inferred from this startup check.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['status', 'ar_summary', 'ar_journal_sha256', 'actual_start_ticks', 'observed_requests']}))
