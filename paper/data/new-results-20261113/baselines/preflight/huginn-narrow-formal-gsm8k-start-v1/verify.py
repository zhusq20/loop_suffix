"""Verify complete same-stack AR and the real final narrow drafter startup."""
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
ar_path = BASE/'runs/dflash-stack-huginn-3.5b-gsm8k-t0-ar-v1'
path = BASE/'runs/dflash-stack-huginn-3.5b-gsm8k-t0-dflash-v1'
ar = read_run(ar_path, protocol)
ar_evidence = validate_run_artifacts(ar)
assert len(ar['rows']) == 300
meta = read(path/'metadata.json')
validate_evaluation_sources(path)
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity']
assert all(meta[key] == ar['meta'][key] for key in common)
assert meta['model'] == 'huginn-3.5b' and meta['dataset'] == 'gsm8k' and meta['temperature'] == 0
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
assert ar_exit['exit_code'] == 0 and ar_exit['pid'] == 182221
assert ar_exit['finished_unix'] <= launch['started_unix'] and launch['pid'] == 219749
proc = Path('/proc/219749')
stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
assert stat[0] in ['R', 'S', 'D'] and stat[1] == '182214'
argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert argv == launch['argv']
assert argv[argv.index('--checkpoint')+1] == str(checkpoint)
assert not {'--pilot', '--trace', '--profile', '--count'} & set(argv)
assert os.sched_getaffinity(219749) == {8, 9, 10, 11}
boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
assert abs(boot+int(stat[19])/os.sysconf('SC_CLK_TCK')-launch['started_unix']) < 3
assert '/usr/lib64/libcuda.so.570.148.08' in proc.joinpath('maps').read_text()
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
assert meta['gpu_uuid'] == 'GPU-861ad2de-b899-21ca-707e-44c0a1ce1029'
assert (meta['gpu_uuid'], 219749) in {(uuid.strip(), int(pid)) for uuid, pid in (line.split(',') for line in apps.splitlines())}
lines = (path/'requests.jsonl').read_bytes().splitlines(keepends=True)
prefix = [json.loads(line) for line in lines[:5] if line.endswith(b'\n')]
assert prefix
for row, reference in zip(prefix, ar['rows']):
    assert row['source_id'] == reference['source_id'] and row['prompt_sha256'] == reference['prompt_sha256']
    assert 1 <= len(row['tokens']) <= 512 and row['stats']['decode_tokens'] == len(row['tokens'])-1
(out/'observed-prefix.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in prefix))
names = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
for name in names:
    assert (BASE/name).read_bytes() == subprocess.check_output(['git', 'show', f'HEAD:experiments/baselines/{name}'], cwd=ROOT)
assert read(BASE/'formal-matrix-status.json')['completed'] == 27
report = dict(status='complete_AR_actual_final_narrow_GPU_decode_started_formal_cell_pending', checked_unix=time.time(),
              ar_summary=ar['summary'], ar_requests=300, ar_journal_sha256=sha(ar_path/'requests.jsonl'),
              ar_execution_evidence=ar_evidence, ar_actual_exit=ar_exit, training_binding=binding,
              checkpoint_loading=loading, common_config_keys=common, method_metadata_sha256=sha(path/'metadata.json'),
              method_launcher=launch, actual_parent=182214, actual_start_ticks=stat[19],
              actual_native_driver_map='/usr/lib64/libcuda.so.570.148.08',
              observed_requests=len(prefix), observed_prefix_sha256=sha(out/'observed-prefix.jsonl'),
              formal_outputs_unchanged=True, verification_source_sha256=sha(Path(__file__)),
              limitations=['First live GPU deployment of the full 8886-step narrow model; not a completed benchmark cell.',
                           'Only the observed prefix IDs, prompts, counts and actual runtime were checked here.',
                           'Full 300-request timed run, separate profile and greedy diagnostics remain pending.',
                           'No final speedup or GPU-startup retry count is inferred from this startup check.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['status', 'ar_summary', 'ar_journal_sha256', 'actual_start_ticks', 'observed_requests']}))
