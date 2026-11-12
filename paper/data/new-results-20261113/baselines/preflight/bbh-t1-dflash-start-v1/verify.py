"""Check the completed BBH/T1 AR and the actual, still-running DFlash stage."""
import hashlib
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
from dflash_source_identity import validate_evaluation_sources
from result_evidence import validate_run_artifacts


def read(path):
    return json.loads(path.read_text())


out = Path(__file__).parent
assert not (out/'report.json').exists()
outputs = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
before = {name: sha(BASE/name) for name in outputs}
for name in outputs:
    assert (BASE/name).read_bytes() == subprocess.check_output(
        ['git', 'show', f'HEAD:experiments/baselines/{name}'], cwd=ROOT)
assert read(BASE/'formal-matrix-status.json')['completed'] == 25
protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
ar_path = BASE/'runs/dflash-stack-ouro-2.6b-bbh-t1-ar-v1'
df_path = BASE/'runs/dflash-stack-ouro-2.6b-bbh-t1-dflash-v1'
ar = read_run(ar_path, protocol)
ar_evidence = validate_run_artifacts(ar)
assert len(ar['rows']) == 270 and ar['meta']['method'] == 'ar'
assert ar['meta']['model'] == 'ouro-2.6b' and ar['meta']['dataset'] == 'bbh' and ar['meta']['temperature'] == 1
finish = read(BASE/f'logs/{ar_path.name}-finished.json')
meta = read(df_path/'metadata.json')
assert meta['method'] == 'dflash' and not meta['profiling'] and meta['reportable']
assert not (df_path/'complete.json').exists()
validate_evaluation_sources(df_path)
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity']
assert all(meta[key] == ar['meta'][key] for key in common)
checkpoint = BASE/'checkpoints/dflash-ouro-2.6b-formal-v1/checkpoint-009030'
assert Path(meta['checkpoint']) == checkpoint
assert all(sha(checkpoint/name) == expected for name, expected in meta['checkpoint_sha256'].items())
audit_path = BASE/'training/dflash-ouro-2.6b-formal-audit-v1/report.json'
assert sha(audit_path) == 'c19089e7e77c7fa61701414b17a253666137397bf85e29c7aa64644b6b494475'
audit = read(audit_path)
assert audit['status'] == 'complete_training_audit_not_benchmark' and audit['model'] == meta['model']
assert audit['weights_sha256'] == meta['checkpoint_sha256']['model.safetensors']
assert audit['checkpoint_config_sha256'] == meta['checkpoint_sha256']['config.json']
loading = read(df_path/'checkpoint-loading.json')
assert not any(loading.values())
launch = read(BASE/f'logs/{df_path.name}-launcher.json')
assert launch['pid'] == 117735 and finish['finished_unix'] <= launch['started_unix']
proc = Path(f'/proc/{launch["pid"]}')
stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
assert stat[0] in ['R', 'S', 'D'] and stat[19] == '40847532'
argv = proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
assert argv == launch['argv']
assert argv[argv.index('--out')+1] == str(df_path) and argv[argv.index('--checkpoint')+1] == str(checkpoint)
assert argv[argv.index('--gpu')+1] == '2' and argv[argv.index('--cpus')+1] == '8-11'
assert os.sched_getaffinity(launch['pid']) == set(meta['cpu_affinity']) == set(range(8, 12))
assert proc.joinpath('status').read_text().split('PPid:\t')[1].splitlines()[0] == '39190'
assert Path('/proc/39190/task/39190/children').read_text().strip() == str(launch['pid'])
libraries = sorted({line.split()[-1] for line in proc.joinpath('maps').read_text().splitlines() if '/libcuda.so' in line})
assert libraries == ['/usr/lib64/libcuda.so.570.148.08']
env = dict(entry.split('=', 1) for entry in proc.joinpath('environ').read_bytes().decode().split('\0') if '=' in entry)
assert env['LD_LIBRARY_PATH'].split(':')[0] == '/tmp/loopfill-native-driver-570.148.08'
assert env['CUDA_VISIBLE_DEVICES'] == meta['gpu_uuid']
acquisition = read(df_path/'cuda-acquisition.json')
assert acquisition['pid'] == launch['pid'] and acquisition['gpu'] == 2 and acquisition['gpu_uuid'] == meta['gpu_uuid']
contexts = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
assert any(line.split(',')[0].strip() == meta['gpu_uuid'] and line.split(',')[1].strip() == str(launch['pid']) for line in contexts.splitlines())
prefix = (df_path/'requests.jsonl').read_bytes()
rows = [json.loads(line) for line in prefix.splitlines()]
expected = [json.loads(line) for line in (ROOT/'results/loopfill-latency512-20261107/inputs/bbh.jsonl').read_bytes().splitlines()]
assert 0 < len(rows) < 270
assert [row['source_id'] for row in rows] == [row['source_id'] for row in expected[:len(rows)]]
for row, source in zip(rows, expected):
    assert row['prompt_sha256'] == hashlib.sha256(source['prompt'].encode()).hexdigest()
    assert 1 <= len(row['tokens']) <= 512 and row['stats']['decode_tokens'] == len(row['tokens'])-1
allocation = read(BASE/'configs/gpu-allocation.json')
assert allocation['allowed_gpus'] == [2, 3, 4, 6] and allocation['released_gpus'] == [0, 1, 5, 7]
assert before == {name: sha(BASE/name) for name in outputs}
(out/'observed-prefix.jsonl').write_bytes(prefix)
report = dict(status='completed_AR_and_live_DFlash_stage_verified_not_complete_cell', checked_unix=time.time(),
              ar_summary=ar['summary'], ar_journal_sha256=sha(ar_path/'requests.jsonl'), ar_execution_evidence=ar_evidence,
              ar_actual_exit=finish, dflash_launcher=launch, dflash_start_ticks=stat[19],
              common_config_keys=common, metadata_sha256={key: sha(path/'metadata.json') for key, path in [('ar', ar_path), ('dflash', df_path)]},
              source_manifest_sha256={key: sha(path/'source-manifest.json') for key, path in [('ar', ar_path), ('dflash', df_path)]},
              checkpoint_sha256=meta['checkpoint_sha256'], checkpoint_loading=loading, training_audit_sha256=sha(audit_path),
              loaded_libcuda=libraries, ld_library_path=env['LD_LIBRARY_PATH'], cuda_acquisition=acquisition, gpu_contexts=contexts,
              observed_requests=len(rows), observed_prefix_sha256=sha(out/'observed-prefix.jsonl'), allocation=allocation,
              completed_cells=25, results_unchanged=before, verification_source_sha256=sha(Path(__file__)),
              limitations=['DFlash timing and its separate full-task profile remain incomplete; no speed ratio is reported for this cell.',
                           'Actual normal CUDA acquisition and driver mapping verified; no real busy-retry or old-parent self-reexec is claimed.',
                           'Shared-host timing variation is retained without correction.',
                           'The original failed startup remains in its recovery archive.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: report[key] for key in ['status', 'ar_summary', 'ar_journal_sha256', 'observed_requests', 'loaded_libcuda', 'completed_cells']}))
