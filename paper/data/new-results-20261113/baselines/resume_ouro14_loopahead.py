"""Finish the missing Ouro-1.4B LAD on one reserved GPU after its current job.

Uses existing teacher, feature, training, calibration, and evaluation programs.
Every child has an independent parent that preserves its actual exit receipt.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
OUT = BASE / 'logs/loopahead-ouro14-resume-v1'
EXTERNAL = Path('/ossfs/workspace/loop_suffix')
HANDOFF = EXTERNAL / 'results/gpu-balanced-queue-20261112-v1/lock_handoff.py'
MODEL = 'ouro-1.4b'
GPU = 3
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(EXTERNAL / 'scripts'))
from loop_queue_recovery import spawn_supervised, UNKNOWN_EXIT
from native_cuda_env import native_cuda_environment
import orchestrate_formal as formal
import orchestrate_appendix as appendix


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(stage, **extra):
    save(OUT / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
         updated_unix=time.time(), gpu=GPU, **extra))


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / 'controller.lock').open('a+') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(OUT / 'plan.json')
        assert plan['controller_sha256'] == sha(Path(__file__))
        assert plan['handoff_sha256'] == sha(HANDOFF)
        assert plan['recovery_helper_sha256'] == sha(EXTERNAL / 'scripts/loop_queue_recovery.py')
        for name, expected in plan['source_sha256'].items():
            assert sha(BASE / name) == expected, name
        for stage, expected in plan['manifest_sha256'].items():
            assert sha(BASE / f'data/manifests/loopahead-{stage}.jsonl') == expected
        environment, driver = native_cuda_environment()
        os.environ.update(environment)
        save(OUT / 'native-driver.json', driver)
        cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in cards.splitlines()}
        uuid = uuids[GPU]
        lease = None
        while lease is None:
            policy = read(BASE / 'configs/gpu-allocation.json')
            assert GPU in policy['allowed_gpus'] and GPU in policy['reserved_for_development']
            assert policy['reservation_owners'][str(GPU)] == str(OUT)
            occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()
            status('waiting_for_current_gpu_job', gpu_uuid=uuid)
            if uuid not in [x.strip() for x in occupied]:
                candidate = open(f'/tmp/loopfill-{uuid}.lock', 'a+')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
                    if uuid not in occupied:
                        lease = candidate
                    else:
                        candidate.close()
                except BlockingIOError:
                    candidate.close()
            if lease is None:
                time.sleep(10)
        try:
            def run_command(name, argv, env=None):
                for source_name, expected in plan['source_sha256'].items():
                    assert sha(BASE / source_name) == expected, source_name
                assert sha(HANDOFF) == plan['handoff_sha256']
                argv = list(map(str, argv))
                child_env = dict(os.environ, **(env or {}))
                child_env.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                if '--gpu' in argv:
                    assert argv[argv.index('--gpu') + 1] == str(GPU)
                    assert argv[1] == '-u'
                    argv.insert(2, str(HANDOFF))
                    child_env['LOOPFILL_RETRY_LOCK_FD'] = str(lease.fileno())
                launch_path = BASE / f'logs/{name}-launcher.json'
                finish_path = BASE / f'logs/{name}-finished.json'
                if finish_path.exists():
                    done = read(finish_path)
                    assert done['exit_code'] == 0 and done['argv'] == argv
                    return
                assert not launch_path.exists(), f'Unfinished prior child requires inspection: {launch_path}'
                with (BASE / f'logs/{name}.log').open('x') as log:
                    child = spawn_supervised(argv, env=child_env, log=log, lock=lease,
                        launcher_path=launch_path, exit_path=finish_path,
                        metadata=dict(name=name, gpu=GPU, resume_plan=str(OUT/'plan.json')),
                        cwd=BASE.parents[1])
                    status('running', child=child.launch, task=name)
                    while (code := child.poll()) is None:
                        time.sleep(5)
                assert code != UNKNOWN_EXIT, f'No actual exit receipt for {name}; retain evidence and investigate'
                assert code == 0, f'{name} exited {code}; see its preserved log'

            formal.run_command = appendix.run_command = run_command
            appendix.command('smoke_loopahead.py', 'loopahead-ouro-1.4b-appendix-fp32-smoke-v1', GPU,
                             ['--dtype', 'float32'], python=appendix.INFER)
            smoke = read(BASE / 'preflight/loopahead-ouro-1.4b-appendix-fp32-smoke-v1/report.json')
            assert smoke['greedy_ar_equal'] and smoke['repeat_exact']
            # One complete shard keeps every approved example while avoiding repeated model loads.
            # Teacher production is eager here; no unvalidated CUDA-graph adoption.
            for stage in ['train', 'dev', 'confirmation']:
                teacher = BASE / f'data/teachers/loopahead-{stage}/{MODEL}/shard0'
                appendix.command('generate_teachers.py', f'teacher-loopahead-{stage}-{MODEL}-shard0-resume-v1', GPU,
                    ['--manifest', BASE/f'data/manifests/loopahead-{stage}.jsonl', '--shard', 0, '--shards', 1,
                     '--max-batch', 32, '--kv-budget-gib', 70], teacher)
                appendix.command('extract_loopahead_features.py', f'features-loopahead-{stage}-{MODEL}-shard0-resume-v1', GPU,
                    ['--teacher-shard', teacher], BASE/f'features/loopahead-{stage}/{MODEL}/shard0')
            appendix.train_lad(GPU, uuids)
            run_command('audit-loopahead-ouro14-resume-v1', [appendix.INFER, '-u', str(BASE/'audit_loopahead_training.py'),
                '--model', MODEL, '--out', str(BASE/'training/loopahead-ouro-1.4b-formal-audit-v1')])
            for temperature in [0, 1]:
                for task in formal.TASKS:
                    formal.worker(dict(kind='loopahead', model=MODEL, dataset=task, temperature=temperature), [GPU], uuids)
            run_command('collect-after-ouro14-lad-resume-v1', ['/opt/conda/bin/python', str(BASE/'collect_results.py'), '--partial'])
            save(OUT/'complete.json', dict(finished_unix=time.time(), completed_lad_cells=12,
                checkpoint=formal.selected(MODEL), limitation='Formal first-divergence diagnostics remain separate.'))
            status('complete')
        finally:
            lease.close()
            path = BASE/'configs/gpu-allocation.json'
            raw = path.read_bytes()
            policy = json.loads(raw)
            if policy.get('reservation_owners', {}).get(str(GPU)) == str(OUT):
                policy['reserved_for_development'] = [g for g in policy['reserved_for_development'] if g != GPU]
                del policy['reservation_owners'][str(GPU)]
                assert path.read_bytes() == raw
                save(path, policy)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(OUT/'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        status('failed_requires_inspection')
        raise
