"""Gate and run the separate narrow Huginn training on two of our four GPUs.

An undertrained pilot's speed is not a fair capacity comparison. The gates here
require real deployment and data/artifact checks, then train from scratch using
the full approved recipe. This queue never interrupts existing evaluations.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from audit_dflash_training import sha
from gpu_allocation import allowed_gpus, policy
from orchestrate_formal import run_command, save, CPUS
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
GPUS = [3, 6]
OUT = BASE/'logs/huginn-narrow-formal-queue-v1'


def load(path):
    return json.loads(path.read_text())


def feasibility_gate():
    dev = BASE/'preflight/huginn-dflash-dev-compare-v1'
    receipt = BASE/'logs/huginn-dflash-dev-compare-v1-finished.json'
    if not receipt.exists():
        launcher = load(BASE/'logs/huginn-dflash-dev-compare-v1-launcher.json')
        proc = Path(f'/proc/{launcher["pid"]}')
        assert proc.exists(), 'Development process disappeared without an exit receipt'
        assert proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1] == launcher['argv']
        return None
    assert load(receipt)['exit_code'] == 0, 'Development comparison failed; preserve evidence'
    done = load(dev/'complete.json')
    assert done['requests'] == 20 and sha(dev/'report.json') == done['report_sha256']
    for method, digest in done['journals'].items():
        assert sha(dev/f'{method}.jsonl') == digest
    report = load(dev/'report.json')
    assert report['status'] == 'completed_feasibility_only' and not report['reportable']
    assert all(v['profile_all_outputs_identical'] for v in report['comparisons'].values())
    metadata = load(dev/'metadata.json')
    assert sha(dev/'metadata.json') == report['metadata_sha256']
    assert metadata['torch'] == '2.6.0+cu126' and metadata['transformers'] == '4.57.1'
    pilot_audit = BASE/'training/dflash-huginn-narrow-pilot-audit-v1/report.json'
    audit = load(pilot_audit)
    assert audit['status'] == 'complete_training_audit_not_benchmark' and audit['scope'] == 'pilot'
    assert audit['epochs'] == 1 and audit['metrics']['updates'] == 60
    assert audit['actual_exit_code'] == 0 and audit['weights']['parameters'] == 332621056
    assert sha(Path(audit['checkpoint'])/'model.safetensors') == audit['weights_sha256']
    assert metadata['checkpoints']['narrow']['hashes']['model.safetensors'] == audit['weights_sha256']
    plan = BASE/'preflight/huginn-dflash-dev-plan-v1/plan.json'
    assert metadata['plan_sha256'] == sha(plan)
    return dict(dev_report_sha256=sha(dev/'report.json'), pilot_audit_sha256=sha(pilot_audit),
        dev_plan_sha256=sha(plan), comparisons=report['comparisons'],
        reason='Real target training, checkpoint audit and deployment succeeded. Complete the approved50k/6epoch recipe to assess the smaller capacity; pilot speed is not an equal-training architecture comparison.',
        configuration='5layers,H2048,FF5632,16heads,block16,target-width-projections-v1',
        parameters=332621056, from_scratch=True, reuse_teacher_answers=True)


def export(root):
    destination = BASE/'training'/root.name
    destination.mkdir(exist_ok=False)
    for path in root.glob('*.json*'):
        shutil.copy2(path, destination/path.name)
    shutil.copytree(root/'source', destination/'source')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(dict(gpus=GPUS, gate='pilot audit + heldout20 deployment',
            probe='2rank micro4 accumulation4 full longest32 one-update DDP probe',
            train='50k teachers/47387 eligible/6epochs/8886updates/global32, fresh initialization')))
        return
    assert set(GPUS) <= set(allowed_gpus())
    assert set(GPUS) <= set(policy()['reserved_for_development'])
    os.sched_setaffinity(0, set(range(80, 84)))
    with open('/tmp/loopfill-projected-huginn-training.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        freeze_sources(OUT, [Path(__file__), BASE/'train_projected_dflash.py', BASE/'audit_projected_dflash_training.py',
            BASE/'audit_dflash_training.py', BASE/'orchestrate_formal.py', BASE/'gpu_allocation.py',
            BASE/'configs/gpu-allocation.json', BASE/'configs/dflash-huginn-narrow-2048-v1.json',
            BASE/'dflash_impl/projected_draft.py', BASE/'dflash_impl/target.py', BASE/'provenance.py'])
        uuids = {int(i): u.strip() for i, u in (line.split(',') for line in subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
        try:
            while (gate := feasibility_gate()) is None:
                save(OUT/'status.json', dict(stage='waiting_actual_dev_exit', updated_unix=time.time()))
                time.sleep(15)
            save(OUT/'feasibility-gate.json', dict(gate, passed_unix=time.time()))
            while True:
                assert set(GPUS) <= set(allowed_gpus())
                assert set(GPUS) <= set(policy()['reserved_for_development'])
                state = load(BASE/'logs/formal-controller-v1/status.json')
                occupied = {g for job in state['running'].values() for g in job['gpus']}
                busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
                if not set(GPUS)&occupied and all(uuids[g] not in busy for g in GPUS):
                    break
                save(OUT/'status.json', dict(stage='waiting_existing_evaluation_to_finish',
                    required_gpus=GPUS, running=state['running'], updated_unix=time.time()))
                time.sleep(15)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids[g] for g in GPUS),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            common = ['/tmp/loopfill-specforge-venv/bin/torchrun', '--standalone', '--nproc_per_node=2',
                str(BASE/'train_projected_dflash.py'), '--model', 'huginn-3.5b',
                '--teachers', str(BASE/'data/teachers/dflash-train/huginn-3.5b'),
                '--cpu-groups', ','.join(CPUS[g] for g in GPUS), '--micro-batch', '4',
                '--accumulation', '4', '--checkpoint-layers']
            smoke_name = 'dflash-huginn-narrow-ddp2-smoke-v1'
            smoke = BASE/'checkpoints'/smoke_name
            save(OUT/'status.json', dict(stage='ddp_probe', gpus=GPUS, updated_unix=time.time()))
            run_command(smoke_name, common+['--out', str(smoke), '--epochs', '1', '--save-every', '1', '--smoke'], env)
            metrics = load(smoke/'progress.json')
            assert metrics['step'] == 1 and metrics['peak_allocated_bytes'] <= 120*1024**3
            assert load(smoke/'metadata.json')['global_batch'] == 32
            for rank in range(2):
                assert load(smoke/f'checkpoint-000001/complete.json')['step'] == 1
                assert (smoke/f'checkpoint-000001/rng-rank{rank}.pt').stat().st_size > 0
                assert all(json.loads(l)['status'] == 'complete' for l in (smoke/f'reservation-cost-rank{rank}.jsonl').open())
            export(smoke)
            name = 'dflash-huginn-narrow-formal-v1'
            root = BASE/'checkpoints'/name
            save(OUT/'status.json', dict(stage='formal_training', gpus=GPUS, steps=8886, updated_unix=time.time()))
            run_command(name, common+['--out', str(root), '--epochs', '6', '--save-every', '500'], env)
            done, meta = load(root/'complete.json'), load(root/'metadata.json')
            assert done['step'] == meta['total_updates'] == 8886 and done['epochs'] == 6 and not done['smoke']
            assert meta['training_scope'] == 'full50000' and meta['examples'] == 47387
            assert meta['world_size'] == 2 and meta['global_batch'] == 32
            export(root)
            save(OUT/'status.json', dict(stage='cpu_audit', updated_unix=time.time()))
            audit_out = BASE/'training/dflash-huginn-narrow-formal-audit-v1'
            run_command('audit-dflash-huginn-narrow-formal-v1', ['taskset', '-c', '80-83',
                '/tmp/loopfill-specforge-venv/bin/python', str(BASE/'audit_projected_dflash_training.py'),
                '--scope', 'formal', '--out', str(audit_out)], dict(env, CUDA_VISIBLE_DEVICES=''))
            assert load(audit_out/'report.json')['scope'] == 'formal'
            save(OUT/'complete.json', dict(checkpoint=load(root/'latest.json')['checkpoint'],
                audit_sha256=sha(audit_out/'report.json'), finished_unix=time.time()))
            # Only release our temporary scheduling reservations. The user's
            # allowed/released GPU allocation and wide-training deferral remain.
            allocation = policy()
            allocation['reserved_for_development'] = [g for g in allocation['reserved_for_development'] if g not in GPUS]
            allocation['updated_unix'] = time.time()
            save(BASE/'configs/gpu-allocation.json', allocation)
        except Exception:
            save(OUT/'failure.json', dict(error=traceback.format_exc(), finished_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
