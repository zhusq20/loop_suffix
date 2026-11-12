"""Resume independently verifiable numerical checks without an all-72 gate."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from native_cuda_env import native_cuda_environment
from packed_dflash_evidence import read, save, sha
from orchestrate_packed_dflash_supplements import cards, occupied, cpus


def verify_plan(path):
    plan = read(path)
    for name, expected in plan['source_sha256'].items():
        assert sha(path.parent / 'source' / name) == expected, name
    base = Path(plan['base'])
    for name, expected in plan['legacy_source_sha256'].items():
        assert sha(base / name) == expected, name
    return plan


def validate_group(plan, group):
    base = Path(plan['base'])
    sys.path.insert(0, str(base))
    folder = Path(group['out'])
    if group['kind'] == 'development':
        from diagnose_huginn_dflash_dev import validate
        return validate(group['label'], folder)
    if group['kind'] == 'legacy':
        from orchestrate_greedy_diagnostics import validate
        return validate(group['row'], folder, group['agreement'])
    from packed_dflash_evidence import check_run, compare
    supplement = Path(plan['supplement_plan']).parent
    original = read(plan['supplement_plan'])
    job = next(j for j in original['jobs'] if j['id'] == group['job'])
    ar = check_run(supplement / f"runs/{job['id']}-ar", original['suite'], job, stage='ar')
    method = check_run(ar['metadata']['timed_run']['path'], original['suite'], job)
    expected = compare(method['rows'], ar['rows'])
    meta, done = read(folder / 'metadata.json'), read(folder / 'complete.json')
    cases = read(folder / 'diagnosis.json')
    assert sha(folder / 'diagnosis.json') == done['diagnosis_sha256']
    assert read(folder / 'agreement.json') == expected
    assert meta['checkpoint_sha256'] == method['metadata']['checkpoint_sha256']
    assert meta['source_journals'] == {name:dict(path=r['path'],sha256=r['hashes']['requests.jsonl'])
                                    for name,r in [('ar',ar),('dflash',method)]}
    assert done['all_bf16_divergences_reproduced'] and done['all_fp32_argmax_agree']
    assert done['cases'] == len(cases) == len(expected['first_divergences'])
    assert [(c['source_id'],c['position']) for c in cases] == [(c['source_id'],c['position']) for c in expected['first_divergences']]
    assert all('top2_probability_margin' in c[label][mode] for c in cases
               for label in ['bf16','fp32'] for mode in ['ar','dflash'])
    return dict(method='dflash', model=job['model'], benchmark=job['dataset'],
                backend='packed', cases=len(cases), folder=str(folder), result_sha256=sha(folder/'diagnosis.json'))


def main(path):
    plan = verify_plan(path)
    root = path.parent
    sys.path.insert(0, str(root / 'source'))
    from loop_queue_recovery import ObservedWorker, UNKNOWN_EXIT, spawn_supervised
    with (root / 'controller.lock').open('a+') as lock_controller:
        fcntl.flock(lock_controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env, driver = native_cuda_environment()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        save(root/'native-driver.json', driver)
        state = read(root/'dispatch.json') if (root/'dispatch.json').exists() else {}
        groups = {g['id']:g for g in plan['groups']}
        steps = {step['id']:dict(step,group=g['id']) for g in plan['groups'] for step in g['steps']}
        active, validated = {}, dict(plan['reused_groups'])
        if (root/'validated.json').exists():
            validated.update(read(root/'validated.json'))
        for key,step in steps.items():
            launch = root/f'logs/{key}-launcher.json'
            if launch.exists() and state.get(key,{}).get('outcome') not in ('complete','failed','exit_unavailable'):
                record = read(launch)
                state[key] = dict(record,outcome='running')
                active[key] = dict(process=ObservedWorker(record,root/f'logs/{key}-exit.json'),lock=None,log=None)
        while True:
            for key,item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                for name in ['lock','log']:
                    if item[name] is not None:item[name].close()
                state[key].update(outcome='complete' if code == 0 else 'exit_unavailable' if code == UNKNOWN_EXIT else 'failed',
                                  exit_code=None if code == UNKNOWN_EXIT else code, finished_unix=time.time())
                if code == 0:
                    if not Path(steps[key]['marker']).exists():
                        state[key].update(outcome='failed', error='Worker exited zero without its required diagnostic artifact')
                del active[key]
                save(root/'dispatch.json',state)
            for key,g in groups.items():
                if key not in validated and all(state.get(s['id'],{}).get('outcome') == 'complete' for s in g['steps']):
                    try:
                        validated[key] = dict(outcome='validated', report=validate_group(plan,g))
                    except Exception:
                        validated[key] = dict(outcome='needs_inspection',error=traceback.format_exc())
                    save(root/'validated.json',validated)
            parent = read(Path(plan['balanced_queue'])/'status.json')
            supplements = read(Path(plan['supplement_status_root'])/'status.json')
            supplement_state = read(Path(plan['supplement_status_root'])/'dispatch.json')
            pending = []
            for key,step in steps.items():
                if key in state or step['group'] in validated:continue
                if not all(state.get(k,{}).get('outcome') == 'complete' for k in step.get('requires',[])):continue
                g = groups[step['group']]
                if g['kind'] == 'packed' and supplement_state.get(g['job']+'-ar',{}).get('outcome') != 'complete':continue
                pending.append(key)
            # Measurement work keeps priority; numerical replays use free tail capacity.
            priority_pending = parent['pending'] or supplements['ready']
            if not priority_pending and not (root/'STOP').exists():
                policy = read(plan['gpu_policy'])
                allowed = set(policy['allowed_gpus']) - set(policy.get('reserved_for_development',[]))
                busy = occupied()
                for gpu,(uuid,mode) in sorted(cards().items()):
                    if gpu not in allowed or uuid in busy or mode != 'Exclusive_Process' or not pending:continue
                    lease = open(f'/tmp/loopfill-{uuid}.lock','a+')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        lease.close();continue
                    if uuid in occupied():lease.close();continue
                    verify_plan(path)
                    key = pending.pop(0);step = steps[key]
                    assert not Path(step['out']).exists(), f'Preserve and inspect partial diagnostic before retry: {key}'
                    argv = [str(x).replace('{GPU}',str(gpu)).replace('{CPUS}',cpus(gpu)).replace('{LOCK_FD}',str(lease.fileno())) for x in step['argv']]
                    child_env = dict(env,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=os.pathsep.join([str(root/'source'),plan['base']]))
                    if step.get('legacy_lock'):
                        argv.insert(2,str(root/'source/lock_handoff.py'))
                        child_env['LOOPFILL_RETRY_LOCK_FD'] = str(lease.fileno())
                    argv = ['taskset','--cpu-list',cpus(gpu),*argv]
                    log = (root/f'logs/{key}.log').open('x')
                    child = spawn_supervised(argv,env=child_env,log=log,lock=lease,
                        launcher_path=root/f'logs/{key}-launcher.json',exit_path=root/f'logs/{key}-exit.json',
                        metadata=dict(group=step['group'],step=key,gpu=gpu,plan_sha256=sha(path)),cwd=plan['base'])
                    state[key] = dict(child.launch,outcome='running')
                    active[key] = dict(process=child,lock=lease,log=log)
                    busy.add(uuid)
                    save(root/'dispatch.json',state)
            save(root/'dispatch.json',state)
            failures = {k:v['outcome'] for k,v in state.items() if v['outcome'] not in ['running','complete']}
            save(root/'status.json',dict(stage='running' if active else 'waiting_for_measurement_priority',
                controller_pid=os.getpid(),updated_unix=time.time(),expected_groups=len(groups),
                validated_groups=list(validated),running={k:state[k] for k in active},ready=pending,
                priority_pending=priority_pending,failures=failures))
            if len(validated) == len(groups):
                save(root/'complete.json',dict(groups=validated,finished_unix=time.time(),
                    status='complete' if all(v['outcome']=='validated' for v in validated.values()) else 'needs_inspection',
                    limitation=plan['limitation']))
                return
            if not active and all(key in validated or any(s['id'] in failures for s in g['steps'])
                                  for key,g in groups.items()):
                save(root/'complete.json',dict(status='needs_inspection',groups=validated,
                    failures=failures,finished_unix=time.time()))
                return
            time.sleep(10)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    try:main(args.plan.resolve())
    except Exception:
        save(args.plan.parent/'failure.json',dict(error=traceback.format_exc(),failed_unix=time.time()))
        raise
