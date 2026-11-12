"""Preserve the new shared stop event without restarting or signalling jobs."""
import gzip
import hashlib
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    assert not (OUT/'report.json').exists()
    csv_path = BASE/'combined-results/results.csv'
    csv_before = csv_path.read_bytes()
    copied = {}

    def copy(relative, label, compress=False):
        source = BASE/relative
        data = source.read_bytes()
        destination = OUT/'raw'/label
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = gzip.compress(data, mtime=0) if compress else data
        with destination.open('xb') as stream:
            stream.write(payload)
        assert source.read_bytes() == data
        assert (gzip.decompress(payload) if compress else payload) == data
        copied[label] = dict(source=str(source), source_sha256=sha(data),
                             snapshot_sha256=sha(payload), compressed=compress)
        return json.loads(data) if source.suffix == '.json' else None

    roots = dict(dflash='logs/appendix-dflash-diagnostics-v1',
                 loopahead='logs/loopahead-math-fp32-continuation-v1')
    for label, root in roots.items():
        for name in ['STOP','status.json','cancellation.json']:
            copy(root+'/'+name, label+'-'+name)
    exit_records = {}
    for label, folder, stem in [
        ('dflash-controller', roots['dflash']+'/resume-after-reconfirmation-v1', 'controller'),
        ('loopahead-controller', roots['loopahead'], 'controller'),
        ('dflash-gsm-worker', roots['dflash']+'/logs', 'table1-ouro-1.4b-gsm8k-t0-greedy')]:
        launch = copy(folder+'/'+stem+'-launcher.json', label+'-launcher.json')
        done = copy(folder+'/'+stem+'-exit.json', label+'-exit.json')
        assert done['exit_code'] == -15 and done['exit_evidence'] == 'supervisor_waitpid'
        for field in ['pid','start_ticks','argv','started_unix','supervisor_identity']:
            assert done[field] == launch[field]
        assert not Path(f"/proc/{done['pid']}").exists()
        exit_records[label] = dict(pid=done['pid'], exit_code=-15,
                                  finished_unix=done['finished_unix'])

    diagnostic = 'logs/ouro14-packed-evaluation-v1/diagnostics/table1-ouro-1.4b-gsm8k-t0'
    traces = copy(diagnostic+'/trace-progress.json', 'dflash-gsm-trace-progress.json.gz', True)
    for name in ['agreement.json','metadata.json']:
        copy(diagnostic+'/'+name, 'dflash-gsm-'+name)
    assert len(traces) == 244 and sum('trace' in case for case in traces) == 53
    assert not (BASE/diagnostic/'complete.json').exists()
    assert not (BASE/diagnostic/'diagnosis.json').exists()
    assert not (BASE/diagnostic/'progress.json').exists()

    status = copy('logs/ouro14-packed-evaluation-v1/status.json', 'measurement-status.json')
    workers = {}
    for name, run in status['running'].items():
        fields = Path(f"/proc/{run['pid']}/stat").read_text().rsplit(')',1)[1].split()
        assert fields[0] not in ('Z','X') and fields[19] == str(run['start_ticks'])
        workers[name] = dict(pid=run['pid'], start_ticks=fields[19], gpu=run['gpu'])
    assert len(workers) == 7 and not status['failures']
    assert csv_path.read_bytes() == csv_before
    external = Path('/ossfs/workspace/loop_suffix/results/output-diagnostics-cancelled-20261113-045608/cancellation.json')
    save(OUT/'report.json', dict(
        status='external_stop_observed_source_unconfirmed', checked_unix=time.time(),
        copied_artifacts=copied, actual_terminal_receipts=exit_records,
        measurement_workers_live=workers, results_csv_sha256=sha(csv_before),
        completed_measurement_cells=67, new_gpu_runs=0, process_signals_sent=0,
        dflash_partial=dict(expected_first_divergences=244, traced_cases=53,
                            completed_common_prefix_BF16_cases=0, completed_FP32_cases=0),
        loopahead_partial=dict(previous_preserved_FP32_cases=262, unexecuted_continuation_cases=167),
        previously_verified_numerical_groups=32, required_numerical_groups=39,
        external_record=dict(path=str(external), sha256=sha(external.read_bytes()),
                             claimed_user_text='不需要做差异诊断实验', actor_or_conversation_verified=False),
        scope='Preserve actual stops and partial evidence. Direct clarification is pending; no scope reduction or restart is inferred from shared files.'))
    print(json.dumps(dict(saved=str(OUT), measurement_workers_live=len(workers),
                          traced_cases=53, stopped_processes=len(exit_records), new_gpu_runs=0)))


if __name__ == '__main__':
    main()
