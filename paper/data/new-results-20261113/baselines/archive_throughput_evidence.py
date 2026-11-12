"""Archive the twenty completed extra throughput measurements, including caveats.

Only copies and verifies immutable evidence. Original paths, arithmetic results,
and the ten LoopAhead admission-order mismatches remain unchanged.
"""
import argparse
import csv
from pathlib import Path
import shutil
import time

from collect_throughput_results import DF, LA, OLD
from packed_dflash_evidence import read, save, sha

BASE = Path(__file__).resolve().parent
ROOTS = {'external-results': Path('/ossfs/workspace/loop_suffix/results'),
         'workspace': BASE.parents[1]}


def archive(out):
    assert not out.exists(), 'Preserve existing archive; choose a fresh output directory.'
    evidence_path = BASE/'combined-results/throughput-evidence.json'
    csv_path = BASE/'combined-results/throughput-results.csv'
    evidence = read(evidence_path)
    assert evidence['measured_cells'] == 20 and evidence['requests'] == 9600
    assert not evidence['full_protocol_alignment']
    assert evidence['aligned_reference_order_cells'] == evidence['unaligned_reference_order_cells'] == 10
    assert sha(csv_path) == evidence['results_sha256']
    assert sha(BASE/'collect_throughput_results.py') == evidence['source_sha256']
    before = sha(evidence_path)
    with csv_path.open() as stream:
        results = list(csv.DictReader(stream))
    out.mkdir(parents=True)
    files = {}

    def copy(path, expected=None):
        path = Path(path)
        if str(path) in files:
            assert expected is None or files[str(path)]['sha256'] == expected
            return
        locations = [(name,path.relative_to(root)) for name,root in ROOTS.items()
                     if path.is_relative_to(root)]
        assert len(locations) == 1 and path.is_file() and not path.is_symlink(), path
        name, relative = locations[0]
        assert '..' not in relative.parts and path.suffix not in {'.pt','.pth','.safetensors'}
        digest = sha(path)
        assert expected is None or digest == expected, path
        dest = out/'files'/name/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as source, dest.open('xb') as target:
            shutil.copyfileobj(source,target,length=8*1024*1024)
        assert sha(dest) == sha(path) == digest, path
        files[str(path)] = dict(archived=str(dest.resolve()),sha256=digest,bytes=dest.stat().st_size)

    copy(evidence_path,before)
    copy(csv_path,evidence['results_sha256'])
    copy(BASE/'combined-results/throughput-summary.md',evidence['summary_sha256'])
    copy(BASE/'collect_throughput_results.py',evidence['source_sha256'])
    copy(Path(__file__))
    copy(BASE/'packed_dflash_evidence.py')
    for record in evidence['audit_reports'].values():
        copy(record['path'],record['sha256'])
    audit_scripts = [BASE/'audit_completed_dflash_throughput.py',
        BASE/'preflight/user-resume-eight-gpu-v1/audit_loopahead_throughput.py']
    for path, method in zip(audit_scripts,['dflash','loopahead'],strict=True):
        copy(path,read(evidence['audit_reports'][method]['path'])['source_sha256'])
    for root in [DF,LA,OLD]:
        protocol = read(root/'protocol.json')
        copy(root/'protocol.json')
        for name,digest in protocol['source_sha256'].items():
            copy(root/'source'/name,digest)
        for name,digest in protocol.get('reference_sha256',{}).items():
            copy(root/name,digest)
        if (root/'jobs.json').exists():
            copy(root/'jobs.json')
    historical = evidence['historical_measurements']
    copy(historical['path'],historical['sha256'])
    for model,record in evidence['input_alignment'].items():
        assert record['requests'] == 480 and not record['identical_actual_admission_order']
        for field in ['loopahead_input','dflash_input']:
            copy(record[field]['path'],record[field]['sha256'])
        for item in record['historical_job_evidence'].values():
            copy(item['path'],item['sha256'])
    cells = {}
    for row in results:
        key = f'{row["method"]}/{row["model"]}/b{row["batch"]}'
        cell = evidence['cells'][key]
        assert int(row['actual_exit_code']) == cell['actual_exit_code'] == 0
        assert int(row['requests']) == 480 and int(row['max_new_tokens']) == 1024
        aligned = row['method'] == 'dflash'
        assert row['order_matches_historical_reference'] == str(aligned)
        if not aligned:
            assert row['status'] == 'measured_order_mismatch_not_aligned_baseline'
        for path,digest in cell['artifacts'].items():
            copy(path,digest)
        run = Path(row['run'])
        if row['method'] == 'loopahead':
            metadata = read(run/'metadata.json')
            for name,digest in metadata['source_sha256'].items():
                copy(Path(cell['actual_source_root'])/name,digest)
        exits = [Path(p) for p in cell['artifacts'] if p.endswith(('-exit.json','-finished.json'))]
        assert len(exits) == 1
        path = exits[0]
        receipt = read(path)
        assert receipt['exit_code'] == 0
        stem = path.name.removesuffix('-exit.json').removesuffix('-finished.json')
        launcher_path = path.with_name(stem+'-launcher.json')
        launch_source = 'start/PID/argv fields retained in actual exit receipt; no separate launcher'
        if receipt.get('supervisor_identity'):
            argv = receipt['supervisor_identity']['argv']
            spec_path = Path(argv[argv.index('--spec')+1])
            spec = read(spec_path)
            copy(spec_path)
            direct = Path(spec['exit_path'])
            original = read(direct)
            # The suite adds status fields and its later observation time.
            # Preserve both actual receipts rather than replacing either time.
            assert original['exit_code'] == receipt['exit_code'] == 0
            assert all(receipt[k] == v for k,v in original.items() if k != 'finished_unix')
            assert receipt['started_unix'] <= original['finished_unix'] <= receipt['finished_unix']
            copy(direct)
            launcher_path = Path(spec['launcher_path'])
        if launcher_path.exists():
            launch = read(launcher_path)
            assert all(receipt[k] == v for k,v in launch.items())
            copy(launcher_path)
            launch_source = str(launcher_path)
        log = path.with_name(stem+'.log')
        if log.exists():
            copy(log)
        for name in ['preflight.json','progress.json','cuda-acquisition.json','reservation-cost.json','source-manifest.json']:
            if (run/name).exists():
                copy(run/name)
        cells[key] = dict(run=str(run),actual_exit=str(path),actual_exit_code=0,
            launcher_evidence=launch_source,order_matches_historical_reference=aligned,
            status=row['status'])
    assert len(cells) == len(results) == len(evidence['cells']) == 20
    assert sha(evidence_path) == before and sha(csv_path) == evidence['results_sha256']
    manifest = dict(status='completed_extra_throughput_evidence_archived',cells=cells,files=files,
        bytes=sum(v['bytes'] for v in files.values()),measured_cells=20,requests=9600,
        aligned_reference_order_cells=10,unaligned_reference_order_cells=10,full_protocol_alignment=False,
        source_evidence_sha256=before,source_results_sha256=evidence['results_sha256'],
        archiver_sha256=sha(Path(__file__)),checked_unix=time.time(),new_gpu_measurements=0,
        limitations=['The ten LoopAhead admission-order mismatches remain; copies do not make their speedup ratios aligned.',
                     'These are extra max1024 continuous-throughput measurements, separate from the 72-cell max512 matrix.',
                     'Original absolute paths remain inside artifacts; use this mapping to locate archived files.',
                     'Target assets, trained weights and environments retain their separately recorded locations.',
                     'Historical AR/LoopFill references, shared-host timing and single measurements retain their original limitations.'])
    save(out/'manifest.json',manifest)
    print(dict(cells=20,files=len(files),bytes=manifest['bytes'],new_gpu_measurements=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    archive(parser.parse_args().out.resolve())
