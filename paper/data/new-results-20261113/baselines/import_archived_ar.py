"""Preserve the paper's same-host AR measurements with explicit provenance.

No GPU work or new timing is performed. Original journals are copied byte for
byte; the small metadata adapter lets the formal collector validate them.
"""
import json
from pathlib import Path
import shutil
import subprocess
import time

from collect_results import BASE,ROOT,MODELS,TASKS,sha,read_run


def main():
    current={u.strip().removeprefix('GPU-') for u in subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()}
    records=[]
    for temp in [0,1]:
        suite=ROOT/('results/loopfill-latency512-t1-20261108' if temp else 'results/loopfill-latency512-20261107')
        protocol=json.loads((suite/'protocol.json').read_text())
        for rel,expected in protocol['source_sha256'].items():
            assert sha(suite/'source'/rel)==expected
        for model in MODELS:
            for task in TASKS:
                source=suite/f'runs/{model}-{task}-ar'
                old=json.loads((source/'metadata.json').read_text())
                done=json.loads((source/'complete.json').read_text())
                rows=[json.loads(l) for l in (source/'requests.jsonl').open()]
                assert sha(source/'requests.jsonl')==done['requests_sha256']
                assert old['model']['model_id']==model and old['model']['revision']==protocol['models'][model]['revision']
                assert old['gpu_uuid'].removeprefix('GPU-') in current
                assert old['arm']=='ar' and old['batch_size']==old['physical_nodes']==1
                assert not old['pad_decode_rows'] and old['source_sha256']==protocol['source_sha256']
                sampling=old.get('sampling',dict(temperature=0.,top_p=1.,top_k=0,sampling_seed=0))
                assert sampling==dict(temperature=float(temp),top_p=1.,top_k=0,sampling_seed=0)
                source_hashes={name:sha(source/name) for name in ['metadata.json','requests.jsonl','summary.json','complete.json']}
                provenance=dict(kind='archived_same_host_AR',source=str(source.resolve()),
                    source_sha256=source_hashes,imported_unix=time.time(),
                    first_request_completed_unix=rows[0]['completed_unix'],
                    last_request_completed_unix=rows[-1]['completed_unix'],
                    new_measurement=False,reason='Preserve original LoopFill table denominator on verified identical GPU host and frozen software/protocol')
                meta=dict(model=model,model_revision=old['model']['revision'],method='ar',dataset=task,
                    input_sha256=old['inputs_sha256'],count=len(rows),full_count=protocol['inputs'][task]['count'],
                    temperature=float(temp),top_p=1.,top_k=0,seed=0,dtype=old['dtype'],
                    max_new_tokens=old['max_new_tokens'],max_seq=old['max_seq'],stop_audit=old['stop_audit'],
                    gpu_uuid='GPU-'+old['gpu_uuid'].removeprefix('GPU-'),gpu=old['gpu'],
                    cpu_affinity=old['cpu_affinity'],torch=old['torch'],transformers=old['transformers'],
                    graph=old['cuda_graph'],compile=old['torch_compile'],physical_nodes=1,
                    reportable=True,stage='formal',frozen_source_sha256=old['source_sha256'],
                    measurement_provenance=provenance)
                out=BASE/f'runs/archived-ar-{model}-{task}-t{temp}'
                out.mkdir(parents=True,exist_ok=False)
                for name in ['requests.jsonl','summary.json','complete.json']:
                    shutil.copy2(source/name,out/name)
                shutil.copy2(source/'metadata.json',out/'original-metadata.json')
                (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
                run=read_run(out,protocol)
                records.append(dict(model=model,dataset=task,temperature=temp,summary=run['summary'],**provenance))
    report=dict(imported_cells=len(records),records=records,
        limitation='AR and new method timings are from different dates. Full300 same-host GSM8K AR/LoopSpec replay is recorded separately; no unmeasured drift correction is applied.')
    (BASE/'preflight/archived-ar-import.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(imported_cells=len(records),new_gpu_measurements=0)))


if __name__=='__main__':main()
