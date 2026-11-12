import json
import os
from pathlib import Path
import shutil
import sys
import time

BASE=Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0,str(BASE))
from orchestrate_formal import run_command,save
from gpu_allocation import allowed_gpus
assert 3 in allowed_gpus()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-bb9dac43-6fce-fc43-0f10-8f64a80742c3',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
common=['/tmp/loopfill-specforge-venv/bin/python','-u',str(BASE/'train_projected_dflash.py'),'--model','huginn-3.5b','--teachers',str(BASE/'data/teachers/dflash-train/huginn-3.5b'),'--cpu-groups','12-15','--epochs','1','--micro-batch','4','--checkpoint-layers']
smoke_name='dflash-huginn-narrow-micro4-smoke-v1';smoke=BASE/'checkpoints'/smoke_name
run_command(smoke_name,common+['--out',str(smoke),'--accumulation','1','--smoke'],env)
metrics=json.loads((smoke/'progress.json').read_text());assert metrics['step']==1 and metrics['peak_allocated_bytes']<=120*1024**3
name='dflash-huginn-narrow-pilot-v1';out=BASE/'checkpoints'/name
save(BASE/'logs/huginn-narrow-pilot-gate-v1.json',dict(status='micro4_memory_passed_pilot_starting',peak_allocated_bytes=metrics['peak_allocated_bytes'],smoke_update_wall_s=metrics['update_wall_s'],gpu=3,global_batch=32,expected_updates=60,started_unix=time.time()))
run_command(name,common+['--out',str(out),'--accumulation','8','--save-every','50','--selection-manifest',str(BASE/'data/manifests/dflash-pilot.jsonl')],env)
meta=json.loads((out/'metadata.json').read_text());done=json.loads((out/'complete.json').read_text())
assert meta['training_scope']=='pilot2000' and meta['global_batch']==32 and meta['examples']==1893
assert done['step']==meta['total_updates']==60 and not done['smoke']
export=BASE/'training'/name;export.mkdir(exist_ok=False)
for path in out.glob('*.json*'):shutil.copy2(path,export/path.name)
shutil.copytree(out/'source',export/'source')
save(BASE/'logs/huginn-narrow-pilot-complete-v1.json',dict(status='pilot_training_complete_requires_heldout_evaluation',checkpoint=json.loads((out/'latest.json').read_text())['checkpoint'],finished_unix=time.time()))
