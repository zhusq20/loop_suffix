import sys
from pathlib import Path
BASE=Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0,str(BASE))
from orchestrate_formal import run_command
run_command('huginn-dflash-dev-compare-v1',[
 '/tmp/loopfill-dflash-venv/bin/python','-u',str(BASE/'compare_huginn_dflash_dev.py'),
 '--plan',str(BASE/'preflight/huginn-dflash-dev-plan-v1/plan.json'),
 '--narrow',str(BASE/'checkpoints/dflash-huginn-narrow-pilot-v1/checkpoint-000060'),
 '--wide',str(BASE/'checkpoints/dflash-huginn-3.5b-formal-v1/checkpoint-004000'),
 '--gpu','3','--cpus','12-15','--out',str(BASE/'preflight/huginn-dflash-dev-compare-v1')])
