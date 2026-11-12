"""Keep the actual exit receipt for the CPU-only temporary queue observer."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from orchestrate_formal import run_command

run_command('huginn-math-priority-v1', [
    'taskset', '-c', '80-83', '/opt/conda/bin/python', '-u',
    str(BASE / 'prioritize_huginn_math.py'), '--execute',
])
