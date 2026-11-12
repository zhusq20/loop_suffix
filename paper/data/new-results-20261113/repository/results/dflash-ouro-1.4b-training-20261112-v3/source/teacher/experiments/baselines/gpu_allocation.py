"""User-owned GPU allocation, shared by dispatchers and diagnostic queues."""
import json
from pathlib import Path

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')


def policy():
    path = BASE/'configs/gpu-allocation.json'
    value = json.loads(path.read_text()) if path.exists() else {'allowed_gpus': list(range(8))}
    allowed = value['allowed_gpus']
    assert allowed and len(set(allowed)) == len(allowed)
    assert all(type(g) is int and 0 <= g < 8 for g in allowed)
    assert not set(allowed) & set(value.get('released_gpus', []))
    assert set(value.get('reserved_for_development', [])) <= set(allowed)
    return value


def allowed_gpus():
    return policy()['allowed_gpus']


def dispatch_gpus():
    value = policy()
    return [g for g in value['allowed_gpus'] if g not in value.get('reserved_for_development', [])]
