"""Explicit, recorded adoption switch for future teacher jobs only."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent


def teacher_backend_args(model):
    adoption=BASE/'configs/teacher-graph-adoption-v1.json'
    specific=BASE/f'configs/teacher-graph-adoption-{model}-v1.json'
    if specific.exists():adoption=specific
    if adoption.exists():
        record=json.loads(adoption.read_text())
        if model in record['enabled_models']:
            proof=Path(record['proof'])
            assert hashlib.sha256((proof/'report.json').read_bytes()).hexdigest()==record['proof_sha256']
            assert json.loads((proof/'report.json').read_text())['status']=='complete'
            return ['--packed-graph-proof',str(proof)]
    if model=='huginn-3.5b':
        return ['--production-cache-proof',str(BASE/'preflight/huginn-production-cache-v1')]
    return []
