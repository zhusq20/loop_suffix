"""Bind diagnostic replays to the original inputs and archived source bytes.

The B4 paper check is a separate completion/max1024 experiment. It must never
silently fall back to the formal chat/max512 HumanEval+ input file.
"""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SUITE = BASE.parents[1]/'results/loopfill-latency512-20261107'
B4_INPUT_SHA = 'a9e455e5983275b9ef82716e6c32dae97b15e07a0c14a2810b40c65362635e8d'
B4_EVALUATORS = {
    'ar': 'edd47657bfda4fd5c87e6b9018b29492b29644951b71e6b7c0736adf18b814d0',
    'loopahead': '14d75e6557012807624c81047dcb0924c38481dafa7db15993807b019858e955',
}
STARTUP_EVALUATOR = 'bb90a31ed60c950213a03b6962c8ac806e7bb0c6c3b727558359226c6baa69e7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def diagnostic_input(meta, ar_meta, paper_check=False):
    assert meta['prompt_format'] == ar_meta.get('prompt_format', 'chat')
    assert meta['input_sha256'] == ar_meta['input_sha256']
    if not paper_check:
        assert not meta.get('paper_check') and not ar_meta.get('paper_check'), 'B4 requires explicit --paper-check'
        path = SUITE/f'inputs/{meta["dataset"]}.jsonl'
    else:
        expected = dict(model='ouro-2.6b', model_revision='1ed04250da1a9936042725d302e81c8fa2ab5abd',
            dataset='paper-humaneval164', prompt_format='completion', input_sha256=B4_INPUT_SHA,
            count=164, full_count=164, temperature=0, top_p=1, top_k=0, seed=0,
            dtype='bfloat16', max_seq=1536, max_new_tokens=1024, graph=False,
            compile=False, trace=False, pilot=False, dev_calibration=False,
            native_only=False, random_lad=False, profile_dev=False,
            paper_check=True, reportable=False, admission_sha256=None)
        for label, item in [('ar', ar_meta), ('loopahead', meta)]:
            assert item['method'] == label
            for key, value in expected.items():
                assert item[key] == value, f'B4 {label}/{key}'
            assert item['stop_audit']['stop_ids'] == [0]
        assert ar_meta['physical_nodes'] == 1
        assert meta.get('physical_nodes', 32) == 32
        assert meta['checkpoint_sha256'] == 'c445ce786366dace9097d0969b90e195dd920bfcc3f90babec9ba900063a5630'
        assert meta['config'] == dict(total=4, first=1, second=2, future=3, retain=32,
            execute=16, guard=2, stride=1, family='ouro', temperature=0, top_p=1)
        path = BASE/'data/paper-inputs/humaneval.jsonl'
        source = read(path.with_name('humaneval-source.json'))
        isolation = read(path.with_name('humaneval-isolation-audit.json'))
        assert source['input_sha256'] == isolation['input_sha256'] == B4_INPUT_SHA
        assert source['revision'] == '463c980b59e818ace59f6f9803cd92c749ceae61'
        assert source['count'] == 164 and isolation['passed'] and not isolation['matches']
    assert digest(path) == meta['input_sha256'], 'Diagnostic input hash differs from source run'
    return path


def validate_replay_sources(folder, paper_check=False):
    manifest = read(folder/'source-manifest.json')
    for rel, sha in manifest.items():
        assert digest(folder/'source'/rel) == sha, f'Archived source changed: {rel}'
        if paper_check and rel == 'evaluate_loopahead.py':
            method = read(folder/'metadata.json')['method']
            assert sha == B4_EVALUATORS[method], 'Unknown historical B4 evaluator'
            assert digest(BASE/rel) in {B4_EVALUATORS['ar'], STARTUP_EVALUATOR}, 'Live evaluator requires a new source audit'
            # The older B4 LA evaluator allocates32 physical nodes for both
            # methods. B4 AR already used the corrected one-node evaluator.
            # Other changes add metadata and admission assertions (unused in
            # B4). Runtime engine/decode/LAD bytes must still match exactly.
            # STARTUP_EVALUATOR changes only main's CUDA acquisition; its
            # complete non-main AST equals the corrected B4 AR evaluator.
        elif rel == 'evaluate_loopahead.py' and sha == B4_EVALUATORS['ar'] and digest(BASE/rel) == STARTUP_EVALUATOR:
            def without_main(path):
                tree = ast.parse(path.read_text())
                tree.body = [node for node in tree.body
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'main']
                return ast.dump(tree, include_attributes=False)
            assert without_main(folder/'source'/rel) == without_main(BASE/rel), 'Change extends beyond CUDA startup'
        else:
            assert digest(BASE/rel) == sha, f'Live source changed: {rel}'


def input_provenance(meta, ar_meta, paper_check=False):
    path = diagnostic_input(meta, ar_meta, paper_check)
    result = dict(scope='B4-paper-check' if paper_check else 'frozen-benchmark-input',
        path=str(path), sha256=digest(path), prompt_format=meta['prompt_format'],
        max_new_tokens=meta['max_new_tokens'], stop_ids=meta['stop_audit']['stop_ids'])
    if paper_check:
        result['historical_evaluator_sha256'] = B4_EVALUATORS
        result['physical_nodes'] = dict(ar=1, loopahead=32)
        result['source_manifest_sha256'] = digest(path.with_name('humaneval-source.json'))
        result['isolation_audit_sha256'] = digest(path.with_name('humaneval-isolation-audit.json'))
    return result
