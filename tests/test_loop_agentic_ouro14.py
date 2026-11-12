"""Keep the third-model extension causal and require complete measurement evidence."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from loop_agentic_ouro14 import checked_checkpoint, tokenize_requests
from loop_agentic_finalize import finalize
from loop_agentic_suite import ARMS, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_extension_preserves_historical_messages_and_order():
    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt):
            assert add_generation_prompt
            return [1] + list(messages[-1]['content'].encode())

    rows = [dict(source_id=str(i), messages=[dict(role='user', content=f'prior error {i}')],
                 phase=phase, tokenizations={'old-model': {'prompt_token_ids': [999]}})
            for i, phase in [(3, 'late'), (1, 'early'), (2, 'middle')]]
    before = json.dumps(rows)
    actual = tokenize_requests(rows, Tokenizer())
    assert json.dumps(rows) == before
    assert [r['source_id'] for r in actual] == ['3', '1', '2']
    assert [r['messages'] for r in actual] == [r['messages'] for r in rows]
    assert [r['phase'] for r in actual] == ['late', 'early', 'middle']
    assert all(r['model'] == 'ouro-1.4b' and 'tokenizations' not in r for r in actual)
    assert actual[0]['prompt_token_ids'] != [999]


def test_extension_fails_on_long_context_instead_of_reselecting():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]*3549

    with pytest.raises(ValueError, match='Unsupported full context: long'):
        tokenize_requests([dict(source_id='long', messages=[])], Tokenizer())


def test_training_gate_rejects_nonzero_receipt(tmp_path):
    write(tmp_path/'complete.json', dict(status='complete', parameters=163597568))
    write(tmp_path/'checks/training-audit.json', dict(status='passed', parameters=163597568))
    identity = dict(pid=123, argv=['teacher'], started_unix=1)
    write(tmp_path/'teachers-launcher.json', identity)
    write(tmp_path/'teachers-exit.json', dict(identity, exit_code=1))
    with pytest.raises(AssertionError):
        checked_checkpoint(dict(training_root=str(tmp_path)), {})


def test_report_uses_extension_model_without_relabeling_old_models(tmp_path):
    jobs = [dict(id=f'ouro-1.4b-{arm}-t{t}', model='ouro-1.4b', arm=arm, temperature=t)
            for t in [0, 1] for arm in ARMS]
    write(tmp_path/'jobs.json', jobs)
    write(tmp_path/'inputs/manifest.json', dict(dataset='SQL', count=100, instances=78,
        phase_counts=dict(early=74, middle=13, late=13), semantics='Fixed replay'))
    assert summarize(tmp_path) == []
    tex = (tmp_path/'table1-agentic.tex').read_text()
    assert 'ouro-1.4b' in tex and 'ouro-2.6b' not in tex and 'huginn' not in tex
    assert '0/14' in (tmp_path/'report.md').read_text()


def test_finalizer_rejects_missing_or_duplicate_methods(tmp_path):
    write(tmp_path/'protocol.json', dict(models={'ouro-1.4b': {}},
        temperatures=[0, 1], main_arms=ARMS))
    write(tmp_path/'status.json', dict(status='complete'))
    write(tmp_path/'jobs.json', [dict(model='ouro-1.4b', temperature=0, arm='ar')]*14)
    with pytest.raises(ValueError, match='All planned'):
        finalize(tmp_path, tmp_path/'repository')
