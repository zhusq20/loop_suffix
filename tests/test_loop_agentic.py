"""Causal request extraction and diagnostic metric definitions for agent traces."""
import json
from types import SimpleNamespace

import pytest

from scripts.loop_agentic_inputs import causal_candidates
from scripts.loop_agentic_worker import RetrievalProbe, digest_tokens


def trajectory():
    messages = [dict(role='system',content='system'),dict(role='user',content='task')]
    for i in range(9):
        messages.extend([dict(role='assistant',content=f'ANSWER_{i}'),
                         dict(role='user',content=f'OBSERVATION_{i}')])
    return messages


def test_each_request_contains_only_its_causal_prefix():
    messages=trajectory()
    rows=causal_candidates(messages)
    assert len(rows)==9
    assert [r['phase'] for r in rows]==['early']*3+['middle']*3+['late']*3
    for i,row in enumerate(rows):
        assert row['messages']==messages[:row['message_index']]
        text=json.dumps(row['messages'])
        assert f'ANSWER_{i}' not in text
        assert f'OBSERVATION_{i}' not in text
        if i:
            assert f'ANSWER_{i-1}' in text and f'OBSERVATION_{i-1}' in text
    assert messages==trajectory()


def test_unknown_tool_payload_cannot_be_silently_dropped():
    messages=trajectory()
    messages[2]['tool_calls']=[dict(id='call')]
    with pytest.raises(ValueError,match='Unexpected message fields'):
        causal_candidates(messages)


def test_probe_separates_nonempty_from_correct_and_child_queries():
    draft=SimpleNamespace(token_ids=[3,4,8],parents=[-1,0,-1],match_len=5)
    cache=SimpleNamespace(responses={7:[1,2]},speculate=lambda *a,**kw:draft)
    probe=RetrievalProbe(cache)
    assert probe.speculate(7,[1,2]) is draft
    probe.speculate(7,[1,2])
    probe.observe('child_suffix_query',cap=4)
    totals=probe.summarize([1,2,3,9])
    assert totals['root']['nonempty']==1
    assert totals['root']['first_token_covered']==1
    assert totals['root']['matching_path_tokens']==1
    assert totals['child']['queries']==1
    assert 'matching_path_tokens' not in totals['child']
    assert probe.queries==[]
    probe.speculate(7,[1,2])
    assert probe.summarize([1,2,9])['root']['first_token_covered']==0


def test_token_hash_detects_order_changes():
    assert digest_tokens([1,2,3])!=digest_tokens([1,3,2])
    assert digest_tokens([1,2,3])==digest_tokens([1,2,3])
