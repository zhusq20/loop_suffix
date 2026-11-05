from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loop_draft_trace import DraftTrace
from loop_suffix_sources_probe import SourceProbe, analyze, paths


def draft(token,score):
    return SimpleNamespace(token_ids=[token],parents=[-1],probs=[.8],match_len=3,score=score)


@pytest.mark.parametrize('local_score,history_score,winner',[(1,1,'local'),(2,1,'local'),(1,2,'history')])
def test_probe_calls_each_provider_once_and_preserves_exact_winner(local_score,history_score,winner):
    called=[];local=draft(2,local_score);history=draft(3,history_score)
    def retrieve(name,value):
        def fn(*a,**kw):called.append(name);return value
        return fn
    cache=SimpleNamespace(local=SimpleNamespace(speculate=retrieve('local',local)),
        history=SimpleNamespace(speculate=retrieve('history',history)),responses={7:[1]},publications=[])
    trace=DraftTrace(4);trace.counts['calls']=5
    probe=SourceProbe(cache,{7:trace})
    result=probe.speculate(7,[1],max_spec_tokens=4)
    assert called==['local','history']
    assert result is (local if winner=='local' else history)
    probe.observe(7,'child_suffix_query',cap=4,path=(1,9),rank=0)
    assert probe.queries[0]['tick']==5 and probe.queries[0]['parent']==(1,9)


def test_branch_paths_preserve_full_prefix():
    assert paths((1,2),dict(tokens=[3,4,5],parents=[-1,-1,0]))==[(1,2,3),(1,2,4),(1,2,3,5)]


def test_only_new_earlier_candidates_on_final_path_count():
    def snap(token):return dict(tokens=[token],parents=[-1])
    queries=[
        dict(request_id=7,parent=(1,),tick=3,kind='child',winner='local',local=snap(9),history=snap(2),completed_history=2),
        dict(request_id=7,parent=(1,2),tick=4,kind='child',winner='local',local=snap(3),history=snap(3),completed_history=2),
        dict(request_id=7,parent=(1,2,3),tick=6,kind='child',winner='local',local=snap(9),history=snap(4),completed_history=2),
        dict(request_id=7,parent=(1,9),tick=0,kind='child',winner='local',local=snap(8),history=snap(3),completed_history=0),
    ]
    tokens=[1,2,3,4,5]
    trace=SimpleNamespace(depth=4,finish=lambda *a:None,
        entries=[dict(path=tuple(tokens[:pos+1]),step=3,tick=pos+6,terminal=False) for pos in (1,2,3)],
        nodes={tuple(tokens[:pos+1]):dict(first_proposal=pos+3,allocations=[dict(tick=pos+3)]) for pos in (1,2,3)})
    r=analyze(queries,{7:trace},[dict(request_id=7,tokens=tokens,stats={'speculate_calls':4})])
    assert r['counts']['one_round_wait_with_new_earlier_candidate']==1
    assert r['examples'][0]['position']==1 and r['examples'][0]['lead_ticks']==1


def test_real_cache_probe_preserves_output_and_compute():
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    from llm_memory_editability.loop_budget_decode import budget_tree_decode
    from test_loop_budget_decode import BudgetFakeLoop,config
    cfg=config(8,scoring='raw',top_m=2,suffix_room=4)
    args=([1,2,3,1,2],cfg,32,set())
    expected,baseline=budget_tree_decode(BudgetFakeLoop(4,1),*args,CompletionSuffixCache.arctic(),7)
    cache=CompletionSuffixCache.arctic();trace=DraftTrace(4);probe=SourceProbe(cache,{7:trace})
    cache.speculate=probe.speculate
    actual,stats=budget_tree_decode(BudgetFakeLoop(4,1),*args,cache,7,
        observer=lambda event,**data:probe.observe(7,event,**data))
    assert actual==expected
    for key in ('calls','rows','nodes','wasted_rows','speculate_calls'):
        assert stats[key]==baseline[key]
    report=analyze(probe.queries,{7:trace},[dict(request_id=7,tokens=actual,stats=stats)])
    assert report['counts']['queries']==stats['speculate_calls']
