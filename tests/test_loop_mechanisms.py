"""Check stage-clock accounting, frozen sampling, and handover denominators."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loop_stage_timing import StageClock, operation_label
from loop_mechanism_suite import select_ids
from loop_mechanism_analyze import readiness


def test_nested_host_scopes_are_exclusive_and_unwind_on_exception():
    values=iter([0.,2.,5.,10.])
    clock=StageClock(clock=lambda:next(values))
    with pytest.raises(RuntimeError), clock.scope('parent'):
        with clock.scope('child'):
            raise RuntimeError('measured failure')
    assert clock.host=={'parent':7.,'child':3.}
    assert not clock.stack


def test_combined_readouts_are_not_falsely_attributed_to_shallow_only():
    assert operation_label('readout')==operation_label('readout_top')=='combined_readout'


def test_sample_is_balanced_order_invariant_and_independent_of_output_fields():
    tasks=['coding','translation','summarization','math_reasoning','qa','rag']
    rows=[dict(question_id=i*10+j,category=t) for i,t in enumerate(tasks) for j in range(8)]
    ids,smoke=select_ids(rows)
    assert len(ids)==24 and len(smoke)==6 and set(smoke)<=set(ids)
    assert all(sum(r['question_id'] in ids for r in rows if r['category']==t)==4 for t in tasks)
    altered=[dict(r,output_tokens=999,latency_s=1000) for r in reversed(rows)]
    assert select_ids(altered)==(ids,smoke)


def sample_row():
    return dict(tokens=[1,2,3,4],stats=dict(tokens=3,calls=6,rows=10),
                trace=dict(calls=6,rows=10,root_entries=2,initial_root_wait=4),
                waits=dict(entries=2,proposal_wait=1,ranking_wait=1,full_ready_entries=1),
                wait_entries=[[1,2,1,1],[2,4,0,0]])


def test_readiness_excludes_terminal_handover_and_keeps_initial_wait():
    row=sample_row()
    result,hist=readiness([row],4)
    assert result['entries']==2 and result['decode_tokens']==3
    assert result['full_ready_fraction']==.5
    assert result['mean_progress_fraction']==.75
    assert result['initial_wait']==4 and result['mean_remaining_steps']==1
    assert hist=={2:1,4:1}


def test_readiness_rejects_wait_inconsistency():
    row=copy.deepcopy(sample_row())
    row['wait_entries'][0][1]=3
    with pytest.raises(AssertionError):
        readiness([row],4)


def test_prefill_only_has_no_initial_wait_or_handover():
    row=dict(tokens=[1],stats=dict(tokens=0,calls=0,rows=0),trace={},waits={},wait_entries=[])
    result,hist=readiness([row],4)
    assert result['initial_wait']==0 and result['entries']==0
    assert result['full_ready_fraction'] is None and not hist
