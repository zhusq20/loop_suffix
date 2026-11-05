"""Readout ablations preserve full-depth verification and the node budget."""
import pytest

from scripts.loop_batch_bench import arm_config, parser
from llm_memory_editability.loop_budget_decode import budget_tree_decode
from test_loop_budget_decode import BudgetFakeLoop, BoundedSuffix
from test_loop_tree_decode import greedy


@pytest.mark.parametrize('depths,top_k', [((1,),4), ((1,2),2)])
def test_requested_readouts_execute_and_full_depth_output_stays_greedy(depths, top_k):
    cfg=arm_config('budget-raw',4,1,2,4,draft_node_budget=16,
        model_top_k=top_k,model_readout_depths=depths,child_suffix_caps=[4]*top_k)
    calls=[]
    class RecordingLoop(BudgetFakeLoop):
        def readout_top(self,tree,groups,full,m):
            calls.append((tuple(d for d,g in groups),m,len(full)))
            return super().readout_top(tree,groups,full,m)
    prompt=[1,2,3,4,5,6,1,2]
    output,stats=budget_tree_decode(RecordingLoop(4,1),prompt,cfg,60,set(),BoundedSuffix(),0)
    assert output==greedy(prompt,60,set())
    assert calls and all(d==depths and k==top_k for d,k,_ in calls)
    assert sum(full for _,_,full in calls)>=len(output)-1
    assert stats['max_rows']<=17 and stats['peak_draft_nodes']<=16


def test_cli_top4_and_single_readout_and_unchanged_default():
    args=parser().parse_args(['--model','ouro-2.6b','--model-path','unused','--questions','unused',
        '--out','unused','--model-top-k','4','--model-readout-depths','1'])
    assert args.model_top_k==4 and args.model_readout_depths==[1]
    assert arm_config('budget-raw',4,1,2,4).readout_depths==(1,2)


@pytest.mark.parametrize('depths',[(2,),(1,1),(1,4),()])
def test_invalid_depth_override_rejected(depths):
    with pytest.raises(ValueError,match='readout depths'):
        arm_config('budget-raw',4,1,2,4,model_readout_depths=depths)


def test_nonbudget_depth_override_is_not_silently_ignored():
    with pytest.raises(ValueError,match='requires budget arms'):
        arm_config('loopspec-dual',4,1,2,4,model_readout_depths=(1,))
