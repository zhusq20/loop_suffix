"""Logical draft ceilings must exclude roots and preserve target verification."""
import pytest
import torch

from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig, loopspec_decode
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_tree_decode import tree_decode
from llm_memory_editability.loop_batch_decode import batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
from scripts.loop_batch_bench import arm_config
from test_loop_loopspec_decode import DualFake
from test_loop_tree_decode import FakeLoop, greedy
from test_loop_optimized_decode import setup, requests


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('N',[0,1,2,8,16,32,64])
@pytest.mark.parametrize('stop',[set(),{5}])
def test_loopspec_cap_preserves_greedy_ancestry_and_peak(N,stop):
    cfg=LoopSpecConfig('loopspec-capped',32,4,8,draft_node_budget=N)
    events=[]
    prompt=[1,2,3]
    tokens,stats=loopspec_decode(DualFake(32,4),prompt,cfg,80,stop,max_nodes=cfg.physical_nodes,
                               observer=lambda event,**data:events.append(data))
    assert tokens==greedy(prompt,80,stop)
    assert stats['peak_draft_nodes']<=N
    assert stats['max_rows']<=N+1
    assert stats['cap_skips']==0
    for e in events:
        paths={tuple(p) for p,_ in e['lanes']}
        root=min(paths,key=len)
        assert len(paths)-1<=N
        assert all(p==root or p[:-1] in paths for p in paths)
    if N==0:assert stats['peak_draft_nodes']==0


def test_nonbinding_cap_preserves_native_schedule():
    native=LoopSpecConfig('loopspec-dual',32,4,8)
    cap=LoopSpecConfig('loopspec-capped',32,4,8,draft_node_budget=128)
    a,sa=loopspec_decode(DualFake(32,4),[1,2,3],native,80,set(),max_nodes=128)
    b,sb=loopspec_decode(DualFake(32,4),[1,2,3],cap,80,set(),max_nodes=128)
    assert a==b
    assert sa['calls']==sb['calls'] and sa['rows']==sb['rows']
    assert sb['budget_rejections']==0


def test_native_wfd_single_chain_pipeline_fits_smallest_tested_budget():
    # With T=32,d=4, the native top1 pipeline has <=8 live nodes including root.
    cfg=ArmConfig('wavefront',32,4,1<<30,False,True)
    for prompt in [[1,2,3],[4,6,1],[2,1,2]]:
        tokens,s=tree_decode(FakeLoop(32,4),prompt,cfg,100,set(),max_nodes=256)
        assert tokens==greedy(prompt,100,set())
        assert s['max_rows']<=8


@pytest.mark.parametrize('N',[0,1,8,16,32,64])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_capped_loopspec_packed_and_graphs(N,device):
    make=setup('huginn',device,torch.bfloat16 if device=='cuda' else torch.float32)
    cfg=arm_config('loopspec-capped',4,1,2,4,draft_node_budget=N)
    ar=arm_config('ar',4,1,2,4)
    refs,_=batch_decode(requests(),[make(),make()],ar,schedule='continuous')
    pool=SharedKVPool(2);slots=[make(pool),make(pool)]
    engine=PackedLoopExecutor(slots,cuda_graph=device=='cuda',max_graphs=32)
    batch_decode(requests(),slots,cfg,schedule='continuous',executor=engine,prefill_chunk_size=2)
    engine.graphs.freeze()
    rows,s=batch_decode(requests(),slots,cfg,schedule='continuous',executor=engine,prefill_chunk_size=2)
    assert [r['tokens'] for r in rows]==[r['tokens'] for r in refs]
    assert all(r['stats']['peak_draft_nodes']<=N for r in rows)
    if device=='cuda':assert s['cuda_graph_replays']>0


def test_cli_budget_has_same_logical_meaning_for_all_capped_methods():
    for name in ['budget-raw','budget-shallow','budget-suffix','loopspec-capped']:
        cfg=arm_config(name,32,4,8,4,draft_node_budget=16,model_top_k=2)
        assert cfg.draft_node_budget==16
    assert arm_config('loopspec-dual',32,4,8,4,draft_node_budget=16).draft_node_budget is None
