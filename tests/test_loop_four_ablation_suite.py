"""Check the scientific matrix and executable flags before spending GPU time."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import loop_four_ablation_suite as suite
from loop_batch_bench import arm_config, parser
from dataclasses import asdict


def test_matrix_has_only_authorized_cells():
    jobs = suite.cells()
    assert len(jobs) == len({j['key'] for j in jobs}) == 36
    for model in suite.MODELS:
        assert len([j for j in jobs if j['model']==model and j['batch']==16])==14
        b1 = [j for j in jobs if j['model']==model and j['batch']==1]
        assert {j['nodes'] for j in b1}=={8,16,32,64}
        assert all(j['readouts']==2 and j['top_k']==2 and j['child_q']==4 and j['arm']=='budget-online' for j in b1)


def test_every_command_reconstructs_exact_frozen_config():
    p = dict(source='/source', python=sys.executable, models={}, mother_configs={})
    for m, T, d, deep in [('huginn-3.5b',32,4,8),('ouro-2.6b',4,1,2)]:
        p['models'][m] = dict(path='/weights', inputs='/inputs', smoke_inputs='/short',max_seq=2688)
        p['mother_configs'][m] = asdict(arm_config('budget-online',T,d,deep,4,
            draft_node_budget=16, model_top_k=2, child_suffix_caps=[4,4],suffix_c_initial=.7))
        p['mother_configs'][m]['readout_depths'] = [d,deep]
        p['mother_configs'][m]['child_suffix_caps'] = [4,4]
    for j in suite.cells():
        for smoke in [False,True]:
            cmd = suite.command(p,dict(cpus='0-3'),j,Path('/out'),smoke)
            args = parser().parse_args(cmd[5:])
            expected = suite.expected_config(p,j)
            assert args.batch_sizes == ('2' if smoke else str(j['batch']))
            assert args.max_new_tokens == (16 if smoke else 1024)
            assert args.model_readout_depths == expected['readout_depths']
            assert args.model_top_k == expected['top_m']
            assert args.child_suffix_caps == expected['child_suffix_caps']
            assert args.draft_node_budget == expected['draft_node_budget']
            assert args.suffix_c_initial == expected['c_initial']
            assert not args.cuda_graph and not args.pad_decode_rows


def test_gamma_and_latency_use_weighted_counters():
    rows = []
    for tokens,calls,latency in [(10,20,2.),(30,30,4.)]:
        stats = dict(tokens=tokens,calls=calls,rows=100,readout_rows=20,wasted_rows=10,
                     speculate_calls=5,child_suffix_queries=3,selected_draft_nodes_sum=12,
                     selection_events=2,peak_draft_nodes=8)
        rows.append(dict(stats=stats,latency_s=latency+.5,ttft_s=.5))
    result = suite.metrics(rows,4,1)
    assert result['gamma']==3.2
    assert result['decode_ms_per_token']==150.
    assert result['mean_decode_latency_s']==3.
    assert suite.metrics(rows,4,16)['mean_decode_latency_s'] is None
