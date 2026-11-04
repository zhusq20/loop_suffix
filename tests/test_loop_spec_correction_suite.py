"""The focused experiment must cover Q without silently expanding its scope."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from loop_spec_correction_suite import q_variants, n_variants, BASELINES, baseline_command


def test_focused_q_matrix_and_adopted_control():
    variants=q_variants()
    assert len(variants)==len({v['name'] for v in variants})==11
    assert variants[0]['name']=='raw-top2'
    assert variants[0]['suffix_q']==64 and variants[0]['top_k']==2
    for k in [1,2]:
        assert {v['suffix_q'] for v in variants if v['arm']=='budget-raw' and v['top_k']==k}=={4,16,64}
    assert {v['suffix_q'] for v in variants if v['arm']=='budget-suffix'}=={4,16,64}
    assert {v['top_k'] for v in variants if v['arm']=='budget-shallow'}=={1,2}
    assert all(v['nodes']==64 and v['child']=='all' and not v['positive'] for v in variants)
    assert BASELINES==('ar','wavefront','loopspec-dual','budget-suffix')


def test_cross_model_gpu_uses_requested_model_environment_and_physical_cpu_affinity():
    def template(model,gpu):
        return dict(model=model,gpu=gpu,command=['taskset','--cpu-list','0-3',f'/{model}/python',
            'bench.py','--model',model,'--model-path',f'/{model}/weights','--arms','ar',
            '--batch-sizes','1','--out','old'])
    job=dict(model='huginn',arm='budget-raw',batch=8,key='huginn-b8-q16')
    command=baseline_command([template('ouro',6),template('huginn',7)],
                             dict(gpu=6,cpus='56-59'),job,Path('/suite'))
    assert '/huginn/python' in command and '/ouro/python' not in command
    assert command[command.index('--model-path')+1]=='/huginn/weights'
    assert command[command.index('--cpu-list')+1]=='56-59'
    assert command[command.index('--batch-sizes')+1]=='8'
    assert command[command.index('--out')+1]=='/suite/runs/huginn-b8-q16'


def test_matched_n_matrix_keeps_q_fixed_and_labels_modified_baseline():
    variants=n_variants()
    assert len(variants)==len({v['name'] for v in variants})==16
    for n in [8,16,32,64]:
        group=[v for v in variants if v['nodes']==n]
        assert {v['arm'] for v in group}=={'budget-raw','budget-shallow','budget-suffix','loopspec-capped'}
        assert all(v['suffix_q']==4 and v['top_k']==2 and not v['positive'] for v in group)
