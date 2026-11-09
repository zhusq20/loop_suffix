"""Freeze the user-requested single-seed B16 matrix and its accounting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import loop_tree_sampling_suite as suite  # noqa: E402


def test_four_arms_use_b16_seed_zero_fixed_coefficient_and_no_trace():
    p = dict(source='/source', python='/python',
             models={m: dict(path='/model', inputs='/full', smoke_inputs='/short', max_seq=2688)
                     for m in suite.MODELS},
             lanes={m: dict(cpus='0-3') for m in suite.MODELS})
    commands = []
    for model in suite.MODELS:
        for variant, settings in suite.VARIANTS.items():
            job = dict(model=model, variant=variant, arm='budget-raw', batch=16)
            cmd = suite.command(p, job, '/output')
            commands.append(cmd)
            for flag, value in [('batch-sizes', '16'), ('max-new-tokens', '1024'),
                                ('draft-node-budget', '16'), ('draft-seed', '0'),
                                ('suffix-c-initial', '1.0'), ('tree-selection', settings['selection'])]:
                assert cmd[cmd.index('--'+flag)+1] == value
            assert ('--existing-child-probs' in cmd) == settings['exact']
            assert '--trace-controller' not in cmd and '--profile' not in cmd
            assert '--no-cuda-graph' in cmd and '--no-pad-decode-rows' in cmd
    assert len(commands) == 8
