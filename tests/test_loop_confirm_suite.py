"""Protect paired repeats from mismatched reuse and accidentally changed arms."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_confirm_suite import ARMS, BATCHES, PLACEMENT, arm_order, command, key, mismatch_reason


def fixture_data():
    protocol = dict(execution_source_sha256={'bench.py': 'sha'}, questions_sha256='data',
                    execution_args=dict(dtype='bfloat16', schedule='continuous'),
                    configs={'budget-online': dict(c_initial=.7, eta=.1, draft_node_budget=16)})
    job = dict(arm='budget-online', batch=8, repeat=0, gpu_uuid='physical-gpu')
    meta = dict(source_sha256={'bench.py': 'sha'}, questions_sha256='data', gpu_uuid='physical-gpu',
        args=dict(dtype='bfloat16', schedule='continuous', arms='budget-online', batch_sizes='8',
                  repeat_id=0, max_new_tokens=1024, limit=None), configs=copy.deepcopy(protocol['configs']))
    return protocol, job, meta


def test_reuse_requires_same_source_gpu_order_repeat_and_online_configuration():
    protocol, job, meta = fixture_data()
    assert mismatch_reason(meta, protocol, job) is None
    prefixed = dict(job, gpu_uuid='GPU-physical-gpu')
    assert mismatch_reason(meta, protocol, prefixed) is None
    for field, value in [('source_sha256', {}), ('questions_sha256', 'other'), ('gpu_uuid', 'other')]:
        altered = copy.deepcopy(meta)
        altered[field] = value
        assert mismatch_reason(altered, protocol, job)
    for field, value in [('repeat_id', 1), ('limit', 8), ('max_new_tokens', 16), ('schedule', 'fixed')]:
        altered = copy.deepcopy(meta)
        altered['args'][field] = value
        assert mismatch_reason(altered, protocol, job)
    altered = copy.deepcopy(meta)
    altered['configs']['budget-online']['c_initial'] = 1.
    assert mismatch_reason(altered, protocol, job) == 'arm_config'


def test_three_repeats_cover_exactly_requested_batches_and_distinct_artifacts():
    assert BATCHES == [1, 2, 4, 8, 16]
    assert set(PLACEMENT) == set(BATCHES)
    assert len(set(PLACEMENT.values())) == len(BATCHES)
    assert len({key(b, a, r) for b in BATCHES for a in ARMS for r in range(3)}) == 105
    for repeat in range(3):
        assert set(arm_order(repeat)) == set(ARMS)
    assert arm_order(1) == list(reversed(ARMS))
    assert arm_order(2) != arm_order(0)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('smoke', [False, True])
def test_commands_preserve_backend_and_use_the_right_initial_weight(arm, smoke):
    options = ['--cpu-list', '--questions', '--arms', '--batch-sizes', '--repeat-id',
               '--order-seeds', '--suffix-c-initial', '--max-new-tokens', '--reference-file', '--out']
    template = ['taskset', '--cpu-list', '0-3', '/python', '/old/bench.py']
    for option in options[1:]:
        template.extend([option, 'old'])
    template += ['--draft-node-budget', '16', '--suffix-max-draft', '4', '--cuda-graph']
    protocol = dict(template_command=template, source='/frozen', questions='/inputs')
    job = dict(arm=arm, batch=16, repeat=2, key=key(16, arm, 2))
    cmd = command(protocol, dict(cpus='12-15'), job, Path('/suite'), smoke)
    assert cmd[4] == '/frozen/scripts/loop_batch_bench.py'
    assert cmd[cmd.index('--cpu-list') + 1] == '12-15'
    assert cmd[cmd.index('--suffix-c-initial') + 1] == ('0.7' if arm == 'budget-online' else '1.0')
    assert cmd[cmd.index('--repeat-id') + 1] == '2'
    assert cmd[cmd.index('--max-new-tokens') + 1] == ('16' if smoke else '1024')
    assert ('--limit' in cmd) == smoke
    assert cmd[cmd.index('--draft-node-budget') + 1] == '16'
    assert '--cuda-graph' in cmd
    assert template[4] == '/old/bench.py'
