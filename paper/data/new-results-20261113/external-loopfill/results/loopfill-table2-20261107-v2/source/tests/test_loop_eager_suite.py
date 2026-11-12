"""Prevent the second-model suite from using Huginn artifacts or changing timing flags."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_eager_suite import ARMS, BATCHES, command, job_key


@pytest.mark.parametrize('model', ['huginn-3.5b', 'ouro-2.6b'])
@pytest.mark.parametrize('smoke', [False, True])
def test_model_reference_and_artifact_routing(model, smoke):
    template = ['taskset', '--cpu-list', '0-3', '/python', '/old/bench.py']
    for flag in ['--questions', '--arms', '--batch-sizes', '--repeat-id', '--order-seeds',
                 '--suffix-c-initial', '--max-new-tokens', '--reference-file', '--out']:
        template += [flag, 'old']
    template += ['--model', model, '--schedule', 'continuous', '--draft-node-budget', '16',
                 '--suffix-max-draft', '4', '@/frozen/configs/loopfill-eager.args']
    protocol = dict(template_command=template, source='/frozen', questions='/inputs',
                    execution_args={'model': model})
    original = copy.deepcopy(protocol)
    keys = set()
    for batch in BATCHES:
        for arm in ARMS:
            key = job_key(protocol, batch, arm, 0)
            keys.add(key)
            job = dict(key=key, batch=batch, arm=arm, repeat=0)
            cmd = command(protocol, {'cpus': '4-7'}, job, Path('/suite'), smoke)
            assert cmd[cmd.index('--model') + 1] == model
            ref = '/suite/' + ('validation/references/' if smoke else 'references/') + model + '.jsonl'
            assert cmd[cmd.index('--reference-file') + 1] == ref
            assert cmd[cmd.index('--out') + 1] == '/suite/' + ('validation/' if smoke else 'runs/') + key
            assert cmd[cmd.index('--batch-sizes') + 1] == str(batch)
            assert cmd[cmd.index('--max-new-tokens') + 1] == ('16' if smoke else '1024')
            assert cmd[cmd.index('--suffix-c-initial') + 1] == ('0.7' if arm == 'budget-online' else '1.0')
            assert cmd[-1 if not smoke else -3] == '@/frozen/configs/loopfill-eager.args'
            assert '--cuda-graph' not in cmd
    assert len(keys) == 30
    assert protocol == original
