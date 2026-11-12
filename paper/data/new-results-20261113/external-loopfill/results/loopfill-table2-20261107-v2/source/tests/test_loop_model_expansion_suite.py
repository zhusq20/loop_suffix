"""Expansion protocols preserve per-model references, inputs and method identities."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_model_expansion_suite import command, validation_rows
from loop_batch_bench import arm_config, load_requests
from llm_memory_editability.loop_model_registry import MODELS


def test_stratified_preflight_contains_every_task_and_length_extremes():
    categories = ['writing', 'coding', 'translation', 'summarization', 'math_reasoning', 'qa', 'rag']
    rows = [dict(question_id=i, category=categories[i % 7], prompt_token_ids=[1] * (i+1)) for i in range(480)]
    selected = validation_rows(rows)
    assert len(selected) == 8
    assert {r['category'] for r in selected} >= set(categories) - {'coding'}
    assert {0,479} <= {r['question_id'] for r in selected}


@pytest.mark.parametrize('phase', ['validate','references','run'])
def test_expansion_commands_route_fresh_references(phase):
    flags = ['--questions','--arms','--batch-sizes','--repeat-id','--order-seeds',
             '--suffix-c-initial','--max-new-tokens','--reference-file','--out']
    template = ['taskset','--cpu-list','0-3','/python','/bench.py']
    for flag in flags:
        template += [flag,'old']
    protocol = dict(template_command=template, source='/frozen', questions='/inputs/raven.jsonl',
                    validation_questions='/inputs/short.jsonl', validation_ids=list(range(8)),
                    execution_args={'model':'raven-llama-3.2'})
    job = dict(key='raven-llama-3.2-b2-r0-budget-online', arm='budget-online', batch=2, repeat=0)
    cmd = command(protocol, {'cpus':'4-7'}, job, Path('/suite'), phase)
    assert cmd[cmd.index('--questions')+1] == ('/inputs/short.jsonl' if phase=='validate' else '/inputs/raven.jsonl')
    assert cmd[cmd.index('--reference-file')+1] == '/suite/'+('validation/' if phase=='validate' else '')+'references/raven-llama-3.2.jsonl'
    assert ('--reference-only' in cmd) == (phase=='references')
    assert cmd[cmd.index('--suffix-c-initial')+1] == '0.7'


def test_throughput_successor_skips_serial_generation_but_retains_short_checks():
    flags = ['--questions','--arms','--batch-sizes','--repeat-id','--order-seeds',
             '--suffix-c-initial','--max-new-tokens','--reference-file','--out']
    template = ['taskset','--cpu-list','0-3','/python','/bench.py']
    for flag in flags:
        template += [flag,'old']
    protocol = dict(template_command=template, source='/frozen', questions='/inputs/raven.jsonl',
                    validation_questions='/inputs/short.jsonl', validation_ids=list(range(8)),
                    execution_args={'model':'raven-llama-3.2','check':False}, require_serial_reference=False)
    job = dict(key='raven-llama-3.2-b2-r0-ar', arm='ar', batch=2, repeat=0)
    cmd = command(protocol, {'cpus':'4-7'}, job, Path('/suite'), 'run')
    assert not {'--check','--reference-only','--reference-file','--limit'} & set(cmd)
    assert cmd[cmd.index('--arms')+1] == 'ar'
    assert cmd[cmd.index('--max-new-tokens')+1] == '1024'
    short = command(protocol, {'cpus':'4-7'}, job, Path('/suite'), 'validate')
    assert '--check' in short and '--reference-file' in short
    with pytest.raises(ValueError, match='disabled'):
        command(protocol, {'cpus':'4-7'}, job, Path('/suite'), 'references')


def test_method_readout_overrides_are_independent():
    for name in ('raven-llama-3.2','raven-olmo-2-0425','raven-tinyllama-3t'):
        s=MODELS[name]
        loop = arm_config('loopspec-dual', s.depth_total, *s.loopfill_readout_depths, 4,
                          loopspec_depths=(2,8), wavefront_depth=4)
        assert loop.draft_depth==2 and loop.second_depth==8 and loop.physical_nodes==499
        wave = arm_config('wavefront', s.depth_total, *s.loopfill_readout_depths, 4, wavefront_depth=4)
        assert wave.draft_depth==4
        budget = arm_config('budget-raw', s.depth_total, 4, 8, 4,
                            model_readout_depths=(4,8), loopspec_depths=s.loopspec_depths)
        assert budget.readout_depths==(4,8)


def test_input_model_identity_context_and_invalid_ids(tmp_path):
    import json
    tokenizer = SimpleNamespace(eos_token_id=2, get_vocab=lambda:{'eos':2})
    args = SimpleNamespace(model='raven-tinyllama-3t', prompt_format='raw', limit=None,
                           max_seq=2816, suffix_max_draft=4, max_new_tokens=1024)
    p=tmp_path/'inputs.jsonl'
    for row, message in [(dict(model_id='ouro-1.4b',prompt_token_ids=[1]),'another model'),
                         (dict(prompt_token_ids=[65505]),'invalid input token'),
                         (dict(prompt_token_ids=[1]*1100),'exceeds model context')]:
        p.write_text(json.dumps(dict(question_id=3,**row))+'\n')
        with pytest.raises(ValueError,match=message):
            load_requests(p,tokenizer,args,65505,65505,32000)
    p.write_text(json.dumps(dict(question_id=3,prompt_token_ids=[1]*1024))+'\n')
    requests,_ = load_requests(p,tokenizer,args,65505,65505,32000)
    assert requests[0].max_new_tokens==1024 and requests[0].stop=={2}


def test_six_model_table_builder_uses_ratio_of_counters_and_keeps_failures(tmp_path):
    import json
    import csv
    from loop_accepted_length import summarize_models
    from loop_eager_suite import ARMS, BATCHES
    suite=tmp_path/'suite'
    (suite/'jobs').mkdir(parents=True)
    target=suite/'run'
    target.mkdir()
    model='raven-llama-3.2'
    protocol=dict(request_ids=[1,2], questions_sha256='input-hash', execution_source_sha256={'source':'hash'},
                  execution_args=dict(model=model,prompt_format='raw'))
    (suite/'protocol.json').write_text(json.dumps(protocol))
    for batch in BATCHES:
        for arm in ARMS:
            key=f'{model}-b{batch}-{arm}'
            complete=(batch==1 and arm=='ar')
            job=dict(key=key, model=model,batch=batch,arm=arm,repeat=0,
                     outcome='complete' if complete else 'oom', output=str(target))
            (suite/'jobs'/f'{key}.json').write_text(json.dumps(job))
    (target/'metadata.json').write_text(json.dumps(dict(configs={'ar':{'depth_total':32}},
         questions_sha256='input-hash',source_sha256={'source':'hash'})))
    rows=[dict(request_id=1,tokens=[1],n_tokens=1,category='writing',stats={'tokens':0,'calls':0}),
          dict(request_id=2,tokens=[1,2,3],n_tokens=3,category='qa',stats={'tokens':2,'calls':64})]
    (target/'ar-b1-seed0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (target/'ar-b1-seed0-summary.json').write_text(json.dumps(dict(output_tokens=4,wall_s=2)))
    (target/'complete.json').write_text('{}')
    summarize_models([suite],tmp_path/'tables')
    with (tmp_path/'tables/measurements.csv').open() as f:
        metrics=list(csv.DictReader(f))
    ar=next(r for r in metrics if r['outcome']=='complete')
    assert ar['gamma']=='1.0' and ar['throughput']=='2.0'
    assert all(not r['gamma'] and not r['throughput'] for r in metrics if r['outcome']=='oom')
    with (suite/'accepted-length/overall.csv').open() as f:
        measured=next(r for r in csv.DictReader(f) if r['outcome']=='complete')
    assert measured['prefill_only_requests']=='1'
    assert 'oom' in (tmp_path/'tables/gamma.md').read_text()
