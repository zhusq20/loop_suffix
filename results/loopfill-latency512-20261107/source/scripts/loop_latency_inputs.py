"""Freeze the six latency workloads without loading a model or generating answers."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import requests
import yaml

SEED = 20261107
COUNTS = {'gsm8k': 300, 'math500': 500, 'bbh': 270,
          'humanevalplus': 164, 'mbppplus': 378, 'specbench': 480}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank(dataset, identity, purpose='sample'):
    return sha(f'{SEED}:{purpose}:{dataset}:{identity}'.encode())


def choose(rows, dataset, count):
    if len(rows) < count:
        raise ValueError(f'{dataset}: need {count}, have {len(rows)}')
    return sorted(rows, key=lambda r: (rank(dataset, r['source_id']), r['source_id']))[:count]


def prepare(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = out / 'sources'
    raw.mkdir()
    provenance = {}

    def fetch(name, url):
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.content
        (raw / name).write_bytes(data)
        provenance[name] = {'url': url, 'sha256': sha(data), 'bytes': len(data)}
        return data

    revisions = {
        'openai/grade-school-math': '3101c7d5072418e28b9008a6636bde82a006892c',
        'suzgunmirac/BIG-Bench-Hard': '9ee07bd481feebf959a6b59d61ea57bdcf30964d',
        'EleutherAI/lm-evaluation-harness': 'd6de81643928d653435c431bae19945d41d32520'}
    base = 'https://raw.githubusercontent.com/'
    harness = base + 'EleutherAI/lm-evaluation-harness/' + revisions['EleutherAI/lm-evaluation-harness']
    gsm_yaml = fetch('gsm8k-cot.yaml', harness + '/lm_eval/tasks/gsm8k/gsm8k-cot.yaml')
    demonstrations = yaml.safe_load(gsm_yaml)['fewshot_config']['samples'][:3]
    gsm_prefix = '\n\n'.join(f"Q: {r['question']}\nA: {r['target']}" for r in demonstrations)
    gsm = [json.loads(line) for line in fetch('gsm8k-test.jsonl', base +
           'openai/grade-school-math/' + revisions['openai/grade-school-math'] +
           '/grade_school_math/data/test.jsonl').decode().splitlines()]
    demo_questions = {r['question'].strip() for r in demonstrations}
    groups = {}
    groups['gsm8k'] = choose([dict(source_id=f'gsm8k/test/{i}', category='gsm8k',
        prompt=f"{gsm_prefix}\n\nQ: {r['question']}\nA:", gold=r['answer'])
        for i, r in enumerate(gsm) if r['question'].strip() not in demo_questions], 'gsm8k', 300)

    math_source = fetch('minerva-utils.py', harness + '/lm_eval/tasks/minerva_math/utils.py')
    node = next(n for n in ast.parse(math_source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'list_fewshot_samples')
    samples = ast.literal_eval(next(n for n in node.body if isinstance(n, ast.Return)).value)
    math_prefix = '\n\n'.join(f"Problem:\n{r['problem']}\n\nSolution: {r['solution']}" for r in samples)
    math_rows = [json.loads(line) for line in fetch('math500-test.jsonl',
        'https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/'
        '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl').decode().splitlines()]
    groups['math500'] = [dict(source_id=str(r.get('unique_id', f'math500/{i}')),
        category=r.get('subject', 'math500'), prompt=f"{math_prefix}\n\nProblem:\n{r['problem']}\n\nSolution:",
        gold=r['answer']) for i, r in enumerate(math_rows)]

    bbh_root = base + 'suzgunmirac/BIG-Bench-Hard/' + revisions['suzgunmirac/BIG-Bench-Hard']
    tasks = sorted('boolean_expressions causal_judgement date_understanding disambiguation_qa '
        'dyck_languages formal_fallacies geometric_shapes hyperbaton logical_deduction_five_objects '
        'logical_deduction_seven_objects logical_deduction_three_objects movie_recommendation '
        'multistep_arithmetic_two navigate object_counting penguins_in_a_table '
        'reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks '
        'sports_understanding temporal_sequences tracking_shuffled_objects_five_objects '
        'tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects '
        'web_of_lies word_sorting'.split())
    assert len(tasks) == 27

    def bbh_task(task):
        data = json.loads(fetch(f'bbh-{task}.json', f'{bbh_root}/bbh/{task}.json'))['examples']
        prompt = fetch(f'bbh-{task}.txt', f'{bbh_root}/cot-prompts/{task}.txt').decode()
        # Drop the authors' canary/header; preserve task description and three CoT examples.
        prompt = prompt.split('-----', 1)[-1].strip()
        eligible = [dict(source_id=f'bbh/{task}/{i}', category=task,
                        prompt=f"{prompt}\n\nQ: {r['input']}\nA: Let's think step by step.",
                        gold=r['target']) for i, r in enumerate(data) if r['input'] not in prompt]
        return choose(eligible, 'bbh/' + task, 10)

    with ThreadPoolExecutor(max_workers=8) as pool:
        groups['bbh'] = [r for selected in pool.map(bbh_task, tasks) for r in selected]
    # Plus data are frozen releases, not the 500-example original MBPP test split.
    import pyarrow.parquet as pq
    datasets = {
        'humanevalplus': ('d32357cf319e50e9c8d8dab5ea876c72b0fd321b',
                         'data/test-00000-of-00001-5973903632b82d40.parquet'),
        'mbppplus': ('b2d74c91837c3f2a20c1299ae98133cbe7cfa077',
                     'data/test-00000-of-00001-d5781c9c51e02795.parquet')}
    for dataset, (revision, path) in datasets.items():
        rows = pq.read_table(io.BytesIO(fetch(dataset + '.parquet',
            f'https://huggingface.co/datasets/evalplus/{dataset}/resolve/{revision}/{path}'))).to_pylist()
        groups[dataset] = [dict(source_id=r['task_id'], category=dataset,
                               prompt=r['prompt'], gold=None) for r in rows]
    spec_path = Path(__file__).resolve().parents[1] / 'results/loop-spec-reuse-20261030/raw/data/spec_bench_question.jsonl'
    data = spec_path.read_bytes()
    (raw / 'specbench480.jsonl').write_bytes(data)
    provenance['specbench480.jsonl'] = {'source': str(spec_path), 'sha256': sha(data)}
    groups['specbench'] = [dict(source_id=f"specbench/{r['question_id']}", category=r['category'],
        prompt=r['turns'][0], gold=None) for r in map(json.loads, data.decode().splitlines())]
    manifests = {}
    for dataset, rows in groups.items():
        assert len(rows) == COUNTS[dataset], (dataset, len(rows))
        assert len({r['source_id'] for r in rows}) == len(rows)
        rows.sort(key=lambda r: (rank(dataset, r['source_id'], 'order'), r['source_id']))
        for i, row in enumerate(rows):
            row.update(question_id=i, dataset=dataset, max_new_tokens=512,
                       prompt_sha256=sha(row['prompt'].encode()))
        path = out / (dataset + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        manifests[dataset] = {'count': len(rows), 'sha256': sha(path.read_bytes()),
                              'source_ids': [r['source_id'] for r in rows]}
    (out / 'manifest.json').write_text(json.dumps(dict(seed=SEED, revisions=revisions,
        sources=provenance, datasets=manifests,
        selection='SHA256(seed:sample:dataset:source_id), lowest ranks; BBH ten per task',
        ordering='independent SHA256(seed:order:dataset:source_id)',
        prompt_policy='GSM three fixed harness CoT examples; MATH four fixed Minerva examples; '
        'BBH three published CoT examples; EvalPlus released prompts; SpecBench first turn. '
        'All wrapped in each model released chat template. EOS/chat-end only; max 512. '
        'This is a frozen local protocol, not exact reproduction of other papers.'), indent=2))
    print(json.dumps({k: len(v) for k, v in groups.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    prepare(parser.parse_args().out)
