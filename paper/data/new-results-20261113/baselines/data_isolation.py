"""Declared exact/near-duplicate protection of evaluation questions and demonstrations."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def token_grams(text):
    # Preserve punctuation and mathematical operators, including +, -, = and LaTeX commands.
    tokens = re.findall(r'\w+|[^\w\s]', normalize(text), re.UNICODE)
    return {hashlib.blake2b('\x1f'.join(tokens[i:i + 5]).encode(), digest_size=8).digest()
            for i in range(max(0, len(tokens) - 4))}


class IsolationIndex:
    def __init__(self, rows=(), bbh_boilerplate=False):
        self.rows, self.grams, self.exact = [], [], {}
        self.short = []
        self.inverted = defaultdict(list)
        self.ignored = {}
        if bbh_boilerplate:
            counts, sizes = defaultdict(Counter), Counter()
            for row in rows:
                if row['bucket'].startswith('bbh/'):
                    counts[row['bucket']].update(token_grams(row['text']))
                    sizes[row['bucket']] += 1
            self.ignored = {bucket: {g for g, n in count.items() if n >= math.ceil(.2 * sizes[bucket])}
                            for bucket, count in counts.items()}
        for row in rows:
            self.add(row)

    def add(self, row):
        row = dict(row)
        text = normalize(row['text'])
        row['normalized'] = text
        grams = token_grams(text) - self.ignored.get(row['bucket'], set())
        i = len(self.rows)
        self.rows.append(row)
        self.grams.append(grams)
        self.exact.setdefault(text, i)
        if len(grams) < 8 and len(text) >= 20:
            self.short.append(i)
        for gram in grams:
            self.inverted[gram].append(i)

    def match(self, text):
        text = normalize(text)
        if text in self.exact:
            row = self.rows[self.exact[text]]
            return dict(reason='normalized_exact', protected_id=row['id'], bucket=row['bucket'])
        for i in self.short:
            row = self.rows[i]
            if row['normalized'] in text:
                return dict(reason='short_question_containment', protected_id=row['id'], bucket=row['bucket'])
        grams = token_grams(text)
        candidates = Counter(i for gram in grams for i in self.inverted.get(gram, ()))
        for i, shared in sorted(candidates.items(), key=lambda x: (-x[1], x[0])):
            if shared < 8:
                continue
            row, known = self.rows[i], self.grams[i]
            filtered = grams - self.ignored.get(row['bucket'], set())
            union = len(known) + len(filtered) - shared
            jaccard = shared / union if union else 0.
            # Additional conservative containment protection catches wrapped/copied questions.
            contained = len(known) >= 8 and row['normalized'] in text
            if jaccard >= .9 or contained:
                return dict(reason='question_containment' if contained else '5gram_jaccard',
                            protected_id=row['id'], bucket=row['bucket'], jaccard=jaccard,
                            shared_5grams=shared)
        return None


def protected_rows(root):
    import pyarrow.parquet as pq
    import yaml
    source = root / 'results/loopfill-latency512-20261107/inputs/sources'
    result, sources = [], {}

    def load(name):
        path = source / name
        data = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return data

    def add(identity, bucket, text):
        if isinstance(text, str) and text.strip():
            result.append(dict(id=identity, bucket=bucket, text=text))

    gsm = [json.loads(line) for line in load('gsm8k-test.jsonl').decode().splitlines()]
    assert len(gsm) == 1319
    for i, row in enumerate(gsm):
        add(f'gsm8k/test/{i}', 'gsm8k', row['question'])
    demos = yaml.safe_load(load('gsm8k-cot.yaml'))['fewshot_config']['samples'][:3]
    for i, row in enumerate(demos):
        add(f'gsm8k/demo/{i}', 'gsm8k-demonstrations', row['question'])
    math_rows = [json.loads(line) for line in load('math500-test.jsonl').decode().splitlines()]
    assert len(math_rows) == 500
    for i, row in enumerate(math_rows):
        add(f'math500/{i}', 'math500', row['problem'])
    tree = ast.parse(load('minerva-utils.py'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'list_fewshot_samples')
    demos = ast.literal_eval(next(n for n in node.body if isinstance(n, ast.Return)).value)
    for i, row in enumerate(demos):
        add(f'math500/demo/{i}', 'math-demonstrations', row['problem'])
    for path in sorted(source.glob('bbh-*.json')):
        task = path.stem[4:]
        for i, row in enumerate(json.loads(load(path.name))['examples']):
            add(f'bbh/{task}/{i}', 'bbh/' + task, row['input'])
        demo = load(f'bbh-{task}.txt').decode().split('-----', 1)[-1]
        for i, text in enumerate(re.findall(r'(?:^|\n)Q:\s*(.*?)\nA:', demo, flags=re.S)):
            add(f'bbh/{task}/demo/{i}', 'bbh/' + task, text)
    for dataset, expected in [('humanevalplus', 164), ('mbppplus', 378)]:
        load(dataset + '.parquet')
        rows = pq.read_table(source / (dataset + '.parquet')).to_pylist()
        assert len(rows) == expected
        for row in rows:
            add(f'{dataset}/{row["task_id"]}', dataset, row['prompt'])
            try:
                parsed = ast.parse(row['prompt'])
                for i, node in enumerate(ast.walk(parsed)):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = ast.get_docstring(node)
                        if doc:
                            add(f'{dataset}/{row["task_id"]}/docstring/{i}', dataset, doc)
            except SyntaxError:
                pass
    spec = [json.loads(line) for line in load('specbench480.jsonl').decode().splitlines()]
    assert len(spec) == 480
    for row in spec:
        for i, turn in enumerate(row['turns']):
            add(f'specbench/{row["question_id"]}/turn/{i}', 'specbench/' + row['category'], turn)
    # Protect all original MBPP examples too, including the larger original-paper evaluation set.
    path = root / 'experiments/baselines/data/raw/mbpp-all.jsonl'
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        add(f'mbpp-original/{row["task_id"]}', 'mbpp-original-all', row['text'])
    return result, sources


def write_protection(root, out):
    rows, hashes = protected_rows(root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protected.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = dict(count=len(rows), counts=dict(Counter(r['bucket'] for r in rows)), source_sha256=hashes,
                  protected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  policy='NFKC/casefold/whitespace exact; punctuation-preserving token5gram '
                         'Jaccard>=.9 with>=8 shared; BBH per-task boilerplate>=20% omitted; '
                         'additional literal question containment; all original MBPP and all SpecBench turns protected',
                  limitation='Lexical isolation cannot prove absence of semantic paraphrases.')
    (out / 'protection-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return rows
