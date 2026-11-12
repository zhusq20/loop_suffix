"""Observe real on-path candidate verification; diagnostic time is not throughput."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_memory_editability.loop_prob_tree import Calibration, ProbConfig, MIN_COUNT
from llm_memory_editability.loop_prob_decode import prob_tree_decode
from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache


class AuditCalibration(Calibration):
    def __init__(self, base):
        super().__init__(base.shallow, base.suffix, base.insitu, base.joint)
        self.trace = []
        self.nesting = 0

    def _lookup(self, table, key, levels, default=0.0):
        kind = next(name for name in ['shallow', 'suffix', 'insitu', 'joint'] if getattr(self, name) is table)
        fine_n = 0
        for level, keep in enumerate(levels):
            hits = n = 0
            for k, (h, m) in table.items():
                if keep(k, key):
                    hits += h
                    n += m
            if level == 0:
                fine_n = n
            if n >= MIN_COUNT:
                self.trace.append(dict(table=kind, level=level, hits=hits, count=n, fine_count=fine_n))
                return hits / n
        self.trace.append(dict(table=kind, level=None, fine_count=fine_n, count=n, default=default))
        return default


def wrap(name):
    original = getattr(Calibration, name)
    def method(self, *args):
        if self.nesting == 0:
            self._cache.clear()
            self.trace = []
        self.nesting += 1
        try:
            return original(self, *args)
        finally:
            self.nesting -= 1
    return method


for name in ['shallow_prob', 'suffix_prob', 'suffix_prob_insitu', 'joint_prob']:
    setattr(AuditCalibration, name, wrap(name))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ['model', 'model-path', 'questions', 'calibration', 'out']:
        p.add_argument('--' + field, required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = list(map(json.loads, Path(a.questions).read_text().splitlines()))
    rows.sort(key=lambda r: r['question_id'])
    random.Random(0).shuffle(rows)
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, trust_remote_code=True,
                local_files_only=True, torch_dtype=torch.bfloat16).eval().to('cuda')
    ouro = a.model.startswith('ouro')
    T, d, deep = (4, 1, 2) if ouro else (32, 4, 8)
    if ouro:
        model.model.total_ut_steps = T
    backbone = OuroLoop(model, T, 4096, 1024) if ouro else HuginnLoop(model, T, 4096, 1024, 0, 4096, extra_depths=(deep,))
    calibration = AuditCalibration(Calibration.from_json(json.loads(Path(a.calibration).read_text())['dev']))
    cfg = ProbConfig('probe-C.01', T, d, (d, deep), tau=.01, joint=True, prune=True, suffix_calibration='insitu')
    eos = tok.eos_token_id if ouro else model.config.eos_token_id
    stop = set(eos if isinstance(eos, list) else [eos])
    stop.add(tok.convert_tokens_to_ids('<|im_end|>' if ouro else '<|end_turn|>'))
    cache = CompletionSuffixCache.arctic()
    meta = dict(model=a.model, calibration_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),
                questions_sha256=hashlib.sha256(Path(a.questions).read_bytes()).hexdigest(),
                tau=.01, dtype='bfloat16', max_new_tokens=256, instrumented=True,
                events='score-query weighted; labeled only after parent reached the verified greedy prefix')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    with (out/'events.jsonl').open('x') as events, (out/'requests.jsonl').open('x') as requests:
        for row in rows:
            pending, verified = {}, {}
            counts = dict(scored=0, labeled=0, discarded_unverified=0)
            def emit(record, gold, position):
                record.update(gold=gold, hit=record['token'] == gold, output_position=position,
                              request_id=row['question_id'])
                events.write(json.dumps(record) + '\n')
                counts['labeled'] += 1
            def observe(kind, **data):
                if kind == 'reset':
                    counts['discarded_unverified'] += sum(map(len, pending.values()))
                    pending.clear()
                    verified.clear()
                elif kind == 'score':
                    counts['scored'] += 1
                    record = dict(data, lookup=list(calibration.trace))
                    parent = data['parent']
                    if parent in verified:
                        emit(record, *verified[parent])
                    else:
                        pending.setdefault(parent, []).append(record)
                elif kind == 'verify':
                    parent = data['parent']
                    verified[parent] = (data['gold'], data['output_position'])
                    for record in pending.pop(parent, []):
                        emit(record, *verified[parent])
            start = time.monotonic()
            tokens, stats = prob_tree_decode(backbone, row['prompt_token_ids'], cfg, calibration,
                                256, stop, cache, row['question_id'], max_nodes=1024, observer=observe)
            counts['discarded_unverified'] += sum(map(len, pending.values()))
            assert counts['scored'] == counts['labeled'] + counts['discarded_unverified']
            record = dict(request_id=row['question_id'], tokens=tokens, stats=stats,
                          counts=counts, instrumented_wall_s=time.monotonic()-start)
            requests.write(json.dumps(record)+'\n')
            requests.flush()
            events.flush()
            print(json.dumps(dict(request_id=row['question_id'], tokens=len(tokens), **counts)), flush=True)
    (out/'complete.json').write_text(json.dumps({'requests':len(rows),'history':cache.publications}))


if __name__ == '__main__':
    main()
