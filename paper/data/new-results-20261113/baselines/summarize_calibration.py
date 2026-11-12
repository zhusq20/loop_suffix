"""Validate and summarize completed calibration journals, never paper baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def journal(path):
    return [json.loads(x) for x in path.open() if x.strip()]


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return dict(output_position_0based=i, ar_token=x, loopspec_token=y)
    if len(a) != len(b):
        i = min(len(a), len(b))
        return dict(output_position_0based=i, ar_token=a[i] if i < len(a) else None,
                    loopspec_token=b[i] if i < len(b) else None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, choices=[20,300], default=20)
    count = parser.parse_args().count
    result, differences = [], []
    for model, depth in [('huginn-3.5b', 32), ('ouro-2.6b', 4)]:
        methods = {}
        for method in ['ar', 'loopspec-dual']:
            name = f'{model}-gsm8k-{method}'
            run = BASE / 'preflight' / (name + f'-n{count}')
            path = run / 'requests.jsonl'
            complete, metadata = read(run / 'complete.json'), read(run / 'metadata.json')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == complete['requests_sha256']
            rows = journal(path)
            assert len(rows) == complete['requests'] == metadata['count'] == count
            assert metadata['max_new_tokens'] == 512 and metadata['full_benchmark'] == (count == 300)
            historic = journal(ROOT / 'results/loopfill-latency512-20261107/runs'
                               / name / 'requests.jsonl')[:len(rows)]
            assert [(r['source_id'], r['prompt_sha256']) for r in rows] == [
                (r['source_id'], r['prompt_sha256']) for r in historic]
            seconds = sum(r['stats']['decode_s'] for r in rows)
            tokens = sum(r['stats']['tokens'] for r in rows)
            calls = sum(r['stats']['calls'] for r in rows)
            summary = read(run / 'summary.json')
            assert abs(seconds - summary['decode_s']) < 1e-7
            assert tokens == summary['decode_tokens'] and calls == summary['loop_steps']
            assert all(len(r['tokens']) == r['stats']['tokens'] + 1 for r in rows)
            if method == 'ar':
                assert all(r['stats']['calls'] == depth * r['stats']['tokens'] for r in rows)
            methods[method] = dict(rows=rows, decode_s=seconds, tok_s=tokens / seconds,
                gamma=depth * tokens / calls,
                repeat_sequences=sum(a['tokens'] == b['tokens'] for a, b in zip(rows, historic)),
                repeat_calls=sum(a['stats']['calls'] == b['stats']['calls'] for a, b in zip(rows, historic)))
        ar, ls = methods['ar'], methods['loopspec-dual']
        matches = 0
        for a, b in zip(ar['rows'], ls['rows']):
            assert (a['source_id'], a['prompt_sha256']) == (b['source_id'], b['prompt_sha256'])
            difference = first_difference(a['tokens'], b['tokens'])
            if difference is None:
                matches += 1
            else:
                differences.append(dict(model=model, source_id=a['source_id'], **difference,
                    diagnosis='unresolved; same-method historical replay is exact, cause not yet established'))
        result.append(dict(model=model, requests=count, ar_tok_s=ar['tok_s'],
            loopspec_tok_s=ls['tok_s'], eta_latency=ar['decode_s'] / ls['decode_s'],
            eta_throughput=ls['tok_s'] / ar['tok_s'], gamma=ls['gamma'],
            ar_vs_loopspec_sequence_matches=matches,
            ar_repeat_sequences=ar['repeat_sequences'], loopspec_repeat_sequences=ls['repeat_sequences'],
            ar_repeat_calls=ar['repeat_calls'], loopspec_repeat_calls=ls['repeat_calls'],
            status='full300_environment_calibration' if count == 300 else 'calibration_only_NOT_full_benchmark'))
    out = BASE / 'preflight'
    with (out / f'calibration-n{count}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (out / f'calibration-first-divergences-n{count}.json').write_text(
        json.dumps(differences, ensure_ascii=False, indent=2) + '\n')
    lines = [f'# GSM8K {count}题环境校验（新运行）', '',
        '四个运行均完成，逐题日志校验和、题序、计数与汇总已核对。max512、BF16、Graph OFF。',
        '这是环境校验，不是DFlash/LoopAhead的结果。', '',
        '| 模型 | AR tok/s | LoopSpec tok/s | η时间比 | η吞吐比 | γ | AR与LoopSpec整序列相同 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result:
        lines.append(f"| {r['model']} | {r['ar_tok_s']:.6f} | {r['loopspec_tok_s']:.6f} | "
                     f"{r['eta_latency']:.6f} | {r['eta_throughput']:.6f} | {r['gamma']:.6f} | "
                     f"{r['ar_vs_loopspec_sequence_matches']}/{count} |")
    lines += ['', '与同方法历史运行的相同题目相比：']
    for r in result:
        lines.append(f"- {r['model']}: AR输出{r['ar_repeat_sequences']}/{count}、循环数{r['ar_repeat_calls']}/{count}一致；"
                     f"LoopSpec输出{r['loopspec_repeat_sequences']}/{count}、循环数{r['loopspec_repeat_calls']}/{count}一致。")
    lines += ['', '同方法可重复与AR/LoopSpec逐token一致是不同的检查。旧AR/LoopSpec差异尚未做共同前缀诊断，不能直接归因于BF16。',
        f'第一分歧位置见 calibration-first-divergences-n{count}.json。',
        '论文参考γ=4.88/3.59来自全300题；20题子集不能用于完整配置验收。',
        '用户已授权完整校验及后续任务；无需再次确认。', '']
    (out / f'calibration-n{count}.md').write_text('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
