"""Publish the already measured twenty continuous-throughput cells, without GPU work."""
import argparse
import csv
import itertools
import math
from pathlib import Path
import time

from packed_dflash_evidence import read, rows, save, sha

BASE = Path(__file__).resolve().parent
DF = Path('/ossfs/workspace/loop_suffix/results/dflash-table12-packed-20261112-v2')
LA = Path('/ossfs/workspace/loop_suffix/results/loopahead-table2-20261112-v2')
OLD = BASE.parents[1]/'results/loopfill-table2-20261107-v2'
MODELS = ('huginn-3.5b', 'ouro-2.6b')
BATCHES = (1, 2, 4, 8, 16)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    reports = dict(dflash=BASE/'preflight/completed-dflash-throughput-v2/report.json',
                   loopahead=BASE/'preflight/user-resume-eight-gpu-v1/loopahead-throughput-audit.json')
    audits = {k: read(v) for k, v in reports.items()}
    sources = dict(dflash=BASE/'audit_completed_dflash_throughput.py',
                   loopahead=reports['loopahead'].with_name('audit_loopahead_throughput.py'))
    protocols = {name: read(root/'protocol.json') for name, root in [('dflash', DF), ('loopahead', LA)]}
    for method, root in [('dflash', DF), ('loopahead', LA)]:
        assert audits[method]['status'] == 'complete_throughput_artifact_audit'
        assert sha(sources[method]) == audits[method]['source_sha256']
        assert sha(root/'protocol.json') == audits[method]['protocol_sha256']
        for relative, expected in protocols[method]['source_sha256'].items():
            assert sha(root/'source'/relative) == expected, relative
    assert sha(OLD/'protocol.json') == protocols['loopahead']['table2_protocol_sha256']
    assert sha(OLD/'measurements.csv') == protocols['dflash']['reference_sha256']['references/table2-measurements.csv']
    assert sha(DF/'references/table2-measurements.csv') == sha(OLD/'measurements.csv')
    old_protocol = read(OLD/'protocol.json')
    for relative, expected in old_protocol['source_sha256'].items():
        assert sha(OLD/'source'/relative) == expected
    historical = list(csv.DictReader((OLD/'measurements.csv').open()))
    references = {(r['model'], int(r['batch']), r['arm']): r for r in historical}
    jobs = {j['id']: j for j in read(DF/'jobs.json') if j['kind'] == 'throughput'}
    input_evidence = {}
    for model in MODELS:
        meta = read(LA/f'runs/{model}-b1/metadata.json')
        original = rows(meta['input_path'])
        job = jobs[f'table2-{model}-b1']
        adapted = rows(DF/job['inputs'])
        assert [(r['question_id'], r['prompt_token_ids']) for r in original] == [
            (r['source_id'], r['prompt_tokens']) for r in adapted]
        assert len(original) == len(adapted) == 480
        assert sha(meta['input_path']) == meta['input_sha256'] == protocols['loopahead']['models'][model]['input_sha256']
        assert sha(DF/job['inputs']) == job['inputs_sha256']
        input_evidence[model] = dict(requests=480, identical_request_order_and_prompt_tokens=True,
            loopahead_input=dict(path=meta['input_path'], sha256=sha(meta['input_path'])),
            dflash_input=dict(path=str(DF/job['inputs']), sha256=sha(DF/job['inputs'])))

    results, proof = [], {}
    for method in ['dflash', 'loopahead']:
        audited = audits[method]['evidence'] if method == 'dflash' else audits[method]['cells']
        assert len(audited) == 10
        for cell in audited:
            assert cell['actual_exit_code'] == 0
            for filename, expected in cell['artifacts'].items():
                assert sha(filename) == expected, filename
            if method == 'dflash':
                old_row = next(r for r in audits[method]['results'] if Path(r['run']).parent.name == cell['id'])
                run = Path(old_row['run'])
                model, batch = old_row['model'], old_row['batch']
                job = jobs[cell['id']]
                assert job['max_new_tokens'] == 1024 and job['temperature'] == 0
                checkpoint = Path(protocols[method]['models'][model]['checkpoint'])
                for name, expected in cell['checkpoint_sha256'].items():
                    assert sha(checkpoint/name) == expected
                denominator = lambda record: record['stats']['target_verifications']
                depth = 1  # A DFlash verification already includes all target loops.
            else:
                old_row = cell['row']
                model, batch = old_row['model'], int(old_row['batch'])
                run = LA/'runs'/cell['id']
                meta = read(run/'metadata.json')
                assert meta['max_new_tokens'] == 1024 and meta['temperature'] == 0
                assert sha(meta['checkpoint']) == protocols[method]['models'][model]['sha256']
                for name, expected in meta['source_sha256'].items():
                    assert sha(Path(cell['actual_source_root'])/name) == expected
                assert cell['placement_only_source_change'] in ([], ['configs/gpu-allocation.json'])
                denominator = lambda record: record['stats']['calls']
                depth = meta['config']['total']
            meta, summary = read(run/'metadata.json'), read(run/'summary.json')
            records = rows(run/'requests.jsonl')
            assert len(records) == summary['requests'] == 480
            assert len({r['request_id'] for r in records}) == 480
            assert all(1 <= len(r['tokens']) <= 1024 for r in records)
            assert summary['concurrency'] == batch
            assert summary['input_tokens'] == summary['prefill_chunk_tokens']
            assert not meta['graph'] and not meta['compile'] and summary['core_padding_rows'] == 0
            assert summary['output_tokens'] == sum(len(r['tokens']) for r in records)
            throughput = summary['output_tokens']/summary['wall_s']
            gamma = depth*sum(len(r['tokens'])-1 for r in records)/sum(denominator(r) for r in records)
            assert math.isclose(throughput, summary['output_tput'], rel_tol=1e-12)
            assert math.isclose(gamma, summary['gamma'], rel_tol=1e-12)
            ar, lf = [float(references[(model, batch, arm)]['output_tput']) for arm in ['ar', 'budget-online']]
            source_throughput = old_row['output_tok_s' if method=='dflash' else 'output_tput']
            assert float(source_throughput) == throughput
            assert float(old_row['gamma']) == summary['gamma']
            assert float(old_row['ratio_to_historical_ar']) == throughput/ar
            assert float(old_row['ratio_to_historical_loopfill']) == throughput/lf
            result = dict(method=method, model=model, benchmark='specbench', temperature=0, batch=batch,
                requests=480, max_new_tokens=1024, input_tokens=summary['input_tokens'],
                output_tokens=summary['output_tokens'], wall_s=summary['wall_s'],
                output_tok_s=throughput, gamma=summary['gamma'], historical_ar_tok_s=ar,
                historical_loopfill_tok_s=lf, ratio_to_historical_ar=throughput/ar,
                ratio_to_historical_loopfill=throughput/lf, actual_exit_code=0, run=str(run),
                status='measured_with_declared_limitations',
                timing='continuous finite-stream E2E including prefill,scheduling,fsynced journal',
                reference_policy='same-host historical AR/LoopFill; cross-date; no drift correction')
            results.append(result)
            proof[f'{method}/{model}/b{batch}'] = cell
    expected = set(itertools.product(['dflash', 'loopahead'], MODELS, BATCHES))
    assert {(r['method'], r['model'], r['batch']) for r in results} == expected and len(results) == 20
    results.sort(key=lambda r:(MODELS.index(r['model']), r['batch'], r['method']))
    with (out/'throughput-results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lookup = {(r['method'],r['model'],r['batch']):r for r in results}
    lines = ['# SpecBench 连续批量端到端吞吐', '',
        '已完成20组：两个正文模型 × B=1/2/4/8/16 × 两种方法，每组480题。'
        '两种方法的题目顺序与prompt token逐项相同，全部测量有真实exit0。', '',
        'BF16，温度0，max_new_tokens=1024，max_seq=2688，128-token分块prefill，continuous batching，'
        'Graph/compile/目标行padding关闭，每个worker绑定4个CPU核、单线程；每格一次测量。'
        '吞吐为全部输出token（含prefill产生的首token）除以有限请求流总壁钟，包含prefill、调度及fsync日志写入，排除加载与预热。'
        'γ按decode token计算，排除各请求的prefill首token。', '',
        '表中AR和LoopFill均为原表同机历史测量；新方法相对AR的列是端到端吞吐比，未做跨日期漂移校正。'
        '这20组是额外吞吐测评，与72组max512纯decode延迟矩阵分别报告。', '',
        '| 模型 | B | 历史AR tok/s | 历史LoopFill tok/s | DFlash tok/s | DFlash / AR | DFlash γ | LoopAhead† tok/s | LoopAhead / AR | LoopAhead γ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model, batch in itertools.product(MODELS, BATCHES):
        d, l = lookup['dflash', model, batch], lookup['loopahead', model, batch]
        assert d['historical_ar_tok_s'] == l['historical_ar_tok_s'] and d['historical_loopfill_tok_s'] == l['historical_loopfill_tok_s']
        lines.append(f"| {model} | {batch} | {d['historical_ar_tok_s']:.2f} | {d['historical_loopfill_tok_s']:.2f} | "
            f"{d['output_tok_s']:.2f} | {d['ratio_to_historical_ar']:.2f}× | {d['gamma']:.2f} | "
            f"{l['output_tok_s']:.2f} | {l['ratio_to_historical_ar']:.2f}× | {l['gamma']:.2f} |")
    lines += ['', '† LoopAhead为论文自行实现；Huginn为迁移适配。DFlash使用已完成正式训练的草稿器及官方解码/训练组件，'
        '通过我们的packed完整循环目标适配器执行；Huginn为332.6M投影窄版。两种方法均使用冻结检查点，逐格来源见throughput-evidence.json。', '',
        '限制：共享主机、单次测量、无置信区间；BF16可能产生不同输出，吞吐和γ不证明逐token数值等价或任务质量相同。'
        'Ouro1.4B未包含在这份已完成的额外吞吐矩阵中；其常规延迟矩阵另行完成。'
        '本报告不把历史AR吞吐比写成新测同栈AR加速比。', '',
        '复现汇总：`python experiments/baselines/collect_throughput_results.py`。此命令仅重验与汇总已存测量，不启动GPU任务。', '']
    (out/'throughput-summary.md').write_text('\n'.join(lines))
    save(out/'throughput-evidence.json', dict(status='complete_throughput_artifact_summary', measured_cells=20,
        requests=9600, source_sha256=sha(Path(__file__)),
        audit_reports={k:dict(path=str(v),sha256=sha(v)) for k,v in reports.items()},
        input_alignment=input_evidence, cells=proof,
        historical_measurements=dict(path=str(OLD/'measurements.csv'),sha256=sha(OLD/'measurements.csv')),
        results_sha256=sha(out/'throughput-results.csv'), summary_sha256=sha(out/'throughput-summary.md'),
        new_gpu_runs=0, checked_unix=time.time()))
    print(dict(cells=20, requests=9600, new_gpu_runs=0, output=str(out/'throughput-summary.md')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE/'combined-results')
    main(parser.parse_args().out)
