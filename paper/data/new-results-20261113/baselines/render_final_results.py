"""Stage the complete paper table without overwriting the running collectors.

Requires a real 72-cell collection and complete 39-group numerical evidence.
This prepares only results.csv/table2_rows.md, not the final training-cost or
research-completion audit. It never starts inference or changes source results.
"""
import argparse
import csv
import io
import itertools
import math
from pathlib import Path
import statistics
import time

from audit_numerical_matrix import completion_gate
from packed_dflash_evidence import read, save, sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
MODELS = ['huginn-3.5b','ouro-2.6b','ouro-1.4b']
METHODS = ['dflash','loopahead']
TASKS = dict(gsm8k=300,math500=500,bbh=270,humanevalplus=164,mbppplus=378,specbench=480)
EXPECTED = set(itertools.product(METHODS,MODELS,TASKS,[0,1]))
UNKNOWN_EXITS = {('dflash','huginn-3.5b','math500',0),
                 ('dflash','huginn-3.5b','math500',1),
                 ('dflash','ouro-2.6b','specbench',1)}


def key(row):
    return row['method'],row['model'],row['benchmark'],int(row['temperature'])


def validate_matrix(items):
    assert len(items) == 72 and {key(r) for r in items} == EXPECTED, 'All 72 unique formal cells are required.'
    unknown = set()
    for row in items:
        assert row['status'] == 'complete_with_declared_limitations'
        for name in ['eta','eta_latency','eta_throughput','gamma','method_tok_s','ar_tok_s',
                     'method_decode_tokens','ar_decode_tokens','method_decode_s','ar_decode_s']:
            assert math.isfinite(float(row[name])) and float(row[name]) > 0,name
        for name in ['match_rate','aligned_token_match_rate']:
            assert math.isfinite(float(row[name])) and 0 <= float(row[name]) <= 1
        assert float(row['eta']) == float(row['eta_throughput'])
        assert math.isclose(float(row['eta_latency']),float(row['ar_decode_s'])/float(row['method_decode_s']),rel_tol=1e-12)
        assert math.isclose(float(row['eta_throughput']),float(row['method_tok_s'])/float(row['ar_tok_s']),rel_tol=1e-12)
        for prefix in ['method','ar']:
            assert math.isclose(float(row[prefix+'_tok_s']),
                float(row[prefix+'_decode_tokens'])/float(row[prefix+'_decode_s']),rel_tol=1e-12)
        if row['method'] == 'dflash':
            assert row['engine'] in {'official','official_dflash_with_packed_loop_target'}
            assert row['target_backend'] in {'legacy_HF_full_loop','packed_full_loop'}
            assert math.isfinite(float(row['drafter_time_frac'])) and 0 <= float(row['drafter_time_frac']) <= 1
            assert row['profile_run']
        else:
            assert row['engine'] == 'ours' and row['target_backend'] == 'packed_LoopAhead'
            assert row['drafter_time_frac'] == 'NA'
        if row['method_exit_evidence'] == 'unavailable_artifacts_verified':
            unknown.add(key(row))
        else:
            assert row['method_exit_evidence'] == 'actual_exit0'
    assert unknown == UNKNOWN_EXITS, 'Unknown exits must retain the exact documented scope.'


def normalize(items, numerical):
    """Preserve every measurement and retain original annotation values."""
    completion_gate(numerical['results'],numerical['pending'],numerical['errors'])
    validate_matrix(items)
    normalized = []
    for row in items:
        record = dict(row)
        for field in ['engine','notes','numerical_diagnostics']:
            record['collected_'+field] = row[field]
        record['engine'] = 'official' if row['method'] == 'dflash' else 'ours'
        record['notes'] = row['notes'].replace('numerical diagnostics not all completed. ', '')
        record['notes'] += ' Engine denotes method provenance; target_backend retains the actual adapted runtime.'
        scope = '/'.join(['formal',row['method'],row['model'],row['benchmark']])
        checked = numerical['results'][scope]
        if int(row['temperature']) == 0:
            assert checked['whole_sequence_match_rate_bf16'] == float(row['match_rate'])
            assert checked['requests'] == TASKS[row['benchmark']]
            assert checked['first_divergences'] == checked['fp32_argmax_equal']
            record['numerical_diagnostics'] = 'all_observed_T0_first_divergences_checked'
        else:
            record['numerical_diagnostics'] = 'T0_checks_do_not_establish_T1_distribution'
        assert all(record[k] == v for k,v in row.items() if k not in ['engine','notes','numerical_diagnostics'])
        normalized.append(record)
    return normalized


def table_text(items):
    validate_matrix([{**r,'engine':r.get('collected_engine',r['engine'])} for r in items])
    lookup = {key(row):row for row in items}
    lines = ['| Model | Temp. | Method | GSM8K η γ | MATH-500 η γ | BBH η γ | HumanEval+ η γ | MBPP+ η γ | SpecBench η γ | Mean η γ |',
             '|'+'---|'*10]
    for model,temp,method in itertools.product(MODELS,[0,1],METHODS):
        cells = [lookup[(method,model,task,temp)] for task in TASKS]
        values = []
        for row in cells:
            backend = (' H' if row['target_backend'] == 'legacy_HF_full_loop' else ' P') if method == 'dflash' else ''
            unknown = ' ‡' if key(row) in UNKNOWN_EXITS else ''
            values.append(f"{float(row['eta_latency']):.2f} {float(row['gamma']):.2f}{backend}{unknown}")
        mean = f"{statistics.mean(float(r['eta_latency']) for r in cells):.2f} {statistics.mean(float(r['gamma']) for r in cells):.2f}"
        if len({r['target_backend'] for r in cells}) > 1:
            mean += '（混合实现）'
        lines.append('| '+' | '.join([model,str(temp),'DFlash*' if method == 'dflash' else 'LoopAhead†',*values,mean])+' |')
    lines += ['', 'η沿用原表的总decode时间比；CSV的eta/eta_throughput另报吞吐比。Mean为六任务算术平均。Ouro-1.4B行用于附录表4。', '',
        '* DFlash采用固定的官方草稿器与适配解码逻辑，每格使用自身同栈实测AR。H为HF循环目标实现，P为packed循环目标实现；混合实现的Mean不能称为同一目标后端的统一测量。Huginn为332.6M缩窄适配，Ouro-1.4B为163.6M适配。训练使用50k混合数据，来源为Nemotron V1（代码经TACO恢复）、CodeAlpaca及UltraChat；与原论文配方的差异见DESIGN.md。CSV的engine表示方法来源，实际目标实现单列为target_backend。', '',
        '† LoopAhead为按论文自行实现；Huginn为迁移适配。AR沿用原同主机冻结测量，跨日期时间漂移未校正；比较时应同时查看绝对速度、输出长度及AR来源。', '',
        '‡ 原测速实际退出凭证缺失：Huginn DFlash MATH/T0、T1及Ouro-2.6B DFlash SpecBench/T1。完整测量产物、同栈AR及独立profile已核验；后两者的实际退出0不能补充为原测速的退出凭证。', '',
        '72组均使用冻结题序/题数（300/500/270/164/378/480）、max512、BF16、batch1、Graph OFF。decode计时含草稿、调度及验证，不含prefill。', '',
        '测量来自共享主机，不能声称整机CPU隔离。部分Ouro-1.4B AR/profile运行与本仓库自动Git整理重叠；发现后将该整理限制到CPU80–83，原始时间未作推算修正。具体范围见EVAL_CONFIG.md与preflight/git-housekeeping-affinity-v1/report.json。', '',
        '36个正式T0格子、B4及两组开发对照的首次分歧检查已完成，原BF16整段输出差异仍保留。局部FP32共同前缀检查不等于完整FP32 benchmark；T1同seed一致率仅作描述，不能用来证明采样分布。训练成本和最终文档验收单独交付。', '']
    return '\n'.join(lines)


def render(collection_path, numerical_path, numerical_exit, out):
    assert not out.exists(), 'Use a fresh staging directory; never overwrite a collector or prior report.'
    csv_path = BASE/'combined-results/results.csv'
    snapshot = csv_path.read_bytes()
    items = list(csv.DictReader(io.StringIO(snapshot.decode())))
    validate_matrix(items)
    collection, numerical, exit_record = map(read,[collection_path,numerical_path,numerical_exit])
    assert collection['status'] == 'passed' and collection['completed'] == collection['expected'] == 72
    assert collection['all_previous_rows_unchanged'] and collection['results_sha256'] == sha(csv_path)
    assert collection['evidence_sha256'] == sha(BASE/'combined-results/evidence.json')
    for receipt in collection['collector_exits'].values():
        assert sha(receipt['path']) == receipt['sha256'] and read(receipt['path'])['exit_code'] == 0
    assert type(exit_record['exit_code']) is int and exit_record['exit_code'] == 0
    assert str(BASE/'audit_numerical_matrix.py') in exit_record['argv']
    audit_source = exit_record['source_sha256']
    if isinstance(audit_source,dict):
        audit_source = audit_source['audit_numerical_matrix.py']
    assert audit_source == numerical['source_sha256']['audit_numerical_matrix.py']
    assert Path(exit_record['argv'][exit_record['argv'].index('--out')+1]).resolve() == numerical_path.parent.resolve()
    assert exit_record['started_unix'] <= numerical['checked_unix'] <= exit_record['finished_unix']
    assert numerical['status'] == 'complete_numerical_scope' and numerical['complete']
    assert numerical['unchanged_results_csv_sha256'] == sha(csv_path)
    completion_gate(numerical['results'],numerical['pending'],numerical['errors'])
    for name,digest in numerical['source_sha256'].items():
        assert sha(numerical_path.parent/'source'/name) == digest
    for name,digest in numerical['plan_sha256'].items():
        assert sha(name) == digest
    verified = {}

    def check(path, digest):
        if path not in verified:
            verified[path] = sha(path)
        assert verified[path] == digest,path

    for group in numerical['results'].values():
        for path,digest in group['artifact_sha256'].items():
            check(path,digest)
        for receipt in group['actual_exits']:
            check(receipt['path'],receipt['sha256'])
            assert read(receipt['path'])['exit_code'] == 0
    evidence = read(BASE/'combined-results/evidence.json')
    assert set(evidence['cells']) == {'/'.join(map(str,key(r))) for r in items}
    for row in items:
        proof = evidence['cells']['/'.join(map(str,key(row)))]
        assert proof['agreement']['requests'] == TASKS[row['benchmark']]
        for record in proof['sources'].values():
            for filename,digest in record['sha256'].items():
                check(str(Path(record['path'])/filename),digest)
        if key(row) in UNKNOWN_EXITS:
            assert proof['original_timed_exit']['actual_exit_code'] is None
    output = normalize(items,numerical)
    text = table_text(output)
    assert csv_path.read_bytes() == snapshot
    out.mkdir(parents=True)
    (out/'collected-results.csv').write_bytes(snapshot)
    with (out/'results.csv').open('x') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(output[0]))
        writer.writeheader();writer.writerows(output)
    (out/'table2_rows.md').write_text(text)
    freeze_sources(out,[Path(__file__),BASE/'audit_numerical_matrix.py',BASE/'packed_dflash_evidence.py',BASE/'provenance.py'])
    save(out/'report.json',dict(status='complete_results_staged_not_full_goal_acceptance',cells=72,
        actual_original_exit0_cells=69,unknown_original_exits=sorted(UNKNOWN_EXITS),
        all_measurement_fields_unchanged=True,new_gpu_runs=0,
        required_numerical_groups=39,verified_artifact_sha256=verified,
        input_sha256={str(p):sha(p) for p in [csv_path,collection_path,numerical_path,numerical_exit,BASE/'combined-results/evidence.json']},
        output_sha256={name:sha(out/name) for name in ['results.csv','table2_rows.md','collected-results.csv']},
        checked_unix=time.time(),
        limitations=['Only results CSV and paper rows staged; this does not publish over active collectors.',
                     'Six training audits, final measured costs, T1 evidence and final document review remain separate completion requirements.']))
    print(dict(staged=str(out),cells=72,original_values_preserved=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-report',type=Path,required=True)
    parser.add_argument('--numerical-report',type=Path,required=True)
    parser.add_argument('--numerical-exit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    render(args.collection_report.resolve(),args.numerical_report.resolve(),args.numerical_exit.resolve(),args.out.resolve())
