"""Render measured costs; refuse a final report until all six trainings finish."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import collect_costs
import dflash_training_registry as dflash_registry

BASE = Path(__file__).resolve().parent
MODELS = ['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--out', type=Path, default=BASE/'training_cost.md')
    args = parser.parse_args()
    if args.partial and args.out.resolve() == (BASE/'training_cost.md').resolve():
        parser.error('Partial reports require a separate --out; final training_cost.md is protected')

    collect_costs.main()
    ledger_path = BASE/'training/cost-ledger.json'
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    ledger_hash = hashlib.sha256(ledger_bytes).hexdigest()
    ledger_snapshot = args.out.with_name(args.out.stem + '-ledger.json')
    manifest = read(BASE/'data/manifests/manifest.json')['files']
    audit = read(BASE/'data/manifest-audit.json')
    human_eval = read(BASE/'data/paper-inputs/humaneval-isolation-audit.json')
    assert audit['passed'] and audit['exact_test_and_cross_method_holdout_overlap'] == 0
    assert human_eval['passed'] and not human_eval['matches']
    for name, item in manifest.items():
        assert digest(BASE/f'data/manifests/{name}.jsonl') == item['sha256']

    consumed = set()

    def costs(prefixes):
        result = defaultdict(float)
        reservations = []
        for i, record in enumerate(ledger['reservations']):
            if any(record['path'].startswith(prefix + '/') for prefix in prefixes):
                assert i not in consumed, 'Cost assigned to two report rows'
                consumed.add(i)
                result[record['status']] += record['reserved_gpu_seconds']/3600
                reservations.append(record)
        return dict(result), reservations

    rows = []
    for method in ['dflash', 'loopahead']:
        stages = ['dflash-train'] if method == 'dflash' else [
            'loopahead-train', 'loopahead-dev', 'loopahead-confirmation']
        for model in MODELS:
            generation, generation_records = costs([f'data/teachers/{s}/{model}' for s in stages])
            features, feature_records = costs([f'features/{s}/{model}' for s in stages])
            training_folder = (dflash_registry.training_root(model) if method == 'dflash'
                               else BASE/f'checkpoints/{method}-{model}-formal-v1')
            training, training_records = costs([str(training_folder.relative_to(BASE))])
            counts = {}
            for stage in stages:
                completed = 0
                shard_ids = set()
                for marker in (BASE/f'data/teachers/{stage}/{model}').glob('shard*/complete.json'):
                    meta = read(marker.with_name('metadata.json'))['identity']
                    assert meta['model'] == model and meta['manifest_sha256'] == manifest[stage]['sha256']
                    assert meta['shard'] not in shard_ids
                    shard_ids.add(meta['shard'])
                    completed += read(marker)['requests']
                assert completed <= manifest[stage]['count']
                counts[stage] = dict(completed=completed, expected=manifest[stage]['count'])
            folder = training_folder
            complete = read(folder/'complete.json') if (folder/'complete.json').exists() else None
            details = {}
            if complete:
                metadata = read(folder/'metadata.json')
                assert metadata['model'] == model
                assert metadata['epochs'] == (6 if method == 'dflash' else 3)
                assert complete['step'] == metadata['total_updates']
                assert all(x['completed'] == x['expected'] for x in counts.values())
                completed_reservations = [r for r in training_records if r['status'] == 'complete']
                assert completed_reservations
                ranks = metadata.get('world_size', 1)
                assert len({r['gpu_uuid'] for r in completed_reservations}) == ranks
                if method == 'dflash':
                    assert not complete['smoke'] and metadata['global_batch'] == 32
                    assert metadata['examples'] + len(metadata['excluded_under_32_tokens']) == 50000
                    eligible = metadata['examples']
                    excluded = metadata['excluded_under_32_tokens']
                    selection = 'final checkpoint'
                    if dflash_registry.is_projected(model):
                        # A completed weight save alone cannot certify the new
                        # formal run before its actual exit/export/CPU audit.
                        dflash_registry.validate_projected_completion()
                        selection = 'final narrow Huginn checkpoint;332621056 parameters'
                else:
                    eligible = 9025 - len(metadata['excluded_no_real_64_state_context'])
                    excluded = metadata['excluded_no_real_64_state_context']
                    selection = Path(complete['selected_checkpoint']).name
                input_rows = [json.loads(line) for line in (BASE/f'data/manifests/{method}-train.jsonl').open()]
                excluded_ids = set(excluded)
                assert len(excluded_ids) == len(excluded) and excluded_ids <= {r['source_id'] for r in input_rows}
                eligible_pools = Counter(r['pool'] for r in input_rows if r['source_id'] not in excluded_ids)
                assert sum(eligible_pools.values()) == eligible
                progress = read(folder/'progress.json')
                details = dict(steps=complete['step'], epochs=metadata['epochs'], eligible=eligible,
                    eligible_source_counts=dict(sorted(eligible_pools.items())),
                    gpus=ranks, selection=selection,
                    rank0_peak_allocated_bytes=progress['peak_allocated_bytes'],
                    metadata_sha256=digest(folder/'metadata.json'),
                    complete_sha256=digest(folder/'complete.json'))
            row = dict(method=method, model=model, counts=counts,
                generation_gpu_hours=generation, feature_gpu_hours=features,
                training_gpu_hours=training, details=details,
                status='complete' if complete else 'pending',
                reservation_paths=sorted({r['path'] for r in generation_records + feature_records + training_records}))
            rows.append(row)

    extras = defaultdict(float)
    for i, record in enumerate(ledger['reservations']):
        if i not in consumed:
            extras[(str(Path(record['path']).parent), record['status'])] += record['reserved_gpu_seconds']/3600
    accounted = sum(sum(row[k].values()) for row in rows
                    for k in ['generation_gpu_hours', 'feature_gpu_hours', 'training_gpu_hours']) + sum(extras.values())
    assert abs(accounted - ledger['total_recorded_gpu_hours']) < 1e-8
    if not args.partial:
        assert all(r['status'] == 'complete' for r in rows), 'Six formal trainings must finish before the final cost report'

    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total',
                                        '--format=csv,noheader'], text=True).strip().splitlines()
    lines = ['# 草稿器实测成本', '',
        ('阶段性报告：尚未完成全部正式训练。' if args.partial else '六个正式草稿器均已完成训练。'), '',
        'GPU小时为各GPU实际占用时间之和，包括该作业内的加载、预热、保存及验证；不是纯kernel时间。'
        '只累计已结束的reservation记录，运行中作业不作估算。生成、特征提取、训练分别报告，导出副本不重复计费。', '',
        '| 方法 | 模型 | 完整分片数据条数/目标 | 生成GPU小时 | 特征GPU小时 | 训练GPU小时 | 更新步数 | 状态 |',
        '|---|---|---|---:|---:|---:|---:|---|']

    def measured(group):
        return f"{group['complete']:.6f}" if 'complete' in group else 'NA'

    for row in rows:
        counts = '; '.join(f"{stage.rsplit('-', 1)[1]} {v['completed']}/{v['expected']}" for stage, v in row['counts'].items())
        feature_cost = '—（在线计算，计入训练）' if row['method'] == 'dflash' else measured(row['feature_gpu_hours'])
        lines.append(f"| {row['method']} | {row['model']} | {counts} | {measured(row['generation_gpu_hours'])} | "
                     f"{feature_cost} | {measured(row['training_gpu_hours'])} | {row['details'].get('steps', 'NA')} | {row['status']} |")
    lines += ['', '未完成行中的数字仅为已完成分片的实测小计，不代表该模型的总成本。', '',
              '数据清单与去重：', '']
    for name in ['dflash-train', 'loopahead-train', 'loopahead-dev', 'loopahead-confirmation']:
        entry = manifest[name]
        mixture = '、'.join(f'{pool} {n}' for pool, n in sorted(entry['pools'].items()))
        lines.append(f"- `{name}`：{entry['count']}条；{mixture}；SHA256 `{entry['sha256']}`。")
    lines += ['', f"保护{audit['protected_unique_questions']}个去重测试问题；精确测试/跨方法heldout重叠为0，"
              f"核验{audit['source_hash_checks']}个源文件哈希。近重复规则见data/manifest-audit.json及清单保护策略；"
              '不据此宣称排除了所有语义改写。原始HumanEval补充隔离检查也通过。', '',
              'Nemotron V2受限而使用公开V1；缺失代码按TACO训练元数据恢复，对话份额由UltraChat补足。'
              'LoopAhead的BBH训练份额替换为非测试reasoning数据。详细来源、版本与适配理由见DESIGN.md和loopahead/SPEC.md。', '',
              '已完成训练的记录：', '']
    for row in rows:
        if row['status'] == 'complete':
            d = row['details']
            lines.append(f"- {row['method']} / {row['model']}：有效训练样本{d['eligible']}，{d['epochs']} epochs，"
                         f"{d['gpus']}张GPU，{d['steps']}次更新；选择`{d['selection']}`；"
                         f"rank0峰值allocated显存{d['rank0_peak_allocated_bytes']}字节。")
            lines.append('  有效样本来源：' + '、'.join(f'{pool} {count}' for pool, count in d['eligible_source_counts'].items()) + '。')
    lines += ['', '失败与额外试跑（不混入上表成功作业）：', '',
              'Huginn旧宽版中断训练保留原始failed退出/成本记录，原因是用户资源与架构调整；其费用单列，不转记为窄版成功训练。', '',
              '| 作业 | 状态 | GPU小时 |', '|---|---|---:|']
    for row in rows:
        for category in ['generation', 'feature', 'training']:
            for status, hours in row[f'{category}_gpu_hours'].items():
                if status != 'complete':
                    lines.append(f"| {row['method']}/{row['model']}/{category} | {status} | {hours:.6f} |")
    for (path, status), hours in sorted(extras.items()):
        lines.append(f'| `{path}` | {status} | {hours:.6f} |')
    lines += ['', f"全部已结束的生成/特征/训练记录合计：{accounted:.6f} GPU小时。"
              '此口径不包括CPU工作、benchmark推理、环境校验和独立数值探针；这些日志另存preflight/runs。', '',
              'GPU型号按本机nvidia-smi实录保留；用户将集群称为8×H200，系统显示如下：', '', '```text',
              *hardware, '```', '', f'原始账本快照：`{ledger_snapshot.name}`，SHA256 `{ledger_hash}`。',
              '可复现命令：`python experiments/baselines/render_training_costs.py`。', '']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ledger_snapshot.write_bytes(ledger_bytes)
    args.out.write_text('\n'.join(lines))
    detail_path = args.out.with_suffix('.json')
    detail_path.write_text(json.dumps(dict(partial=args.partial, rows=rows,
        ledger_sha256=ledger_hash, ledger_snapshot=str(ledger_snapshot), total_recorded_gpu_hours=accounted,
        hardware=hardware), indent=2) + '\n')
    print(json.dumps(dict(output=str(args.out), details=str(detail_path), complete=sum(r['status'] == 'complete' for r in rows))))


if __name__ == '__main__':
    main()
