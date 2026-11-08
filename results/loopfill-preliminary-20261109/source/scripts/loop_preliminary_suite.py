"""Freeze, collect, validate, replay and plot the 2026-11-08 preliminary plan.

GPU workers collect independent reference requests. Suffix replay always runs
in the frozen request order, after collection, using completion-only history.
The 12-request pilot is rerun from scratch in the formal 480-request phase.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from llm_memory_editability.loop_preliminary import (
    ProbeConfig, bootstrap_summary, replay_request, request_totals,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = '1ed04250da1a9936042725d302e81c8fa2ab5abd'
SOURCES = ['scripts/loop_preliminary_suite.py', 'scripts/loop_suffix_spec_readouts.py',
           'src/llm_memory_editability/loop_preliminary.py',
           'src/llm_memory_editability/loop_batch_suffix.py',
           'src/llm_memory_editability/loop_model_registry.py',
           'src/llm_memory_editability/__init__.py',
           'tests/test_loop_preliminary.py', 'docs/loopfill-preliminary-plan.md']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def gpu_info():
    lines = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True).splitlines()
    return [dict(index=int(a.strip()), uuid=b.strip(), memory_mb=float(c), utilization=float(d))
            for a, b, c, d in csv.reader(lines)]


def prepare(args):
    from dataclasses import asdict
    from loop_suffix_spec_readouts import load_questions
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'{out} already exists; resume using run')
    rows = load_questions(Path(args.questions))
    if len(rows) != 480 or len({r['question_id'] for r in rows}) != 480:
        raise ValueError('the formal protocol requires exactly 480 unique SpecBench requests')
    model = Path(args.model_path).resolve()
    if model.name != MODEL_REVISION:
        raise ValueError('model path must identify the pinned Ouro-2.6B revision')
    suffix = Path(args.suffix_lib).resolve()
    revision = subprocess.check_output(['git', '-C', str(suffix), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != 'aca5d9a8a62474035c15d114d40a01abc8c94b51':
        raise ValueError('wrong ArcticInference revision')
    out.mkdir(parents=True)
    for name in SOURCES:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    shutil.copy2(args.questions, out / 'questions.jsonl')
    # Copy the suffix implementation and native module to freeze local changes too.
    shutil.copytree(suffix / 'arctic_inference', out / 'suffix' / 'arctic_inference',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {name: sha(out / 'source' / name) for name in SOURCES}
    suffix_hashes = {str(p.relative_to(out / 'suffix')): sha(p)
                     for p in sorted((out / 'suffix').rglob('*')) if p.is_file()}
    model_hashes = {p.name: sha(p) for p in sorted(model.iterdir()) if p.is_file()}
    selected_gpus = [int(v) for v in args.gpus.split(',')]
    lanes = [g for g in gpu_info() if g['index'] in selected_gpus]
    if len(lanes) != len(selected_gpus):
        raise ValueError('GPU list contains missing or duplicate devices')
    protocol = dict(name=out.name, created_unix=time.time(), python=sys.executable,
        model='ouro-2.6b', model_revision=MODEL_REVISION, model_path=str(model),
        model_sha256=model_hashes, questions_sha256=sha(out / 'questions.jsonl'),
        ordered_question_ids=[r['question_id'] for r in rows], request_count=480,
        pilot_count=12, max_new_tokens=512, depth_total=4, seed=0,
        dtype='bfloat16', attention='sdpa', prompt_format='chat', prompt_truncation=False,
        reference='Native Ouro AR with separate per-loop KV; online normed hidden-state top1',
        readout_shape='Each head separately [1,H], exactly as the original AR collector',
        position_policy='Column i predicts output[i]; exclude prefill column 0',
        history='Current prompt + confirmed output; prior completed outputs only; sorted question_id',
        cache=asdict(ProbeConfig()), bootstrap_repeats=2000, bootstrap_seed=0,
        suffix_revision=revision, source_sha256=source_hashes, suffix_sha256=suffix_hashes,
        lanes=lanes, formal_includes_pilot_outputs=False,
        pilot_validation='All 12 full-length requests independently regenerated with original AR entrypoint',
        timing_claims=False)
    save(out / 'protocol.json', protocol)
    save(out / 'status.json', dict(state='prepared', completed_requests=0, planned_requests=480))
    print(json.dumps(dict(out=str(out), protocol_sha256=sha(out / 'protocol.json'))), flush=True)


def verify_frozen(out):
    protocol = read(out / 'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        if sha(out / 'source' / name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    for name, expected in protocol['suffix_sha256'].items():
        if sha(out / 'suffix' / name) != expected:
            raise ValueError(f'frozen suffix dependency changed: {name}')
    if sha(out / 'questions.jsonl') != protocol['questions_sha256']:
        raise ValueError('frozen input changed')
    return protocol


def load_record(out, phase, qid):
    directory = out / phase / 'readouts'
    meta = read(directory / f'{qid}.json')
    path = directory / f'{qid}.npz'
    if meta['protocol_sha256'] != sha(out / 'protocol.json') or sha(path) != meta['npz_sha256']:
        raise ValueError(f'record identity/hash mismatch: {phase}/{qid}')
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in data.files}
    if result['predictions'].shape != (4, len(result['output'])):
        raise ValueError(f'misaligned readout shape: {qid}')
    if not np.array_equal(result['predictions'][-1], result['output']):
        raise ValueError(f'full-depth token mismatch: {qid}')
    return meta, result


def worker(args):
    out = Path(args.out).resolve()
    protocol = verify_frozen(out)
    from loop_suffix_spec_readouts import OuroRunner, load_questions
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids, tokenize_prompt
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(protocol['seed'])
    rows = load_questions(out / 'questions.jsonl')
    if args.phase == 'pilot':
        rows = rows[:protocol['pilot_count']]
    rows = rows[args.shard::args.num_shards]
    directory = out / args.phase / 'readouts'
    directory.mkdir(parents=True, exist_ok=True)
    unfinished = []
    for row in rows:
        if (directory / f"{row['question_id']}.json").exists():
            load_record(out, args.phase, row['question_id'])
        else:
            unfinished.append(row)
    if unfinished:
        tok = AutoTokenizer.from_pretrained(protocol['model_path'], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(protocol['model_path'], trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        model.eval()
        model.model.total_ut_steps = 4
        runner = OuroRunner(model, 4)
        stop, stop_audit = resolve_stop_ids(MODELS['ouro-2.6b'], tok,
            model.config.eos_token_id, getattr(model.generation_config, 'eos_token_id', None),
            model.config.vocab_size, 'chat')
        save(out / args.phase / f'worker-{args.shard}.json', dict(pid=os.getpid(),
            gpu=os.environ['CUDA_VISIBLE_DEVICES'], torch=torch.__version__,
            transformers=transformers.__version__, attention=model.config._attn_implementation,
            stop_audit=stop_audit, cpu_affinity=sorted(os.sched_getaffinity(0))))
        for row in unfinished:
            qid = row['question_id']
            prompt = tokenize_prompt(MODELS['ouro-2.6b'], tok, row['text'])
            if len(prompt) + protocol['max_new_tokens'] > MODELS['ouro-2.6b'].context_limit:
                raise ValueError(f'context overflow: {qid}; no truncation permitted')
            save(out / args.phase / f'progress-{args.shard}.json',
                 dict(question_id=qid, state='generating', started_unix=time.time()))
            ids = torch.tensor([prompt], device='cuda')
            start = time.perf_counter()
            output, predictions = runner.generate_with_readouts(ids, protocol['max_new_tokens'], stop)
            validated = None
            if args.phase == 'pilot':
                reference = runner.generate(ids, protocol['max_new_tokens'], stop)
                validated = output == reference
                if not validated:
                    save(directory / f'{qid}.mismatch.json',
                         dict(reference=reference, captured=output))
                    raise ValueError(f'independent AR token mismatch: {qid}')
            temp = directory / f'{qid}.tmp.npz'
            np.savez_compressed(temp, prompt=np.asarray(prompt, dtype=np.int32),
                                output=np.asarray(output, dtype=np.int32), predictions=predictions)
            path = directory / f'{qid}.npz'
            temp.replace(path)
            meta = dict(question_id=qid, category=row['category'], prompt_tokens=len(prompt),
                output_tokens=len(output), decode_positions=len(output) - 1,
                ended_eos=output[-1] in stop, cap_stopped=len(output) == protocol['max_new_tokens']
                and output[-1] not in stop, independent_ar_exact_match=validated,
                elapsed_s=time.perf_counter() - start, npz_sha256=sha(path),
                protocol_sha256=sha(out / 'protocol.json'), shard=args.shard,
                gpu=os.environ['CUDA_VISIBLE_DEVICES'])
            save(directory / f'{qid}.json', meta)
            print(json.dumps(meta), flush=True)
            torch.cuda.empty_cache()
    save(out / args.phase / f'worker-{args.shard}-complete.json',
         dict(requests=len(rows), finished_unix=time.time()))


def analyze(out, phase):
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    protocol = verify_frozen(out)
    ids = protocol['ordered_question_ids']
    if phase == 'pilot':
        ids = ids[:protocol['pilot_count']]
    config = ProbeConfig(**protocol['cache'])
    cache = CompletionSuffixCache.arctic(config.max_tree_depth, config.max_cached_requests)
    dest = out / phase
    requests = []
    positions_path = dest / 'positions.jsonl'
    with positions_path.with_suffix('.jsonl.tmp').open('w') as f:
        for qid in ids:
            meta, data = load_record(out, phase, qid)
            if phase == 'pilot' and meta['independent_ar_exact_match'] is not True:
                raise ValueError(f'pilot independent reference not validated: {qid}')
            rows = list(replay_request(cache, qid, data['prompt'], data['output'],
                                      data['predictions'], config))
            for row in rows:
                row['category'] = meta['category']
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
            requests.append(dict(question_id=qid, category=meta['category'],
                                 output_tokens=meta['output_tokens'], **request_totals(rows)))
            print(f'[{phase} replay] {len(requests)}/{len(ids)} qid={qid}', flush=True)
        f.flush()
        os.fsync(f.fileno())
    positions_path.with_suffix('.jsonl.tmp').replace(positions_path)
    if [p['request_id'] for p in cache.publications] != ids:
        raise ValueError('history publication order differs from the frozen order')
    save(dest / 'requests.json', requests)
    summary = bootstrap_summary(requests, protocol['bootstrap_repeats'], protocol['bootstrap_seed'])
    summary.update(phase=phase, protocol_sha256=sha(out / 'protocol.json'),
                   positions_sha256=sha(positions_path), publications=cache.publications)
    save(dest / 'summary.json', summary)
    grouped = {}
    for category in sorted({r['category'] for r in requests}):
        subset = [r for r in requests if r['category'] == category]
        if sum(r['positions'] for r in subset):
            grouped[category] = bootstrap_summary(subset, protocol['bootstrap_repeats'], 0)
    save(dest / 'by-category.json', grouped)
    with (dest / 'metrics.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'component', 'mean', 'ci95_low', 'ci95_high'])
        for metric, value in summary['metrics'].items():
            if value['mean'] is not None:
                for i, mean in enumerate(value['mean']):
                    writer.writerow([metric, i + 1, mean, *value['ci95'][i]])
    plot(dest, summary)
    save(dest / 'complete.json', dict(requests=len(ids), positions=summary['positions'],
                                    summary_sha256=sha(dest / 'summary.json'), finished_unix=time.time()))
    return summary


def plot(dest, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.72), layout='constrained')
    metrics = summary['metrics']
    for key, label, color, marker in [('agreement', 'Single loop', '#3477a3', 'o'),
                                      ('coverage', 'Cumulative', '#d6782a', 's')]:
        mean = np.asarray(metrics[key]['mean']) * 100
        ci = np.asarray(metrics[key]['ci95']) * 100
        axes[0].plot([1, 2, 3], mean, marker=marker, markersize=3, linewidth=1,
                     color=color, label=label)
        axes[0].fill_between([1, 2, 3], ci[:, 0], ci[:, 1], color=color, alpha=.15)
    axes[0].set(xlabel='Loop depth', ylabel='Agreement / coverage (%)', xticks=[1, 2, 3],
                ylim=(0, 100), title='(a) Shallow agreement')
    axes[0].legend(loc='lower right', frameon=False, handlelength=1)
    means = np.asarray(metrics['lengths']['mean'])
    ci = np.asarray(metrics['lengths']['ci95'])
    axes[1].bar(range(3), means, color=['#a9b4bd', '#609bbf', '#e4a366'], width=.65)
    # CI segments are drawn independently so skewed percentile intervals need not bracket mean.
    axes[1].vlines(range(3), ci[:, 0], ci[:, 1], color='#333333', linewidth=.8)
    for i, (lo, hi) in enumerate(ci):
        axes[1].hlines([lo, hi], i - .08, i + .08, color='#333333', linewidth=.8)
    axes[1].set(xticks=range(3), xticklabels=['Suffix', '+Shallow', '+Requery'],
                ylabel='Correct draft length', title='(b) Suffix synergy', ylim=(0, None))
    axes[1].tick_params(axis='x', labelsize=6, pad=2)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(dest / 'preliminary.pdf')
    fig.savefig(dest / 'preliminary.png', dpi=300)
    plt.close(fig)
    # A plain-text caption is packaged with each standalone scientific plot.
    caption = ('Ouro-2.6B, T=4, greedy, SpecBench first turns, at most 512 new tokens. '
        'Prefill token excluded. Left: per-loop top-1 agreement and cumulative candidate coverage. '
        'Right: consecutive reference tokens covered by Suffix, Suffix plus depth-1 top-1, '
        'and the latter plus one suffix requery at that top-1. N=16, Q=4; common candidates '
        'are preserved. Actual candidate counts differ. Shading/intervals: paired request '
        'bootstrap, 2,000 resamples, seed 0. Candidate coverage is not measured speedup. '
        f"Phase: {summary['phase']}; requests={summary['requests']}; positions={summary['positions']}.\n")
    (dest / 'caption.txt').write_text(caption)


def run_phase(out, phase, protocol):
    lanes = protocol['lanes']
    directory = out / phase
    directory.mkdir(exist_ok=True)
    expected = protocol['pilot_count'] if phase == 'pilot' else protocol['request_count']
    active, pending, outcomes = {}, list(range(len(lanes))), {}
    env_base = dict(os.environ, PYTHONPATH=str(out / 'source' / 'src') + ':' + str(out / 'suffix'),
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    while pending or active:
        for shard, task in list(active.items()):
            code = task['proc'].poll()
            if code is None:
                continue
            task['log'].close()
            task['lock'].close()
            outcomes[shard] = code
            del active[shard]
        states = {g['uuid']: g for g in gpu_info()}
        if not any(outcomes.values()):
            for shard in list(pending):
                uuid = lanes[shard]['uuid']
                state = states[uuid]
                if state['memory_mb'] > 256 or state['utilization'] > 10:
                    continue
                lock = open(f'/tmp/loopfill-{uuid}.lock', 'a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                # Repeat the occupancy check after lock acquisition.
                state = next(g for g in gpu_info() if g['uuid'] == uuid)
                if state['memory_mb'] > 256 or state['utilization'] > 10:
                    lock.close()
                    continue
                log = (logs / f'{phase}-{shard}.log').open('a')
                command = [sys.executable, str(out / 'source' / 'scripts' / Path(__file__).name),
                    'worker', '--out', str(out), '--phase', phase, '--shard', str(shard),
                    '--num-shards', str(len(lanes))]
                proc = subprocess.Popen(command, env=env_base | {'CUDA_VISIBLE_DEVICES': uuid},
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                active[shard] = dict(proc=proc, lock=lock, log=log)
                pending.remove(shard)
                save(directory / f'process-{shard}.json', dict(pid=proc.pid, gpu=uuid,
                     command=command, started_unix=time.time()))
        completed = len(list((directory / 'readouts').glob('*.json')))
        status = dict(state='running', phase=phase, updated_unix=time.time(),
            completed_requests=completed, planned_requests=expected,
            active={s: t['proc'].pid for s, t in active.items()}, pending=pending, outcomes=outcomes,
            pilot_validated=(out / 'pilot' / 'complete.json').exists())
        save(out / 'status.json', status)
        if any(outcomes.values()) and not active:
            raise RuntimeError(f'{phase} workers failed: {outcomes}; inspect logs')
        if pending or active:
            time.sleep(10)
    save(out / 'status.json', dict(state='analyzing', phase=phase, completed_requests=expected,
                                  planned_requests=expected, updated_unix=time.time()))
    return analyze(out, phase)


def run(out):
    out = Path(out).resolve()
    lock = (out / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = verify_frozen(out)
    # Verify checkpoint contents once before spawning workers.
    for name, expected in protocol['model_sha256'].items():
        if sha(Path(protocol['model_path']) / name) != expected:
            raise ValueError(f'checkpoint changed: {name}')
    # A lost controller must never duplicate surviving workers.
    for phase in ('pilot', 'full'):
        for path in (out / phase).glob('process-*.json'):
            pid = read(path)['pid']
            cmdline = Path(f'/proc/{pid}/cmdline')
            if cmdline.exists() and str(out).encode() in cmdline.read_bytes():
                raise RuntimeError(f'worker {pid} still alive; refusing duplicate supervisor')
    save(out / 'launcher.json', dict(pid=os.getpid(), started_unix=time.time()))
    try:
        if not (out / 'pilot' / 'complete.json').exists():
            run_phase(out, 'pilot', protocol)
        # Validate the pilot evidence again when resuming.
        for qid in protocol['ordered_question_ids'][:12]:
            meta, _ = load_record(out, 'pilot', qid)
            if meta['independent_ar_exact_match'] is not True:
                raise ValueError('pilot reference check not passed')
        summary = run_phase(out, 'full', protocol)
        save(out / 'status.json', dict(state='complete', phase='full', completed_requests=480,
            planned_requests=480, positions=summary['positions'], updated_unix=time.time()))
    except BaseException:
        save(out / 'status.json', dict(state='failed', error=traceback.format_exc(), updated_unix=time.time()))
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--out', required=True)
    prep.add_argument('--questions', required=True)
    prep.add_argument('--model-path', required=True)
    prep.add_argument('--suffix-lib', required=True)
    prep.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    run_p = sub.add_parser('run')
    run_p.add_argument('--out', required=True)
    work = sub.add_parser('worker')
    work.add_argument('--out', required=True)
    work.add_argument('--phase', required=True, choices=['pilot', 'full'])
    work.add_argument('--shard', type=int, required=True)
    work.add_argument('--num-shards', type=int, required=True)
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--out', required=True)
    analysis.add_argument('--phase', required=True, choices=['pilot', 'full'])
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'worker':
        worker(args)
    elif args.command == 'analyze':
        analyze(Path(args.out).resolve(), args.phase)
    else:
        run(args.out)


if __name__ == '__main__':
    main()
