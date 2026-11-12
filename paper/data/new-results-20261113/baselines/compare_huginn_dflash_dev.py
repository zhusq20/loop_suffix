"""Held-out feasibility comparison of Huginn DFlash widths, never a table cell.

The pilot and interrupted wide checkpoint have unequal training budgets. This
diagnostic can establish deployment compatibility and measure their current
acceptance/cost; it cannot establish the fully trained narrow model's quality.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from evaluate_dflash import digest, save
from gpu_allocation import allowed_gpus
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
QUOTAS = dict(nemotron_math=8, nemotron_code=4, codealpaca=2, ultrachat=6)
SEED = 20261111


def selected_rows():
    directory = BASE/'data/manifests'
    manifest = json.loads((directory/'manifest.json').read_text())['files']
    path = directory/'dflash-dev.jsonl'
    assert digest(path) == manifest['dflash-dev']['sha256']
    rows = [json.loads(line) for line in path.open()]
    assert len(rows) == 1000 and len({r['source_id'] for r in rows}) == 1000
    assert Counter(r['pool'] for r in rows) == {k: v*50 for k, v in QUOTAS.items()}
    selected = []
    key = lambda row: hashlib.sha256(f"{SEED}:{row['source_id']}".encode()).hexdigest()
    for pool, count in QUOTAS.items():
        selected.extend(sorted((r for r in rows if r['pool'] == pool), key=key)[:count])
    selected.sort(key=key)
    ids = {r['source_id'] for r in selected}
    normalized = {r['normalized_question_sha256'] for r in selected}
    exclusions = {}
    for name in ['dflash-train', 'dflash-pilot', 'loopahead-train']:
        p = directory/f'{name}.jsonl'
        assert digest(p) == manifest[name]['sha256']
        excluded = [json.loads(line) for line in p.open()]
        assert not ids & {r['source_id'] for r in excluded}
        assert not normalized & {r['normalized_question_sha256'] for r in excluded}
        exclusions[name] = dict(count=len(excluded), sha256=digest(p))
    assert all(r['split'] == 'dev' and r['method'] == 'dflash' and r['prompt_format'] == 'chat' for r in selected)
    plan = dict(model='huginn-3.5b', stage='heldout_feasibility', reportable=False,
        manifest_path=str(path), manifest_sha256=digest(path), count=20,
        selection='Lowest SHA256(seed:source_id) in each source pool, then same hash order',
        selection_seed=SEED, pool_counts=QUOTAS, exclusion_manifests=exclusions,
        max_new_tokens=128, temperature=0., graph=False, compile=False,
        rows=[dict(source_id=r['source_id'], pool=r['pool'],
            prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),
            normalized_question_sha256=r['normalized_question_sha256']) for r in selected],
        limitations=[
            '20 held-out prompts and max128 only; not the max512 benchmark or final quality.',
            'Pilot2000 epoch1 versus interrupted wide50k step4000: unequal training budgets.',
            'Finite-precision sequence disagreement is recorded, not assumed to be numerical.',
            'GPU exclusively held; shared host CPU/memory contention is not isolated.',
            'Stratified source mix is fixed before decoding; no test benchmark used for selection.'])
    return selected, plan


def summarize(records):
    stats = [r['stats'] for r in records]
    tokens = sum(s['decode_tokens'] for s in stats)
    seconds = sum(s['decode_s'] for s in stats)
    calls = sum(s['target_verifications'] for s in stats)
    accepted = sum(sum(s['accepted_draft_tokens']) for s in stats)
    draft_s = sum(s['draft_gpu_s'] or 0 for s in stats)
    return dict(requests=len(records), decode_tokens=tokens, decode_s=seconds,
        tok_s=tokens/seconds if seconds else None, target_verifications=calls,
        gamma=tokens/calls if calls else None, mean_raw_accepted=accepted/calls if calls else None,
        draft_gpu_s=draft_s, draft_fraction=draft_s/seconds if seconds else None)


def run(args, uuid, rows, plan):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoConfig
    sys.path.insert(0, str(BASE/'third_party/dflash'))
    from dflash.model import DFlashDraftModel
    from dflash_impl.projected_draft import inference_draft_class
    from dflash_impl.target import LoopTarget
    from dflash_impl.decode import generate
    suite = ROOT/'results/loopfill-latency512-20261107'
    sys.path.insert(0, str(suite/'source/src'))
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt, resolve_stop_ids
    spec = MODELS['huginn-3.5b']
    protocol = json.loads((suite/'protocol.json').read_text())
    path = Path(protocol['models']['huginn-3.5b']['path'])
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    generation = json.loads((path/'generation_config.json').read_text())
    stops, stop_audit = resolve_stop_ids(spec, tok, config.eos_token_id,
        generation.get('eos_token_id'), config.vocab_size, 'chat')
    prompts = [tokenize_prompt(spec, tok, r['prompt'], 'chat') for r in rows]
    assert all(len(p) == r['prompt_lengths']['huginn-3.5b'] for p, r in zip(prompts, rows))
    assert all(len(p)+128+spec.depth_total+4 <= 3072 for p in prompts)
    with (args.out/'inputs.jsonl').open('x') as f:
        for row, prompt in zip(plan['rows'], prompts):
            f.write(json.dumps(dict(row, prompt_tokens=prompt))+'\n')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    target = LoopTarget.load(path, spec.backend, 3072)
    drafts = dict(ar=None)
    checkpoints = {}
    for label, checkpoint, cls in [('narrow', args.narrow, inference_draft_class()),
                                   ('wide', args.wide, DFlashDraftModel)]:
        complete = json.loads((checkpoint/'complete.json').read_text())
        meta = json.loads((checkpoint.parent/'metadata.json').read_text())
        assert meta['model'] == 'huginn-3.5b' and not meta['smoke']
        if label == 'narrow':
            assert meta['training_scope'] == 'pilot2000' and complete['step'] == 60
            assert json.loads((checkpoint.parent/'complete.json').read_text())['step'] == 60
        else:
            assert complete['step'] == 4000 and meta['examples'] == 47387
        draft, info = cls.from_pretrained(checkpoint, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
        assert all(not info[k] for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']), info
        draft = draft.to(target.device).eval()
        assert draft.target_layer_ids == target.expanded_ids and draft.block_size == 16
        assert sum(p.numel() for p in draft.parameters()) == (332621056 if label == 'narrow' else 2116288320)
        drafts[label] = draft
        checkpoints[label] = dict(path=str(checkpoint), step=complete['step'],
            training_examples=meta['examples'], training_epochs_planned=meta['epochs'],
            parameters=sum(p.numel() for p in draft.parameters()),
            hashes={p.name: digest(p) for p in sorted(checkpoint.glob('*.safetensors'))},
            config_sha256=digest(checkpoint/'config.json'),
            training_metadata_sha256=digest(checkpoint.parent/'metadata.json'), loading=info)
    metadata = dict(plan=plan, plan_sha256=digest(args.plan), gpu_uuid=uuid,
        model_revision=spec.revision, cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch=str(torch.__version__), transformers=transformers.__version__, cuda=torch.version.cuda,
        dtype='bfloat16', target_depth=32, target_features=target.expanded_ids, stop_audit=stop_audit,
        max_seq=3072, checkpoints=checkpoints, inputs_sha256=digest(args.out/'inputs.jsonl'),
        method_order='Cycle ar/narrow/wide per request; profiles after uninstrumented passes',
        warmups='First two selected prompts,32 output tokens per method',
        source_sha256=json.loads((args.out/'source-manifest.json').read_text()))
    save(args.out/'metadata.json', metadata)

    def decode(method, prompt, limit, profile=False):
        return generate(target, torch.tensor([prompt], device=target.device), limit, stops,
            draft=drafts[method], temperature=0., seed=0, profile_draft=profile)

    for method in drafts:
        for prompt in prompts[:2]:
            decode(method, prompt, 32)
    records = {k: [] for k in ['ar', 'narrow', 'wide', 'narrow_profile', 'wide_profile']}
    journals = {k: (args.out/f'{k}.jsonl').open('x') for k in records}
    methods = ['ar', 'narrow', 'wide']
    try:
        for index, (row, prompt) in enumerate(zip(rows, prompts)):
            order = methods[index%3:]+methods[:index%3]
            for method in order+['narrow_profile', 'wide_profile']:
                profile = method.endswith('_profile')
                result = decode(method.removesuffix('_profile'), prompt, 128, profile)
                record = dict(source_id=row['source_id'], tokens=result.tokens,
                    stats=result.stats, finished_unix=time.time())
                journals[method].write(json.dumps(record)+'\n')
                journals[method].flush()
                os.fsync(journals[method].fileno())
                records[method].append(record)
                if profile:
                    assert result.tokens == records[method.removesuffix('_profile')][-1]['tokens'], 'Instrumented replay changed output'
            progress = dict(requests=index+1, count=len(rows),
                summaries={k: summarize(v) for k, v in records.items()})
            save(args.out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    finally:
        for f in journals.values():
            f.close()
        target.close()
    summary = {k: summarize(v) for k, v in records.items()}
    comparisons = {}
    for label in ['narrow', 'wide']:
        differences = []
        for ar, other in zip(records['ar'], records[label]):
            assert ar['source_id'] == other['source_id']
            a, b = ar['tokens'], other['tokens']
            if a != b:
                position = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                differences.append(dict(source_id=ar['source_id'], position=position,
                    ar_token=a[position] if position<len(a) else None,
                    dflash_token=b[position] if position<len(b) else None))
        comparisons[label] = dict(eta_latency=summary['ar']['decode_s']/summary[label]['decode_s'],
            eta_throughput=summary[label]['tok_s']/summary['ar']['tok_s'],
            whole_sequence_equal=len(rows)-len(differences), requests=len(rows),
            first_differences=differences, profile_all_outputs_identical=True,
            profile_wall_change_pct=100*(summary[label+'_profile']['decode_s']/summary[label]['decode_s']-1))
    report = dict(status='completed_feasibility_only', reportable=False, summaries=summary,
        comparisons=comparisons, limitations=plan['limitations'],
        checkpoint_hashes=checkpoints, metadata_sha256=digest(args.out/'metadata.json'),
        narrow_vs_wide_throughput=summary['narrow']['tok_s']/summary['wide']['tok_s'],
        finished_unix=time.time())
    save(args.out/'report.json', report)
    save(args.out/'complete.json', dict(requests=len(rows), report_sha256=digest(args.out/'report.json'),
        journals={k: digest(args.out/f'{k}.jsonl') for k in records}, finished_unix=time.time()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-plan', action='store_true')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--narrow', type=Path)
    p.add_argument('--wide', type=Path)
    p.add_argument('--gpu', type=int)
    p.add_argument('--cpus')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    rows, plan = selected_rows()
    if args.prepare_plan:
        assert not args.plan.exists()
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        save(args.plan, plan)
        print(json.dumps(dict(plan=str(args.plan), sha256=digest(args.plan), count=20, quotas=QUOTAS)))
        return
    assert json.loads(args.plan.read_text()) == plan
    assert args.gpu in allowed_gpus() and args.cpus and args.out and args.narrow and args.wide
    import torch  # CPU import before querying/acquiring the exclusive GPU.
    uuids = {int(i): u.strip() for i, u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
    uuid = uuids[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        deadline = time.monotonic()+60
        while True:
            busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
            if uuid not in busy:
                try:
                    torch.cuda.set_device(0)
                    lease = torch.empty(1, device='cuda')
                    break
                except RuntimeError as exc:
                    if 'busy or unavailable' not in str(exc):
                        raise
            assert time.monotonic()<deadline, 'Assigned GPU busy before model loading'
            time.sleep(.2)
        args.out.mkdir(parents=True, exist_ok=False)
        freeze_sources(args.out, [Path(__file__), BASE/'evaluate_dflash.py', BASE/'gpu_allocation.py',
            BASE/'configs/gpu-allocation.json', BASE/'provenance.py', BASE/'dflash_impl/projected_draft.py',
            BASE/'dflash_impl/target.py', BASE/'dflash_impl/decode.py', args.plan])
        started, status = time.time(), 'failed'
        try:
            run(args, uuid, rows, plan)
            status = 'complete'
        except Exception:
            save(args.out/'failure.json', dict(error=traceback.format_exc(), finished_unix=time.time()))
            raise
        finally:
            finished = time.time()
            save(args.out/'reservation-cost.json', dict(gpu_uuid=uuid, started_unix=started,
                finished_unix=finished, reserved_gpu_seconds=finished-started, status=status))


if __name__ == '__main__':
    main()
