"""Resume the preserved Ouro MATH FP32 diagnostic without repeating its prefix.

Forced prefix tokens are diagnostic inputs only; no output here is a benchmark.
"""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from loopahead_diagnostic_inputs import diagnostic_input, input_provenance, validate_replay_sources
from loopahead_partial_resume import load_partial, reconcile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ar', type=Path, required=True)
    parser.add_argument('--loopahead', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observed-margin-report', type=Path,
        help='Verified formal BF16 replay report, when original timed runs omit traces')
    parser.add_argument('--paper-check', action='store_true', help='Replay the separate B4 original HumanEval/max1024 check')
    parser.add_argument('--resume-evidence', type=Path, required=True)
    args = parser.parse_args()
    args.partial = load_partial(args.resume_evidence, args)
    args.out.mkdir(parents=True, exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'loopahead_partial_resume.py', BASE/'diagnose_trained_loopahead.py', BASE/'loopahead_diagnostic_inputs.py', BASE/'provenance.py',
        BASE/'gpu_startup.py', BASE/'gpu_allocation.py', *sorted((BASE/'loopahead').glob('*.py'))])
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        report = dict(status='running', gpu_uuid=uuid, started_unix=time.time(), results=[])
        try:
            from gpu_startup import acquire_cuda_context
            lease,acquisition=acquire_cuda_context(args.gpu,uuid)
            report['cuda_acquisition']=acquisition
            run(args, report)
            report['status'] = 'complete'
        except Exception:
            report.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            report['finished_unix'] = time.time()
            (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def run(args, report):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    suite = ROOT/'results/loopfill-latency512-20261107'
    sys.path[:0] = [str(suite/'source/src'), str(suite/'source/scripts'), '/tmp/loopfill-assets/ArcticInference']
    from loop_batch_bench import enforce_precision
    from loop_latency_worker import drive
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt
    from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop
    from llm_memory_editability.loop_optimized_decode import SharedKVPool
    from loopahead.decode import LoopAheadConfig, decode_steps
    from loopahead.engine import LoopAheadExecutor
    from loopahead.lad import LADConfig, LookaheadDrafter, RequestVocabulary

    meta = read(args.loopahead/'metadata.json')
    ar_meta = read(args.ar/'metadata.json')
    for key in ['model', 'model_revision', 'dataset', 'input_sha256', 'max_seq', 'max_new_tokens', 'stop_audit', 'temperature']:
        assert meta[key] == ar_meta[key], key
    assert meta['prompt_format'] == ar_meta.get('prompt_format','chat')
    assert meta['temperature'] == 0
    assert meta['dtype'] == ar_meta['dtype'] == 'bfloat16'
    input_path = diagnostic_input(meta, ar_meta, args.paper_check)
    report['input_provenance'] = input_provenance(meta, ar_meta, args.paper_check)
    report['dataset'] = meta['dataset']
    if args.paper_check:
        validate_replay_sources(args.ar, True)
        validate_replay_sources(args.loopahead, True)
    checkpoint = Path(meta['checkpoint'])
    assert digest(checkpoint) == meta['checkpoint_sha256']
    rows = {}
    for label, folder in [('ar', args.ar), ('loopahead', args.loopahead)]:
        completion = read(folder/'complete.json')
        assert digest(folder/'requests.jsonl') == completion['requests_sha256']
        rows[label] = [json.loads(line) for line in (folder/'requests.jsonl').open()]
        assert len(rows[label]) == completion['requests']
    assert len(rows['ar']) == len(rows['loopahead'])
    observed = None
    if args.observed_margin_report is not None:
        margins = read(args.observed_margin_report)
        assert margins['status'] == 'complete' and margins['model'] == meta['model']
        assert margins['dataset'] == meta['dataset'] and margins['checkpoint_sha256'] == meta['checkpoint_sha256']
        if args.paper_check:
            assert margins['input_provenance'] == report['input_provenance']
        for label, folder in [('ar',args.ar),('loopahead',args.loopahead)]:
            source = margins['source_journals'][label]
            assert Path(source['path']).resolve() == folder.resolve()
            assert source['sha256'] == digest(folder/'requests.jsonl')
        observed = {(r['source_id'],r['position']):r for r in margins['results']}
        assert len(observed) == len(margins['results']) == margins['cases']
        assert all(r['both_original_prefixes_reproduced'] for r in observed.values())
        report['observed_margin_report'] = dict(path=str(args.observed_margin_report.resolve()),
            sha256=digest(args.observed_margin_report))
    else:
        assert meta['trace'] and ar_meta['trace']
    cases = []
    for ar, la in zip(rows['ar'], rows['loopahead']):
        assert ar['source_id'] == la['source_id'] and ar['prompt_sha256'] == la['prompt_sha256']
        if ar['tokens'] == la['tokens']:
            continue
        position = next((i for i, (a, d) in enumerate(zip(ar['tokens'], la['tokens'])) if a != d), None)
        assert position is not None, 'Length-only disagreement requires separate stop-condition diagnosis'
        if observed is None:
            traces = {label: next(t for t in row['trace'] if t['position'] == position)
                      for label, row in [('ar', ar), ('loopahead', la)]}
        else:
            record = observed[(ar['source_id'],position)]
            traces = {label:record[label] for label in ['ar','loopahead']}
            assert traces['ar']['argmax'] == ar['tokens'][position]
            assert traces['loopahead']['argmax'] == la['tokens'][position]
        cases.append(dict(source_id=ar['source_id'], position=position,
            common_prefix=ar['tokens'][:position], observed_bf16=traces,
            ar_token=ar['tokens'][position], loopahead_token=la['tokens'][position]))
    if observed is not None:
        assert len(cases) == len(observed)
    report.update(model=meta['model'], checkpoint_sha256=meta['checkpoint_sha256'], cases=len(cases),
        source_journals={label: dict(path=str(folder.resolve()), sha256=digest(folder/'requests.jsonl'))
                         for label, folder in [('ar', args.ar), ('loopahead', args.loopahead)]})
    assert cases, 'No mismatches to diagnose'
    reused, pending_cases = reconcile(cases, args.partial)
    report['results'].extend(reused)
    report['resume_provenance'] = args.partial['evidence']
    report['executed_case_count'] = 0
    protocol = read(suite/'protocol.json')
    for rel, sha in protocol['source_sha256'].items():
        assert digest(suite/'source'/rel) == sha
    spec = MODELS[meta['model']]
    assert spec.revision == meta['model_revision']
    path = protocol['models'][meta['model']]['path']
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    prompts = {row['source_id']: tokenize_prompt(spec, tok, row['prompt'], meta['prompt_format'])
               for row in map(json.loads, input_path.open())}
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32).to('cuda').eval()
    model = enforce_precision(model, 'float32')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    lad = LookaheadDrafter(LADConfig(**saved['config']))
    lad.load_state_dict(saved['model'])
    lad = lad.to(device='cuda', dtype=torch.float32).eval()
    cfg = LoopAheadConfig(**meta['config'])
    stops = set(meta['stop_audit']['stop_ids'])

    class Reached(Exception):
        pass

    class Replay(LoopAheadExecutor):
        def _prefill(self, backbone, prompt, depth):
            super()._prefill(backbone, prompt, depth)
            if not self.forced:
                self.observed = self.prefill_logits.detach().float().cpu()
                raise Reached()
            self.vocab = RequestVocabulary(self.head, self.frequent_ids, prompt, self.stops)
            self.vocab.commit([self.forced[0]])
            return self.forced[0]

        def _readout(self, backbone, tree, groups, full):
            reads, finals = super()._readout(backbone, tree, groups, full)
            for node in full:
                if node.path == self.forced:
                    self.observed = finals[node.idx].detach().float().cpu()
                    raise Reached()
                if len(node.path) < len(self.forced) and node.path == self.forced[:len(node.path)]:
                    forced = torch.full_like(finals[node.idx], float('-inf'))
                    forced[self.forced[len(node.path)]] = 0.
                    finals[node.idx] = forced
            return reads, finals

    engines = {}
    for label, physical_nodes in [('serial', 1), ('loopahead', 32)]:
        pool = SharedKVPool(1, pad_decode_rows=False)
        if spec.backend == 'ouro':
            backbone = OuroLoop(model, 4, meta['max_seq'], physical_nodes, kv_factory=pool)
        else:
            backbone = HuginnLoop(model, 32, meta['max_seq'], physical_nodes, 0, meta['max_seq'], extra_depths=(8,), kv_factory=pool)
        backbone.position_limit = meta['max_seq']
        engine = Replay(backbone, lad, saved['frequent_ids'])
        engine.capture_prefill_logits = True
        engines[label] = (backbone, engine)
    with torch.inference_mode():
        for case in pending_cases:
            logits = {}
            for label, (backbone, engine) in engines.items():
                engine.forced = tuple(case['common_prefix'])
                engine.observed = None
                key = f'0/{meta["model"]}/{meta["dataset"]}/{case["source_id"]}'.encode()
                seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
                engine.request(seed, stops)
                config = replace(cfg, retain=1, execute=1, future=0) if label == 'serial' else cfg
                try:
                    drive(backbone, engine, decode_steps(backbone, prompts[case['source_id']], config,
                        meta['max_new_tokens'], stops, generator=engine.generator))
                except Reached:
                    pass
                assert engine.observed is not None
                logits[label] = engine.observed
            entry = dict(case, same_top1=int(logits['serial'].argmax()) == int(logits['loopahead'].argmax()),
                max_abs_logit_difference=float((logits['serial']-logits['loopahead']).abs().max()))
            for label, value in logits.items():
                values, ids = value.topk(3)
                probs = value.softmax(-1)[ids]
                entry[label] = dict(top_ids=ids.tolist(), top_logits=values.tolist(), top_probs=probs.tolist(),
                    top2_probability_margin=float(probs[0]-probs[1]))
            report['results'].append(entry)
            report['executed_case_count'] += 1
            print(json.dumps({k:v for k,v in entry.items() if k not in ['common_prefix', 'observed_bf16']}), flush=True)
    assert len(report['results']) == len(cases)
    assert report['executed_case_count'] == len(cases) - len(reused)
    report['all_fp32_same_top1'] = all(r['same_top1'] for r in report['results'])
    report['limitations'] = ['Forced tokens preserve the observed BF16 common prefix only and are not reported as generated output.',
        'FP32 proposals may change tree shape relative to BF16. This checks each observed first divergence, not complete FP32 benchmark sequences.',
        f'Serial has one physical/live node; LoopAhead retains32 with the actual trained LAD, full target depth, original max{meta["max_new_tokens"]} budget and Graph OFF.']
    report['limitations'].append('First 262 results reuse the preserved interrupted attempt; only the remaining 167 are executed here. Original exit -15 remains explicit in resume_provenance.')
    assert report['all_fp32_same_top1'], 'FP32 common-prefix disagreements require investigation'


if __name__ == '__main__':
    main()
