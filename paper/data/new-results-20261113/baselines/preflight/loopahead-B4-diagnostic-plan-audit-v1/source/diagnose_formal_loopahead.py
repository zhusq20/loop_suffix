"""Record first-divergence BF16 margins by replaying completed greedy runs.

This uses the real AR and trained LoopAhead paths, without forcing any tokens.
Every observed token through the divergence must reproduce the source journal.
These instrumented replays are diagnostics and never contribute benchmark time.
"""
import argparse
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

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(ar_path, la_path, paper_check=False):
    meta, ar_meta = read(la_path/'metadata.json'), read(ar_path/'metadata.json')
    for key in ['model', 'model_revision', 'dataset', 'input_sha256', 'max_seq',
                'max_new_tokens', 'stop_audit', 'temperature', 'dtype']:
        assert meta[key] == ar_meta[key], key
    assert meta['temperature'] == 0 and meta['dtype'] == 'bfloat16'
    assert meta['method'] == 'loopahead' and ar_meta['method'] == 'ar'
    input_path = diagnostic_input(meta, ar_meta, paper_check)
    if not paper_check:
        assert meta['reportable'] and ar_meta['reportable']
    assert not meta['trace'] and not meta['graph'] and not ar_meta['graph']
    assert digest(Path(meta['checkpoint'])) == meta['checkpoint_sha256']
    validate_replay_sources(la_path, paper_check)
    if paper_check:
        validate_replay_sources(ar_path, paper_check)
    rows = {}
    for label, folder in [('ar', ar_path), ('loopahead', la_path)]:
        marker = read(folder/'complete.json')
        assert digest(folder/'requests.jsonl') == marker['requests_sha256']
        rows[label] = [json.loads(line) for line in (folder/'requests.jsonl').open()]
        assert len(rows[label]) == marker['requests']
    if paper_check:
        inputs = [json.loads(line) for line in input_path.open()]
        assert len(inputs) == len(rows['ar']) == len(rows['loopahead']) == 164
        for source, ar in zip(inputs, rows['ar'], strict=True):
            assert source['source_id'] == ar['source_id']
            assert hashlib.sha256(source['prompt'].encode()).hexdigest() == ar['prompt_sha256']
    cases = []
    for ar, la in zip(rows['ar'], rows['loopahead'], strict=True):
        assert ar['source_id'] == la['source_id'] and ar['prompt_sha256'] == la['prompt_sha256']
        if ar['tokens'] == la['tokens']:
            continue
        position = next((i for i, (a, d) in enumerate(zip(ar['tokens'], la['tokens'])) if a != d), None)
        assert position is not None, 'Length-only disagreement requires stop-condition investigation'
        cases.append(dict(source_id=ar['source_id'], position=position,
            ar_tokens=ar['tokens'][:position+1], loopahead_tokens=la['tokens'][:position+1]))
    return meta, cases


def replay(args, meta, cases, report):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    suite = ROOT/'results/loopfill-latency512-20261107'
    protocol = read(suite/'protocol.json')
    for rel, sha in protocol['source_sha256'].items():
        assert digest(suite/'source'/rel) == sha
    sys.path[:0] = [str(suite/'source/src'), str(suite/'source/scripts'), '/tmp/loopfill-assets/ArcticInference']
    from loop_batch_bench import arm_config, enforce_precision
    from loop_latency_worker import drive
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt
    from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop, tree_decode_steps
    from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
    from loopahead.decode import LoopAheadConfig, decode_steps
    from loopahead.engine import LoopAheadExecutor
    from loopahead.lad import LADConfig, LookaheadDrafter

    spec = MODELS[meta['model']]
    assert spec.revision == meta['model_revision']
    input_path = diagnostic_input(meta, read(args.ar/'metadata.json'), args.paper_check)
    path = protocol['models'][meta['model']]['path']
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    inputs = [json.loads(line) for line in input_path.open()]
    prompts = {row['source_id']: tokenize_prompt(spec, tok, row['prompt'], meta['prompt_format']) for row in inputs}
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16).to('cuda').eval()
    model = enforce_precision(model, 'bfloat16')
    saved = torch.load(meta['checkpoint'], map_location='cpu', weights_only=True)
    lad = LookaheadDrafter(LADConfig(**saved['config']))
    lad.load_state_dict(saved['model'])
    lad = lad.to(device='cuda', dtype=torch.bfloat16).eval()
    cfg = LoopAheadConfig(**meta['config'])
    ar_cfg = arm_config('ar', spec.depth_total, *spec.loopfill_readout_depths, 4,
        draft_node_budget=16, model_top_k=2, child_suffix_caps=(4,4), suffix_c_initial=1.,
        suffix_eta=.1, loopspec_depths=spec.loopspec_depths, wavefront_depth=spec.wavefront_depth)
    stops = set(meta['stop_audit']['stop_ids'])

    class Reached(Exception):
        pass

    class ObservedLoopAhead(LoopAheadExecutor):
        observe_prefill = None

        def _prefill(self, backbone, prompt, depth):
            token = super()._prefill(backbone, prompt, depth)
            if self.observe_prefill:
                self.observe_prefill(token, self.prefill_logits)
            return token

    engines = {}
    for label, nodes in [('ar', 1), ('loopahead', 32)]:
        pool = SharedKVPool(1, pad_decode_rows=False)
        if spec.backend == 'ouro':
            backbone = OuroLoop(model, 4, meta['max_seq'], nodes, kv_factory=pool)
        else:
            backbone = HuginnLoop(model, 32, meta['max_seq'], nodes, 0, meta['max_seq'], extra_depths=(8,), kv_factory=pool)
        backbone.position_limit = meta['max_seq']
        engine = PackedLoopExecutor([backbone], cuda_graph=False) if label == 'ar' else ObservedLoopAhead(backbone, lad, saved['frequent_ids'])
        if label == 'loopahead':
            engine.capture_prefill_logits = True
        engines[label] = (backbone, engine)

    def request_seed(source_id):
        key = f'0/{meta["model"]}/{meta["dataset"]}/{source_id}'.encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)

    def describe(logits):
        logits = logits.reshape(-1, logits.shape[-1])[-1].detach().float()
        values, ids = logits.topk(3)
        probs = logits.softmax(-1)[ids]
        return dict(top_ids=ids.tolist(), top_logits=values.tolist(), top_probs=probs.tolist(),
            top2_probability_margin=float(probs[0]-probs[1]), top1_logit_margin=float(values[0]-values[1]),
            argmax=int(logits.argmax()))

    with torch.inference_mode():
        # Preserve the evaluator's two32-token warmups; their times are unused.
        for label, (backbone, engine) in engines.items():
            for row in inputs[:2]:
                prompt = prompts[row['source_id']]
                if label == 'ar':
                    steps = tree_decode_steps(backbone, prompt, ar_cfg, 32, stops, max_nodes=1)
                else:
                    engine.request(request_seed(row['source_id']), stops)
                    steps = decode_steps(backbone, prompt, cfg, 32, stops, generator=engine.generator)
                drive(backbone, engine, steps)
            engine.graphs.freeze()
        for case in cases:
            entry = {k:case[k] for k in ['source_id', 'position']}
            for label, (backbone, engine) in engines.items():
                expected = case[label+'_tokens']
                emitted = []

                def observe(token, logits):
                    index = len(emitted)
                    assert int(token) == expected[index], f'{label} replay differs before/at source divergence: {case["source_id"]}/{index}'
                    emitted.append(int(token))
                    if index == case['position']:
                        entry[label] = describe(logits)
                        assert entry[label]['argmax'] == expected[-1]
                        raise Reached()

                hook = None
                if label == 'ar':
                    head = model.lm_head
                    def head_hook(module, inputs, output):
                        value = output.reshape(-1, output.shape[-1])[-1]
                        observe(int(value.argmax()), value)
                    hook = head.register_forward_hook(head_hook)
                    steps = tree_decode_steps(backbone, prompts[case['source_id']], ar_cfg,
                        meta['max_new_tokens'], stops, max_nodes=1)
                else:
                    engine.observe_prefill = observe
                    engine.request(request_seed(case['source_id']), stops)
                    def observer(event, **data):
                        if event == 'commit':
                            assert data['position'] == len(emitted)
                            observe(data['token'], data['logits'])
                    steps = decode_steps(backbone, prompts[case['source_id']], cfg,
                        meta['max_new_tokens'], stops, generator=engine.generator, observer=observer)
                try:
                    drive(backbone, engine, steps)
                except Reached:
                    pass
                finally:
                    steps.close()
                    if hook is not None:
                        hook.remove()
                    if label == 'loopahead':
                        engine.observe_prefill = None
                assert emitted == expected and label in entry
            entry['both_original_prefixes_reproduced'] = True
            entry['at_least_one_top2_tie'] = any(entry[k]['top2_probability_margin'] == 0 for k in ['ar','loopahead'])
            report['results'].append(entry)
            with (args.out/'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(entry)+'\n')
            print(json.dumps(dict(source_id=case['source_id'], position=case['position'], reproduced=True)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ar', type=Path, required=True)
    parser.add_argument('--loopahead', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cpus')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--paper-check', action='store_true', help='Replay the separate B4 original HumanEval/max1024 check')
    args = parser.parse_args()
    meta, cases = prepare(args.ar, args.loopahead, args.paper_check)
    if args.plan_only:
        print(json.dumps(dict(model=meta['model'], dataset=meta['dataset'], cases=len(cases),
            input_provenance=input_provenance(meta, read(args.ar/'metadata.json'), args.paper_check))))
        return
    if args.gpu is None or args.cpus is None or args.out is None:
        parser.error('--gpu, --cpus and --out are required for replay')
    args.out.mkdir(parents=True, exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(args.out, [Path(__file__), BASE/'loopahead_diagnostic_inputs.py', BASE/'evaluate_loopahead.py', BASE/'provenance.py', *sorted((BASE/'loopahead').glob('*.py'))])
    uuid = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    report = dict(status='running', model=meta['model'], dataset=meta['dataset'], cases=len(cases),
        checkpoint_sha256=meta['checkpoint_sha256'], results=[], gpu_uuid=uuid, started_unix=time.time(),
        input_provenance=input_provenance(meta, read(args.ar/'metadata.json'), args.paper_check),
        source_journals={label:dict(path=str(folder.resolve()), sha256=digest(folder/'requests.jsonl'))
                         for label,folder in [('ar',args.ar),('loopahead',args.loopahead)]},
        limitations=['Instrumented BF16 replay only; timings are excluded from the benchmark.',
                    'Probability ties/margins describe actual reproduced divergences, not proof that all discrepancies disappear in FP32.'])
    with open(f'/tmp/loopfill-{uuid}.lock','a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo,hi = map(int,args.cpus.split('-'))
        os.sched_setaffinity(0,set(range(lo,hi+1)))
        try:
            if cases:
                replay(args,meta,cases,report)
            assert len(report['results']) == len(cases)
            report['status'] = 'complete'
        except Exception:
            report.update(status='failed',error=traceback.format_exc())
            raise
        finally:
            report['finished_unix'] = time.time()
            (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
