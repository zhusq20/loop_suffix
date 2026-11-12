"""Real-model scheduler compatibility with an untrained LAD; no quality claim."""
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['ouro-2.6b', 'ouro-1.4b', 'huginn-3.5b'], required=True)
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--cpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuid = {int(i.strip()): u.strip() for i, u in (line.split(',') for line in devices.splitlines())}[args.gpu]
    with open(f'/tmp/loopfill-{uuid}.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
        assert uuid not in busy
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lo, hi = map(int, args.cpus.split('-'))
        os.sched_setaffinity(0, set(range(lo, hi+1)))
        report = dict(model=args.model, gpu_uuid=uuid, started_unix=time.time(), stage='untrained_LAD_compatibility', dtype=args.dtype)
        try:
            run(args, report)
            report['status'] = 'complete'
        except Exception:
            report.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            report['finished_unix'] = time.time()
            (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2), flush=True)


def run(args, report):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    suite = ROOT/'results/loopfill-latency512-20261107'
    sys.path[:0] = [str(suite/'source/src'), str(suite/'source/scripts'), '/tmp/loopfill-assets/ArcticInference']
    from loop_batch_bench import arm_config, enforce_precision
    from loop_latency_worker import drive
    from llm_memory_editability.loop_model_registry import MODELS
    from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop, tree_decode_steps
    from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
    from loopahead.lad import LookaheadDrafter, LADConfig
    from loopahead.engine import LoopAheadExecutor
    from loopahead.decode import LoopAheadConfig, decode_steps
    protocol = json.loads((suite/'protocol.json').read_text())
    spec = MODELS[args.model]
    path = protocol['models'][args.model]['path']
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, local_files_only=True,
                                               torch_dtype=dtype).to('cuda').eval()
    model = enforce_precision(model, args.dtype)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    prompt = ('We are writing a short Python programming tutorial. The explanation should use ordinary words. '
              'A function takes two integer arguments and returns their sum. Include one example of calling it. '
              'Use meaningful variable names and keep the example small. Explain the return value in a sentence. '
              'The tutorial is for a beginner who has already learned integer arithmetic and variables. '
              'Show how the arguments become the values used by the function body. '
              'Write the function and explanation now.\n')
    ids = tok.encode(prompt, add_special_tokens=False)
    assert len(ids) >= 64
    pool = SharedKVPool(1, pad_decode_rows=False)
    if spec.backend == 'ouro':
        b = OuroLoop(model, 4, 512, 32, kv_factory=pool)
        cfg = LoopAheadConfig()
        h = model.config.hidden_size
    else:
        b = HuginnLoop(model, 32, 512, 32, 0, 512, extra_depths=(8,), kv_factory=pool)
        cfg = LoopAheadConfig(total=32, first=4, second=8, future=2, guard=1, stride=2, family='huginn')
        h = model.config.n_embd
    b.position_limit = 512
    lad = LookaheadDrafter(LADConfig(h)).to(device='cuda', dtype=dtype).eval()
    engine = LoopAheadExecutor(b, lad, list(range(8192)))
    ar_engine = PackedLoopExecutor([b], cuda_graph=False)
    ar_cfg = arm_config('ar', spec.depth_total, *spec.loopfill_readout_depths, 4,
        draft_node_budget=16, model_top_k=2, child_suffix_caps=(4,4), suffix_c_initial=1., suffix_eta=.1,
        loopspec_depths=spec.loopspec_depths, wavefront_depth=spec.wavefront_depth)
    stop = {tok.eos_token_id}
    outputs = {}
    with torch.inference_mode():
        for name in ['ar', 'loopahead', 'repeat', 'sampling']:
            if name == 'ar':
                tokens, stats = drive(b, ar_engine, tree_decode_steps(b, ids, ar_cfg, 64, stop, max_nodes=32))
            else:
                active = replace(cfg, temperature=1.) if name == 'sampling' else cfg
                engine.temperature = active.temperature
                engine.request(0, stop)
                tokens, stats = drive(b, engine, decode_steps(b, ids, active, 64, stop, generator=engine.generator))
            outputs[name] = dict(tokens=tokens, stats=stats)
            (args.out/f'{name}.json').write_text(json.dumps(outputs[name], indent=2)+'\n')
            print(name, len(tokens), stats['decode_s'], flush=True)
    a, d = outputs['ar']['tokens'], outputs['loopahead']['tokens']
    report.update(prompt_tokens=len(ids), greedy_ar_equal=a == d,
        repeat_exact=outputs['repeat']['tokens'] == d,
        first_divergence=next((i for i,(x,y) in enumerate(zip(a,d)) if x != y), None),
        lad_calls=outputs['loopahead']['stats']['lad_calls'], draft_parameters=sum(p.numel() for p in lad.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), limitations=['untrained LAD',
            'synthetic prompt only', 'sampling admission has not been calibrated', 'not a benchmark result'])
    assert report['repeat_exact']
    assert report['lad_calls'] > 0, 'The smoke test must exercise the LAD, not just native proposals'


if __name__ == '__main__':
    main()
