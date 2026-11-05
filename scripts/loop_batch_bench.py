"""Real fixed/continuous-batch throughput; no offline readouts or answers required.

Example (model weights and ArcticInference must already be installed):
  PYTHONPATH=src python scripts/loop_batch_bench.py --model ouro-2.6b \
    --model-path /models/Ouro-2.6B --questions questions.jsonl \
    --arms ar,wavefront,budget-suffix,budget-raw,budget-online --scoring raw \
    --draft-node-budget 64 --batch-sizes 1,4 --out results/batch-ouro
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode
from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, SharedKVPool
from llm_memory_editability.loop_budget_decode import BudgetConfig
from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_prob_tree import Calibration, ProbConfig
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop, tree_decode
from llm_memory_editability.loop_raven_decode import RavenLoop, INIT_POLICY
from llm_memory_editability.loop_model_registry import (
    MODELS, checkpoint_manifest, context_audit, resolve_stop_ids, tokenize_prompt,
)

SPECS = {k: (s.depth_total, *s.loopfill_readout_depths, s.stop_token) for k, s in MODELS.items()}
ARMS = ("ar", "wavefront", "prob-low", "prob-load", "suffix",
        "budget-raw", "budget-online", "budget-fixed", "budget-online-length", "budget-fixed-length",
        "budget-suffix", "budget-shallow", "loopspec-dual", "loopspec-capped")


def parser():
    p = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    p.add_argument("--model", choices=SPECS, required=True)
    p.add_argument("--model-path", required=True, help="local Hugging Face checkpoint directory")
    p.add_argument("--questions", required=True, help="Spec-Bench JSONL or prompt_token_ids JSONL")
    p.add_argument("--calibration", help="frozen calibration JSON, dev table only")
    p.add_argument("--out", required=True)
    p.add_argument("--scoring", choices=("four", "simple", "raw"), default="four")
    p.add_argument("--tau", type=float, help="override probability-tree threshold; ablation only")
    p.add_argument("--draft-node-budget", type=int, default=64,
                   help="N per request for budget-* arms; excludes the confirmed root")
    p.add_argument("--suffix-eta", type=float, default=0.1)
    p.add_argument("--suffix-c-min", type=float, default=0.25)
    p.add_argument("--suffix-c-max", type=float, default=2.0)
    p.add_argument("--suffix-c-initial", type=float, default=1.0,
                   help="initial online coefficient, or constant coefficient for budget-fixed[-length]")
    p.add_argument("--model-top-k", type=int, choices=(1, 2, 4, 8), default=8)
    p.add_argument("--model-readout-depths", type=int, nargs="+",
                   help="budget arms only: intermediate model readout depths; default uses both model-specific depths")
    p.add_argument('--loopspec-depths', type=int, nargs=2,
                   help='LoopSpec first/second readout only; defaults to registry')
    p.add_argument('--wavefront-depth', type=int, help='WFD first readout only; defaults to registry')
    p.add_argument("--child-suffix-policy", choices=("all", "best1", "none"), default="all")
    p.add_argument("--child-suffix-caps", type=int, nargs="+",
                   help="per-model-rank suffix node caps; zero skips that child's query; root cap unchanged")
    p.add_argument("--positive-only", action="store_true")
    p.add_argument("--existing-child-probs", action="store_true",
                   help="budget model arms: score existing children outside top-k with their full-vocabulary softmax probability")
    p.add_argument("--progress-every", type=int, default=0,
                   help="write progress after this many completed requests; zero disables")
    p.add_argument("--trace-controller", action="store_true",
                   help="diagnostic coefficient traces and CPU metadata estimates; timed separately")
    p.add_argument("--arms", default="ar,wavefront,prob-low,prob-load")
    p.add_argument("--batch-sizes", default="1,4,16")
    p.add_argument("--schedule", choices=("fixed", "continuous"), default="fixed",
                   help="continuous fills freed request slots at the next scheduler iteration")
    p.add_argument("--prefill-chunk-size", type=int, default=0,
                   help="positive token count enables prefill/decode interleaving; 0 uses full prompts")
    p.add_argument("--attention-backend", choices=("per-request", "packed"), default="per-request")
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=False,
                   help="capture decode shapes in warmup only; requires packed")
    p.add_argument("--max-cuda-graphs", type=int, default=24)
    p.add_argument("--pad-decode-rows", action=argparse.BooleanOptionalAction, default=False,
                   help="packed backend: bucket execution rows using an isolated KV sink for padding")
    p.add_argument("--verifier-shape-warmup", choices=("complete", "observed"), default="complete",
                   help="with row padding: prime bounded verifier shapes or retain request-observed warmup only")
    p.add_argument("--allocation-guard", choices=("conservative", "disabled"),
                   default="conservative",
                   help="disabled attempts the requested buffers and lets CUDA report actual OOM")
    p.add_argument("--order-seeds", default="0,1,2", help="paired shuffle seeds for full streams")
    p.add_argument("--repeat-id", type=int, default=0,
                   help="timing replicate ID; use a separate output directory, odd IDs reverse arms")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--buffer-policy", choices=("shared", "per-arm"), default="shared",
                   help="per-arm allocates AR=1, budget=N+1 and LoopSpec's schedule bound")
    p.add_argument(
        "--max-nodes", type=int, default=8192, help="per-slot tree cap; logged, never hidden"
    )
    p.add_argument("--suffix-max-draft", type=int, default=64)
    p.add_argument("--prompt-format", choices=("raw", "chat"),
                   help='defaults to the registered model prompt format')
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--seed", type=int, default=0, help="model/init seed, independent of request order"
    )
    p.add_argument("--limit", type=int, help="first N IDs before shuffling; for smoke checks")
    p.add_argument("--warmup-tokens", type=int, default=8)
    p.add_argument("--warmup-request-limit-factor", type=int, choices=(1, 2),
                   help="warmup requests per batch slot; default: 2 with CUDA Graph, otherwise 1")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--check",
        action="store_true",
        help="compare against untimed serial AR; FP32 fails on mismatch",
    )
    p.add_argument("--reference-file", help="locked, resumable serial AR JSONL shared by paired runs")
    p.add_argument('--reference-only', action='store_true',
                   help='generate/resume the untimed serial reference and exit without timing')
    p.add_argument(
        "--profile", action="store_true", help="export a PyTorch trace; results marked profiled"
    )
    return p


def arm_config(name, T, d, deep, suffix_max_draft, *, draft_node_budget=64,
               suffix_eta=0.1, suffix_c_min=0.25, suffix_c_max=2.0, trace_controller=False,
               model_top_k=8, child_suffix_policy="all", positive_only=False, child_suffix_caps=None,
               suffix_c_initial=1.0, model_readout_depths=None, existing_child_probs=False,
               loopspec_depths=None, wavefront_depth=None):
    if model_readout_depths is not None and not name.startswith("budget-"):
        raise ValueError("model readout override requires budget arms")
    if existing_child_probs and (not name.startswith("budget-") or name == "budget-suffix"):
        raise ValueError("existing child probabilities require budget arms with model readouts")
    if name in ("loopspec-dual", "loopspec-capped"):
        d, deep = loopspec_depths or (d, deep)
        return LoopSpecConfig(name, T, d, deep,
                              draft_node_budget=draft_node_budget if name == "loopspec-capped" else None)
    if name.startswith("budget-"):
        return BudgetConfig(
            name, T, d, (d, deep) if model_readout_depths is None else tuple(model_readout_depths),
            draft_node_budget=draft_node_budget,
            scoring={"budget-online": "online_suffix_i", "budget-fixed": "fixed_suffix",
                     "budget-online-length": "online_suffix_length",
                     "budget-fixed-length": "fixed_suffix_length"}.get(name, "raw"),
            eta=suffix_eta, c_min=suffix_c_min, c_max=suffix_c_max,
            use_model=name != "budget-suffix", suffix_room=suffix_max_draft,
            use_suffix=name != "budget-shallow", top_m=model_top_k,
            child_suffix_policy=child_suffix_policy, positive_only=positive_only,
            child_suffix_caps=tuple(child_suffix_caps) if child_suffix_caps is not None else None,
            c_initial=suffix_c_initial,
            existing_child_probs=existing_child_probs,
            trace_controller=trace_controller,
        )
    if name.startswith("prob-"):
        return ProbConfig(
            name,
            T,
            d,
            (d, deep),
            tau=0.01 if name == "prob-low" else 0.1,
            joint=True,
            prune=True,
            suffix_calibration="insitu",
        )
    return ArmConfig(
        name,
        T,
        wavefront_depth if name == 'wavefront' and wavefront_depth is not None else d,
        (1 << 30) if name == "wavefront" else suffix_max_draft,
        name == "suffix",
        name == "wavefront",
    )


def load_requests(path, tokenizer, args, model_eos, generation_eos=None, vocab_size=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r.get("question_id", r.get("request_id", 0)))
    if args.limit is not None:
        rows = rows[: args.limit]
    spec = MODELS[args.model]
    fmt = args.prompt_format or spec.prompt_format
    vocab_size = vocab_size or (max(tokenizer.get_vocab().values()) + 1)
    stop, _ = resolve_stop_ids(spec, tokenizer, model_eos, generation_eos, vocab_size, fmt)
    requests, metadata = [], {}
    for i, row in enumerate(rows):
        rid = row.get("question_id", row.get("request_id", i))
        if row.get('model_id', args.model) != args.model:
            raise ValueError(f'request {rid}: tokenized input belongs to another model')
        if "prompt_token_ids" in row:
            prompt = row["prompt_token_ids"]
        else:
            text = row["turns"][0] if "turns" in row else row["prompt"]
            prompt = tokenize_prompt(spec, tokenizer, text, fmt) if spec.expansion else (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True
                )
                if fmt == "chat"
                else tokenizer(text).input_ids
            )
        n = row.get("max_new_tokens", args.max_new_tokens)
        if any(type(t) is not int or not 0 <= t < vocab_size for t in prompt):
            raise ValueError(f'request {rid}: invalid input token ID')
        # Reserve room for unverified WFD descendants near the output boundary.
        margin = 0 if spec.expansion else SPECS[args.model][0] + args.suffix_max_draft
        if not prompt or n < 1 or len(prompt) + n + margin > args.max_seq:
            raise ValueError(
                f"request {rid}: prompt={len(prompt)}, output={n}, margin={margin}; "
                f"exceeds max_seq={args.max_seq} or invalid budget. No truncation applied."
            )
        if spec.expansion:
            audit = context_audit(spec, rid, prompt, n, max_seq=args.max_seq)
            if not audit['context_ok']:
                raise ValueError(f'request {rid}: exceeds model context; no truncation: {audit}')
        requests.append(BatchRequest(rid, list(prompt), n, stop.copy()))
        metadata[rid] = dict(category=row.get("category"), prompt_tokens=len(prompt))
    if not requests or len(metadata) != len(requests):
        raise ValueError("input must contain nonempty, uniquely identified requests")
    return requests, metadata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def enforce_precision(model, dtype):
    # Some pinned remote-code/Transformers combinations ignore torch_dtype.
    # Preserve positional buffers at their original precision: Huginn stores
    # real-valued cos/sin in float32, while other backbones may use complex RoPE.
    buffers = [(module, name, value) for module in model.modules()
               for name, value in module._buffers.items()]
    model = model.bfloat16() if dtype == "bfloat16" else model.float()
    for module, name, value in buffers:
        module._buffers[name] = value
    actual = {str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    if actual != {str(getattr(torch, dtype))}:
        raise TypeError(f"requested {dtype}, actual parameter dtypes {actual}")
    return model


def serial_reference(path, identity, requests, generate):
    """Reuse only an exact reference identity; persist each finished request outside timing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        header = {"reference_identity": identity}
        expected = {r.request_id for r in requests}
        reference = {}
        if path.exists():
            with path.open() as f:
                if json.loads(next(f)) != header:
                    raise ValueError(f"reference identity mismatch: {path}")
                for line in f:
                    row = json.loads(line)
                    rid = row["request_id"]
                    if rid not in expected or rid in reference:
                        raise ValueError(f"invalid/duplicate reference ID: {rid}")
                    if not row["tokens"] or not all(type(t) is int for t in row["tokens"]):
                        raise ValueError(f"invalid reference tokens: {rid}")
                    reference[rid] = row["tokens"]
        else:
            path.write_text(json.dumps(header) + "\n")
        with path.open("a", buffering=1) as f:
            for index, request in enumerate(requests, 1):
                rid = request.request_id
                if rid not in reference:
                    tokens = generate(request)
                    f.write(json.dumps(dict(request_id=rid, tokens=tokens)) + "\n")
                    reference[rid] = tokens
                print(f"reference {index}/{len(requests)}: request={rid}, "
                      f"tokens={len(reference[rid])}", flush=True)
        return reference


def main():
    args = parser().parse_args()
    spec = MODELS[args.model]
    args.prompt_format = args.prompt_format or spec.prompt_format
    warmup_request_limit_factor = args.warmup_request_limit_factor
    if warmup_request_limit_factor is None:
        warmup_request_limit_factor = 2 if args.cuda_graph else 1
    arms = args.arms.split(",")
    batches = [int(n) for n in args.batch_sizes.split(",")]
    seeds = [int(n) for n in args.order_seeds.split(",")]
    if not set(arms) <= set(ARMS) or min(batches) < 1 or args.max_nodes < 1:
        raise ValueError("invalid arms, batch sizes or max_nodes")
    if any(len(items) != len(set(items)) for items in (arms, batches, seeds)):
        raise ValueError("arms, batch sizes and order seeds must not contain duplicates")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.warmup_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("warmup and generation budgets must be positive")
    if args.tau is not None and not 0 < args.tau < 1:
        raise ValueError("tau must be between zero and one")
    budget_arms = [a for a in arms if a.startswith("budget-")]
    if args.repeat_id < 0:
        raise ValueError("repeat-id must be nonnegative")
    if args.reference_file and not args.check:
        raise ValueError("--reference-file requires --check")
    if args.reference_only and not args.check:
        raise ValueError('--reference-only requires --check')
    if budget_arms:
        if args.calibration or args.tau is not None or args.scoring != "raw":
            raise ValueError("budget arms require --scoring raw and no calibration or tau")
        if args.max_nodes < args.draft_node_budget + 1:
            raise ValueError("--max-nodes must cover --draft-node-budget + 1")
    T, d, deep, _ = SPECS[args.model]

    def config_for(arm):
        config = arm_config(
            arm, T, d, deep, args.suffix_max_draft,
            draft_node_budget=args.draft_node_budget, suffix_eta=args.suffix_eta,
            suffix_c_min=args.suffix_c_min, suffix_c_max=args.suffix_c_max,
            trace_controller=args.trace_controller,
            model_top_k=args.model_top_k, child_suffix_policy=args.child_suffix_policy,
            positive_only=args.positive_only,
            child_suffix_caps=args.child_suffix_caps,
            suffix_c_initial=args.suffix_c_initial,
            model_readout_depths=args.model_readout_depths,
            existing_child_probs=args.existing_child_probs,
            loopspec_depths=args.loopspec_depths or spec.loopspec_depths,
            wavefront_depth=args.wavefront_depth or spec.wavefront_depth,
        )
        if isinstance(config, ProbConfig) and args.tau is not None:
            config = replace(config, tau=args.tau)
        return config

    configs = {arm: config_for(arm) for arm in arms}  # validate before loading weights
    if any(a.startswith("prob-") for a in arms) and args.scoring != "raw" and not args.calibration:
        raise ValueError("probability arms require --calibration")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists():
        raise FileExistsError(f"{out} already contains a benchmark; use a new output directory")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    generation_path = Path(args.model_path) / 'generation_config.json'
    generation_eos = json.loads(generation_path.read_text()).get('eos_token_id') if generation_path.exists() else None
    reqs, req_meta = load_requests(args.questions, tokenizer, args,
                                  getattr(model_config, 'eos_token_id', None), generation_eos,
                                  model_config.vocab_size)
    _, stop_audit = resolve_stop_ids(spec, tokenizer, getattr(model_config, 'eos_token_id', None),
                                    generation_eos, model_config.vocab_size, args.prompt_format)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
        )
        .to(device)
        .eval()
    )
    model = enforce_precision(model, args.dtype)
    T, d, deep, _ = SPECS[args.model]
    if args.model.startswith("ouro"):
        model.model.total_ut_steps = T
        model.early_exit_step = None
        model.early_exit_threshold = None
    calibration = None
    if args.calibration:
        raw = json.loads(Path(args.calibration).read_text())
        if raw.get("model", args.model) != args.model:
            raise ValueError("calibration belongs to a different model")
        calibration = Calibration.from_json(raw["dev"])
    if args.scoring != "four":
        from llm_memory_editability.loop_score_ablation import RawScores, SimpleScores
        calibration = RawScores() if args.scoring == "raw" else SimpleScores(calibration)
        if args.scoring == "simple":
            (out / "simple-scoring-table.json").write_text(json.dumps(calibration.to_json(), indent=2))

    # Allocate only once per concurrency; all methods share the same buffer budget.
    def nodes_for(arm):
        if args.buffer_policy == "shared":
            return args.max_nodes
        if arm == "ar":
            return 1
        if arm.startswith("budget-"):
            return args.draft_node_budget + 1
        if arm in ("loopspec-dual", "loopspec-capped"):
            return configs[arm].physical_nodes
        return args.max_nodes

    if args.prefill_chunk_size < 0 or args.max_cuda_graphs < 1:
        raise ValueError("invalid chunk size or graph count")
    if args.pad_decode_rows and args.attention_backend != "packed":
        raise ValueError("row padding requires packed attention")
    if args.cuda_graph and (args.attention_backend != "packed" or device.type != "cuda"):
        raise ValueError("CUDA graphs require the packed CUDA backend")

    def backbone(nodes=None, kv_factory=None):
        nodes = args.max_nodes if nodes is None else nodes
        if spec.backend == 'ouro':
            b = OuroLoop(model, T, args.max_seq, nodes, kv_factory=kv_factory)
            if spec.expansion:
                b.position_limit = min(args.max_seq, spec.context_limit)
            return b
        if spec.backend == 'raven':
            depths = set(spec.loopfill_readout_depths) | set(args.loopspec_depths or spec.loopspec_depths)
            depths.update(args.model_readout_depths or ())
            # Every explicitly requested depth gets a separate coda prefix.
            return RavenLoop(model, T, args.max_seq, nodes, args.seed,
                             extra_depths=tuple(sorted(depths)), kv_factory=kv_factory)
        if spec.backend != 'huginn':
            raise ValueError(f'Unknown backend: {spec.backend}')
        return HuginnLoop(
            model, T, args.max_seq, nodes, args.seed, args.max_seq, extra_depths=(deep,), kv_factory=kv_factory
        )

    def cache_for(config):
        enabled = config.use_suffix and not (
            isinstance(config, BudgetConfig) and config.draft_node_budget == 0
        )
        return CompletionSuffixCache.arctic() if enabled else None

    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    sources = [Path(__file__).resolve()] + sorted(
        (Path(__file__).resolve().parents[1] / "src/llm_memory_editability").glob("*.py")
    )
    source_hashes = {str(p.relative_to(Path(__file__).resolve().parents[1])): digest(p)
                     for p in sources}
    meta = dict(
        args=vars(args),
        git_revision=revision.stdout.strip(),
        git_dirty=bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        ),
        torch=torch.__version__,
        transformers=__import__('transformers').__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        actual_kv_dtype=str(getattr(torch, args.dtype)),
        actual_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        complex_buffer_dtypes={name: str(value.dtype) for name, value in model.named_buffers()
                               if value.is_complex()},
        buffer_dtypes={name: str(value.dtype) for name, value in model.named_buffers()},
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid)
        if device.type == "cuda" else None,
        source_sha256=source_hashes,
        questions_sha256=digest(args.questions),
        calibration_sha256=digest(args.calibration),
        backend="packed_prefix_and_tree" if args.attention_backend == "packed" else "eager_fused_projections_per_request_tree_attention",
        cuda_graph=args.cuda_graph,
        compile=False,
        prefill="chunked_interleaved" if args.prefill_chunk_size else "sequential",
        warmup_prefill="same_as_measured",
        warmup_request_limit_factor=warmup_request_limit_factor,
        schedule=args.schedule,
        history="cold_start_completion_only",
        profiled=args.profile,
        controller_profiled=args.trace_controller,
        configs={arm: asdict(config) for arm, config in configs.items()},
        checkpoint=checkpoint_manifest(spec, args.model_path) if spec.expansion else None,
        loaded_model_class=f'{type(model).__module__}.{type(model).__name__}',
        stop_audit=stop_audit,
        initialization_policy=INIT_POLICY if spec.backend == 'raven' else 'historical_backend',
        coda_policy='independent_depth_caches; WFD uses its existing shared coda rule',
        physical_nodes={arm: nodes_for(arm) for arm in arms},
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    reference = {}
    if args.check:
        b = None
        config = arm_config("ar", T, d, deep, args.suffix_max_draft)
        def generate(r):
            nonlocal b
            if b is None:
                b = backbone(nodes_for("ar"))
            return tree_decode(
                b, r.prompt, config, r.max_new_tokens, r.stop, max_nodes=nodes_for("ar")
            )[0]
        identity = dict(model=args.model, model_path=str(Path(args.model_path).resolve()),
                        dtype=args.dtype, seed=args.seed, max_seq=args.max_seq,
                        max_nodes=nodes_for("ar"), source_sha256=source_hashes,
                        torch=torch.__version__, cuda=torch.version.cuda,
                        requests_sha256=hashlib.sha256(json.dumps([
                            [r.request_id, r.prompt, r.max_new_tokens, sorted(r.stop)]
                            for r in reqs]).encode()).hexdigest())
        reference_start = time.perf_counter()
        reference_path = args.reference_file or str(out / "serial-ar.jsonl")
        reference = serial_reference(reference_path, identity, reqs, generate)
        meta.update(reference_file=str(Path(reference_path).resolve()),
                    reference_sha256=digest(reference_path),
                    reference_setup_s=time.perf_counter() - reference_start)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        del b
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.reference_only:
        (out / 'reference-complete.json').write_text(json.dumps(dict(
            requests=len(reference), reference_sha256=meta['reference_sha256'])))
        return

    def allocate(B, nodes):
        # A dense cache can be large: fail explicitly rather than silently lowering B/caps.
        pool = SharedKVPool(B, pad_decode_rows=args.pad_decode_rows) if args.attention_backend == "packed" else None
        first = backbone(nodes, pool)
        kv = first.kv
        if kv.K.dtype != getattr(torch, args.dtype):
            raise TypeError("KV cache precision does not match the requested experiment dtype")
        slot_bytes = sum(
            t.numel() * t.element_size() for t in (kv.K, kv.V, kv.TK, kv.TV, first.hidden)
        )
        if hasattr(first, "static"):
            slot_bytes += first.static.numel() * first.static.element_size()
        if hasattr(first, "init"):
            slot_bytes += first.init.numel() * first.init.element_size()
        if device.type == "cuda" and args.allocation_guard == "conservative" and pool is None:
            free, _ = torch.cuda.mem_get_info(device)
            if slot_bytes * (B - 1) > free * 0.85:
                raise MemoryError(
                    f"CAPACITY_GUARD B={B}: dense slot buffers need about "
                    f"{slot_bytes * B / 2**30:.1f} GiB; free={free / 2**30:.1f} GiB. "
                    "Reduce batch size or explicitly change --max-nodes/--max-seq; "
                    "changing caps changes the experiment."
                )
        return [first] + [backbone(nodes, pool) for _ in range(B - 1)], slot_bytes

    for B in batches:
        if args.buffer_policy == "shared":
            slots, slot_bytes = allocate(B, args.max_nodes)
        for order_index, seed in enumerate(seeds):
            ordered = list(reqs)
            random.Random(seed).shuffle(ordered)
            arm_order = arms[order_index % len(arms) :] + arms[: order_index % len(arms)]
            if args.repeat_id % 2:
                arm_order = arm_order[::-1]
            for arm in arm_order:
                config = configs[arm]
                physical_nodes = nodes_for(arm)
                if args.buffer_policy == "per-arm":
                    slots, slot_bytes = allocate(B, physical_nodes)
                warm = [
                    replace(r, max_new_tokens=min(args.warmup_tokens, r.max_new_tokens))
                    for r in ordered[:B * warmup_request_limit_factor]
                ]
                engine = (PackedLoopExecutor(slots, args.cuda_graph, args.max_cuda_graphs)
                          if args.attention_backend == "packed" else None)
                batch_decode(
                    warm, slots, config, calibration, cache_for(config), max_nodes=physical_nodes,
                    schedule=args.schedule, executor=engine, prefill_chunk_size=args.prefill_chunk_size,
                )
                if engine is not None:
                    if (args.pad_decode_rows and args.cuda_graph and engine.pool.max_nodes <= 65
                            and args.verifier_shape_warmup == "complete"):
                        if arm == 'ar':
                            engine.prime_verifier_shapes(1)
                        elif isinstance(config, BudgetConfig) and not config.use_model:
                            engine.prime_verifier_shapes(min(config.draft_node_budget, config.suffix_room) + 1)
                    engine.graphs.freeze()
                cache = cache_for(config)  # warmup outputs never enter measured history
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                key = f"{arm}-b{B}-seed{seed}"
                def progress(completed, elapsed):
                    if len(completed) % args.progress_every and len(completed) != len(ordered):
                        return
                    record = dict(completed_requests=len(completed), total_requests=len(ordered),
                                  completed_output_tokens=sum(r['n_tokens'] for r in completed),
                                  elapsed_s=elapsed, updated_unix=time.time(), phase="measured")
                    target = out / f"{key}-progress.json"
                    temporary = target.with_suffix(".tmp")
                    temporary.write_text(json.dumps(record))
                    temporary.replace(target)
                    print(f"{key}: {len(completed)}/{len(ordered)} completed, {elapsed:.1f}s", flush=True)
                prof = None
                try:
                    if args.profile:
                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if device.type == "cuda":
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                        prof = torch.profiler.profile(activities=activities, record_shapes=True)
                        prof.start()
                    records, summary = batch_decode(
                        ordered, slots, config, calibration, cache, max_nodes=physical_nodes,
                        schedule=args.schedule, executor=engine, prefill_chunk_size=args.prefill_chunk_size,
                        progress_callback=progress if args.progress_every > 0 else None,
                    )
                except Exception as exc:
                    (out / f"{key}-failure.json").write_text(
                        json.dumps(dict(error=repr(exc), arm=arm, batch_size=B, seed=seed))
                    )
                    raise
                finally:
                    if prof is not None:
                        prof.stop()
                        prof.export_chrome_trace(str(out / f"{key}-trace.json"))
                for r in records:
                    r.update(req_meta[r["request_id"]])
                    if reference:
                        ref = reference[r["request_id"]]
                        r["same_as_serial_ar"] = r["tokens"] == ref
                        r["first_mismatch"] = next(
                            (i for i, (x, y) in enumerate(zip(r["tokens"], ref)) if x != y),
                            min(len(ref), len(r["tokens"]))
                            if len(ref) != len(r["tokens"])
                            else None,
                        )
                summary.update(
                    arm=arm,
                    config=asdict(config),
                    order_seed=seed,
                    repeat_id=args.repeat_id,
                    request_order=[r.request_id for r in ordered],
                    max_nodes=physical_nodes,
                    buffer_policy=args.buffer_policy,
                    slot_buffer_bytes=slot_bytes,
                    profiled=args.profile,
                    controller_profiled=args.trace_controller,
                    tree_policy="node_budget" if isinstance(config, BudgetConfig) else (
                        ("loopspec_dual" if config.draft_node_budget is None else "loopspec_capped") if isinstance(config, LoopSpecConfig) else
                        "threshold" if isinstance(config, ProbConfig) else "legacy_arm"
                    ),
                    scoring=config.scoring if isinstance(config, BudgetConfig) else args.scoring,
                    cap_skips=sum(r["stats"].get("cap_skips", 0) for r in records),
                    publications=cache.publications if cache else [],
                )
                if reference:
                    summary["same_as_serial_ar"] = sum(r["same_as_serial_ar"] for r in records)
                if device.type == "cuda":
                    summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                (out / f"{key}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                (out / f"{key}-summary.json").write_text(json.dumps(summary, indent=2))
                print(
                    f"{key}: {summary['output_tput']:.3f} tok/s, "
                    f"{summary['requests']} requests, cap_skips={summary['cap_skips']}",
                    flush=True,
                )
                if (
                    args.check
                    and args.dtype == "float32"
                    and any(not r["same_as_serial_ar"] for r in records)
                ):
                    raise AssertionError(f"{key}: FP32 output differs from serial AR; see JSONL")
                del engine
                if args.buffer_policy == "per-arm":
                    del slots
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        if args.buffer_policy == "shared":
            del slots
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (out / "complete.json").write_text(json.dumps(dict(
        streams=len(arms) * len(batches) * len(seeds), requests_per_stream=len(reqs),
        finished_unix=time.time(), reference_sha256=meta.get("reference_sha256")), indent=2))


def cuda_oom_cause(error):
    """A graph capture-end error can wrap the original typed allocation failure."""
    pending, seen = [error], set()
    while pending:
        exc = pending.pop()
        if id(exc) in seen:
            continue
        seen.add(id(exc))
        if isinstance(exc, torch.OutOfMemoryError):
            return exc
        pending.extend(e for e in (exc.__cause__, exc.__context__) if e is not None)
    return None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Covers allocation and warmup as well as the timed decode. A supervisor
        # runs each cell in a fresh process so the next cell has a clean context.
        oom = cuda_oom_cause(exc)
        if oom is None:
            raise
        args = parser().parse_args()
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure.json").write_text(json.dumps(dict(
            outcome="oom", exception_type=type(oom).__name__, error=str(oom), outer_error=repr(exc),
            model=args.model, arms=args.arms, batch_sizes=args.batch_sizes,
            finished_unix=time.time()), indent=2))
        raise
