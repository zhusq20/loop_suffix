"""Score-coordination headroom: wide fixed-c sweep and oracle-ranking replay (diagnostic).

One job = one model x node budget N x scoring variant over a frozen stratified subset.
It follows the B=1 latency protocol (N/Q/top-k/child caps, cold suffix history per
dataset, only completed outputs published, optional target sampling), but every request
carries a DraftTrace observer: wall time is NOT throughput. Report loop-step counts.

Variants
  fixed-c<c>    constant suffix coefficient, edge score min(1, c q); c=1 is raw scoring
  online-c<c0>  the existing per-request integral controller started at c0
  oracle-first  same proposals, scores and N as fixed-c1, but candidates on the
                reference output path (the fixed-c1 job's output for the same request)
                are selected first; remaining slots follow c=1 score order
  oracle-only   as oracle-first, but only reference-path candidates are kept

Oracle variants rerank only what the decoder itself proposed, so they bound what any
better scoring of the two sources could achieve at this N. They do not add proposals.
If the realized output leaves the reference (BF16 shape effects, sampling), no candidate
is on-path afterwards and selection falls back to c=1 order; such requests are flagged.

Each handover (a correct node becoming the root) is split into
  proposal_wait = T - min(T, ticks since the node was first proposed)
  ranking_wait  = min(T, ticks since first proposal) - progress at handover
so that calls = T + sum(proposal_wait + ranking_wait). ranking_wait is the part that
earlier admission and no budget eviction could have removed; proposal_wait needs
earlier or different proposals. The split is per node: a parent admitted late can also
delay its children's proposals, which is counted as proposal_wait of the children.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from loop_draft_trace import DraftTrace  # noqa: E402
from loop_latency_worker import digest, drive, load_journal, read, save  # noqa: E402
import llm_memory_editability.loop_budget_decode as budget_module  # noqa: E402

_SCORE_SELECT = budget_module.select_candidates
ORACLE_VARIANTS = ("oracle-first", "oracle-only")


def fmt_c(c):
    return format(float(c), "g")


def parse_variant(name):
    if name.startswith("fixed-c"):
        return dict(kind="fixed", c=float(name[len("fixed-c"):]))
    if name.startswith("online-c"):
        return dict(kind="online", c=float(name[len("online-c"):]))
    if name in ORACLE_VARIANTS:
        return dict(kind=name, c=1.0)
    raise ValueError(f"unknown variant {name!r}")


def variant_names(cs, online_c0):
    cs = sorted({float(c) for c in cs} | {1.0})
    return ([f"fixed-c{fmt_c(c)}" for c in cs] + [f"online-c{fmt_c(online_c0)}"]
            + list(ORACLE_VARIANTS))


class OracleRanking:
    """Hindsight reranking of the decoder's own candidate pool; scores are untouched."""

    def __init__(self, reference, only=False):
        self.reference = tuple(int(t) for t in reference)
        self.only = only
        self.counts = Counter()

    def on_path(self, path):
        return len(path) <= len(self.reference) and path == self.reference[: len(path)]

    def __call__(self, pool, root, budget, coefficient, *, positive_only=False,
                 length_conditioned=False):
        # Score every candidate exactly as the decoder would (fills e and p).
        _SCORE_SELECT(pool, root, len(pool), coefficient, positive_only=False,
                      length_conditioned=length_conditioned)
        others = [c for path, c in pool.items() if path != root]
        hits = sorted((c for c in others if self.on_path(c.path)), key=lambda c: len(c.path))
        rest = [] if self.only else sorted(
            (c for c in others if not self.on_path(c.path) and (not positive_only or c.p > 0)),
            key=lambda c: (-c.p, len(c.path), c.path))
        # Reference-path nodes form one chain whose ancestors are also on the path, and
        # the remaining order is the decoder's own, so the selection stays ancestor-closed.
        chosen = (hits + rest)[:budget]
        self.counts["selections"] += 1
        self.counts["on_path_candidates"] += len(hits)
        self.counts["on_path_truncated"] += max(0, len(hits) - budget)
        self.counts["root_on_path"] += self.on_path(root)
        return {root: pool[root], **{c.path: c for c in chosen}}


@contextmanager
def ranking(selector):
    """Temporarily replace the budget decoder's selection rule (None keeps it)."""
    original = budget_module.select_candidates
    if selector is not None:
        budget_module.select_candidates = selector
    try:
        yield
    finally:
        budget_module.select_candidates = original


def wait_decomposition(depth, entries, gold_nodes):
    first = {n["position"]: n["first_proposal"] for n in gold_nodes}
    allocations = {n["position"]: n["allocations"] for n in gold_nodes}
    totals, per_entry = Counter(), []
    for e in entries:
        if e["terminal"]:
            continue
        tick, step, position = e["tick"], e["step"], e["position"]
        proposed = first.get(position)
        reach = 0 if proposed is None or proposed >= tick else min(depth, tick - proposed)
        if step > reach:  # cannot happen if the trace is consistent; never hide it
            totals["reach_violations"] += 1
            reach = step
        proposal_wait, ranking_wait = depth - reach, reach - step
        evicted = any(a["reason"] == "budget" and a["release_tick"] is not None
                      and a["release_tick"] <= tick for a in allocations.get(position, ()))
        totals["entries"] += 1
        totals["wait"] += depth - step
        totals["proposal_wait"] += proposal_wait
        totals["ranking_wait"] += ranking_wait
        totals["ranking_entries"] += ranking_wait > 0
        totals["evicted_entries"] += evicted
        totals["full_ready_entries"] += step == depth
        per_entry.append([position, step, proposal_wait, ranking_wait])
    return dict(totals), per_entry


def traced_decode(run_steps, depth, oracle=None):
    """run_steps(observer) -> (tokens, stats); returns tokens, stats and trace payload."""
    trace = DraftTrace(depth)
    extra = Counter()

    def observe(event, **data):
        if event == "selection_candidates":
            # Only selections where positive-score candidates exceed N can be
            # changed by any monotone reweighting of the two sources.
            positive = sum(v[3] > 0 for p, v in data["candidates"].items() if p != data["root"])
            extra["positive_contested_selections"] += positive > data["budget"]
        trace.observe(event, **data)

    with ranking(oracle):
        tokens, stats = run_steps(observe)
    counts, gold_nodes, entries = trace.finish(tokens, stats)
    counts.update(extra)
    waits, per_entry = wait_decomposition(depth, entries, gold_nodes)
    payload = dict(trace=counts, waits=waits, wait_entries=per_entry,
                   oracle=dict(oracle.counts) if oracle is not None else None)
    return tokens, stats, payload


def budget_config(spec, variant, budget, protocol, arm_config):
    v = parse_variant(variant)
    c = v["c"]
    if v["kind"] == "online":
        lo, hi = protocol["online_c_min"], protocol["online_c_max"]
    else:
        # A constant weight needs bounds only to satisfy BudgetConfig validation.
        lo, hi = min(protocol["online_c_min"], c), max(protocol["online_c_max"], c)
    return arm_config(
        "budget-online" if v["kind"] == "online" else "budget-fixed",
        spec.depth_total, *spec.loopfill_readout_depths, protocol["suffix_max_draft"],
        draft_node_budget=budget, model_top_k=protocol["model_top_k"],
        child_suffix_caps=tuple(protocol["child_suffix_caps"]), suffix_c_initial=c,
        suffix_eta=protocol["suffix_eta"], suffix_c_min=lo, suffix_c_max=hi,
        loopspec_depths=spec.loopspec_depths, wavefront_depth=spec.wavefront_depth)


def load_subset(suite, protocol):
    rows = []
    for dataset in protocol["datasets"]:
        path = suite / "inputs" / f"{dataset}.jsonl"
        if digest(path) != protocol["inputs"][dataset]["sha256"]:
            raise ValueError(f"input checksum changed: {dataset}")
        rows += [dict(json.loads(line), dataset=dataset)
                 for line in path.read_text().splitlines() if line.strip()]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--smoke", action="store_true", help="two requests x 16 tokens, separate folder")
    args = p.parse_args()
    suite = args.suite.resolve()
    protocol = read(suite / "protocol.json")
    job = read(suite / "jobs" / f"{args.job}.json")
    for rel, expected in protocol["source_sha256"].items():
        if digest(Path(protocol["source"]) / rel) != expected:
            raise ValueError(f"frozen source changed: {rel}")
    target = suite / ("validation" if args.smoke else "runs") / args.job
    target.mkdir(parents=True, exist_ok=True)
    texts = load_subset(suite, protocol)
    if args.smoke:
        texts = texts[:2]
    max_new_tokens = 16 if args.smoke else protocol["max_new_tokens"]
    variant, budget = job["variant"], job["budget"]
    oracle_kind = variant if variant in ORACLE_VARIANTS else None
    reference = {}
    if oracle_kind:
        ref_dir = suite / ("validation" if args.smoke else "runs") / job["reference"]
        if not (ref_dir / "complete.json").exists():
            raise RuntimeError(f"oracle reference {job['reference']} is not complete")
        reference = {r["source_id"]: r["tokens"] for r in map(
            json.loads, (ref_dir / "requests.jsonl").read_text().splitlines())}

    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from loop_batch_bench import arm_config, enforce_precision
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    from llm_memory_editability.loop_model_registry import MODELS, resolve_stop_ids, tokenize_prompt
    from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, SharedKVPool
    from llm_memory_editability.loop_sampling import TargetSampler, request_seed
    from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop

    spec = MODELS[job["model"]]
    if spec.backend not in ("ouro", "huginn"):
        raise ValueError(f"unsupported backend for this diagnostic: {spec.backend}")
    config = budget_config(spec, variant, budget, protocol, arm_config)
    T = spec.depth_total
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    model_path = protocol["models"][job["model"]]["path"]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    mc = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    gp = Path(model_path) / "generation_config.json"
    generation_eos = read(gp).get("eos_token_id") if gp.exists() else None
    stop, stop_audit = resolve_stop_ids(spec, tok, getattr(mc, "eos_token_id", None),
                                        generation_eos, mc.vocab_size, "chat")
    prompts = [tokenize_prompt(spec, tok, row["prompt"], "chat") for row in texts]
    required = max(map(len, prompts)) + max_new_tokens + T + 4
    max_seq = min(spec.context_limit, ((required + 255) // 256) * 256)
    if any(len(ids) + max_new_tokens + T + 4 > max_seq for ids in prompts):
        raise ValueError("inputs exceed context; no truncation or deletion")
    nodes = budget + 1
    sampling = {k: protocol[k] for k in ("temperature", "sampling_seed")}
    metadata = dict(job=job, config=asdict(config), sampling=sampling, max_new_tokens=max_new_tokens,
                    max_seq=max_seq, physical_nodes=nodes, stop_audit=stop_audit,
                    inputs_sha256={d: protocol["inputs"][d]["sha256"] for d in protocol["datasets"]},
                    source_sha256=protocol["source_sha256"], dtype="bfloat16", batch_size=1,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                    instrumented=True, formal_throughput=False,
                    history="cold per dataset; only completed outputs of this job are published")
    prior = target / "metadata.json"
    if prior.exists():
        old = read(prior)
        for key in ("config", "sampling", "max_new_tokens", "inputs_sha256", "source_sha256"):
            # Compare through JSON so tuples in the config equal their saved lists.
            if old[key] != json.loads(json.dumps(metadata[key])):
                raise ValueError(f"resume identity mismatch: {key}")
    save(prior, metadata)

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    model = enforce_precision(model, "bfloat16")
    if spec.backend == "ouro":
        model.model.total_ut_steps = T
        model.early_exit_step = model.early_exit_threshold = None
    pool = SharedKVPool(1, pad_decode_rows=False)
    if spec.backend == "ouro":
        b = OuroLoop(model, T, max_seq, nodes, kv_factory=pool)
    else:
        b = HuginnLoop(model, T, max_seq, nodes, 0, max_seq,
                       extra_depths=(spec.loopfill_readout_depths[1],), kv_factory=pool)
    b.position_limit = max_seq
    if b.kv.K.dtype != torch.bfloat16:
        raise ValueError("KV dtype is not BF16")
    engine = PackedLoopExecutor([b], cuda_graph=False)
    engine.graphs.freeze()

    journal = target / "requests.jsonl"
    completed = load_journal(journal, [r["source_id"] for r in texts])
    caches, index_in_dataset = {}, Counter()
    for row, prompt in zip(texts, prompts):
        row["_index"] = index_in_dataset[row["dataset"]]
        index_in_dataset[row["dataset"]] += 1
    # Rebuild only the history that completed requests already published.
    for done, row, prompt in zip(completed, texts, prompts):
        cache = caches.setdefault(row["dataset"], CompletionSuffixCache.arctic())
        cache.start_request(row["_index"], np.asarray(prompt, dtype=np.int32))
        cache.add_active_response(row["_index"], np.asarray(done["tokens"], dtype=np.int32))
        cache.stop_request(row["_index"])

    with journal.open("a") as f:
        for i in range(len(completed), len(texts)):
            row, prompt = texts[i], prompts[i]
            dataset, rid = row["dataset"], row["_index"]
            cache = caches.setdefault(dataset, CompletionSuffixCache.arctic())
            if sampling["temperature"]:
                seed = request_seed(sampling["sampling_seed"], job["model"], dataset, row["source_id"])
                b.target_sampler = TargetSampler(sampling["temperature"], seed, max_seq + 2, b.kv.K.device)
            else:
                b.target_sampler = None
            oracle = None
            if oracle_kind:
                oracle = OracleRanking(reference[row["source_id"]], only=oracle_kind == "oracle-only")

            def run_steps(observer):
                steps = budget_module.budget_tree_decode_steps(
                    b, prompt, config, max_new_tokens, stop, cache=cache, req_id=rid,
                    max_nodes=nodes, observer=observer)
                with torch.inference_mode():
                    return drive(b, engine, steps)

            tokens, stats, payload = traced_decode(run_steps, T, oracle)
            record = dict(source_id=row["source_id"], dataset=dataset, category=row.get("category"),
                          prompt_tokens=len(prompt), tokens=tokens, stats=stats, **payload,
                          same_as_reference=(tokens == reference[row["source_id"]]) if oracle_kind else None,
                          stop_reason="eos" if tokens[-1] in stop else "length",
                          completed_unix=time.time())
            f.write(json.dumps(record) + "\n")
            f.flush()
            completed.append(record)
            tokens_done = sum(r["stats"]["tokens"] for r in completed)
            calls_done = sum(r["stats"]["calls"] for r in completed)
            save(target / "progress.json", dict(completed_requests=len(completed), total_requests=len(texts),
                 gamma=T * tokens_done / calls_done if calls_done else None, updated_unix=time.time()))
            print(f"{args.job} {i + 1}/{len(texts)} tokens={len(tokens)} calls={stats['calls']}", flush=True)
    tokens_done = sum(r["stats"]["tokens"] for r in completed)
    calls_done = sum(r["stats"]["calls"] for r in completed)
    save(target / "summary.json", dict(requests=len(completed), decode_tokens=tokens_done,
         loop_steps=calls_done, gamma=T * tokens_done / calls_done if calls_done else None,
         instrumented=True, formal_throughput=False))
    save(target / "complete.json", dict(requests=len(completed), finished_unix=time.time(),
                                        requests_sha256=digest(journal)))


if __name__ == "__main__":
    main()
