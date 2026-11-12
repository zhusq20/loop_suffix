"""Wall-clock benchmark of loop speculative decoding with suffix-tree drafts and shallow readouts.

One process on one dedicated GPU. For each benchmark question (n per category, evenly
spaced by question_id, taken from an offline run of loop_suffix_spec_readouts.py with the
same prompt ids and stop rule) every decoder runs the same request:

  wfd-ar, wfd          WaveFront Decoding's AR baseline and scheduler (reference code)
  ar, shallow, suffix  loop_tree_decode: AR, shallow chain only, suffix tree only
  wavefront            loop_tree_decode: WFD's schedule in our engine (shallow chain, no
                       draft cap; Huginn drafts through the shared coda cache as in WFD)
  both, both-path      suffix tree + shallow; fills on every live node / on the predicted path

All arms share one copy of the weights. Arm order rotates per question and every arm gets
an untimed short warm-up on the question first. The suffix cache's global tree holds the
offline greedy outputs of all earlier questions (question_id order), as in the offline
replay; each run's own response is evicted afterwards. For loop_tree_decode arms the block
calls are compared with replay_request on the offline readouts.

Timing hygiene: refuses to start while another compute process is on the GPU; no sampling
threads in this process; GC collected before and disabled during each timed run; W&B is
written only after the last question. GPU clocks, temperature, power, throttle reasons,
compute processes and host load are recorded before and after each question; the host-side
scripts/loop_suffix_spec_env_monitor.py samples continuously and --summarize --monitor joins it.

In check mode (fp32) every arm must reproduce AR, and `wavefront` is compared with the
reference `wfd` run token by token and in block calls. `--summarize --merge-old <jsonl>`
pairs a wavefront-only run with the arms of an earlier run, question by question.

Protocols: results/loop-suffix-spec-wallclock-20261031/protocol.md,
results/loop-wfd-same-engine-20261031/protocol.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pwd
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_prob_tree_replay import MODELS as PROB_MODELS  # noqa: E402
from loop_prob_tree_replay import v3_configs  # noqa: E402
from loop_spec_sibling import spread_subset  # noqa: E402
from loop_suffix_spec_readouts import MODELS  # noqa: E402
from loop_suffix_spec_simulate import MAIN, load_requests  # noqa: E402

from llm_memory_editability.loop_prob_decode import prob_tree_decode  # noqa: E402
from llm_memory_editability.loop_prob_tree import Calibration, ProbReplay  # noqa: E402
from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request  # noqa: E402
from llm_memory_editability.loop_tree_decode import (  # noqa: E402
    DraftTree,
    HuginnLoop,
    LoopKV,
    OuroLoop,
    make_rows,
    tree_attention,
    tree_decode,
)

OURS = {  # name: (use_suffix, use_fill, fill_scope, uncapped draft, shared coda)
    "ar": (False, False, "all", False, False),
    "shallow": (False, True, "all", False, False),
    "wavefront": (False, True, "all", True, True),
    "suffix": (True, False, "all", False, False),
    "both": (True, True, "all", False, False),
    "both-path": (True, True, "path", False, False),
}
# probability-threshold decoder (loop_prob_decode): the replay setting each arm runs,
# frozen on the selection split (results/loop-spec-prob-tree-20261031/report.md)
PROB = {"prob-low": "v3 C t0.01", "prob-load": "v3 C t0.1"}
UNCAPPED = 1 << 30  # max_draft above any output length
WFD_ARMS = ("wfd-ar", "wfd")
THROTTLE_BAD = 0x4 | 0x8 | 0x20 | 0x40 | 0x80  # SW power cap, HW slowdown, thermal, power brake


class GpuEnv:
    """NVML view of the (only) visible GPU, plus host load."""

    def __init__(self):
        import pynvml

        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
        self.reasons = self.reasons or pynvml.nvmlDeviceGetCurrentClocksThrottleReasons

    def uuid(self) -> str:
        u = self.nv.nvmlDeviceGetUUID(self.h)
        return u.decode() if isinstance(u, bytes) else u

    def procs(self) -> list[list[int]]:
        out = self.nv.nvmlDeviceGetComputeRunningProcesses(self.h)
        return [[p.pid, (p.usedGpuMemory or 0) >> 20] for p in out]

    def snapshot(self) -> dict:
        nv, h = self.nv, self.h

        def safe(f, *a):
            try:
                return f(h, *a)
            except nv.NVMLError:
                return None

        util, mem = safe(nv.nvmlDeviceGetUtilizationRates), safe(nv.nvmlDeviceGetMemoryInfo)
        power = safe(nv.nvmlDeviceGetPowerUsage)
        return {
            "time": time.time(),
            "gpu_util": None if util is None else util.gpu,
            "mem_used_mib": None if mem is None else mem.used >> 20,
            "sm_mhz": safe(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM),
            "mem_mhz": safe(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_MEM),
            "temp_c": safe(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU),
            "power_w": None if power is None else power / 1000,
            "throttle_reasons": safe(self.reasons),
            "compute_procs": self.procs(),
            "loadavg": os.getloadavg(),
        }


def timed(fn):
    """Run fn between CUDA syncs with GC off; returns (result, wall_s, cpu_s, start, end)."""
    gc.collect()
    torch.cuda.synchronize()
    gc.disable()
    start = time.time()
    c0, t0 = time.process_time(), time.perf_counter()
    try:
        result = fn()
        torch.cuda.synchronize()
    finally:
        t1, c1 = time.perf_counter(), time.process_time()
        gc.enable()
    return result, t1 - t0, c1 - c0, start, time.time()


def truncate(tokens, stop: set[int], max_new: int) -> list[int]:
    out = []
    for t in tokens[:max_new]:
        out.append(int(t))
        if t in stop:
            break
    return out


def attention_self_check(kv0: LoopKV) -> float:
    """FlashAttention varlen + tree merge against the math path, on the benchmark GPU."""
    H, D = kv0.K.shape[2:]
    dev, dt = kv0.K.device, kv0.K.dtype
    kv = LoopKV(6, 512, 16, H, D, dt, dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    for t in (kv.K, kv.V, kv.TK, kv.TV):
        t.normal_(generator=gen)
    kv.length = 301
    tree = DraftTree(16, 16)
    tree.reset(kv.length)
    nodes = []
    for i, p in enumerate([-1, 0, 0, 1, 3, 2, 5]):
        nodes.append(tree.add(i + 1, None if p < 0 else nodes[p], 0, "suffix"))
    rows = make_rows(kv, tree, nodes, [0, 2, 2, 4, 4, 4, 0])
    q, k, v = torch.randn(3, len(nodes), H, D, device=dev, dtype=dt, generator=gen).unbind(0)
    a = tree_attention(q, k, v, kv, rows, 1, fast=True).float()
    b = tree_attention(q, k, v, kv, rows, 1, fast=False).float()
    return float((a - b).abs().max())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MAIN))
    p.add_argument("--runs-dir", required=True, help="offline setting dir: readouts/, index-*")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hf-root", default="/hf/hub")
    p.add_argument("--wfd-root", default="/wfd")
    p.add_argument("--suffix-lib", default="/arctic")
    p.add_argument("--prompt-format", choices=["raw", "chat"], required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--max-prompt-tokens", type=int, required=True, help="of the offline run")
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--question-ids", default=None, help="run only these (comma-separated)")
    p.add_argument("--arms", default="wfd-ar,wfd,ar,shallow,suffix,both,both-path")
    p.add_argument("--draft-depth", type=int, default=0, help="0: the model's main d")
    p.add_argument("--max-draft", type=int, default=16)
    p.add_argument("--spec-factor", type=float, default=1.0)
    p.add_argument("--prune-top", type=int, default=0)
    p.add_argument("--max-nodes", type=int, default=256)
    p.add_argument("--prob-calibration", default=None, help="loop_prob_tree calibration JSON")
    p.add_argument("--prob-max-nodes", type=int, default=8192)
    p.add_argument("--flush-every", type=int, default=16, help="wavefront: see loop_tree_decode")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup-tokens", type=int, default=8)
    p.add_argument("--mode", choices=["bench", "check"], default="bench")
    p.add_argument("--allow-shared-gpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--tracking-defaults", default="/repo/configs/experiment-tracking-defaults.json")
    p.add_argument("--summarize", action="store_true", help="only rebuild summary.json")
    p.add_argument("--monitor", default=None, help="host monitor JSONL for --summarize")
    p.add_argument("--own-user", default=None, help="host user name of this job (monitor)")
    p.add_argument("--batch", default="loop-suffix-spec-wallclock-20261031", help="W&B group")
    p.add_argument("--merge-old", default=None, help="earlier bench JSONL for --summarize")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    tag = f"{args.mode}-{'float32' if args.mode == 'check' else 'bfloat16'}"
    rec_path = out_dir / f"{tag}.jsonl"
    if args.summarize:
        summarize(rec_path, out_dir / f"{tag}-summary.json", args.monitor, args.own_user)
        if args.merge_old:
            summarize(
                rec_path, out_dir / f"{tag}-merged-summary.json", args.monitor, args.own_user,
                Path(args.merge_old),
            )  # fmt: skip
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    env = GpuEnv()
    foreign, busy = env.procs(), []
    for _ in range(6):
        busy.append(env.snapshot()["gpu_util"] or 0)
        time.sleep(0.5)
    if (foreign or max(busy) > 0) and not args.allow_shared_gpu:
        raise SystemExit(f"GPU not idle before start: procs={foreign} util={busy}")

    T = MAIN[args.model]["T"]
    d = args.draft_depth or MAIN[args.model]["d_main"]
    K = args.max_draft
    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in OURS and a not in WFD_ARMS and a not in PROB]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    cfgs = {
        a: ArmConfig(
            a,
            T,
            d,
            0 if a == "ar" else UNCAPPED if OURS[a][3] else K,
            OURS[a][0],
            OURS[a][1],
            prune_top=(args.prune_top or None) if OURS[a][1] and not OURS[a][3] else None,
            spec_factor=args.spec_factor,
        )  # fmt: skip
        for a in arms
        if a in OURS
    }

    reqs = load_requests(Path(args.runs_dir))
    by_id = {r["question_id"]: r for r in reqs}
    rows = spread_subset(
        [{"question_id": r["question_id"], "category": r["category"]} for r in reqs],
        1 if args.mode == "check" else args.per_category,
    )
    max_new = args.max_new_tokens
    # in-flight drafts reach at most max(K, T) positions past the last committed token
    pcfgs, calib, deep = {}, None, ()
    if any(a in PROB for a in arms):
        if args.prob_calibration is None:
            raise SystemExit("prob arms need --prob-calibration")
        ps = PROB_MODELS[args.model]
        if (ps["T"], ps["d"]) != (T, d):
            raise SystemExit("prob arms run the replay's T and d")
        named = {c.name: c for c in v3_configs(T, d, (d,), (d, ps["deep"]))}
        pcfgs = {a: named[PROB[a]] for a in arms if a in PROB}
        calib = Calibration.from_json(json.loads(Path(args.prob_calibration).read_text())["dev"])
        deep = (ps["deep"],)
    # prob drafts may run past the frontier by a suffix draft and its fills
    margin = 256 if pcfgs else max(K, T) + 2
    max_seq = max(len(by_id[r["question_id"]]["prompt"]) for r in rows) + max_new + margin
    # an uncapped chain keeps every token of a step in the tree: up to max_new + T / d + 1
    node_cap = {a: max_new + T + 8 if OURS[a][3] else args.max_nodes for a in cfgs}
    n_nodes = max([args.max_nodes, *node_cap.values()] + ([args.prob_max_nodes] if pcfgs else []))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[args.model]
    path = Path(args.hf_root) / spec["repo"] / "snapshots" / spec["revision"]
    dtype = torch.float32 if args.mode == "check" else torch.bfloat16
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=dtype)
    model = model.to("cuda").eval()
    our_pids = [pid for pid, _ in env.procs()]
    if args.model.startswith("ouro"):
        eos = tok.eos_token_id if tok.eos_token_id is not None else model.config.eos_token_id
        backbone = OuroLoop(model, T, max_seq, n_nodes)
    else:
        eos = getattr(model.config, "eos_token_id", None)
        init_len = args.max_prompt_tokens + max_new + 8  # as loop_suffix_spec_readouts.py
        backbone = HuginnLoop(model, T, max_seq, n_nodes, args.seed, init_len, extra_depths=deep)
    stop = {eos}
    wfd_stop = eos  # WFD stops at one token id: the end-of-turn token under chat
    if args.prompt_format == "chat":
        wfd_stop = tok.convert_tokens_to_ids(spec["end_turn"])
        stop.add(wfd_stop)
    lens = [len(r["output"]) for r in reqs]
    ended = all(int(r["output"][-1]) in stop for r in reqs if len(r["output"]) < max_new)
    if args.mode == "bench" and (max(lens) > max_new or not ended):
        raise SystemExit("offline outputs do not match --max-new-tokens / stop: wrong setting?")

    wfd = None
    if any(a in WFD_ARMS for a in arms):
        sys.path.insert(0, args.wfd_root)
        from src.decoding.huginn_blocks import HuginnBlocks
        from src.decoding.ouro_blocks import OuroBlocks
        from src.decoding.policy import SsdPolicy
        from src.decoding.scheduler import generate_wavefront_dynamic

        cls = OuroBlocks if args.model.startswith("ouro") else HuginnBlocks
        blocks = cls(model, T_total=T, max_seq_len=max_seq)
        if isinstance(backbone, HuginnLoop):
            # same per-position initial latent as the offline readouts and HuginnLoop
            table = backbone.init
            blocks._init_state_tokens = lambda pos: table[torch.as_tensor(pos, device=table.device)]
            blocks._init_state_prefill = lambda b, n: table[:n].unsqueeze(0).expand(b, -1, -1)
        wfd = (blocks, SsdPolicy(T_total=T, T_draft=d, mode="greedy"), generate_wavefront_dynamic)

    sys.path.insert(0, args.suffix_lib)
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)
    fed = 0

    def feed_below(qid: int) -> None:
        nonlocal fed
        while fed < len(reqs) and reqs[fed]["question_id"] < qid:
            r, rid = reqs[fed], ("offline", reqs[fed]["question_id"])
            cache.start_request(rid, r["prompt"].astype(np.int32))
            cache.add_active_response(rid, r["output"].astype(np.int32))
            cache.stop_request(rid)
            fed += 1

    def evict(rid) -> None:
        if rid in cache.cached_requests:
            cache.evict_cached_response(rid)

    def run(arm: str, prompt: list[int], n_new: int, rid) -> dict:
        if arm in WFD_ARMS:
            blocks, policy, generate = wfd
            ids = torch.tensor(prompt, device="cuda")
            if arm == "wfd-ar":
                fn = lambda: blocks.ar_generate(ids, n_new, wfd_stop, policy.make_sampler())  # noqa: E731
            else:
                fn = lambda: generate(blocks, ids, n_new, policy, eos_token_id=wfd_stop)  # noqa: E731
            (toks, trace), wall, cpu, t0, t1 = timed(fn)
            res = {"tokens": truncate(toks, stop, n_new), "raw_len": len(toks)}
            if arm == "wfd":
                res.update(calls=trace.n_r_calls, accepted=trace.n_drafts_accepted)
                res.update(rejected=trace.n_drafts_rejected)
        elif arm in PROB:
            (toks, st), wall, cpu, t0, t1 = timed(
                lambda: prob_tree_decode(
                    backbone, prompt, pcfgs[arm], calib, n_new, stop, cache, rid,
                    max_nodes=args.prob_max_nodes, flush_every=args.flush_every,
                )
            )  # fmt: skip
            evict(rid)
            res = st | {"tokens": toks, "new_tokens": st["tokens"]}
        else:
            cfg, scope, wf = cfgs[arm], OURS[arm][2], OURS[arm][3]
            (toks, st), wall, cpu, t0, t1 = timed(
                lambda: tree_decode(
                    backbone, prompt, cfg, n_new, stop, cache if cfg.use_suffix else None, rid,
                    max_nodes=node_cap[arm], fill_scope=scope, shared_coda=OURS[arm][4],
                    flush_every=args.flush_every if wf else None,
                )
            )  # fmt: skip
            evict(rid)
            res = st | {"tokens": toks, "new_tokens": st["tokens"]}
        timing = {"n_tokens": len(res["tokens"]), "wall_s": wall, "cpu_s": cpu}
        return res | timing | {"start": t0, "end": t1}

    attn_err = None
    if backbone.kv.fast:
        attn_err = attention_self_check(backbone.kv)
        if attn_err > 2e-2:
            raise SystemExit(f"flash/math tree attention mismatch: {attn_err}")
    meta = {
        "args": vars(args),
        "T": T,
        "d": d,
        "max_seq": max_seq,
        "stop": sorted(stop),
        "our_gpu_pids": our_pids,
        "uid": os.getuid(),
        "max_nodes": node_cap,
        "attention_self_check_max_abs_err": attn_err,
        "gpu_uuid": env.uuid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "env_start": env.snapshot(),
        "questions": [r["question_id"] for r in rows],
    }
    (out_dir / f"{tag}-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[wallclock] {args.model} {tag}: {len(rows)} questions, arms {arms}, "
          f"max_seq {max_seq}, attention check {attn_err}", flush=True)  # fmt: skip

    done = set()
    if rec_path.exists():
        done = {json.loads(x)["question_id"] for x in rec_path.read_text().splitlines() if x}
    # a shard keeps the full question list, so max_seq and the arm order stay the same
    if args.question_ids:
        done |= {r["question_id"] for r in rows} - {int(x) for x in args.question_ids.split(",")}
    mismatches = 0
    with rec_path.open("a") as fh:
        for qi, row in enumerate(rows):
            qid = row["question_id"]
            if qid in done:
                continue
            req = by_id[qid]
            prompt = [int(t) for t in req["prompt"]]
            offline = truncate(req["output"].tolist(), stop, max_new)  # check mode may cut
            feed_below(qid)
            replay = {}
            for a, cfg in cfgs.items():
                st = replay_request(
                    qid, prompt, offline, req["readouts"], cfg, cache if cfg.use_suffix else None
                )
                evict(qid)
                replay[a] = st["calls"]
            if pcfgs:
                z = np.load(Path(args.runs_dir) / "readouts" / f"{qid}.npz")
                at = {int(t): i for i, t in enumerate(z["depths"])}
                pids = {r: z["ids"][at[r]] for r in at if r < T}
                pprobs = {r: z["probs"][at[r]].astype(np.float32) for r in at if r < T}
                for a, pc in pcfgs.items():
                    st = ProbReplay(qid, prompt, offline, pids, pprobs, pc, cache, calib).run()
                    evict(qid)
                    replay[a] = st["calls"]
            order = arms[qi % len(arms) :] + arms[: qi % len(arms)]
            env_before = env.snapshot()
            for a in order:
                run(a, prompt, args.warmup_tokens, ("warm", qid))
            wfd_prefill = None
            if wfd is not None:
                ids = torch.tensor([prompt], device="cuda")
                _, wfd_prefill, *_ = timed(lambda ids=ids: wfd[0].prefill(ids, T, d))
            results = defaultdict(list)
            for rep in range(args.repeats):
                for a in order:
                    res = run(a, prompt, max_new, qid)
                    res["same_as_offline"] = res["tokens"] == offline
                    if rep:
                        res.pop("tokens")
                    if a in replay:
                        res["replay_calls"] = replay[a]
                    results[a].append(res)
            ref = results["ar"][0]["tokens"] if "ar" in results else offline
            for runs in results.values():
                runs[0]["same_as_ar"] = runs[0]["tokens"] == ref
                if args.mode == "check" and not runs[0]["same_as_ar"]:
                    mismatches += 1
            vs_wfd = None
            if "wavefront" in results and "wfd" in results:
                ours, theirs = results["wavefront"][0], results["wfd"][0]
                vs_wfd = {
                    "same_tokens": ours["tokens"] == theirs["tokens"],
                    "calls": [ours["calls"], theirs["calls"]],
                }
            rec = {
                "question_id": qid,
                "category": row["category"],
                "prompt_len": len(prompt),
                "offline_len": len(offline),
                "order": order,
                "wfd_prefill_s": wfd_prefill,
                "wavefront_vs_wfd": vs_wfd,
                "env_before": env_before,
                "env_after": env.snapshot(),
                "runs": results,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            brief = {
                a: f"{(len(r[0]['tokens']) - 1) / r[0]['wall_s']:.1f}t/s"
                + ("" if r[0]["same_as_ar"] else "*")
                for a, r in results.items()
            }
            vs = "" if vs_wfd is None else f" wavefront-vs-wfd {vs_wfd}"
            print(
                f"[wallclock] {qi + 1}/{len(rows)} q{qid} {row['category']} {brief}{vs}", flush=True
            )
    summary = summarize(rec_path, out_dir / f"{tag}-summary.json", None)
    log_wandb(args, meta, summary, rec_path)
    if args.mode == "check" and mismatches:
        raise SystemExit(f"[check] {mismatches} arm outputs differ from AR")


def _own_users(meta: dict, own_user: str | None) -> set[str]:
    """Names the host monitor may show for this job's uid. Inside the container the
    benchmark is PID 1 and the uid usually has no passwd entry, so the host-side PID is
    unknown here: the monitor's processes are matched by user instead."""
    uid = meta.get("uid", os.getuid())
    names = {str(uid)}
    if own_user:
        names.add(own_user)
    try:
        names.add(pwd.getpwuid(uid).pw_name)
    except KeyError:
        pass
    return names


def _merge_monitor(recs: list[dict], monitor_path: str | None, meta: dict, own: set[str]) -> None:
    """Per question, from the host monitor: compute processes of other users on our GPU,
    any second process of our own user there at the same time, the busiest watched CPU,
    and the GPU's SM clock range during the timed runs."""
    if not monitor_path:
        return
    samples = [json.loads(x) for x in Path(monitor_path).read_text().splitlines() if x]
    uuid = meta.get("gpu_uuid")
    for rec in recs:
        t0 = min(r["start"] for runs in rec["runs"].values() for r in runs)
        t1 = max(r["end"] for runs in rec["runs"].values() for r in runs)
        window = [s for s in samples if t0 <= s["time"] <= t1]
        gpus = [g for s in window for g in s["gpus"] if g["uuid"] == uuid]
        on_gpu = [[p for p in s["procs"] if p["gpu_uuid"] == uuid] for s in window]
        rec["monitor"] = {
            "samples": len(window),
            "foreign_pids": sorted({p["pid"] for ps in on_gpu for p in ps if p["user"] not in own}),
            "own_pids": sorted({p["pid"] for ps in on_gpu for p in ps if p["user"] in own}),
            "own_concurrent_max": max(
                (sum(p["user"] in own for p in ps) for ps in on_gpu), default=0
            ),
            "cpu_busy_max": max(
                (max(s["cpu_busy"].values(), default=0) for s in window), default=None
            ),
            "sm_mhz_range": [min(g["sm_mhz"] for g in gpus), max(g["sm_mhz"] for g in gpus)]
            if gpus
            else None,
        }


def _merge_old(recs: list[dict], old_path: Path) -> None:
    """Add the arms of an earlier run of the same questions (timed in another session).
    Called after the monitor merge, so flags and monitor windows describe this run only;
    same_as_ar is recomputed against the earlier AR output."""
    old = {}
    for line in old_path.read_text().splitlines():
        if line:
            rec = json.loads(line)
            old[rec["question_id"]] = rec
    for rec in recs:
        o = old[rec["question_id"]]
        for arm, runs in o["runs"].items():
            rec["runs"].setdefault(arm, runs)
        if rec.get("wfd_prefill_s") is None:
            rec["wfd_prefill_s"] = o.get("wfd_prefill_s")
        ref = rec["runs"]["ar"][0]["tokens"] if "ar" in rec["runs"] else None
        for runs in rec["runs"].values():
            if ref is not None and "tokens" in runs[0]:
                runs[0]["same_as_ar"] = runs[0]["tokens"] == ref


def summarize(
    rec_path: Path,
    out_path: Path,
    monitor_path: str | None,
    own_user: str | None = None,
    merge_old: Path | None = None,
) -> dict:
    recs = [json.loads(x) for x in rec_path.read_text().splitlines() if x]
    meta_path = rec_path.with_name(rec_path.stem + "-meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    _merge_monitor(recs, monitor_path, meta, _own_users(meta, own_user))
    if merge_old is not None:
        _merge_old(recs, merge_old)
    ours = set(meta.get("our_gpu_pids", []))

    def agg(subset: list[dict]) -> dict:
        out = {}
        for arm in subset[0]["runs"] if subset else []:
            s = defaultdict(float)
            for rec in subset:
                for r in rec["runs"][arm]:
                    s["tokens"] += r["n_tokens"]
                    s["new"] += r["n_tokens"] - 1
                    s["wall"] += r["wall_s"]
                    s["cpu"] += r["cpu_s"]
                    pre = r.get("prefill_s", rec.get("wfd_prefill_s") or 0.0)
                    s["decode"] += r.get("decode_s", r["wall_s"] - pre)
                    s["calls"] += r.get("calls", 0)
                    s["rows"] += r.get("rows", 0)
                    s["readout_rows"] += r.get("readout_rows", 0)
                    s["replay_calls"] += r.get("replay_calls", 0)
                    s["speculate"] += r.get("speculate_s", 0.0)
                    s["runs"] += 1
                first = rec["runs"][arm][0]
                s["same_as_offline"] += first["same_as_offline"]
                s["same_as_ar"] += first["same_as_ar"]
                s["calls_eq_replay"] += first.get("calls") == first.get("replay_calls")
            out[arm] = {
                "questions": len(subset),
                "runs": int(s["runs"]),
                "e2e_tok_per_s": s["tokens"] / s["wall"],
                "decode_tok_per_s": s["new"] / s["decode"],
                "cpu_over_wall": s["cpu"] / s["wall"],
                "calls_per_token": s["calls"] / s["new"] if s["calls"] else None,
                "ms_per_call": 1000 * s["decode"] / s["calls"] if s["calls"] else None,
                "rows_per_call": s["rows"] / s["calls"] if s["rows"] else None,
                "rows_per_token": s["rows"] / s["new"] if s["rows"] else None,
                "readout_rows_per_call": s["readout_rows"] / s["calls"] if s["calls"] else None,
                "replay_calls_per_token": s["replay_calls"] / s["new"]
                if s["replay_calls"]
                else None,
                "speculate_share_of_decode": s["speculate"] / s["decode"],
                "same_as_offline": s["same_as_offline"] / len(subset),
                "same_as_ar": s["same_as_ar"] / len(subset),
                "calls_eq_replay": s["calls_eq_replay"] / len(subset),
            }
        for base in ("ar", "wfd-ar", "wfd", "wavefront", "shallow", "both-path"):
            if base in out:
                for v in out.values():
                    v[f"decode_speedup_vs_{base}"] = (
                        v["decode_tok_per_s"] / out[base]["decode_tok_per_s"]
                    )
                    v[f"e2e_speedup_vs_{base}"] = v["e2e_tok_per_s"] / out[base]["e2e_tok_per_s"]
        return out

    def flags(rec: dict) -> list[str]:
        f = []
        for snap in (rec["env_before"], rec["env_after"]):
            pids = {p for p, _ in snap["compute_procs"]} - ours
            if pids:
                f.append(f"foreign_gpu_procs:{sorted(pids)}")
            if (snap["throttle_reasons"] or 0) & THROTTLE_BAD:
                f.append(f"throttle:{snap['throttle_reasons']:#x}")
        mon = rec.get("monitor", {})
        if mon.get("foreign_pids"):
            f.append(f"monitor_foreign:{mon['foreign_pids']}")
        if mon.get("own_concurrent_max", 0) > 1:
            f.append(f"monitor_own_concurrent:{mon['own_pids']}")
        return sorted(set(f))

    by_cat = defaultdict(list)
    for rec in recs:
        by_cat[rec["category"]].append(rec)
    clean = [r for r in recs if not flags(r)]
    vs = [r["wavefront_vs_wfd"] for r in recs if r.get("wavefront_vs_wfd")]
    summary = {
        "questions": len(recs),
        "wavefront_vs_wfd": {
            "questions": len(vs),
            "same_tokens": sum(v["same_tokens"] for v in vs),
            "same_calls": sum(v["calls"][0] == v["calls"][1] for v in vs),
            "same_both": sum(v["same_tokens"] and v["calls"][0] == v["calls"][1] for v in vs),
            "calls_ours_total": sum(v["calls"][0] for v in vs),
            "calls_wfd_total": sum(v["calls"][1] for v in vs),
        }
        if vs
        else None,
        "overall": agg(recs),
        "clean_only": agg(clean) if clean else {},
        "by_category": {c: agg(v) for c, v in sorted(by_cat.items())},
        "flagged": {r["question_id"]: flags(r) for r in recs if flags(r)},
        "monitor": {r["question_id"]: r["monitor"] for r in recs if "monitor" in r},
        "env": {
            "sm_mhz_min": min(
                r[k]["sm_mhz"] or 0 for r in recs for k in ("env_before", "env_after")
            ),
            "temp_c_max": max(
                r[k]["temp_c"] or 0 for r in recs for k in ("env_before", "env_after")
            ),
            "loadavg1_max": max(
                r[k]["loadavg"][0] for r in recs for k in ("env_before", "env_after")
            ),
        }
        if recs
        else {},
    }
    out_path.write_text(json.dumps(summary, indent=1))
    if summary["wavefront_vs_wfd"]:
        print(f"wavefront vs wfd: {summary['wavefront_vs_wfd']}")
    print(f"clean questions {len(clean)}/{len(recs)}")
    for arm, v in summary["overall"].items():
        print(
            f"{arm:10s} decode {v['decode_tok_per_s']:7.1f} tok/s"
            f"  x{v.get('decode_speedup_vs_ar', float('nan')):.2f} vs ar"
            f"  x{v.get('decode_speedup_vs_wavefront', float('nan')):.2f} vs wavefront"
            f"  x{v.get('decode_speedup_vs_wfd-ar', float('nan')):.2f} vs wfd-ar"
            f"  calls/tok {v['calls_per_token'] or float('nan'):.2f}"
            f" (replay {v['replay_calls_per_token'] or float('nan'):.2f})"
            f"  ms/call {v['ms_per_call'] or float('nan'):.2f}"
            f"  same-as-ar {v['same_as_ar']:.2f}"
        )
    return summary


def log_wandb(args, meta: dict, summary: dict, rec_path: Path) -> None:
    """Offline W&B run written after timing, so no logger runs during the timed loop."""
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    defaults = json.loads(Path(args.tracking_defaults).read_text())
    run = wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        name=f"{args.batch}-{args.model}-{Path(args.runs_dir).parent.name}-{rec_path.stem}",
        group=args.batch,
        job_type="inference-benchmark",
        config=meta | {"training_updates": 0},
        dir=str(rec_path.parent),
    )
    for i, line in enumerate(rec_path.read_text().splitlines()):
        rec = json.loads(line)
        log = {"question": i, "question_id": rec["question_id"]}
        for arm, runs in rec["runs"].items():
            r = runs[0]
            log[f"{arm}/decode_tok_per_s"] = (len(r["tokens"]) - 1) / max(
                1e-9, r.get("decode_s", r["wall_s"] - (rec["wfd_prefill_s"] or 0.0))
            )
            log[f"{arm}/same_as_ar"] = float(r["same_as_ar"])
        run.log(log)
    run.summary.update({f"{a}/{k}": v for a, s in summary["overall"].items() for k, v in s.items()})
    run.finish()


if __name__ == "__main__":
    main()
