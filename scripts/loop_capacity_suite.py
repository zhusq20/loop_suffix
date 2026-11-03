"""Capacity-only preflight, then a full six-arm run at each model's common batch.

The backend and main benchmark come from an existing frozen suite. Capacity tests
produce no throughput measurements. Synthetic late-context/tree states exercise
allocation shapes; they are never treated as correctness or speed results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_suite(suite):
    protocol = json.loads((suite / "protocol.json").read_text())
    source = Path(protocol["source"])
    for name, expected in json.loads((suite / "source-hashes.json").read_text()).items():
        if digest(source / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, expected in json.loads((suite / "input-hashes.json").read_text()).items():
        if digest(name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    return protocol


def guarded_batch_limit(slot_bytes, free_after_first, fraction=0.85):
    """Same allocation inequality as loop_batch_bench.allocate, including B=1."""
    if slot_bytes <= 0 or free_after_first < 0 or not 0 < fraction <= 1:
        raise ValueError("invalid memory accounting")
    return 1 + math.floor(free_after_first * fraction / slot_bytes)


def slot_bytes(backbone):
    kv = backbone.kv
    tensors = [kv.K, kv.V, kv.TK, kv.TV, backbone.hidden]
    tensors += [getattr(backbone, n) for n in ("static", "init") if hasattr(backbone, n)]
    return sum(t.numel() * t.element_size() for t in tensors)


def physical_nodes(arm, config, protocol):
    if arm == "ar":
        return 1
    if arm.startswith("budget-"):
        return protocol["draft_node_budget"] + 1
    if arm == "loopspec-dual":
        return config.physical_nodes
    return protocol["physical_nodes"][arm]


def stress_tree_shapes(backbones, config, prefix_length, nodes):
    """Exercise full live-row bounds at a late context using synthetic KV values.

WFD's physical history buffer is not its live width: at most T/d rows advance,
with up to 16 unflushed ancestors. Budget arms use N+1 live rows. LoopSpec uses
its registered live bound. Readout groups conservatively include every live row.
"""
    from llm_memory_editability.loop_batch_decode import FusedLoopExecutor
    from llm_memory_editability.loop_decode_steps import ModelCall
    from llm_memory_editability.loop_tree_decode import DraftTree

    arm, T, d = config.name, config.depth_total, config.draft_depth
    if arm == "ar":
        live_count, retired = 1, 0
    elif arm == "wavefront":
        live_count, retired = T // d, 16
    elif arm == "loopspec-dual":
        live_count, retired = config.live_bound, 0
    else:
        live_count, retired = config.draft_node_budget + 1, 0
    count = live_count + retired
    if count > nodes:
        raise ValueError("stress exceeds physical tree capacity")
    jobs, trees, lives = [], [], []
    for b in backbones:
        if prefix_length + count > b.kv.max_seq:
            raise ValueError("stress positions exceed the main allocation")
        # Buffers are already fully allocated. Zero the simulated committed tail;
        # its values do not change the attention allocation shape.
        b.kv.K[:, b.kv.length:prefix_length].zero_()
        b.kv.V[:, b.kv.length:prefix_length].zero_()
        b.kv.length = prefix_length
        tree = DraftTree(nodes, 1 << 30)
        tree.reset(prefix_length)
        parent = None
        for _ in range(count):
            parent = tree.add(1, parent, 0, "capacity_synthetic")
        live = tree.nodes[retired:]
        trees.append(tree)
        lives.append(live)
        jobs.append((b, ModelCall("embed", (tree, tree.take_fresh()))))
    engine = FusedLoopExecutor(backbones)
    engine.execute("embed", jobs)
    for live in lives:
        for i, n in enumerate(live):
            n.step = (i * d) % T if arm == "wavefront" else i % T
    engine.execute("advance", [
        (b, ModelCall("advance", (t, live)))
        for b, t, live in zip(backbones, trees, lives)
    ])
    if arm.startswith("budget-") or arm == "loopspec-dual":
        deep = config.second_depth if arm == "loopspec-dual" else config.readout_depths[-1]
        jobs = []
        for b, t, live in zip(backbones, trees, lives):
            split = max(1, len(live) // 2)
            groups = [(d, live[:split]), (deep, live[split:-1])]
            jobs.append((b, ModelCall("readout_top", (t, groups, live[-1:],
                                                       0 if arm == "loopspec-dual" else 8))))
        engine.execute("readout_top", jobs)
    else:
        engine.execute("readout", [
            (b, ModelCall("readout", (t, live[:-1], live[-1:], 1, False)))
            for b, t, live in zip(backbones, trees, lives)
        ])
    backbones[0].synchronize()
    return dict(live_rows_per_request=live_count, retained_tree_nodes=count,
                committed_prefix_length=prefix_length, synthetic=True)


def probe(args):
    suite, out = Path(args.suite), Path(args.out)
    protocol = verify_suite(suite)
    source = Path(protocol["source"])
    sys.path[:0] = [str(source / "scripts"), str(source / "src")]
    import gc
    import torch
    import loop_batch_bench as bench
    from transformers import AutoModelForCausalLM

    out.mkdir(parents=True, exist_ok=False)
    state = dict(phase="loading", model=args.model, started_unix=time.time(),
                 guard_fraction=0.85, estimates={}, attempts=[], speed_measurement=False)

    def update(**fields):
        state.update(fields, updated_unix=time.time())
        save(out / "status.json", state)
        print(json.dumps(fields, ensure_ascii=False), flush=True)

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    revision = protocol["model_revisions"][args.model]
    model_path = Path("/root/.cache/huggingface/hub") / ("models--" + revision[0]) / "snapshots" / revision[1]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    model = bench.enforce_precision(model, "bfloat16")
    T, d, deep, _ = bench.SPECS[args.model]
    if args.model.startswith("ouro"):
        model.model.total_ut_steps = T
    audit = json.loads((suite / "input-audit.json").read_text())[args.model]["specbench480"]
    max_seq = audit["allocated_positions"]
    input_path = Path(protocol["inputs"][f"specbench480/{args.model}"])
    rows = [json.loads(s) for s in input_path.read_text().splitlines()]
    longest = max(rows, key=lambda r: len(r["prompt_token_ids"]))
    prompt = longest["prompt_token_ids"]
    configs = {arm: bench.arm_config(
        arm, T, d, deep, 64, draft_node_budget=protocol["draft_node_budget"],
        suffix_eta=protocol["controller"]["eta"],
        suffix_c_min=protocol["controller"]["c_min"],
        suffix_c_max=protocol["controller"]["c_max"],
    ) for arm in protocol["arms"]}
    update(phase="estimating", gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
           actual_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
           max_seq=max_seq, max_output_tokens=protocol["main"]["max_new_tokens"],
           longest_prompt_id=longest["question_id"], longest_prompt_tokens=len(prompt),
           input_sha256=digest(input_path), source_hashes_sha256=digest(suite / "source-hashes.json"))

    def backbone(nodes):
        if args.model.startswith("ouro"):
            return bench.OuroLoop(model, T, max_seq, nodes)
        return bench.HuginnLoop(model, T, max_seq, nodes, 0, max_seq, extra_depths=(deep,))

    def clean():
        gc.collect()
        torch.cuda.empty_cache()

    with torch.inference_mode():
        for arm, cfg in configs.items():
            clean()
            b = backbone(physical_nodes(arm, cfg, protocol))
            if b.kv.K.dtype != torch.bfloat16:
                raise TypeError("KV precision mismatch")
            size = slot_bytes(b)
            free, total = torch.cuda.mem_get_info()
            state["estimates"][arm] = dict(slot_bytes=size, free_after_first=free,
                total_memory=total, max_batch_under_guard=guarded_batch_limit(size, free),
                physical_nodes=physical_nodes(arm, cfg, protocol))
            del b
            update(estimates=state["estimates"])
        clean()
        bound = min(x["max_batch_under_guard"] for x in state["estimates"].values())
        candidate = min(bound, args.max_batch)
        # Most restrictive static allocation first. Other arms must still pass
        # their own real short decode and saturated-tree shape checks.
        ordered = sorted(configs, key=lambda a: state["estimates"][a]["max_batch_under_guard"])

        def attempt(arm, batch):
            cfg = configs[arm]
            nodes = physical_nodes(arm, cfg, protocol)
            record = dict(arm=arm, batch=batch, started_unix=time.time())
            state["attempts"].append(record)
            update(phase="checking", current_arm=arm, current_batch=batch,
                   attempts=state["attempts"])
            torch.cuda.reset_peak_memory_stats()
            slots = []
            try:
                first = backbone(nodes)
                slots.append(first)
                free, _ = torch.cuda.mem_get_info()
                if slot_bytes(first) * (batch - 1) > free * 0.85:
                    record.update(outcome="capacity_guard", free_after_first=free)
                    return False
                slots.extend(backbone(nodes) for _ in range(batch - 1))
                requests = [bench.BatchRequest(i, list(prompt), 8, set()) for i in range(batch)]
                cache = bench.CompletionSuffixCache.arctic() if cfg.use_suffix else None
                outputs, summary = bench.batch_decode(requests, slots, cfg, cache=cache, max_nodes=nodes)
                if summary["max_active_requests"] != batch or len(outputs) != batch:
                    raise RuntimeError("capacity trial did not fill the declared batch")
                late_prefix = len(prompt) + protocol["main"]["max_new_tokens"] - 1
                shapes = stress_tree_shapes(slots, cfg, late_prefix, nodes)
                record.update(outcome="passed", actual_batch=batch,
                              short_decode_tokens_per_request=8, stress=shapes)
                return True
            except torch.OutOfMemoryError as exc:
                record.update(outcome="cuda_oom", error=str(exc))
                return False
            finally:
                record.update(finished_unix=time.time(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
                update(attempts=state["attempts"])

        while candidate > 0:
            passed = True
            for arm in ordered:
                clean()
                if not attempt(arm, candidate):
                    passed = False
                    break
            clean()
            if passed:
                break
            candidate -= 1
        if candidate < 1:
            raise RuntimeError("no common batch passed capacity checks")
        if candidate == bound and bound < args.max_batch:
            # Confirm the rejection using the real guard, not just arithmetic.
            if attempt(ordered[0], candidate + 1):
                raise RuntimeError("capacity estimate changed; re-evaluate before selection")
            clean()
        selection = dict(model=args.model, selected_batch=candidate,
            all_arms_passed=True, maximum_search_batch=args.max_batch,
            search_capped=candidate == args.max_batch,
            allocation_guard_fraction=0.85, max_seq=max_seq,
            main_output_budget=protocol["main"]["max_new_tokens"],
            interpretation="Largest common batch passing the unchanged main allocation guard and capacity stresses. Not an absolute hardware OOM limit or a throughput result; full main run remains the final validation.",
            estimates=state["estimates"], attempts=state["attempts"])
        save(out / "selection.json", selection)
        update(phase="complete", selected_batch=candidate)


def run(args):
    suite, out = Path(args.suite).resolve(), Path(args.out).resolve()
    if (suite / "probe-cancelled.json").exists():
        raise RuntimeError("Capacity probing was cancelled; use loop_batch_sweep.py for all requested batches")
    protocol = verify_suite(suite)
    out.mkdir(parents=True, exist_ok=True)
    guard = (out / "supervisor.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "protocol.json").exists():
        raise FileExistsError("use a fresh output directory; interrupted artifacts are retained")
    script = out / "source" / Path(__file__).name
    script.parent.mkdir()
    shutil.copy2(__file__, script)
    save(out / "protocol.json", dict(parent_suite=str(suite), selection="largest common batch across six arms per model",
        max_batch=args.max_batch, guard_fraction=0.85, source=str(script), source_sha256=digest(script),
        parent_source_hashes_sha256=digest(suite / "source-hashes.json"),
        capacity="fully allocate main buffers, longest prompt short decode, synthetic late-context/live-tree stress; no throughput sweep",
        main="SpecBench480, 1024 output tokens, six arms; reuse parent's exact serial AR references",
        lanes=protocol["probe_lanes"], created_unix=time.time()))
    state = dict(phase="running", supervisor_pid=os.getpid(), started_unix=time.time(), lanes={})
    mutex = threading.Lock()

    def update(gpu=None, **fields):
        with mutex:
            (state if gpu is None else state["lanes"].setdefault(str(gpu), {})).update(fields)
            state["updated_unix"] = time.time()
            save(out / "status.json", state)

    def worker(lane):
        model, gpu = lane["model"], lane["gpu"]
        with open(f"/tmp/loopfill-suite-gpu{gpu}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            occupied = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            if occupied:
                raise RuntimeError(f"GPU {gpu} occupied: {occupied}")
            venv = protocol["model_revisions"][model][2]
            python = f"/root/.cache/loopfill/{venv}/bin/python"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
                       PYTHONPATH=str(Path(protocol["source"]) / "src"))
            probe_out = out / "checks" / model
            cmd = [python, str(script), "probe", "--suite", str(suite), "--out", str(probe_out),
                   "--model", model, "--max-batch", str(args.max_batch)]
            with (out / f"capacity-{model}.log").open("x") as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(gpu, phase="capacity", model=model, pid=child.pid, command=cmd)
                rc = child.wait()
            if rc:
                raise RuntimeError(f"capacity check failed for {model}; see capacity-{model}.log")
            selection = json.loads((probe_out / "selection.json").read_text())
            B = selection["selected_batch"]
            if B in protocol["main"]["batch_sizes"]:
                update(gpu, phase="existing_main_covers_selected_batch", pid=None, selected_batch=B)
                return
            verify_suite(suite)
            source = Path(protocol["source"])
            parent_job = json.loads((suite / "jobs" / f"main-{model}-b1-six-arms.json").read_text())
            cmd = parent_job["command"].copy()
            main_out = out / "runs" / f"main-{model}-b{B}-six-arms"
            cmd[cmd.index("--batch-sizes") + 1] = str(B)
            cmd[cmd.index("--out") + 1] = str(main_out)
            jobpath = out / "jobs" / f"main-{model}-b{B}-six-arms.json"
            job = dict(model=model, gpu=gpu, batch=B, phase="main", arms=protocol["arms"],
                       output=str(main_out), command=cmd, started_unix=time.time())
            with (out / f"main-{model}-b{B}.log").open("x") as log:
                child = subprocess.Popen(cmd, cwd=source, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                job["pid"] = child.pid
                save(jobpath, job)
                update(gpu, phase="main", selected_batch=B, pid=child.pid, output=str(main_out), command=cmd,
                       note="Reference setup/wait precedes all timed streams")
                rc = child.wait()
            job.update(returncode=rc, finished_unix=time.time(), outcome="complete" if rc == 0 else "failed")
            save(jobpath, job)
            if rc:
                raise RuntimeError(f"main run failed at selected B={B}: {model}; no silent batch reduction")
            sys.path.insert(0, str(source / "scripts"))
            from loop_online_suite import audit_run
            expected = {r["question_id"] for r in map(json.loads, Path(protocol["inputs"][f"specbench480/{model}"]).read_text().splitlines())}
            audit_run(main_out, expected, True)
            update(gpu, phase="complete", pid=None)

    update()
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(worker, lane): lane for lane in protocol["probe_lanes"]}
        for future in concurrent.futures.as_completed(futures):
            lane = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
                update(lane["gpu"], phase="failed", pid=None, error=str(exc))
    update(phase="failed" if errors else "complete", errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "probe"))
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=("ouro-2.6b", "huginn-3.5b"))
    parser.add_argument("--max-batch", type=int, default=64)
    args = parser.parse_args()
    if args.max_batch < 1 or (args.mode == "probe" and not args.model):
        parser.error("positive max-batch and a model for probe are required")
    (run if args.mode == "run" else probe)(args)
