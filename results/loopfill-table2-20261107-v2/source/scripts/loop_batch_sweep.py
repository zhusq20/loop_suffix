"""Full SpecBench480 matrix with continuous batching and fixed GPU resources."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import concurrent.futures
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from loop_capacity_suite import digest, save, verify_suite

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
LANES = [
    dict(model="ouro-2.6b", gpu=0, cpus="0-3", batches=[1, 16, 256], shard=0),
    dict(model="ouro-2.6b", gpu=1, cpus="4-7", batches=[4, 64], shard=1),
    dict(model="ouro-2.6b", gpu=4, cpus="48-51", batches=[2, 32], shard=2),
    dict(model="ouro-2.6b", gpu=6, cpus="56-59", batches=[8, 128], shard=3),
    dict(model="huginn-3.5b", gpu=2, cpus="8-11", batches=[1, 16, 256], shard=0),
    dict(model="huginn-3.5b", gpu=3, cpus="12-15", batches=[4, 64], shard=1),
    dict(model="huginn-3.5b", gpu=5, cpus="52-55", batches=[2, 32], shard=2),
    dict(model="huginn-3.5b", gpu=7, cpus="60-63", batches=[8, 128], shard=3),
]
GPU_FIELDS = "index,uuid,name,compute_mode,clocks.sm,clocks.mem,clocks.applications.graphics,clocks.applications.memory,power.limit,temperature.gpu,memory.used,utilization.gpu"


def gpu_snapshot(gpu):
    output = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"], text=True)
    return dict(zip(GPU_FIELDS.split(","), next(csv.reader([output.strip()], skipinitialspace=True))))


def check_resources(lane):
    snap = gpu_snapshot(lane["gpu"])
    expected = {"uuid": lane["uuid"], "compute_mode": "Exclusive_Process",
                "clocks.applications.graphics": "1980", "clocks.applications.memory": "3201"}
    if any(snap[k] != v for k, v in expected.items()) or float(snap["power.limit"]) != 700:
        raise RuntimeError(f"fixed GPU resource settings changed: {snap}")
    occupied = subprocess.check_output(["nvidia-smi", f"--id={lane['gpu']}",
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if occupied:
        raise RuntimeError(f"GPU {lane['gpu']} occupied: {occupied}")
    return snap


def cells(model, arms, batches=BATCHES):
    # Deliberately no per-arm upper bound, search, break, or OOM-based pruning.
    return [dict(key=f"{model}-b{batch}-{arm}", model=model, batch=batch, arm=arm)
            for batch in batches for arm in arms]


def classify(returncode, out):
    failure = out / "failure.json"
    if returncode != 0 and failure.exists():
        data = json.loads(failure.read_text())
        if data.get("outcome") == "oom" and data.get("exception_type") == "OutOfMemoryError":
            return "oom"
    return "complete" if returncode == 0 else "failed"


def run_cells(planned, execute):
    results = []
    for cell in planned:
        # Cell failures (including OOM) never remove any other cell from the queue.
        try:
            results.append(execute(cell))
        except Exception as exc:
            results.append(dict(**cell, outcome="failed", error=repr(exc)))
    return results


def prepare(suite, out, schedule="continuous", optimized=False):
    if optimized and schedule != "continuous":
        raise ValueError("the optimized full480 protocol uses continuous scheduling")
    parent = verify_suite(suite)
    if out.exists():
        raise FileExistsError(f"use a fresh output directory: {out}")
    out.mkdir(parents=True)
    source = out / "source"
    shutil.copytree(parent["source"], source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    repo = Path(__file__).resolve().parents[1]
    # Execution backend changes are isolated from the serial reference decoder.
    for name in ("loop_batch_bench.py", "loop_batch_sweep.py", "loop_capacity_suite.py"):
        shutil.copy2(repo / "scripts" / name, source / "scripts" / name)
    changed_reference_sources = {"scripts/loop_batch_bench.py"}
    for name in ("loop_batch_decode.py", "loop_optimized_decode.py", "loop_tree_decode.py"):
        relative = "src/llm_memory_editability/" + name
        changed_reference_sources.add(relative)
        shutil.copy2(repo / relative, source / relative)
    # Only optional storage/factory constructor parameters may change in the
    # serial backbone. Its prefill, attention, and token-decoding bodies must match.
    def serial_ast(path):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in ("LoopKV", "OuroLoop", "HuginnLoop"):
                node.body = [f for f in node.body if not (isinstance(f, ast.FunctionDef) and f.name == "__init__")]
        return ast.dump(tree, include_attributes=False)
    relative = "src/llm_memory_editability/loop_tree_decode.py"
    if serial_ast(Path(parent["source"]) / relative) != serial_ast(source / relative):
        raise ValueError("serial reference implementation changed beyond storage constructors")
    for name in ("test_loop_continuous_batch.py", "test_loop_batch_decode.py", "test_loop_batch_sweep.py", "test_loop_optimized_decode.py"):
        shutil.copy2(repo / "tests" / name, source / "tests" / name)
    shutil.copy2(repo / "configs/loopfill-online-suffix.json", source / "configs/loopfill-online-suffix.json")
    changes = []
    for name in sorted(changed_reference_sources):
        old = Path(parent["source"]) / name
        delta = subprocess.run(["diff", "-u", str(old) if old.exists() else "/dev/null",
                                str(source / name)], capture_output=True, text=True)
        changes.append(delta.stdout)
    (out / "benchmark-changes.diff").write_text("".join(changes))
    inputs = {}
    for model in parent["model_revisions"]:
        rows = [json.loads(line) for line in Path(parent["inputs"][f"specbench480/{model}"]).read_text().splitlines()]
        if len(rows) != 480 or len({r["question_id"] for r in rows}) != 480:
            raise ValueError(f"expected 480 unique requests: {model}")
        dst = out / "inputs" / f"specbench480-{model}.jsonl"
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(parent["inputs"][f"specbench480/{model}"], dst)
        inputs[f"specbench480/{model}"] = str(dst)
    lanes = []
    for lane in LANES:
        snap = gpu_snapshot(lane["gpu"])
        lane = dict(lane, uuid=snap["uuid"])
        check_resources(lane)
        lanes.append(lane)
    protocol = dict(parent, version=f"full480-{schedule}-batch-sweep-v2", parent_suite=str(suite),
        source=str(source), inputs=inputs, created_unix=time.time(), lanes=lanes,
        main=dict(parent["main"], streams=108, batch_sizes=BATCHES, schedule=schedule),
        schedule=schedule,
        limits=["BF16 output differences retained; no unconditional losslessness claim",
                "one seed / one timing repeat", "finite queue; no external arrival or queueing latency model",
                "sequential unchunked prefill; eager per-request attention"],
        scheduling_policy="FCFS finite queue; refill free slots each iteration; sequential unchunked prefill; no arrival-rate simulation" if schedule == "continuous" else "fixed batches",
        timing="end-to-end from first admission through final synchronization; sequential prefill and all scheduling included; reference/loading/warmup outside timing",
        probe=dict(enabled=False), probe_lanes=[], main_lanes=[], allocation_guard="disabled",
        failure_policy="actual CUDA OOM recorded per model/batch/arm; every other cell still attempted",
        resource_policy=dict(gpu="one fixed UUID per model/batch across all arms; eight exclusive GPUs in parallel",
            graphics_clock_mhz=1980, memory_application_clock_mhz=3201, power_limit_w=700,
            cpu="fixed four CPU affinity per GPU", torch_threads=1, omp_threads=1,
            process="fresh process for every cell; no MPS or GPU sharing", telemetry_seconds=15),
        paired_gpu_policy="all six arms for each model/batch use the same GPU UUID; every GPU has identical model, memory and settings")
    protocol["execution"] = dict(attention_backend="packed" if optimized else "per-request",
        prefill_chunk_size=128 if optimized else 0, cuda_graph=optimized,
        max_cuda_graphs=24, warmup_tokens=32 if optimized else 8)
    if optimized:
        protocol.update(version="full480-continuous-packed-chunked-graphs-v3",
            limits=protocol["limits"][:-1] + ["dense KV allocation; no paging", "CUDA graphs cover warmed decode shapes only; eager misses counted", "prefill chunks run eager; CPU scheduler remains outside graphs"],
            scheduling_policy="FCFS continuous slots; round-robin 128-token prefill chunks alternate with shared decode advances",
            timing="end-to-end including chunked prefill, CPU scheduling and decode; loading, reference and graph warmup excluded",
            graph_warmup="32 output tokens on first min(2B,480) shuffled requests; same chunked/continuous scheduler includes slot refills; capture up to 24 frequent shapes with bucketed attention groups/query/tree widths; no measured capture")
    save(out / "protocol.json", protocol)
    save(out / "source-hashes.json", {str(p.relative_to(source)): digest(p)
        for p in sorted(source.rglob("*")) if p.is_file() and "__pycache__" not in p.parts})
    save(out / "input-hashes.json", {p: digest(p) for p in inputs.values()})
    shutil.copy2(suite / "input-audit.json", out / "input-audit.json")
    # Reuse only completed reference lines. The old writer may still be active;
    # truncate at its last newline, never modify or append to its reference file.
    # AR decode, weights, input tokens, and all generation settings are unchanged.
    provenance = []
    for model in parent["model_revisions"]:
        old = suite / "references" / f"{model}.jsonl"
        if not old.exists():
            continue
        data = old.read_bytes()
        data = data[:data.rfind(b"\n") + 1]
        lines = data.splitlines()
        header = json.loads(lines[0])
        identity = header["reference_identity"]
        old_hashes = identity["source_sha256"]
        for name, expected in old_hashes.items():
            if digest(Path(parent["source"]) / name) != expected:
                raise ValueError(f"reference source mismatch: {name}")
            if name not in changed_reference_sources and digest(source / name) != expected:
                raise ValueError(f"reference decoding code changed: {name}")
        reference_sources = [source / "scripts/loop_batch_bench.py"] + sorted((source / "src/llm_memory_editability").glob("*.py"))
        identity["source_sha256"] = {str(p.relative_to(source)): digest(p) for p in reference_sources}
        target = out / "references" / f"{model}.jsonl"
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(json.dumps(header).encode() + b"\n" + b"\n".join(lines[1:]) + (b"\n" if lines[1:] else b""))
        provenance.append(dict(model=model, parent_reference=str(old), completed_rows=len(lines)-1,
            parent_snapshot_sha256=__import__("hashlib").sha256(data).hexdigest(),
            changed_sources=sorted(changed_reference_sources),
            change="reference header migrated for execution backend and optional shared-storage constructors; serial AR operation bodies AST-verified unchanged; weights and input identity unchanged"))
        (out / "references" / f"{model}-parent-snapshot.jsonl").write_bytes(data)
        expected_ids = {json.loads(line)["question_id"] for line in Path(inputs[f"specbench480/{model}"]).read_text().splitlines()}
        reference_rows = [json.loads(line) for line in lines[1:]]
        ids = [row["request_id"] for row in reference_rows]
        if len(ids) != len(set(ids)) or not set(ids) <= expected_ids:
            raise ValueError("invalid cached reference IDs")
        if len(ids) == 480:
            save(out / "references" / f"{model}-complete.json", dict(
                requests=480, sha256=digest(target), reused_from=str(old), finished_unix=time.time()))
    save(out / "reference-provenance.json", provenance)
    for lane in lanes:
        for cell in cells(lane["model"], protocol["arms"], lane["batches"]):
            save(out / "jobs" / (cell["key"] + ".json"), dict(cell, outcome="pending", schedule=schedule,
                gpu=lane["gpu"], gpu_uuid=lane["uuid"]))
    print(json.dumps(dict(out=str(out), planned=108, lanes=lanes, references=provenance), indent=2))


def reference_shard():
    """Generate independent serial-AR requests in four disjoint shards per model."""
    shard = int(sys.argv[3])
    sys.argv = [sys.argv[0]] + sys.argv[4:]
    import loop_batch_bench as bench
    original = bench.serial_reference

    def generate_shard(path, identity, requests, generate):
        path = Path(path)
        selected = requests[shard::4]
        expected = {r.request_id for r in selected}
        target = path.with_name(f"{path.stem}-shard{shard}.jsonl")
        header = dict(reference_identity=identity)
        if not target.exists():
            cached = []
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if rows[0] != header:
                    raise ValueError("parent reference identity mismatch")
                cached = [r for r in rows[1:] if r["request_id"] in expected]
            target.parent.mkdir(exist_ok=True)
            target.write_text("".join(json.dumps(row) + "\n" for row in [header, *cached]))
        original(target, identity, selected, generate)
        # The reference setup process must not enter a timed benchmark.
        raise SystemExit(0)

    bench.serial_reference = generate_shard
    bench.main()


def merge_references(out, model):
    header, combined = None, {}
    for shard in range(4):
        path = out / "references" / f"{model}-shard{shard}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if header is None:
            header = rows[0]
        if rows[0] != header:
            raise ValueError("reference shard identities differ")
        for row in rows[1:]:
            rid = row["request_id"]
            if rid in combined:
                raise ValueError("duplicate reference ID across shards")
            combined[rid] = row
    questions = [json.loads(line) for line in (out / "inputs" / f"specbench480-{model}.jsonl").read_text().splitlines()]
    expected = [r["question_id"] for r in questions]
    if len(expected) != 480 or set(combined) != set(expected):
        raise ValueError("incomplete reference shard coverage")
    target = out / "references" / f"{model}.jsonl"
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in [header, *[combined[rid] for rid in expected]]))
    tmp.replace(target)
    save(out / "references" / f"{model}-complete.json", dict(requests=480, sha256=digest(target), finished_unix=time.time()))


def summarize(out):
    rows = []
    for path in sorted((out / "jobs").glob("*.json")):
        if path.name.startswith("reference-"):
            continue
        job = json.loads(path.read_text())
        row = {k: job.get(k) for k in ("key", "model", "batch", "arm", "gpu", "gpu_uuid", "outcome")}
        row.update(schedule=job.get("schedule"), tok_s=None, wall_s=None, output_tokens=None,
                   requests=None, same_as_serial_ar=None, average_active_requests=None, slot_utilization=None,
                   prefill_policy=None, prefill_chunk_size=None, attention_policy=None,
                   cuda_graph_shapes=None, cuda_graph_replays=None, cuda_graph_eager_misses=None,
                   cuda_graph_capture_s=None)
        if job["outcome"] == "complete":
            s = json.loads((Path(job["output"]) / f"{job['arm']}-b{job['batch']}-seed0-summary.json").read_text())
            row.update(tok_s=s["output_tput"], wall_s=s["wall_s"], output_tokens=s["output_tokens"],
                       requests=s["requests"], same_as_serial_ar=s.get("same_as_serial_ar"),
                       schedule=s["schedule"], average_active_requests=s.get("average_active_requests"),
                       slot_utilization=s.get("slot_utilization"))
            for name in ("prefill_policy", "prefill_chunk_size", "attention_policy", "cuda_graph_shapes",
                         "cuda_graph_replays", "cuda_graph_eager_misses", "cuda_graph_capture_s"):
                row[name] = s.get(name)
        row["error"] = job.get("error")
        rows.append(row)
    save(out / "summary.json", dict(updated_unix=time.time(), counts=dict(Counter(r["outcome"] for r in rows)), measurements=rows))
    with (out / "measurements.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    protocol = verify_suite(out)
    audit = json.loads((out / "input-audit.json").read_text())
    source = Path(protocol["source"])
    guard = (out / "supervisor.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(phase="running", schedule=protocol["schedule"], supervisor_pid=os.getpid(),
                 started_unix=time.time(), planned_cells=108, lanes={})
    mutex = threading.RLock()
    stopped = threading.Event()
    barriers = {model: threading.Barrier(4, action=lambda m=model: merge_references(out, m))
                for model in protocol["model_revisions"]}

    def update(gpu=None, **fields):
        with mutex:
            (state if gpu is None else state["lanes"].setdefault(str(gpu), {})).update(fields)
            state["updated_unix"] = time.time()
            save(out / "status.json", state)
            summarize(out)

    def heartbeat():
        while not stopped.wait(15):
            try:
                with (out / "gpu-telemetry.jsonl").open("a") as f:
                    f.write(json.dumps(dict(time=time.time(), gpus=[gpu_snapshot(l["gpu"]) for l in protocol["lanes"]])) + "\n")
                update()
            except Exception as exc:
                update(telemetry_error=repr(exc))

    def worker(lane):
        gpu, model = lane["gpu"], lane["model"]
        with open(f"/tmp/loopfill-suite-gpu{gpu}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_resources(lane)
            def execute(cell, reference_only=False):
                jobpath = out / "jobs" / (cell["key"] + ".json")
                job = json.loads(jobpath.read_text())
                if not reference_only and job["outcome"] in ("complete", "oom", "failed"):
                    return job
                target = out / ("reference-setup" if reference_only else "runs") / cell["key"]
                if target.exists():
                    raise RuntimeError(f"interrupted cell retained for diagnosis: {target}")
                logpath = out / "logs" / (cell["key"] + ".log")
                logpath.parent.mkdir(exist_ok=True)
                revision = protocol["model_revisions"][model]
                modelpath = Path("/root/.cache/huggingface/hub") / ("models--" + revision[0]) / "snapshots" / revision[1]
                cmd = ["taskset", "--cpu-list", lane["cpus"], f"/root/.cache/loopfill/{revision[2]}/bin/python",
                    str(source / "scripts/loop_batch_bench.py"), "--model", model, "--model-path", str(modelpath),
                    "--questions", protocol["inputs"][f"specbench480/{model}"], "--arms", cell["arm"],
                    "--batch-sizes", str(cell["batch"]), "--schedule", protocol["schedule"],
                    "--order-seeds", "0", "--repeat-id", "0",
                    "--max-new-tokens", "1024", "--max-seq", str(audit[model]["specbench480"]["allocated_positions"]),
                    "--max-nodes", "1152", "--buffer-policy", "per-arm", "--allocation-guard", "disabled",
                    "--scoring", "raw", "--draft-node-budget", "64", "--suffix-eta", "0.1",
                    "--suffix-c-min", "0.25", "--suffix-c-max", "2", "--dtype", "bfloat16", "--threads", "1",
                    "--check", "--reference-file", str(out / "references" / f"{model}.jsonl"), "--out", str(target)]
                execution = protocol.get("execution", {})
                for option in ("attention_backend", "prefill_chunk_size", "max_cuda_graphs", "warmup_tokens"):
                    if option in execution:
                        cmd += ["--" + option.replace("_", "-"), str(execution[option])]
                if execution.get("cuda_graph"):
                    cmd.append("--cuda-graph")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           PYTHONPATH=str(source / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
                env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
                if reference_only:
                    bench_index = cmd.index(str(source / "scripts/loop_batch_bench.py"))
                    cmd[bench_index:bench_index + 1] = [str(source / "scripts/loop_batch_sweep.py"),
                        "reference", "--shard", str(lane["shard"])]
                job.update(command=cmd, output=str(target), log=str(logpath), cpu_affinity=lane["cpus"],
                           schedule=protocol["schedule"], started_unix=time.time(), outcome="running")
                try:
                    verify_suite(out)
                    job["resource_before"] = check_resources(lane)
                    with logpath.open("x") as log:
                        child = subprocess.Popen(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True)
                        job["pid"] = child.pid
                        save(jobpath, job)
                        update(gpu, current_cell=cell["key"], pid=child.pid, phase="running")
                        rc = child.wait()
                    job.update(returncode=rc, outcome=classify(rc, target), resource_after=gpu_snapshot(gpu))
                    if job["outcome"] == "complete" and not reference_only:
                        from loop_online_suite import audit_run
                        expected = {json.loads(line)["question_id"] for line in Path(protocol["inputs"][f"specbench480/{model}"]).read_text().splitlines()}
                        summaries = audit_run(target, expected, True)
                        if any(s["schedule"] != protocol["schedule"] for s in summaries):
                            raise ValueError("mixed scheduling policies in one suite")
                        if execution.get("cuda_graph") and any(
                            not s.get("cuda_graph") or s.get("attention_policy") != "packed_prefix_and_tree"
                            or s.get("prefill_chunk_size") != execution["prefill_chunk_size"] for s in summaries
                        ):
                            raise ValueError("execution backend differs from frozen protocol")
                    elif job["outcome"] != "complete" and (target / "failure.json").exists():
                        job["error"] = json.loads((target / "failure.json").read_text())["error"]
                    elif job["outcome"] != "complete":
                        job["error"] = logpath.read_text()[-4000:]
                except Exception as exc:
                    job.update(outcome="failed", error=repr(exc))
                job["finished_unix"] = time.time()
                save(jobpath, job)
                update(gpu, last_cell=cell["key"], last_outcome=job["outcome"], pid=None)
                return job
            complete_ref = out / "references" / f"{model}-complete.json"
            if complete_ref.exists():
                marker = json.loads(complete_ref.read_text())
                if marker["sha256"] != digest(out / "references" / f"{model}.jsonl"):
                    raise ValueError("completed serial reference changed")
            else:
                ref_cell = dict(key=f"reference-{model}-shard{lane['shard']}", model=model, batch=1, arm="ar")
                ref_job = out / "jobs" / (ref_cell["key"] + ".json")
                # Reference setup is separate from the 108 measured outcomes.
                save(ref_job, dict(ref_cell, outcome="pending", gpu=gpu, gpu_uuid=lane["uuid"]))
                ref_result = execute(ref_cell, reference_only=True)
                save(out / "reference-setup" / (ref_cell["key"] + ".json"), ref_result)
                ref_job.unlink()
                if ref_result["outcome"] != "complete":
                    barriers[model].abort()
                    raise RuntimeError(f"reference shard failed: {ref_result.get('error')}")
                update(gpu, phase="waiting_for_reference_shards", pid=None)
                barriers[model].wait()
            results = run_cells(cells(model, protocol["arms"], lane["batches"]), execute)
            # Persist unexpected scheduler errors too; never leave failed cells pending.
            for result in results:
                if "finished_unix" not in result:
                    path = out / "jobs" / (result["key"] + ".json")
                    record = json.loads(path.read_text())
                    record.update(result, finished_unix=time.time())
                    save(path, record)
            update(gpu, phase="complete", current_cell=None, pid=None)

    update()
    threading.Thread(target=heartbeat, daemon=True).start()
    errors = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, lane) for lane in protocol["lanes"]]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(repr(exc))
                    for barrier in barriers.values():
                        barrier.abort()
        counts = Counter(json.loads(p.read_text())["outcome"] for p in (out / "jobs").glob("*.json"))
        update(phase="finished_with_errors" if errors or counts["failed"] else "complete", errors=errors, counts=dict(counts))
    finally:
        stopped.set()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reference":
        reference_shard()
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--suite", type=Path, default=Path("results/loopfill-batch480-20261104"))
    parser.add_argument("--out", type=Path, default=Path("results/loopfill-continuous480-20261104"))
    parser.add_argument("--schedule", choices=("fixed", "continuous"), default="continuous",
                        help="prepare only; run always follows its frozen protocol")
    parser.add_argument("--optimized", action="store_true", help="prepare shared chunked/packed/graph execution for all arms")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.suite.resolve(), args.out.resolve(), args.schedule, args.optimized)
    elif args.action == "run":
        run(args.out.resolve())
    else:
        summarize(args.out.resolve())
