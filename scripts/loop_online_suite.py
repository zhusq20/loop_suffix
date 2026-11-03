"""Frozen 24-stream comparison plus independent, short-output batch exploration.

prepare runs input-only audits and snapshots the source. run starts four paired
main lanes and two batch-probe lanes, with per-GPU locks and durable job records.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

REPO = Path(__file__).resolve().parents[1]
ARMS = ["ar", "wavefront", "loopspec-dual", "budget-suffix", "budget-raw", "budget-online"]
MODELS = {
    "ouro-2.6b": ("ByteDance--Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd", "venv"),
    "huginn-3.5b": ("tomg-group-umd--huginn-0125", "bb6621b65e90b6a4b9b29ef88dc83866d450470c", "huginn-venv"),
}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def model_path(name):
    repo, revision, _ = MODELS[name]
    return Path("/root/.cache/huggingface/hub") / ("models--" + repo) / "snapshots" / revision


def prepare(root):
    from transformers import AutoTokenizer
    if (root / "protocol.json").exists():
        raise FileExistsError("This suite is already frozen; use a new output root")
    questions = REPO / "results/loop-spec-reuse-20261030/raw/data/spec_bench_question.jsonl"
    load = REPO / "results/loopfill-icml-20261103/external-calibration/load192.jsonl"
    source = root / "source"
    for folder in ("src", "scripts", "configs", "tests"):
        shutil.copytree(REPO / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(REPO / name, source / name)
    save(root / "source-hashes.json", {
        str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()
    })
    subprocess.run(["git", "diff", "--binary"], cwd=REPO, stdout=(root / "tracked-source.diff").open("w"), check=True)
    gpu_lines = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"
    ], text=True)
    audit = {}
    input_files = {}
    for model in MODELS:
        tok = AutoTokenizer.from_pretrained(model_path(model), trust_remote_code=True, local_files_only=True)
        audit[model] = {}
        for dataset, path, expected in (("specbench480", questions, 480), ("load192", load, 192)):
            rows = read_rows(path)
            assert len(rows) == expected == len({r["question_id"] for r in rows})
            tokenized = []
            lengths = []
            for row in rows:
                ids = tok.apply_chat_template([{"role": "user", "content": row["turns"][0]}],
                                               add_generation_prompt=True)
                lengths.append({"request_id": row["question_id"], "prompt_tokens": len(ids)})
                tokenized.append(dict(question_id=row["question_id"], prompt_token_ids=ids,
                                      category=row.get("category")))
            required = max(r["prompt_tokens"] for r in lengths) + 1024 + (68 if model.startswith("ouro") else 96)
            capacity = ((required + 255) // 256) * 256
            if capacity > 4096:
                raise ValueError(f"{model}/{dataset}: requires {capacity}; no input truncation")
            target = root / "inputs" / f"{dataset}-{model}.jsonl"
            target.parent.mkdir(exist_ok=True)
            target.write_text("".join(json.dumps(r) + "\n" for r in tokenized))
            input_files[f"{dataset}/{model}"] = str(target)
            audit[model][dataset] = dict(requests=expected, required_positions=required,
                                        allocated_positions=capacity, lengths=lengths,
                                        original_sha256=digest(path), tokenized_sha256=digest(target))
    save(root / "input-audit.json", audit)
    protocol = dict(
        version="online-six-arms-20261104", created_unix=time.time(), arms=ARMS,
        source=str(source), model_revisions=MODELS, gpu_inventory=gpu_lines,
        inputs=input_files, audit="input-audit.json", draft_node_budget=64,
        controller=dict(eta=.1, c_min=.25, c_max=2., initial=1., scope="request"),
        readouts={"ouro-2.6b": [1, 2], "huginn-3.5b": [4, 8]},
        loopspec=dict(implementation="same_engine_greedy_reimplementation",
                      reference_repo="https://github.com/kaist-flexml-lab/loopspec",
                      reference_commit="41aa710ed7c9ce908cffa03597be5d1df19d7b56",
                      schedule="q1_top1_then_distinct_q2_top1_keep_both_until_final",
                      huginn_scope="schedule_transfer_to_Huginn_not_official_Raven_checkpoint"),
        main=dict(streams=24, requests_per_stream=480, batch_sizes=[1, 4],
                  order_seeds=[0], repeat_ids=[0], max_new_tokens=1024, check_serial_ar=True),
        probe=dict(dataset="external_load192", batch_sizes=[8, 16, 32, 64, 4, 1],
                   max_new_tokens=128, allocated_for_output_budget=1024,
                   order_seed=0, repeat_id=0, check_serial_ar=False,
                   stop="per-arm capacity guard or CUDA OOM; other arms continue",
                   interpretation="short-output throughput exploration; not a 1024-token or statistically confirmed saturation curve"),
        main_lanes=[dict(gpu=i, model=list(MODELS)[i // 2], batch=[1, 4][i % 2]) for i in range(4)],
        probe_lanes=[dict(gpu=i + 4, model=model) for i, model in enumerate(MODELS)],
        buffer_policy="per-arm", physical_nodes=dict(ar=1, wavefront=1152,
            **{"budget-suffix": 65, "budget-raw": 65, "budget-online": 65,
               "loopspec-dual": {"ouro-2.6b": 15, "huginn-3.5b": 109}}),
        paired_gpu_policy="all methods within each model/B share GPU UUID; B lanes use distinct identical-model GPUs, recorded",
        timing="end-to-end, sequential prefill and fixed batch; model/allocation/tokenization/warmup/reference outside timing",
        limits=["BF16 differences retained, no unconditional losslessness claim", "one seed/one repeat gives no timing-repeat uncertainty estimate",
                "historical Spec-Bench exposure disclosed", "capacity guard is conservative, not a measured hardware maximum"],
    )
    # The frozen top-level config is also checked; no automatic parameter selection.
    save(root / "protocol.json", protocol)
    save(root / "input-hashes.json", {path: digest(path) for path in input_files.values()})
    for model, (_, _, venv) in MODELS.items():
        with (root / f"environment-{model}.txt").open("w") as output:
            subprocess.run([f"/root/.cache/loopfill/{venv}/bin/python", "-m", "pip", "freeze"],
                           stdout=output, check=True)
    print(json.dumps({"root": str(root), "main_streams": 24,
                      "capacity": {m: {d: v["allocated_positions"] for d, v in ds.items()}
                                   for m, ds in audit.items()}}, indent=2))


def audit_run(out, expected, require_check):
    meta = json.loads((out / "metadata.json").read_text())
    if meta["questions_sha256"] != digest(meta["args"]["questions"]):
        raise ValueError("input hash mismatch")
    summaries = []
    for arm in meta["args"]["arms"].split(","):
        key = f"{arm}-b{meta['args']['batch_sizes']}-seed0"
        summary = json.loads((out / (key + "-summary.json")).read_text())
        rows = read_rows(out / (key + ".jsonl"))
        if len(rows) != len(expected) or {r["request_id"] for r in rows} != expected:
            raise ValueError("incomplete/duplicate request coverage")
        if summary["output_tokens"] != sum(len(r["tokens"]) for r in rows) or summary["wall_s"] <= 0:
            raise ValueError("invalid output/timing accounting")
        if summary["max_active_requests"] != min(int(meta["args"]["batch_sizes"]), len(expected)):
            raise ValueError("declared batch was not filled by active requests")
        if require_check and any("same_as_serial_ar" not in r for r in rows):
            raise ValueError("missing serial-AR comparison")
        summaries.append(summary)
    if not (out / "complete.json").exists():
        raise ValueError("missing completion marker")
    return summaries


def run(root):
    if (root / "probe-cancelled.json").exists():
        raise RuntimeError("This workflow was superseded by the full480 eight-GPU matrix; use loop_batch_sweep.py")
    protocol = json.loads((root / "protocol.json").read_text())
    audit = json.loads((root / "input-audit.json").read_text())
    source = Path(protocol["source"])
    guard = (root / "supervisor.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mutex = threading.RLock()
    status = dict(phase="running", supervisor_pid=os.getpid(), started_unix=time.time(),
                  planned_main_streams=24, lanes={})
    stopped = threading.Event()

    def update(gpu=None, **data):
        with mutex:
            (status if gpu is None else status["lanes"].setdefault(str(gpu), {})).update(data)
            status["updated_unix"] = time.time()
            status["completed_main_streams"] = sum(1 for _ in (root / "runs").glob("main-*/*-summary.json"))
            status["completed_probe_streams"] = sum(1 for _ in (root / "runs").glob("probe-*/*-summary.json"))
            refs = {}
            for p in (root / "references").glob("*.jsonl"):
                refs[p.stem] = max(0, sum(1 for _ in p.open()) - 1)
            status["serial_reference_requests"] = refs
            save(root / "status.json", status)

    def heartbeat():
        while not stopped.wait(15):
            update()

    def verify():
        for name, sha in json.loads((root / "source-hashes.json").read_text()).items():
            if digest(source / name) != sha:
                raise RuntimeError(f"frozen source changed: {name}")
        for name, sha in json.loads((root / "input-hashes.json").read_text()).items():
            if digest(name) != sha:
                raise RuntimeError(f"frozen input changed: {name}")

    def cell(lane, phase, batch, arms):
        verify()
        model, gpu = lane["model"], lane["gpu"]
        dataset = "specbench480" if phase == "main" else "load192"
        key = f"{phase}-{model}-b{batch}-" + ("six-arms" if phase == "main" else arms[0])
        out = root / "runs" / key
        jobpath = root / "jobs" / (key + ".json")
        logpath = root / "logs" / (key + ".log")
        questions = protocol["inputs"][f"{dataset}/{model}"]
        expected = {r["question_id"] for r in read_rows(questions)}
        if (out / "complete.json").exists():
            audit_run(out, expected, phase == "main")
            return "complete"
        if out.exists():
            old = json.loads(jobpath.read_text()) if jobpath.exists() else {}
            if old.get("outcome") in ("capacity_guard", "cuda_oom"):
                return old["outcome"]
            raise RuntimeError(f"incomplete run retained; diagnose before a new attempt: {out}")
        cmd = [f"/root/.cache/loopfill/{MODELS[model][2]}/bin/python",
               str(source / "scripts/loop_batch_bench.py"), "--model", model,
               "--model-path", str(model_path(model)), "--questions", questions,
               "--arms", ",".join(arms), "--scoring", "raw", "--draft-node-budget", "64",
               "--suffix-eta", "0.1", "--suffix-c-min", "0.25", "--suffix-c-max", "2",
               "--max-nodes", "1152", "--buffer-policy", "per-arm",
               "--max-seq", str(audit[model][dataset]["allocated_positions"]),
               "--batch-sizes", str(batch), "--order-seeds", "0", "--repeat-id", "0",
               "--max-new-tokens", "1024" if phase == "main" else "128",
               "--dtype", "bfloat16", "--out", str(out)]
        if phase == "main":
            cmd += ["--check", "--reference-file", str(root / "references" / (model + ".jsonl"))]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(source / "src"),
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
        for folder in ("logs", "runs", "jobs"):
            (root / folder).mkdir(exist_ok=True)
        record = dict(key=key, phase=phase, model=model, gpu=gpu, batch=batch, arms=arms,
                      started_unix=time.time(), command=cmd, output=str(out))
        with logpath.open("x") as log:
            child = subprocess.Popen(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record["pid"] = child.pid
            save(jobpath, record)
            update(gpu, phase=phase, current_cell=key, pid=child.pid)
            rc = child.wait()
        record.update(returncode=rc, finished_unix=time.time())
        output = logpath.read_text()
        outcome = ("complete" if rc == 0 else "capacity_guard" if "CAPACITY_GUARD" in output
                   else "cuda_oom" if "CUDA out of memory" in output else "failed")
        record["outcome"] = outcome
        save(jobpath, record)
        if outcome == "complete":
            audit_run(out, expected, phase == "main")
        elif phase == "main" or outcome == "failed":
            raise RuntimeError(f"{key}: {outcome}; see {logpath}")
        update(gpu, last_cell=key, last_outcome=outcome, pid=None)
        return outcome

    def worker(lane, phase):
        gpu = lane["gpu"]
        with open(f"/tmp/loopfill-suite-gpu{gpu}.lock", "a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            occupied = subprocess.check_output([
                "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
            ], text=True).strip()
            if occupied:
                raise RuntimeError(f"GPU {gpu} occupied by {occupied}; no overlapping job launched")
            if phase == "main":
                cell(lane, phase, lane["batch"], ARMS)
            else:
                blocked = {}
                for batch in protocol["probe"]["batch_sizes"]:
                    for arm in ARMS:
                        if arm in blocked and batch >= blocked[arm]["batch"]:
                            continue
                        outcome = cell(lane, phase, batch, [arm])
                        if outcome != "complete":
                            blocked[arm] = dict(batch=batch, reason=outcome)
                            save(root / f"probe-boundaries-{lane['model']}.json", blocked)
            update(gpu, phase=phase + "_complete", current_cell=None)

    threading.Thread(target=heartbeat, daemon=True).start()
    update()
    errors = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(worker, lane, phase): lane
                       for phase, lanes in (("main", protocol["main_lanes"]), ("probe", protocol["probe_lanes"]))
                       for lane in lanes}
            for future in concurrent.futures.as_completed(futures):
                lane = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    update(lane["gpu"], phase="failed", error=str(exc))
                summarize(root)
        update(phase="failed" if errors else "complete", errors=errors)
    finally:
        stopped.set()


def summarize(root):
    rows = []
    for jobpath in sorted((root / "jobs").glob("*.json")):
        job = json.loads(jobpath.read_text())
        out = Path(job["output"])
        for p in sorted(out.glob("*-summary.json")):
            s = json.loads(p.read_text())
            rows.append(dict(phase=job["phase"], model=job["model"], batch=job["batch"],
                gpu=job["gpu"], arm=s["arm"], requests=s["requests"], wall_s=s["wall_s"],
                output_tokens=s["output_tokens"], tok_s=s["output_tput"],
                same_as_serial_ar=s.get("same_as_serial_ar"),
                peak_allocated_gib=s["peak_allocated_bytes"] / 2**30,
                core_rows=s["core_rows"], rows_per_token=s["core_rows"] / s["output_tokens"],
                max_core_rows=s["max_core_rows"], source=str(p)))
            rows[-1]["max_active_requests"] = s["max_active_requests"]
    baseline = {(r["phase"], r["model"], r["batch"]): r["tok_s"] for r in rows if r["arm"] == "ar"}
    for r in rows:
        ar = baseline.get((r["phase"], r["model"], r["batch"]))
        r["speedup_vs_ar"] = r["tok_s"] / ar if ar else None
    save(root / "summary.json", dict(complete_main_streams=sum(r["phase"] == "main" for r in rows),
        complete_probe_streams=sum(r["phase"] == "probe" for r in rows), measurements=rows,
        interpretation="One repeat; probe outputs capped at 128. Capacity guard is not a measured OOM or saturation."))
    if rows:
        with (root / "measurements.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--out", default="results/loopfill-online-20261104")
    args = parser.parse_args()
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "run": run, "summarize": summarize}[args.action](destination)
