"""Frozen 2x2 ablation of zero filtering and existing-child model probabilities."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import threading
import time

from loop_batch_sweep import check_resources, gpu_snapshot

VARIANTS = {
    "raw": [],
    "positive": ["--positive-only"],
    "exact": ["--existing-child-probs"],
    "exact-positive": ["--existing-child-probs", "--positive-only"],
}
COUNTS = ("tokens", "calls", "rows", "readout_rows", "speculate_calls", "wasted_rows",
          "wasted_budget_rows", "selected_draft_nodes_sum", "selected_zero_nodes_sum",
          "absent_rescores", "absent_positive_rescores", "existing_probability_values")


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def set_option(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def prepare(out):
    repo = Path(__file__).resolve().parents[1]
    parent = repo / "results/loopfill-f1top4-vs-dualtop2-ouro-20261105"
    old = json.loads((parent / "protocol.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    for directory in ("src", "scripts", "configs", "tests"):
        shutil.copytree(repo / directory, source / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(repo / "pyproject.toml", source / "pyproject.toml")
    (out / "inputs").mkdir()
    (out / "logs").mkdir()
    template = next(j["command"] for j in old["jobs"] if j["arm"] == "timing-r0-f1f2-top2")
    jobs = []
    model_settings = {}
    # Three paired timing blocks per model, each on one GPU. A fourth GPU
    # collects diagnostic traces; no trace wall time enters the timing table.
    orders = [list(VARIANTS), list(reversed(VARIANTS)), ["exact", "raw", "exact-positive", "positive"]]
    for model_index, short in enumerate(("ouro", "huginn")):
        full = json.loads((repo / f"results/loopfill-eager-b1-b16-{short}-20261105/protocol.json").read_text())
        settings = full["execution_args"]
        original_inputs = Path(full["questions"])
        if digest(original_inputs) != full["questions_sha256"]:
            raise ValueError("parent input hash changed")
        indexed = {r["question_id"]: r for r in read_rows(original_inputs)}
        selected = [indexed[rid] for rid in old["request_ids"]]
        categories = {"translation", "summarization", "math_reasoning", "qa", "rag"}
        assert dict(Counter(r["category"] if r["category"] in categories else "mt_bench"
                            for r in selected)) == old["category_counts"]
        inputs = out / "inputs" / f"{short}.jsonl"
        inputs.write_text("".join(json.dumps(r) + "\n" for r in selected))
        model_settings[short] = dict(model=settings["model"], model_path=settings["model_path"],
                                    depth_total=4 if short == "ouro" else 32,
                                    readout_depths=[1, 2] if short == "ouro" else [4, 8],
                                    questions=str(inputs), questions_sha256=digest(inputs),
                                    input_provenance=str(original_inputs),
                                    input_provenance_sha256=digest(original_inputs))
        for block in range(4):
            gpu = model_index * 4 + block
            snap = gpu_snapshot(gpu)
            lane = dict(gpu=gpu, uuid=snap["uuid"], cpus=f"{gpu * 4}-{gpu * 4 + 3}")
            check_resources(lane)
            kind, repeat = ("timing", block) if block < 3 else ("trace", 0)
            for variant in orders[block] if block < 3 else VARIANTS:
                key = f"{short}-{kind}-r{repeat}-{variant}"
                command = list(template)
                command[2] = lane["cpus"]
                command[4] = str(source / "scripts" / ("loop_batch_bench.py" if kind == "timing" else "loop_draft_trace.py"))
                for flag, value in {
                    "--model": settings["model"], "--model-path": settings["model_path"],
                    "--questions": inputs, "--out": out / "runs" / key,
                    "--repeat-id": repeat,
                }.items():
                    set_option(command, flag, value)
                # The old template explicitly selected the Ouro readout depths.
                at = command.index("--model-readout-depths")
                del command[at:at + 3]
                command = ["@" + str(source / "configs/loopfill-eager.args") if v.startswith("@") else v
                           for v in command]
                command.extend(VARIANTS[variant])
                jobs.append(dict(arm=key, model=short, variant=variant, kind=kind, repeat=repeat,
                                 lane=lane, command=command, outcome="pending"))
    hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*")) if p.is_file()}
    save(out / "source-sha256.json", hashes)
    protocol = dict(
        purpose=__doc__, created_unix=time.time(), source=str(source), request_ids=old["request_ids"],
        category_counts=old["category_counts"], selection=old["selection"],
        models=model_settings, variants=VARIANTS, batch=4, max_new_tokens=256, jobs=jobs,
        common=dict(node_budget=16, suffix_q=4, top_k=2, scoring="raw", coefficient=1,
                    child_suffix_caps=[4, 4], source_policy="winner_local_history", dtype="bfloat16",
                    schedule="continuous", attention="packed", prefill_chunk_size=128,
                    cuda_graph=False, pad_decode_rows=False, warmup_requests=8, warmup_tokens=32,
                    order_seed=0, history="cold_start_completion_only"),
        timing="Three blocks per model; all four variants share one GPU per block, varied order. Trace runs separate.",
        hypotheses=["Filtering zero path scores reduces rows without excessive extra loop steps.",
                    "Full-vocabulary probabilities for existing children improve ranking under fixed N and top-k.",
                    "Combination may add little when exact probabilities remove artificial zero scores."],
        limitations=["24 previously examined requests; exploratory, not independent full-suite validation.",
                     "BF16 variants may generate different outputs and completion-history states.",
                     "Identical-output subsets and common prefixes are post hoc diagnostics, not controlled causal estimates.",
                     "Repetitions describe fixed-input timing variation across three paired GPUs, not request-order robustness.",
                     "Rows measure node-loops, not FLOPs; gamma is not wall-clock speedup."],
    )
    save(out / "protocol.json", protocol)
    print(json.dumps(dict(out=str(out), jobs=len(jobs), gpus=8)))


def run(out):
    protocol = json.loads((out / "protocol.json").read_text())
    source = Path(protocol["source"])
    for rel, expected in json.loads((out / "source-sha256.json").read_text()).items():
        if digest(source / rel) != expected:
            raise ValueError(f"frozen source changed: {rel}")
    for model in protocol["models"].values():
        if digest(Path(model["questions"])) != model["questions_sha256"]:
            raise ValueError("frozen inputs changed")
    if (out / "status.json").exists():
        raise FileExistsError("suite already started; preserve existing runs")
    state = dict(phase="running", pid=os.getpid(), jobs=protocol["jobs"])
    lock = threading.Lock()

    def update():
        with lock:
            state["updated_unix"] = time.time()
            save(out / "status.json", state)

    def lane_run(jobs):
        lane = jobs[0]["lane"]
        with open(f"/tmp/loopfill-suite-gpu{lane['gpu']}.lock", "a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            for job in jobs:
                try:
                    job["resource_before"] = check_resources(lane)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               PYTHONPATH=str(source / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                    with (out / "logs" / (job["arm"] + ".log")).open("x") as log:
                        child = subprocess.Popen(job["command"], cwd=source, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        job.update(outcome="running", pid=child.pid, started_unix=time.time())
                        update()
                        rc = child.wait()
                    success = rc == 0 and (out / "runs" / job["arm"] / "complete.json").exists()
                    job.update(returncode=rc, outcome="complete" if success else "failed",
                               finished_unix=time.time(), resource_after=gpu_snapshot(lane["gpu"]))
                except Exception as exc:
                    job.update(outcome="failed", error=repr(exc), finished_unix=time.time())
                update()

    update()
    groups = {}
    for job in state["jobs"]:
        groups.setdefault(job["lane"]["gpu"], []).append(job)
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        list(pool.map(lane_run, groups.values()))
    state["phase"] = "complete" if all(j["outcome"] == "complete" for j in state["jobs"]) else "failed"
    update()
    if state["phase"] == "complete":
        analyze(out)


def work(rows, depth):
    counts = Counter()
    requests = 0
    for row in rows:
        requests += 1
        counts.update({k: row["stats"].get(k, 0) for k in COUNTS})
    result = dict(requests=requests, **counts)
    if counts["tokens"]:
        result.update(gamma=depth * counts["tokens"] / counts["calls"],
                      calls_per_token=counts["calls"] / counts["tokens"],
                      rows_per_token=counts["rows"] / counts["tokens"],
                      readout_rows_per_token=counts["readout_rows"] / counts["tokens"])
    return result


def analyze(out):
    protocol = json.loads((out / "protocol.json").read_text())
    runs, table, hashes, summaries, traces = {}, [], {}, {}, {}
    for job in protocol["jobs"]:
        path = out / "runs" / job["arm"]
        assert (path / "complete.json").exists(), path
        rows = read_rows(path / "budget-raw-b4-seed0.jsonl")
        assert len(rows) == len(protocol["request_ids"])
        assert {r["request_id"] for r in rows} == set(protocol["request_ids"])
        rr = {r["request_id"]: r for r in rows}
        summary = json.loads((path / "budget-raw-b4-seed0-summary.json").read_text())
        cfg, variant = summary["config"], job["variant"]
        assert cfg["positive_only"] == ("--positive-only" in VARIANTS[variant])
        assert cfg["existing_child_probs"] == ("--existing-child-probs" in VARIANTS[variant])
        assert cfg["top_m"] == 2 and cfg["draft_node_budget"] == 16 and cfg["suffix_room"] == 4
        assert cfg["readout_depths"] == protocol["models"][job["model"]]["readout_depths"]
        assert cfg["scoring"] == "raw" and cfg["child_suffix_caps"] == [4, 4]
        assert not summary["profiled"] and not summary["controller_profiled"]
        depth = cfg["depth_total"]
        entry = dict(model=job["model"], variant=variant, kind=job["kind"], repeat=job["repeat"],
                     gpu=job["lane"]["gpu"], **work(rows, depth))
        if job["kind"] == "timing":
            assert not summary.get("diagnostic", False)
            entry.update(output_tokens=summary["output_tokens"], wall_s=summary["wall_s"],
                         tokens_per_s=summary["output_tput"])
        else:
            assert summary["diagnostic"]
            trace = json.loads((path / "budget-raw-b4-lifecycle-summary.json").read_text())["totals"]
            assert trace["calls"] == entry["calls"] and trace["rows"] == entry["rows"]
            entry.update(root_steps={str(k): trace.get(f"root_step_{k}", 0) for k in range(depth + 1)},
                         correct_path_budget_waste=trace["correct_path_budget_waste"],
                         delayed_correct_admissions=trace["delayed_correct_admissions"])
            traces[job["arm"]] = {(e["request_id"], e["position"]): e for e in
                read_rows(path / "budget-raw-b4-root-entries.jsonl") if not e["terminal"]}
        runs[job["arm"]], summaries[job["arm"]] = rr, entry
        table.append(entry)
        for file in path.glob("*.json*"):
            hashes[str(file.relative_to(out))] = digest(file)
    timing, pairs, stability, common_work, prefix_comparisons = [], [], {}, {}, {}
    for model, settings in protocol["models"].items():
        depth = settings["depth_total"]
        common = set(protocol["request_ids"])
        base = runs[f"{model}-timing-r0-raw"]
        for key, rr in runs.items():
            if key.startswith(model + "-"):
                common &= {rid for rid in base if base[rid]["tokens"] == rr[rid]["tokens"]}
        common_work[model] = dict(request_ids=sorted(common), variants={v: work(
            (runs[f"{model}-trace-r0-{v}"][rid] for rid in sorted(common)), depth) for v in VARIANTS})
        for variant in VARIANTS:
            measured = [summaries[f"{model}-timing-r{repeat}-{variant}"] for repeat in range(3)]
            rates = [r["tokens_per_s"] for r in measured]
            timing.append(dict(model=model, variant=variant, repeats=3, tokens_per_s_mean=mean(rates),
                               tokens_per_s_stdev=stdev(rates), wall_s_mean=mean(r["wall_s"] for r in measured)))
            own = runs[f"{model}-timing-r0-{variant}"]
            stability[f"{model}-{variant}"] = {}
            for kind, repeat in [("timing", 0), ("timing", 1), ("timing", 2), ("trace", 0)]:
                key = f"{model}-{kind}-r{repeat}-{variant}"
                rr = runs[key]
                stability[f"{model}-{variant}"][key] = dict(
                    identical_outputs=sum(rr[rid]["tokens"] == own[rid]["tokens"] for rid in own),
                    identical_compute=sum(all(rr[rid]["stats"].get(k, 0) == own[rid]["stats"].get(k, 0)
                                               for k in COUNTS) for rid in own))
            if variant == "raw":
                continue
            for repeat in range(3):
                raw_key, new_key = f"{model}-timing-r{repeat}-raw", f"{model}-timing-r{repeat}-{variant}"
                raw, new = summaries[raw_key], summaries[new_key]
                same = sorted(rid for rid in base if runs[raw_key][rid]["tokens"] == runs[new_key][rid]["tokens"])
                pairs.append(dict(model=model, variant=variant, repeat=repeat,
                    throughput_change_pct=100 * (new["tokens_per_s"] / raw["tokens_per_s"] - 1),
                    wall_change_pct=100 * (new["wall_s"] / raw["wall_s"] - 1),
                    identical_output_request_ids=same,
                    identical_output_work={v: work((runs[k][rid] for rid in same), depth)
                                           for v, k in [("raw", raw_key), (variant, new_key)]}))
            raw_key, new_key = f"{model}-trace-r0-raw", f"{model}-trace-r0-{variant}"
            counts = Counter()
            for (rid, pos), entry in traces[raw_key].items():
                other = traces[new_key].get((rid, pos))
                if other is None or runs[raw_key][rid]["tokens"][:pos + 1] != runs[new_key][rid]["tokens"][:pos + 1]:
                    continue
                delta = other["step"] - entry["step"]
                counts["positions"] += 1
                counts["more_ready" if delta > 0 else "less_ready" if delta < 0 else "same_ready"] += 1
                counts["remaining_wait_change"] -= delta
            prefix_comparisons[f"{model}-{variant}"] = dict(counts)
    report = dict(protocol=str(out / "protocol.json"), timing_summary=timing, paired_timing=pairs,
                  runs=table, output_and_compute_stability=stability, common_identical_output_work=common_work,
                  common_prefix_readiness=prefix_comparisons, caveats=protocol["limitations"],
                  analyzer_sha256=digest(Path(__file__)))
    save(out / "analysis.json", report)
    save(out / "analysis-input-sha256.json", hashes)
    with (out / "timing.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(timing[0]))
        writer.writeheader()
        writer.writerows(timing)
    print(json.dumps(dict(timing_summary=timing, analysis=str(out / "analysis.json")), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.action](args.out.resolve())
