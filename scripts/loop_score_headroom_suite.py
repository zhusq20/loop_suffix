"""Queue and analysis for the score-coordination headroom diagnostic.

prepare  freeze source, a stratified per-dataset subset, and one job per
         model x N x variant (oracle jobs depend on that model/N's fixed-c1 job)
run      work-conserving GPU queue; resumable; never runs two jobs on one card
analyze  gamma, rows, handover-wait decomposition, c sensitivity and oracle headroom

Example
  python scripts/loop_score_headroom_suite.py prepare --out results/score-headroom-t0 \
      --assets /tmp/loopfill-assets --python /tmp/loopfill-venv/bin/python
  python results/score-headroom-t0/source/scripts/loop_score_headroom_suite.py run \
      --out results/score-headroom-t0 --gpus 0,1,2,3,4,5,6,7
  python scripts/loop_score_headroom_suite.py analyze --out results/score-headroom-t0
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "src"))

from loop_latency_worker import digest, read, save  # noqa: E402
from loop_score_headroom import ORACLE_VARIANTS, parse_variant, variant_names  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = ["huginn-3.5b", "ouro-1.4b", "ouro-2.6b"]
DATASETS = ["gsm8k", "math500", "bbh", "humanevalplus", "mbppplus", "specbench"]
DEFAULT_CS = [0.001, 0.05, 0.25, 0.5, 0.7, 1, 2, 4, 1000]
SCRIPTS = ["loop_score_headroom.py", "loop_score_headroom_suite.py", "loop_latency_worker.py",
           "loop_draft_trace.py", "loop_batch_bench.py"]


def stratified_subset(rows, k):
    """Round-robin over categories, each in sha256(source_id) order; keep file order."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.get("category") or "_"].append(
            (hashlib.sha256(str(row["source_id"]).encode()).hexdigest(), i))
    for g in groups.values():
        g.sort()
    chosen, depth = [], 0
    while len(chosen) < min(k, len(rows)):
        for name in sorted(groups):
            if depth < len(groups[name]) and len(chosen) < k:
                chosen.append(groups[name][depth][1])
        depth += 1
    return [rows[i] for i in sorted(chosen)]


def job_key(model, budget, variant):
    return f"{model}-n{budget}-{variant}"


def make_jobs(models, budgets, cs, online_c0):
    jobs = []
    for model in models:
        for budget in budgets:
            for variant in variant_names(cs, online_c0):
                jobs.append(dict(key=job_key(model, budget, variant), model=model, budget=budget,
                                 variant=variant, outcome="pending", failures=0,
                                 reference=job_key(model, budget, "fixed-c1")
                                 if variant in ORACLE_VARIANTS else None))
    return jobs


def prepare(out, assets, python, models, inputs_dir, datasets, per_dataset, budgets, cs,
            online_c0, temperature, sampling_seed, max_new_tokens, model_specs=None):
    out = Path(out).resolve()
    if (out / "protocol.json").exists():
        raise FileExistsError("protocol exists; use run to resume")
    if model_specs is None:
        from llm_memory_editability.loop_model_registry import MODELS as model_specs
    source = out / "source"
    source.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "scripts").mkdir()
    for name in SCRIPTS:
        shutil.copy2(REPO / "scripts" / name, source / "scripts" / name)
    hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*.py"))}
    (out / "inputs").mkdir()
    inputs = {}
    for dataset in datasets:
        rows = [json.loads(line) for line in (Path(inputs_dir) / f"{dataset}.jsonl").read_text().splitlines()
                if line.strip()]
        subset = stratified_subset(rows, per_dataset)
        # Dataset-local IDs include integers and overlap between datasets.
        # Oracle reference lookup and cross-variant pairing require global IDs.
        subset = [dict(r, original_source_id=r["source_id"],
                       source_id=f"{dataset}/{r['source_id']}") for r in subset]
        if len({r["source_id"] for r in subset}) != len(subset):
            raise ValueError(f"duplicate source IDs in {dataset}")
        path = out / "inputs" / f"{dataset}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        inputs[dataset] = dict(count=len(subset), sha256=digest(path),
                               categories=dict(Counter(r.get("category") for r in subset)),
                               source=str((Path(inputs_dir) / f"{dataset}.jsonl").resolve()),
                               source_sha256=digest(Path(inputs_dir) / f"{dataset}.jsonl"))
    assets = Path(assets).resolve()
    model_info = {}
    for m in models:
        info = read(assets / f"{m}.json") if (assets / f"{m}.json").exists() else {}
        model_info[m] = dict(info, depth_total=model_specs[m].depth_total,
                             readout_depths=list(model_specs[m].loopfill_readout_depths))
    jobs = make_jobs(models, budgets, cs, online_c0)
    protocol = dict(
        name=out.name, created_unix=time.time(), purpose=read_doc(), source=str(source),
        source_sha256=hashes, python=str(Path(python).absolute()), assets=str(assets),
        models=model_info, datasets=list(datasets), inputs=inputs, per_dataset=per_dataset,
        budgets=list(budgets), fixed_cs=sorted({float(c) for c in cs} | {1.0}), online_c0=online_c0,
        variants=variant_names(cs, online_c0), max_new_tokens=max_new_tokens,
        temperature=temperature, sampling_seed=sampling_seed, batch_size=1,
        suffix_max_draft=4, model_top_k=2, child_suffix_caps=[4, 4], suffix_eta=0.1,
        online_c_min=0.25, online_c_max=2.0, dtype="bfloat16", cuda_graph=False,
        history="cold per dataset per job; only completed outputs of that job are published",
        oracle="rerank own candidate pool; reference = same model/N fixed-c1 output; c=1 order otherwise",
        limits=["instrumented diagnostic: loop-step counts only, not throughput",
                "stratified subset; suffix history is shorter than in the full-dataset runs",
                "BF16 tree shapes may change outputs; oracle falls back after leaving the reference",
                "single run; no request-order or timing repeats"])
    save(out / "protocol.json", protocol)
    for folder in ("jobs", "runs", "validation", "logs"):
        (out / folder).mkdir(exist_ok=True)
    for job in jobs:
        save(out / "jobs" / f"{job['key']}.json", job)
    print(json.dumps(dict(suite=str(out), jobs=len(jobs),
                          requests_per_job=sum(v["count"] for v in inputs.values()))), flush=True)
    return protocol


def read_doc():
    return (__doc__ or "").strip().splitlines()[0]


def gpu_state():
    fields = ["index", "uuid", "memory.used", "utilization.gpu"]
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                      "--format=csv,noheader,nounits"], text=True)
    rows = [dict(zip(fields, [v.strip() for v in line.split(",")])) for line in output.splitlines()]
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    return {r["index"]: dict(r, busy=r["uuid"] in busy) for r in rows}


def ready(job, jobs):
    if job["outcome"] != "pending":
        return False
    return job["reference"] is None or jobs[job["reference"]]["outcome"] == "complete"


def run(out, gpus, retry_failed=False, max_failures=2):
    out = Path(out).resolve()
    guard = (out / "supervisor.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(out / "protocol.json")
    source = Path(protocol["source"])
    for rel, expected in protocol["source_sha256"].items():
        if digest(source / rel) != expected:
            raise ValueError(f"frozen source changed: {rel}")
    path_of = lambda key: out / "jobs" / f"{key}.json"  # noqa: E731
    for p in (out / "jobs").glob("*.json"):
        job = read(p)
        if job["outcome"] == "running":
            try:
                os.kill(job["pid"], 0)
            except ProcessLookupError:
                job["outcome"] = "pending"
            else:
                raise RuntimeError(f"worker {job['pid']} still alive; refusing duplicate controller")
        if retry_failed and job["outcome"] in ("failed", "blocked"):
            job.update(outcome="pending", failures=0)
        save(p, job)
    env_base = dict(os.environ, PYTHONPATH=":".join([str(source / "src"), str(source / "scripts"),
                    str(Path(protocol["assets"]) / "ArcticInference")]),
                    PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                    TOKENIZERS_PARALLELISM="false")
    active = {}
    while True:
        for gpu, task in list(active.items()):
            code = task["process"].poll()
            if code is None:
                continue
            task["log"].close()
            job = read(path_of(task["key"]))
            done = code == 0 and (out / "runs" / job["key"] / "complete.json").exists()
            job["failures"] += not done
            job.update(outcome="complete" if done else
                       ("failed" if job["failures"] >= max_failures else "pending"),
                       exit_code=code, ended_unix=time.time())
            save(path_of(job["key"]), job)
            fcntl.flock(task["lock"], fcntl.LOCK_UN)
            task["lock"].close()
            del active[gpu]
        jobs = {j["key"]: j for j in map(read, sorted((out / "jobs").glob("*.json")))}
        for job in jobs.values():
            ref = jobs.get(job["reference"]) if job["reference"] else None
            if job["outcome"] == "pending" and ref and ref["outcome"] in ("failed", "blocked"):
                job["outcome"] = "blocked"
                save(path_of(job["key"]), job)
        # Reference jobs first so oracle jobs can start as early as possible.
        queue = sorted((j for j in jobs.values() if ready(j, jobs)),
                       key=lambda j: (j["variant"] != "fixed-c1", j["variant"] in ORACLE_VARIANTS, j["key"]))
        states = gpu_state()
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            state = states[str(gpu)]
            if state["busy"] or float(state["memory.used"]) > 256:
                continue
            lock = open(f"/tmp/loopfill-{state['uuid']}.lock", "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            job = queue.pop(0)
            log = (out / "logs" / f"{job['key']}.log").open("a")
            command = [protocol["python"], str(source / "scripts/loop_score_headroom.py"),
                       "--suite", str(out), "--job", job["key"]]
            process = subprocess.Popen(command, env=dict(env_base, CUDA_VISIBLE_DEVICES=state["uuid"]),
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True)
            job.update(outcome="running", gpu=gpu, gpu_uuid=state["uuid"], pid=process.pid,
                       started_unix=time.time())
            save(path_of(job["key"]), job)
            active[gpu] = dict(process=process, key=job["key"], lock=lock, log=log)
        counts = Counter(j["outcome"] for j in jobs.values())
        save(out / "status.json", dict(updated_unix=time.time(), counts=dict(counts),
                                       active={str(g): t["key"] for g, t in active.items()}))
        if not active and not any(ready(j, jobs) for j in jobs.values()):
            print(json.dumps(dict(counts)), flush=True)
            if all(j["outcome"] == "complete" for j in jobs.values()):
                analyze(out)
            return
        time.sleep(10)


# ---------------------------------------------------------------- analysis


def rows_of(out, key):
    path = out / "runs" / key / "requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(a, b):
    return None if not b or a is None else 100 * (a / b - 1)


def metrics(rows, depth, reference=None):
    s, t, w, o = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        s.update({k: r["stats"].get(k, 0) for k in (
            "tokens", "calls", "rows", "wasted_rows", "wasted_budget_rows", "covered_tokens",
            "selection_events", "selected_draft_nodes_sum", "selected_zero_nodes_sum")})
        t.update({k: v for k, v in r["trace"].items() if isinstance(v, (int, float))})
        w.update(r["waits"])
        if r.get("oracle"):
            o.update(r["oracle"])
    tokens, calls = s["tokens"], s["calls"]
    m = dict(requests=len(rows), decode_tokens=tokens, loop_steps=calls,
             gamma=depth * tokens / calls if calls else None,
             rows_per_token=s["rows"] / tokens if tokens else None,
             wasted_rows_per_token=s["wasted_rows"] / tokens if tokens else None,
             budget_evicted_rows_per_token=s["wasted_budget_rows"] / tokens if tokens else None,
             coverage_pct=100 * s["covered_tokens"] / tokens if tokens else None,
             mean_selected_nodes=s["selected_draft_nodes_sum"] / s["selection_events"]
             if s["selection_events"] else None,
             mean_handover_progress=(depth * w["entries"] - w["wait"]) / w["entries"] if w["entries"] else None,
             full_ready_handover_pct=100 * w["full_ready_entries"] / w["entries"] if w["entries"] else None,
             proposal_wait_share_of_steps_pct=100 * w["proposal_wait"] / calls if calls else None,
             ranking_wait_share_of_steps_pct=100 * w["ranking_wait"] / calls if calls else None,
             ranking_delayed_handover_pct=100 * w["ranking_entries"] / w["entries"] if w["entries"] else None,
             reach_violations=w["reach_violations"],
             contested_selection_pct=100 * t["contested_selections"] / t["selections"] if t["selections"] else None,
             positive_contested_selection_pct=100 * t["positive_contested_selections"] / t["selections"]
             if t["selections"] else None,
             selection_changed_at_c0_25_pct=100 * t["changed_c0.25"] / t["selections"] if t["selections"] else None,
             selection_changed_at_c2_pct=100 * t["changed_c2.0"] / t["selections"] if t["selections"] else None,
             selected_suffix_scored_pct=100 * t["selected_suffix_entries"] / t["selected_entries"]
             if t["selected_entries"] else None,
             selected_zero_score_pct=100 * t["selected_zero_entries"] / t["selected_entries"]
             if t["selected_entries"] else None)
    if o:
        m.update(oracle_truncated_per_selection=o["on_path_truncated"] / o["selections"],
                 oracle_root_off_reference_pct=100 * (1 - o["root_on_path"] / o["selections"]))
    finals = sorted(r["stats"]["c_final"] for r in rows if r["stats"].get("controller_updates"))
    if finals:
        q = lambda f: finals[int(f * (len(finals) - 1))]  # noqa: E731
        m.update(c_final_p10=q(0.1), c_final_median=q(0.5), c_final_p90=q(0.9))
    if reference is not None:
        m["same_output_as_c1"] = sum(r["tokens"] == reference[r["source_id"]] for r in rows)
    return m


def analyze(out):
    out = Path(out).resolve()
    protocol = read(out / "protocol.json")
    table, headlines, per_dataset, missing = [], [], [], []
    for model in protocol["models"]:
        depth = protocol["models"][model]["depth_total"]
        for budget in protocol["budgets"]:
            runs = {}
            for variant in protocol["variants"]:
                key = job_key(model, budget, variant)
                if (out / "runs" / key / "complete.json").exists():
                    runs[variant] = rows_of(out, key)
                else:
                    missing.append(key)
            if "fixed-c1" not in runs:
                continue
            ref = {r["source_id"]: r["tokens"] for r in runs["fixed-c1"]}
            # Requests whose output every completed variant reproduces exactly.
            common = {sid for sid in ref if all(
                {r["source_id"]: r["tokens"] for r in rows}.get(sid) == ref[sid] for rows in runs.values())}
            stats = {}
            for variant, rows in runs.items():
                m = metrics(rows, depth, ref)
                same = [r for r in rows if r["tokens"] == ref[r["source_id"]]]
                m["gamma_common"] = metrics([r for r in rows if r["source_id"] in common], depth)["gamma"]
                m["gamma_same_as_c1"] = metrics(same, depth)["gamma"] if same else None
                base_same = [r for r in runs["fixed-c1"] if r["source_id"] in {x["source_id"] for x in same}]
                m["steps_vs_c1_same_output_pct"] = pct(sum(r["stats"]["calls"] for r in same),
                                                      sum(r["stats"]["calls"] for r in base_same))
                stats[variant] = m
                table.append(dict(model=model, budget=budget, variant=variant,
                                  common_requests=len(common), **m))
                by = defaultdict(list)
                for r in rows:
                    by[r["dataset"]].append(r)
                for dataset, group in sorted(by.items()):
                    per_dataset.append(dict(model=model, budget=budget, variant=variant, dataset=dataset,
                                            gamma=metrics(group, depth)["gamma"]))
            fixed = {parse_variant(v)["c"]: m for v, m in stats.items() if v.startswith("fixed-c")}
            online = next((m for v, m in stats.items() if v.startswith("online-c")), None)
            best_c = max(fixed, key=lambda c: fixed[c]["gamma"])
            worst_c = min(fixed, key=lambda c: fixed[c]["gamma"])
            c07 = fixed.get(protocol["online_c0"])
            base = stats["fixed-c1"]
            first, only = stats.get("oracle-first"), stats.get("oracle-only")
            headlines.append(dict(
                model=model, budget=budget, requests=base["requests"], common_requests=len(common),
                gamma_c1=base["gamma"], best_fixed_c=best_c, gamma_best_fixed=fixed[best_c]["gamma"],
                worst_fixed_c=worst_c, gamma_worst_fixed=fixed[worst_c]["gamma"],
                fixed_c_gamma_range_pct=pct(fixed[best_c]["gamma"], fixed[worst_c]["gamma"]),
                fixed_c_gamma_range_common_pct=pct(max(m["gamma_common"] or 0 for m in fixed.values()),
                                                   min(m["gamma_common"] or 0 for m in fixed.values())),
                online_vs_c1_pct=pct(online and online["gamma"], base["gamma"]),
                online_vs_same_fixed_c0_pct=pct(online and online["gamma"], c07 and c07["gamma"]),
                online_vs_best_fixed_pct=pct(online and online["gamma"], fixed[best_c]["gamma"]),
                oracle_first_vs_c1_pct=pct(first and first["gamma"], base["gamma"]),
                oracle_first_vs_best_fixed_pct=pct(first and first["gamma"], fixed[best_c]["gamma"]),
                oracle_first_vs_c1_common_pct=pct(first and first["gamma_common"], base["gamma_common"]),
                oracle_first_same_output=first and first["same_output_as_c1"],
                oracle_only_rows_vs_c1_pct=pct(only and only["rows_per_token"], base["rows_per_token"]),
                oracle_only_vs_c1_gamma_pct=pct(only and only["gamma"], base["gamma"]),
                c1_ranking_wait_share_pct=base["ranking_wait_share_of_steps_pct"],
                c1_proposal_wait_share_pct=base["proposal_wait_share_of_steps_pct"],
                c1_contested_selection_pct=base["contested_selection_pct"],
                c1_positive_contested_selection_pct=base["positive_contested_selection_pct"],
                c1_selection_changed_at_c0_25_pct=base["selection_changed_at_c0_25_pct"]))
    report = dict(protocol=str(out / "protocol.json"), headlines=headlines, table=table,
                  per_dataset=per_dataset, missing_jobs=missing,
                  definitions=dict(
                      gamma="T * sum(decode tokens) / sum(loop steps)",
                      gamma_common="gamma on requests whose output every variant reproduces exactly",
                      ranking_wait="loop steps a correct node waited although proposed early enough",
                      proposal_wait="loop steps lost because the correct node was proposed late or never",
                      contested_selection="selection boundaries with more candidates than N",
                      positive_contested_selection="boundaries with more positive-score candidates than N",
                      selection_changed="same pool re-ranked at another c changes the selected set"),
                  limits=protocol["limits"], analyzer_sha256=digest(Path(__file__)))
    save(out / "analysis.json", report)
    with (out / "table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
        writer.writeheader()
        writer.writerows(table)
    (out / "report.md").write_text(render(protocol, headlines, table, per_dataset, missing))
    print(json.dumps(headlines, indent=2))
    return report


def cell(v):
    if v is None:
        return "—"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def pc(v):
    return "—" if v is None else f"{v:+.1f}%"


def render(protocol, headlines, table, per_dataset, missing):
    lines = [f"# Score headroom: {protocol['name']}", "",
             f"T={protocol['temperature']}, B=1, N={protocol['budgets']}, "
             f"{protocol['per_dataset']} requests x {len(protocol['datasets'])} datasets, "
             f"max {protocol['max_new_tokens']} new tokens. Instrumented diagnostic: "
             "loop-step counts only, never throughput.", ""]
    lines += ["## Headline", "",
              "| model | N | γ c=1 | best fixed c (γ) | fixed-c γ range | online vs c=1 | online vs fixed c0 "
              "| oracle-first vs c=1 | oracle-first same output | oracle-only rows vs c=1 "
              "| c=1 ranking wait | c=1 proposal wait | c=1 positive-contested selections "
              "| c=1 selection changed at c=0.25 |",
              "|" + "---|" * 14]
    for h in headlines:
        lines.append("| " + " | ".join([
            h["model"], str(h["budget"]), cell(h["gamma_c1"]),
            f"{h['best_fixed_c']:g} ({cell(h['gamma_best_fixed'])})",
            pc(h['fixed_c_gamma_range_pct']), pc(h['online_vs_c1_pct']),
            pc(h['online_vs_same_fixed_c0_pct']), pc(h['oracle_first_vs_c1_pct']),
            f"{h['oracle_first_same_output']}/{h['requests']}" if h["oracle_first_same_output"] is not None else "—",
            pc(h['oracle_only_rows_vs_c1_pct']), f"{cell(h['c1_ranking_wait_share_pct'])}%",
            f"{cell(h['c1_proposal_wait_share_pct'])}%",
            f"{cell(h['c1_positive_contested_selection_pct'])}%",
            f"{cell(h['c1_selection_changed_at_c0_25_pct'])}%"]) + " |")
    lines += ["", "Ranking/proposal wait: share of all loop steps. Ranking wait is the part better "
              "admission (any scoring) could have removed for that node; it is an upper bound per "
              "node, while oracle-first measures the realized gain. Percent changes are pooled γ ratios.",
              "", "## All variants", "",
              "| model | N | variant | γ | γ common | steps vs c=1 (same output) | rows/token | handover progress "
              "| ranking wait % | contested % | suffix-scored selected % | c_final median |",
              "|" + "---|" * 12]
    for r in table:
        lines.append("| " + " | ".join([
            r["model"], str(r["budget"]), r["variant"], cell(r["gamma"]), cell(r["gamma_common"]),
            pc(r['steps_vs_c1_same_output_pct']), cell(r["rows_per_token"]),
            cell(r["mean_handover_progress"]), cell(r["ranking_wait_share_of_steps_pct"]),
            cell(r["contested_selection_pct"]), cell(r["selected_suffix_scored_pct"]),
            cell(r.get("c_final_median"))]) + " |")
    datasets = sorted({r["dataset"] for r in per_dataset})
    lines += ["", "## γ by dataset", "", "| model | N | variant | " + " | ".join(datasets) + " |",
              "|" + "---|" * (3 + len(datasets))]
    grouped = defaultdict(dict)
    for r in per_dataset:
        grouped[(r["model"], r["budget"], r["variant"])][r["dataset"]] = r["gamma"]
    for (model, budget, variant), values in grouped.items():
        lines.append(f"| {model} | {budget} | {variant} | "
                     + " | ".join(cell(values.get(d)) for d in datasets) + " |")
    if missing:
        lines += ["", "Missing jobs: " + ", ".join(missing)]
    lines += ["", "Limits: " + "; ".join(protocol["limits"]), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("prepare", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--assets", default="/tmp/loopfill-assets",
                   help="contains <model>.json with 'path' and ArcticInference/")
    p.add_argument("--python", default="/tmp/loopfill-venv/bin/python")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--inputs-dir", default=str(REPO / "results/loopfill-latency512-t1-20261108/inputs"))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--per-dataset", type=int, default=20)
    p.add_argument("--budgets", default="16", help="comma-separated N values, e.g. 4,8,16")
    p.add_argument("--cs", default=",".join(map(str, DEFAULT_CS)), help="fixed suffix coefficients; 1 is always added")
    p.add_argument("--online-c0", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--sampling-seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--retry-failed", action="store_true")
    a = p.parse_args()
    if a.action == "prepare":
        if a.temperature not in (0.0, 1.0):
            raise ValueError("supported temperatures: 0 (greedy) or 1")
        prepare(a.out, a.assets, a.python, a.models.split(","), a.inputs_dir, a.datasets.split(","),
                a.per_dataset, [int(n) for n in a.budgets.split(",")], [float(c) for c in a.cs.split(",")],
                a.online_c0, a.temperature, a.sampling_seed, a.max_new_tokens)
    elif a.action == "run":
        run(a.out, [int(g) for g in a.gpus.split(",")], a.retry_failed)
    else:
        analyze(a.out)


if __name__ == "__main__":
    main()
