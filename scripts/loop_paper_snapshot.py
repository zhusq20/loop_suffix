"""Capture paper evidence without modifying experiments; verify without GPUs.

JSONL files are gzip-compressed byte for byte. The manifest maps each archived
file to its original path, original hash and capture time. Only finished request
streams are included; live queues contribute status/progress snapshots only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


class Snapshot:
    def __init__(self, out, repo, baseline):
        self.out, self.repo, self.baseline = out, repo, baseline
        self.files, self.runs = {}, []
        self.created = utc()
        out.mkdir(parents=True, exist_ok=False)

    def relative(self, source):
        source = source.absolute()
        if source.is_relative_to(self.repo):
            return Path("repository") / source.relative_to(self.repo)
        if source.is_relative_to(self.baseline):
            return Path("baselines") / source.relative_to(self.baseline)
        raise ValueError(f"Unmapped source: {source}")

    def copy(self, source, group, destination=None, stable=True):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        rel = destination or self.relative(source)
        if source.suffix == ".jsonl":
            rel = Path(str(rel) + ".gz")
        if str(rel) in self.files:
            return rel
        data = source.read_bytes()
        if source.suffix == ".json":
            json.loads(data)
        elif source.suffix == ".jsonl":
            for line in data.splitlines():
                json.loads(line)
        if stable and sha(source.read_bytes()) != sha(data):
            raise ValueError(f"Source changed during capture: {source}")
        encoded = gzip.compress(data, mtime=0) if source.suffix == ".jsonl" else data
        target = self.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        self.files[str(rel)] = dict(path=str(rel), group=group, source=str(source),
            source_sha256=sha(data), source_bytes=len(data), sha256=sha(encoded),
            bytes=len(encoded), encoding="gzip" if encoded != data else "identity",
            captured_utc=utc(), stable_source=stable)
        return rel

    def tree(self, root, group, recursive=True, stable=True):
        extensions = {".json", ".jsonl", ".csv", ".md", ".tex", ".py", ".toml", ".pdf", ".png"}
        for source in sorted(root.rglob("*") if recursive else root.glob("*")):
            if source.is_file() and source.suffix in extensions and "__pycache__" not in source.parts:
                self.copy(source, group, stable=stable)

    def root_files(self, root, group, names):
        for name in names.split():
            self.copy(root / name, group)

    def run(self, root, group, status="complete", journal="requests.jsonl", expected=None):
        record_path = root / journal
        records = [json.loads(line) for line in record_path.read_bytes().splitlines()]
        complete = read(root / "complete.json") if (root / "complete.json").exists() else {}
        summary = read(root / "summary.json") if (root / "summary.json").exists() else {}
        expected = expected or complete.get("requests") or summary.get("requests") or summary.get("completed_requests")
        if expected is None or len(records) != expected:
            raise ValueError(f"Request count mismatch: {root}: {len(records)} / {expected}")
        ids = [r.get("source_id", r.get("request_id")) for r in records]
        if None in ids or len(set(ids)) != len(ids):
            raise ValueError(f"Missing/duplicate request IDs: {root}")
        for evidence in [complete, summary]:
            if evidence.get("requests_sha256"):
                assert sha(record_path.read_bytes()) == evidence["requests_sha256"], root
            if evidence.get("summary_sha256"):
                assert sha((root / "summary.json").read_bytes()) == evidence["summary_sha256"], root
        output_tokens = sum(len(r["tokens"]) for r in records)
        decode_tokens = output_tokens - len(records)
        for row in records:
            for key in ("tokens", "decode_tokens"):
                if key in row["stats"]:
                    assert row["stats"][key] == len(row["tokens"]) - 1, (root, key)
        for key, value in [("output_tokens", output_tokens), ("decode_tokens", decode_tokens)]:
            if key in summary:
                assert summary[key] == value, (root, key)
        if "decode_s" in summary:
            seconds = sum(r["stats"]["decode_s"] for r in records)
            assert math.isclose(seconds, summary["decode_s"], rel_tol=1e-10, abs_tol=1e-8), root
        if "output_tput" in summary:
            assert math.isclose(summary["output_tput"], output_tokens / summary["wall_s"], rel_tol=1e-10), root
        # Copy only the selected journal, immutable metadata and completion evidence.
        for name in [journal, "complete.json", "summary.json", "metadata.json", "source-manifest.json",
                     "preflight.json", "validation.json", "reservation-cost.json", "cuda-acquisition.json"]:
            if (root / name).exists():
                self.copy(root / name, group)
        if (root / "source").is_dir():
            self.tree(root / "source", group)
        archived_journal = self.relative(record_path)
        self.runs.append(dict(group=group, source=str(root), status=status,
            journal=str(archived_journal) + ".gz", requests=len(records),
            output_tokens=output_tokens, decode_tokens=decode_tokens,
            completion_hash_verified=bool(complete.get("requests_sha256") or summary.get("requests_sha256"))))

    def finish(self):
        save(self.out / "run-audits.json", dict(passed=True, runs=self.runs,
            note="Integrity/aggregate audit; original statuses and missing process-exit evidence are preserved."))
        data = (self.out / "run-audits.json").read_bytes()
        self.files["run-audits.json"] = dict(path="run-audits.json", group="archive-audit",
            sha256=sha(data), bytes=len(data), encoding="identity")
        save(self.out / "manifest.json", dict(version=1, created_utc=self.created,
            finished_utc=utc(), exporter_sha256=sha(Path(__file__).read_bytes()),
            policy="Point-in-time evidence. Original backend, denominators and completion statuses remain distinct. "
                   "JSONL is losslessly compressed. Live request streams, weights and caches are excluded.",
            files=list(self.files.values()), audited_request_streams=len(self.runs),
            audited_requests=sum(r["requests"] for r in self.runs)))


def capture(repo, baseline, out, costs):
    snap = Snapshot(out, repo, baseline)
    # Packed DFlash: include preserved artifacts but do not promote missing exits.
    suite = repo / "results/dflash-table12-packed-20261112-v2"
    snap.root_files(suite, "dflash-packed", "protocol.json jobs.json dispatch.json status.json measurements.csv measurements.json report.md implementation-validation.json gpu-placement-amendment.json")
    for name in ["inputs", "references", "source"]:
        snap.tree(suite / name, "dflash-packed")
    dispatch = read(suite / "dispatch.json")
    for job in read(suite / "jobs.json"):
        state = dispatch[job["id"]]
        if job["kind"] == "validate":
            snap.tree(suite / state["out"], "dflash-validation")
        elif state["outcome"] in ["complete", "complete_artifact_exit_unavailable"]:
            snap.run(suite / state["out"], "dflash-packed", state["outcome"], expected=job["count"])
    for root in [suite / "logs", repo / "results/gpu-balanced-queue-20261112-v1"]:
        for source in sorted(root.glob("*-exit.json")):
            snap.copy(source, "dflash-exit-evidence")
    snap.root_files(repo / "results/gpu-balanced-queue-20261112-v1", "dflash-exit-evidence",
        "status.json finished.json retry-audit.json retry-status.json recovery-amendment.json controller.py")

    suite = repo / "results/loopahead-table2-20261112-v2"
    snap.root_files(suite, "loopahead-table2", "protocol.json status.json measurements.csv gpu-placement-amendment.json launch-verification.json hardware-at-launch.csv")
    for name in ["source", "preflight"]:
        snap.tree(suite / name, "loopahead-table2")
    for run in sorted((suite / "runs").iterdir()):
        snap.run(run, "loopahead-table2", expected=480)
    for source in sorted((suite / "logs").glob("*-finished.json")):
        snap.copy(source, "loopahead-table2")
    for source in sorted((repo / "results/gpu-rebalance-20261112-v1").glob("*-finished.json")):
        snap.copy(source, "loopahead-table2")

    # Native/HF DFlash and LoopAhead latency rows retain their own AR denominators.
    snap.root_files(baseline, "baseline-table1", "results.csv formal-agreement.json formal-matrix-status.json table2_rows.md EVAL_CONFIG.md")
    rows = list(csv.DictReader((baseline / "results.csv").open()))
    selected = {Path(row[k]) for row in rows for k in ["method_run", "ar_run"]}
    for run in list(selected):
        profile = run.with_name(run.name.replace("-dflash-v1", "-dflash-profile-v1"))
        if profile != run and (profile / "complete.json").exists():
            selected.add(profile)
    for run in sorted(selected):
        group = "baseline-profile" if "-profile-" in run.name else "baseline-table1"
        snap.run(run, group)
    snap.tree(baseline / "preflight/user-resume-eight-gpu-v1", "baseline-reuse")
    for name in ["status.json", "finished.json", "plan.json"]:
        snap.copy(repo / "results/loopahead-t1-resume-20261112-v1" / name, "baseline-table1")

    suite = repo / "results/loopfill-agentic-sql-table1-20261112"
    snap.root_files(suite, "agentic-sql", "protocol.json jobs.json status.json measurements.csv phase-measurements.csv report.md table1-agentic.tex trace-protocol.json trace-status.json")
    for name in ["inputs", "checks", "source", "trace-source", "traces"]:
        snap.tree(suite / name, "agentic-sql")
    for job in read(suite / "jobs.json"):
        snap.run(suite / "runs" / job["id"], "agentic-sql", expected=100)

    suite = repo / "results/loopfill-mechanisms-20261109/timing"
    snap.root_files(suite, "mechanism-timing", "protocol.json runtime.json status.json measurements.csv stages.csv pairs.csv audit.json report.md stage-timing.pdf stage-timing.png")
    snap.tree(suite / "inputs", "mechanism-timing")
    for path in sorted((suite / "jobs").glob("*.json")):
        job = read(path)
        if job["phase"] == "measurement" and job["outcome"] == "complete":
            snap.copy(path, "mechanism-timing")
            snap.run(Path(job["output"]), "mechanism-timing", expected=24,
                journal=f"{job['arm']}-b{job['batch']}-seed0.jsonl")

    # Training curves and completion audits, including the newly finished 1.4B LAD.
    names = ["dflash-huginn-narrow-formal-v1", "dflash-ouro-2.6b-formal-v1",
             "loopahead-huginn-3.5b-formal-v1", "loopahead-ouro-2.6b-formal-v1",
             "loopahead-ouro-1.4b-formal-v1"]
    for name in names:
        root = baseline / "training" / name
        assert (root / "complete.json").exists(), root
        snap.tree(root, "drafter-training")
    for root in sorted((baseline / "training").glob("*-formal-audit-*")):
        snap.tree(root, "drafter-training-audits")
    for root in sorted((baseline / "training").glob("*-data-v1")):
        snap.tree(root, "training-data-audits")
    snap.root_files(baseline, "training-protocol", "DESIGN.md loopahead/SPEC.md data/manifest-audit.json data/protection-report.json data/manifests/manifest.json data/paper-inputs/humaneval-isolation-audit.json")
    for name in ["render_training_costs.py", "collect_costs.py", "dflash_training_registry.py", "collect_results.py"]:
        snap.copy(baseline / name, "analysis-source")
    for name in ["dflash-huginn-narrow-2048-v1.json", "loopahead-lad.json", "loopahead-ouro-2.6b-admission-v1.json", "loopahead-sampling-dev16.json"]:
        snap.copy(baseline / "configs" / name, "training-protocol")
    for file in costs.iterdir():
        snap.copy(file, "training-costs", Path("training-costs") / file.name)

    # Only completed supplement runs. A completed AR is not a completed pair.
    root = baseline / "logs/packed-dflash-supplement-v1"
    for name in ["status.json", "dispatch.json", "plan.json"]:
        snap.copy(root / name, "supplement-snapshot", stable=False)
    snap.tree(root / "source", "supplement-snapshot")
    for marker in sorted((root / "runs").glob("*/complete.json")):
        run = marker.parent
        exit_file = root / "logs" / (run.name + "-exit.json")
        if exit_file.exists() and read(exit_file)["exit_code"] == 0:
            snap.run(run, "supplement-completed-stage", "stage_complete_pair_may_be_pending")
            snap.copy(exit_file, "supplement-completed-stage")

    root = baseline / "logs/baseline-diagnostics-resume-v1"
    for name in ["status.json", "plan.json"]:
        snap.copy(root / name, "diagnostics-snapshot", stable=False)
    for name in ["formal-greedy-margins", "subset-sensitivity-gsm8k-v1"]:
        snap.tree(baseline / "preflight" / name, "diagnostics-completed")
    for root in [repo / "results/dflash-ouro-1.4b-training-20261112-v3",
                 baseline / "logs/loopahead-ouro14-resume-v1",
                 baseline / "logs/ouro14-packed-evaluation-v1"]:
        for name in ["status.json", "plan.json"]:
            snap.copy(root / name, "ongoing-snapshot", stable=False)
    snap.copy(repo / "results/dflash-ouro-1.4b-training-20261112-v3/teachers/shard0/progress.json",
              "ongoing-snapshot", stable=False)
    for marker in sorted((baseline / "preflight").glob("loopahead-trained-ouro-1.4b-*/complete.json")):
        snap.run(marker.parent, "ouro14-preflight", "pilot_not_formal")
    snap.root_files(repo / "results/dflash-packed-backend-20261112", "backend-probe", "comparison.csv validation.json")
    snap.tree(repo / "results/config-consistency-audit-20261112", "configuration-audit")
    snap.finish()


def verify(out):
    manifest = read(out / "manifest.json")
    for item in manifest["files"]:
        data = (out / item["path"]).read_bytes()
        assert len(data) == item["bytes"] and sha(data) == item["sha256"], item["path"]
        decoded = gzip.decompress(data) if item["encoding"] == "gzip" else data
        if "source_sha256" in item:
            assert sha(decoded) == item["source_sha256"], item["path"]
        if item["path"].endswith(".json"):
            json.loads(decoded)
        elif item["path"].endswith(".jsonl.gz"):
            for line in decoded.splitlines():
                json.loads(line)
        elif item["path"].endswith(".csv"):
            list(csv.DictReader(io.StringIO(decoded.decode())))
    audits = read(out / "run-audits.json")
    for run in audits["runs"]:
        with gzip.open(out / run["journal"], "rt") as stream:
            records = [json.loads(line) for line in stream]
        assert len(records) == run["requests"]
        assert sum(len(r["tokens"]) for r in records) == run["output_tokens"]
    result = dict(passed=True, files=len(manifest["files"]),
        bytes=sum(f["bytes"] for f in manifest["files"]),
        request_streams=len(audits["runs"]), requests=sum(r["requests"] for r in audits["runs"]))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "verify"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--costs", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        if args.baseline_root is None or args.costs is None:
            parser.error("capture requires --baseline-root and --costs")
        capture(Path(__file__).resolve().parents[1], args.baseline_root.absolute(), args.out.absolute(), args.costs)
    verify(args.out)
