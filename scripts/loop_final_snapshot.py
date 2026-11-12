"""Archive the final and historical new-baseline evidence without running models.

Large JSON/log files and all JSONL streams use lossless gzip. Original paths and
hashes are retained, and verification resolves them through the manifest rather
than requiring the original machine. The earlier dated snapshot is preserved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import io
import json
import math
from pathlib import Path
import re

from loop_paper_snapshot import Snapshot, read, save, sha, utc


class FinalSnapshot(Snapshot):
    def relative(self, source):
        try:
            return super().relative(source)
        except ValueError:
            return Path("external-loopfill") / Path(source).relative_to(self.baseline.parents[1])

    def copy(self, source, group, destination=None, stable=True):
        source = Path(source)
        if source.suffix not in {".json", ".log"} or source.stat().st_size < 1024**2:
            return super().copy(source, group, destination, stable)
        rel = Path(str(destination or self.relative(source)) + ".gz")
        if str(rel) in self.files:
            return rel
        data = source.read_bytes()
        if source.suffix == ".json":
            json.loads(data)
        assert not stable or sha(source.read_bytes()) == sha(data), source
        encoded = gzip.compress(data, mtime=0)
        target = self.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        self.files[str(rel)] = dict(path=str(rel), group=group, source=str(source),
            source_sha256=sha(data), source_bytes=len(data), sha256=sha(encoded),
            bytes=len(encoded), encoding="gzip", captured_utc=utc(), stable_source=stable)
        return rel

    def tree(self, root, group, recursive=True, stable=True):
        extensions = {".json", ".jsonl", ".csv", ".md", ".tex", ".py", ".toml",
                      ".pdf", ".png", ".log", ".txt", ".sh", ".args"}
        paths = root.rglob("*") if recursive else root.glob("*")
        for source in sorted(paths):
            if "__pycache__" in source.parts or "wandb" in source.parts or source.is_symlink():
                continue
            if source.is_file() and (source.suffix in extensions or source.name.startswith("LICENSE")
                                     or source.name == "STOP"):
                self.copy(source, group, stable=stable)


def capture(repo, baseline, out):
    snap = FinalSnapshot(out, repo, baseline)
    # Current collectors and every retained legacy/pilot run and audit remain
    # separate from the 72 canonical cells; copying does not promote their status.
    snap.tree(baseline, "baseline-protocol-and-collectors", recursive=False)
    for name in ["combined-results", "configs", "runs", "preflight", "logs",
                 "training", "loopahead", "dflash_impl"]:
        print(f"Capturing baseline {name}", flush=True)
        snap.tree(baseline / name, "baseline-" + name)
    for name in ["manifests", "paper-inputs"]:
        snap.tree(baseline / "data" / name, "input-provenance")
    snap.tree(baseline / "data", "input-provenance", recursive=False)

    # Dated repository suites, including superseded attempts and cancellation
    # records, retain their own status. Large training assets are handled below.
    for suite in sorted((repo / "results").iterdir()):
        if not suite.is_dir() or not any(d in suite.name for d in ["20261112", "20261113"]):
            continue
        if suite.name == "dflash-ouro-1.4b-training-20261112-v3":
            continue
        print(f"Capturing repository {suite.name}", flush=True)
        snap.tree(suite, "repository-new-experiments")
    training = repo / "results/dflash-ouro-1.4b-training-20261112-v3"
    snap.tree(training, "ouro14-dflash-training", recursive=False)
    for name in ["source", "training", "checks", "logs"]:
        snap.tree(training / name, "ouro14-dflash-training")
    # Teacher generation is a training input; retain audits and costs, not the
    # 50k generated training examples or optimizer/feature/weight binaries.
    for path in sorted((training / "teachers").rglob("*")):
        if path.is_file() and path.suffix == ".json":
            snap.copy(path, "ouro14-teacher-audit")
    costs = read(baseline / "training_cost.json")
    assert not costs["partial"] and len(costs["rows"]) == 6
    for row in costs["rows"]:
        assert row["status"] == "complete"
        for name in row.get("reservation_paths", []):
            path = Path(name)
            if not path.is_absolute():
                path = baseline / path
            if path.exists():
                snap.copy(path, "training-cost-ledger")
    finish_capture(snap)


def finish_capture(snap):
    repo, baseline, out = snap.repo, snap.baseline, snap.out
    # Preserve all immutable dependencies in the completed throughput archive,
    # including reference inputs and historical source used by its audit.
    evidence_manifest = baseline / "external-evidence/extra-throughput-v1/manifest.json"
    for original, record in read(evidence_manifest)["files"].items():
        source = Path(original)
        archived = Path(record["archived"])
        assert sha(archived.read_bytes()) == record["sha256"]
        if source.exists() and sha(source.read_bytes()) == record["sha256"]:
            snap.copy(source, "throughput-dependencies")
        else:
            snap.copy(archived, "historical-throughput-dependencies")
    snap.copy(evidence_manifest, "throughput-dependencies")

    rows = list(csv.DictReader((baseline / "combined-results/results.csv").open()))
    assert len(rows) == 72
    selected = {Path(row[k]) for row in rows for k in ["method_run", "ar_run", "profile_run"]
                if row.get(k)}
    for root in sorted(selected):
        snap.run(root, "canonical-latency-evidence")
    throughput = list(csv.DictReader((baseline / "combined-results/throughput-results.csv").open()))
    assert len(throughput) == 20
    for row in throughput:
        snap.run(Path(row["run"]), "throughput-evidence", row["status"], expected=480)
    for name, expected in [("loopfill-agentic-sql-table1-20261112", 28),
                           ("loopfill-agentic-sql-ouro14-20261112", 14)]:
        suite = repo / "results" / name
        jobs = read(suite / "jobs.json")
        assert len(jobs) == expected and read(suite / "checks/final-audit.json")["passed"]
        for job in jobs:
            snap.run(suite / "runs" / job["id"], "sql-evidence", expected=100)
    for path in (repo / "paper/data/new-results-20261112/licenses").iterdir():
        snap.copy(path, "licenses", Path("licenses") / path.name)
    snap.finish()
    manifest = read(out / "manifest.json")
    manifest.update(exporter_sha256=sha(Path(__file__).read_bytes()),
        scope="Final 72 latency cells, 20 throughput cells, 42 SQL cells, six drafter training runs, "
              "and retained historical runs/audits/cancellations. Earlier paper data remains versioned.",
        exclusions=["target and drafter weights (published separately)", "optimizer and feature caches",
                    "50k teacher/training input records", "Python bytecode and W&B runtime state"],
        original_repository=str(repo), original_baseline=str(baseline))
    save(out / "manifest.json", manifest)


def resume_capture(repo, baseline, out):
    """Recover a copy interrupted before its manifest, verifying every saved byte."""
    snap = FinalSnapshot.__new__(FinalSnapshot)
    snap.repo, snap.baseline, snap.out = repo, baseline, out
    snap.files, snap.runs = {}, []
    snap.created = datetime.fromtimestamp(out.stat().st_mtime, timezone.utc).isoformat()
    roots = {"repository": repo, "baselines": baseline,
             "external-loopfill": baseline.parents[1],
             "licenses": repo / "paper/data/new-results-20261112/licenses"}
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out)
        if rel.parts[0] not in roots:
            continue
        encoded = p.read_bytes()
        compressed = p.name.endswith((".json.gz", ".jsonl.gz", ".log.gz"))
        data = gzip.decompress(encoded) if compressed else encoded
        relative_source = Path(*rel.parts[1:])
        if compressed:
            relative_source = relative_source.with_suffix("")
        source = roots[rel.parts[0]] / relative_source
        assert source.is_file() and sha(source.read_bytes()) == sha(data), source
        snap.files[str(rel)] = dict(path=str(rel), group="retained-experiment-evidence",
            source=str(source), source_sha256=sha(data), source_bytes=len(data),
            sha256=sha(encoded), bytes=len(encoded), encoding="gzip" if compressed else "identity",
            captured_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            stable_source=True, source_rechecked_at_resume=True)
    print(f"Recovered {len(snap.files)} byte-verified files", flush=True)
    finish_capture(snap)


def verify(out):
    manifest = read(out / "manifest.json")
    sources = {}
    # Report paths only, never a matched credential value.
    secret = re.compile(rb"hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|"
                        rb"github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}|"
                        rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
    for item in manifest["files"]:
        data = (out / item["path"]).read_bytes()
        assert len(data) == item["bytes"] and sha(data) == item["sha256"], item["path"]
        assert len(data) < 50 * 1024**2, item["path"]
        raw = gzip.decompress(data) if item["encoding"] == "gzip" else data
        assert not secret.search(raw), f"Credential-like content: {item['path']}"
        if "source" in item:
            assert sha(raw) == item["source_sha256"], item["path"]
            sources[item["source"]] = item

    def data(source):
        item = sources[str(source)]
        blob = (out / item["path"]).read_bytes()
        return gzip.decompress(blob) if item["encoding"] == "gzip" else blob

    def load(source):
        return json.loads(data(source))

    def close(a, b):
        assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)

    totals = {}
    audits = read(out / "run-audits.json")
    for item in audits["runs"]:
        records = [json.loads(line) for line in data(Path(item["source"]) / "requests.jsonl").splitlines()]
        assert len(records) == item["requests"]
        tokens = sum(len(r["tokens"]) - 1 for r in records)
        assert tokens == item["decode_tokens"]
        ids = [r.get("source_id", r.get("request_id")) for r in records]
        assert None not in ids and len(ids) == len(set(ids))
        value = dict(decode_tokens=tokens, requests=len(records),
                     ids=ids, prompts=[r.get("prompt_sha256") for r in records])
        if all("decode_s" in r["stats"] for r in records):
            value["decode_s"] = sum(r["stats"]["decode_s"] for r in records)
        summary = load(Path(item["source"]) / "summary.json")
        for key in ["decode_s", "decode_tokens", "requests"]:
            if key in summary and key in value:
                close(summary[key], value[key])
        totals[item["source"]] = value

    base = Path(manifest["original_baseline"])
    rows = list(csv.DictReader(io.StringIO(data(base / "combined-results/results.csv").decode())))
    status = load(base / "combined-results/status.json")
    assert status["completed"] == status["expected"] == len(rows) == 72 and not status["missing"]
    assert sha(data(base / "combined-results/results.csv")) == status["results_sha256"]
    for row in rows:
        method, ar = (totals[row[k]] for k in ["method_run", "ar_run"])
        for prefix, value in [("method", method), ("ar", ar)]:
            close(row[prefix + "_decode_s"], value["decode_s"])
            close(row[prefix + "_decode_tokens"], value["decode_tokens"])
            close(row[prefix + "_tok_s"], value["decode_tokens"] / value["decode_s"])
        close(row["eta_latency"], ar["decode_s"] / method["decode_s"])
        close(row["eta_throughput"], float(row["method_tok_s"]) / float(row["ar_tok_s"]))
        assert method["ids"] == ar["ids"] and method["prompts"] == ar["prompts"]
    through = list(csv.DictReader(io.StringIO(data(base / "combined-results/throughput-results.csv").decode())))
    assert len(through) == 20
    for row in through:
        assert totals[row["run"]]["requests"] == 480
        summary = load(Path(row["run"]) / "summary.json")
        for key in ["output_tput", "wall_s", "gamma"]:
            if key in row and key in summary:
                close(row[key], summary[key])
    repo = Path(manifest["original_repository"])
    for name, count in [("loopfill-agentic-sql-table1-20261112", 28),
                        ("loopfill-agentic-sql-ouro14-20261112", 14)]:
        suite = repo / "results" / name
        assert load(suite / "checks/final-audit.json")["passed"]
        table = list(csv.DictReader(io.StringIO(data(suite / "measurements.csv").decode())))
        assert len(table) == count
        for row in table:
            summary = load(suite / "runs" / row["id"] / "summary.json")
            for key in ["decode_s", "decode_tokens", "decode_token_s", "gamma"]:
                close(row[key], summary[key])
    costs = load(base / "training_cost.json")
    assert len(costs["rows"]) == 6 and not costs["partial"]
    assert sha(data(base / "training_cost-ledger.json")) == costs["ledger_sha256"]
    result = dict(passed=True, files=len(manifest["files"]),
        bytes=sum(v["bytes"] for v in manifest["files"]),
        audited_streams=len(audits["runs"]), audited_requests=sum(v["requests"] for v in audits["runs"]),
        canonical_latency_cells=72, throughput_cells=20, sql_cells=42, completed_training_runs=6,
        large_file_and_credential_checks_passed=True, original_bytes_and_metric_totals_verified=True,
        output_divergence_diagnostics="cancelled_by_user; existing historical evidence preserved",
        validator_sha256=sha(Path(__file__).read_bytes()))
    save(out / "validation.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "resume", "verify"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    if args.action in {"capture", "resume"}:
        if args.baseline_root is None:
            parser.error("capture requires --baseline-root")
        action = capture if args.action == "capture" else resume_capture
        action(Path(__file__).resolve().parents[1], args.baseline_root.absolute(), args.out.absolute())
    verify(args.out)
