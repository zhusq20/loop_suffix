"""Publish the six selected experiment drafters and verify remote weight hashes.

Credentials are read from HF_TOKEN or a hidden prompt, never saved. Only selected
inference weights are uploaded; optimizer states and frozen target weights are
excluded. Requires huggingface_hub and the original local training artifacts.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import disable_progress_bars

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    disable_progress_bars()
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token (hidden): ")
    api = HfApi(token=token)
    identity = api.whoami()
    assert identity["name"] == args.namespace, "Token owner does not match requested namespace"
    print("Authenticated namespace:", identity["name"], flush=True)
    repo = Path(__file__).resolve().parents[1]
    base = args.baseline_root.resolve()
    import sys
    sys.path.insert(0, str(repo / "src"))
    from llm_memory_editability.loop_model_registry import MODELS

    selected = [
        ("dflash", "huginn-3.5b", "dflash-huginn-narrow-formal-v1", "008886"),
        ("dflash", "ouro-2.6b", "dflash-ouro-2.6b-formal-v1", "009030"),
        ("dflash", "ouro-1.4b", None, "009108"),
        ("loopahead", "huginn-3.5b", "loopahead-huginn-3.5b-formal-v1", "013500"),
        ("loopahead", "ouro-2.6b", "loopahead-ouro-2.6b-formal-v1", "013000"),
        ("loopahead", "ouro-1.4b", "loopahead-ouro-1.4b-formal-v1", "013533"),
    ]
    published = []
    for method, model, name, step in selected:
        if name is None:
            training = repo / "results/dflash-ouro-1.4b-training-20261112-v3/training"
            checkpoint = training / ("checkpoint-" + step)
        else:
            training = base / "training" / name
            checkpoint = base / "checkpoints" / name / ("checkpoint-" + step)
        metadata = json.loads((training / "metadata.json").read_text())
        complete = json.loads((training / "complete.json").read_text())
        weight = checkpoint / ("model.safetensors" if method == "dflash" else "lad.pt")
        if method == "loopahead":
            assert Path(complete["selected_checkpoint"]) == checkpoint
        assert json.loads((checkpoint / "complete.json").read_text())["step"] == int(step)
        repo_id = f"{args.namespace}/loopfill-{method}-{model}"
        weight_hash = digest(weight)
        print(f"Preparing {repo_id}: {weight.stat().st_size} bytes", flush=True)
        with tempfile.TemporaryDirectory(prefix="loopfill-hf-model-") as folder:
            stage = Path(folder)
            shutil.copy2(checkpoint / "complete.json", stage / "checkpoint-complete.json")
            shutil.copy2(training / "metadata.json", stage / "training-metadata.json")
            shutil.copy2(training / "complete.json", stage / "training-complete.json")
            if method == "dflash":
                shutil.copy2(checkpoint / "config.json", stage / "config.json")
                for license_name in ["LICENSE.dflash", "LICENSE.SpecForge"]:
                    shutil.copy2(repo / "baselines/vendor" / license_name, stage / license_name)
            else:
                save(stage / "lad-config.json", metadata["config"])
                for filename in ["best.json", "confirmation.json", "frequent-token-ids.json"]:
                    if (training / filename).exists():
                        shutil.copy2(training / filename, stage / filename)
                admission = base / "configs" / f"loopahead-{model}-admission-v1.json"
                if admission.exists():
                    shutil.copy2(admission, stage / "admission.json")
            provenance = dict(method=method, target_model=model,
                target_repository=MODELS[model].repo_id, target_revision=MODELS[model].revision,
                checkpoint_step=int(step), original_checkpoint=str(checkpoint),
                weight_file=weight.name, weight_sha256=weight_hash, weight_bytes=weight.stat().st_size,
                parameters=metadata.get("draft_parameters", metadata.get("parameters")),
                training_epochs=metadata["epochs"], training_updates=metadata["total_updates"],
                code_repository="https://github.com/zhusq20/loop_suffix",
                code_base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                implementation_sha256={str(p.relative_to(repo)): digest(p)
                    for p in sorted((repo / "baselines").rglob("*.py"))},
                scope="Selected inference drafter used in the reported experiments; requires the frozen target model.")
            save(stage / "artifact-provenance.json", provenance)
            card = f"""---
tags:
- pytorch
- speculative-decoding
- {method}
- loopfill
---

# LoopFill experiments: {method} for {model}

This repository contains the **selected trained drafter** used by the LoopFill
baseline experiments. It requires [{MODELS[model].repo_id}](https://huggingface.co/{MODELS[model].repo_id})
at revision `{MODELS[model].revision}`. Target weights and tokenizer are downloaded
separately from their original repository.

| Property | Value |
| --- | --- |
| Method | {method} |
| Selected checkpoint step | {int(step)} |
| Trainable parameters | {provenance['parameters']} |
| Training epochs / final updates | {metadata['epochs']} / {metadata['total_updates']} |
| Weight file | `{weight.name}` |
| SHA256 | `{weight_hash}` |

## Download and use

```python
from huggingface_hub import snapshot_download
checkpoint = snapshot_download("{repo_id}")
```

Use the custom implementation in [zhusq20/loop_suffix](https://github.com/zhusq20/loop_suffix)
and follow its [standalone inference guide](https://github.com/zhusq20/loop_suffix/blob/main/docs/standalone-baselines.md).
Pass the downloaded directory as `--checkpoint` to `python run_baseline.py infer`
with `--method {method}` and `--model {model}`. A local matching target model and
JSONL prompts are also required. This artifact is not an independent chat model
or a generic AutoModelForCausalLM checkpoint.

The byte-identical weights, training metadata, selection evidence and source
hashes are preserved in this repository. The GitHub experiment archive documents
the benchmark inputs, backend, AR denominator, timing boundaries and limitations.
LoopAhead is our paper-based implementation, not official author code/weights.
DFlash includes our recurrent-target adaptations. Single-seed timing results do
not establish cross-seed stability or task-quality equivalence.

Optimizer/RNG states and intermediate training checkpoints are not included.
The original target and third-party implementation terms remain applicable;
the repository records upstream code license notices where relevant.
"""
            (stage / "README.md").write_text(card)
            files = {p.name: dict(sha256=digest(p), bytes=p.stat().st_size) for p in stage.iterdir()}
            files[weight.name] = dict(sha256=weight_hash, bytes=weight.stat().st_size)
            save(stage / "SHA256SUMS.json", files)
            api.create_repo(repo_id, repo_type="model", private=not args.public, exist_ok=True)
            before = api.model_info(repo_id)
            assert before.private == (not args.public), "Existing repository visibility differs"
            operations = [CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=p)
                          for p in sorted(stage.iterdir())]
            operations.append(CommitOperationAdd(path_in_repo=weight.name, path_or_fileobj=weight))
            commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=operations,
                commit_message=f"Publish selected {method} checkpoint for {model}", parent_commit=before.sha)
            info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
            remote = next(f for f in info.siblings if f.rfilename == weight.name)
            assert remote.size == weight.stat().st_size and remote.lfs.sha256 == weight_hash
            assert digest(weight) == weight_hash
            published.append(dict(repo_id=repo_id, url=f"https://huggingface.co/{repo_id}",
                revision=commit.oid, public=not info.private, method=method, target_model=model,
                checkpoint_step=int(step), weight_file=weight.name, weight_sha256=weight_hash,
                weight_bytes=weight.stat().st_size, remote_lfs_sha256_verified=True,
                files=files))
            save(args.out, dict(status="complete" if len(published) == 6 else "uploading",
                                namespace=args.namespace, models=published))
            print(f"Verified {repo_id}@{commit.oid}", flush=True)
    print("Published and verified all six selected drafters.", flush=True)


if __name__ == "__main__":
    main()
