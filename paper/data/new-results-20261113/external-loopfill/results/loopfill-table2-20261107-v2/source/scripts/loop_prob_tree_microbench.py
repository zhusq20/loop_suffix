"""Block-call cost against the number of live rows in the tree-decoding engine.

Builds synthetic draft trees in loop_tree_decode's backbones and times one block call
(backbone.advance) and one readout (backbone.readout) while the committed prefix,
the number of live rows and how the rows spread over loop depths vary. Rows are
either all at one depth (one slot group) or spread over every depth below T, which
is how a deep draft tree looks. The node steps are reset after each call, so every
timed call sees the same tree.

Protocol: results/loop-spec-prob-tree-20261031/protocol.md (S3)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_suffix_spec_readouts import MODELS  # noqa: E402
from loop_suffix_spec_simulate import MAIN  # noqa: E402

from llm_memory_editability.loop_tree_decode import DraftTree, HuginnLoop, OuroLoop  # noqa: E402

ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)  # up to a saturated GPU
READS = (0, 1, 4, 16, 64)


def build_tree(tree: DraftTree, n: int, branching: int, base_pos: int) -> list:
    """Root plus n - 1 nodes, each parent having up to `branching` children."""
    tree.reset(base_pos)
    nodes = [tree.add(1000, None, 0, "root")]
    i = 0
    while len(nodes) < n:
        parent = nodes[i // branching]
        nodes.append(tree.add(1001 + len(nodes), parent, 0, "suffix"))
        i += 1
    return nodes


def timed(fn, sync, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(1000 * (time.perf_counter() - t0))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MAIN))
    p.add_argument("--hf-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="256,1024,2048")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--branching", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    spec = MODELS[args.model]
    T, d = MAIN[args.model]["T"], MAIN[args.model]["d_main"]
    path = Path(args.hf_root) / spec["repo"] / "snapshots" / spec["revision"]
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    prefixes = [int(x) for x in args.prefix.split(",")]
    max_nodes = max(ROWS) + 8
    max_seq = max(prefixes) + 64
    if args.model.startswith("ouro"):
        backbone = OuroLoop(model, T, max_seq, max_nodes)
    else:
        backbone = HuginnLoop(model, T, max_seq, max_nodes, args.seed, max_seq + 8)
    rng = np.random.default_rng(args.seed)
    records = []
    with torch.no_grad():
        for P in prefixes:
            prompt = [int(t) for t in rng.integers(100, 20000, size=P)]
            backbone.prefill(prompt, d)
            for n in ROWS:
                tree = DraftTree(max_nodes, 64)
                nodes = build_tree(tree, n, args.branching, backbone.length)
                backbone.embed(tree, tree.take_fresh())
                for spread in ("one-depth", "all-depths"):
                    steps = [0] * n if spread == "one-depth" else [i % (T - 1) for i in range(n)]

                    def call(nodes=nodes, steps=steps, tree=tree):
                        for node, s in zip(nodes, steps, strict=True):
                            node.step = s
                        backbone.advance(tree, nodes)

                    timed(call, backbone.synchronize, args.warmup)
                    ms = timed(call, backbone.synchronize, args.repeats)
                    stat = {"ms_median": float(np.median(ms)), "ms_mean": float(np.mean(ms))}
                    records.append(
                        {"kind": "advance", "prefix": P, "rows": n, "spread": spread} | stat
                    )
                for k in READS:
                    if k > n:
                        continue
                    for node in nodes:
                        node.step = d

                    def read(nodes=nodes, k=k, tree=tree):
                        backbone.readout(tree, nodes[:k], [], 8)

                    timed(read, backbone.synchronize, args.warmup)
                    ms = timed(read, backbone.synchronize, args.repeats)
                    stat = {"ms_median": float(np.median(ms)), "ms_mean": float(np.mean(ms))}
                    records.append({"kind": "readout", "prefix": P, "rows": n, "reads": k} | stat)
                adv = [r for r in records if r["kind"] == "advance" and r["prefix"] == P]
                shown = ", ".join(f"{r['spread']} {r['ms_median']:.2f}" for r in adv[-2:])
                print(f"[microbench] {args.model} P={P} rows={n}: {shown} ms", flush=True)
    meta = {"model": args.model, "T": T, "d": d, "branching": args.branching,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}  # fmt: skip
    Path(args.out).write_text(json.dumps({"meta": meta, "records": records}, indent=1))


if __name__ == "__main__":
    main()
