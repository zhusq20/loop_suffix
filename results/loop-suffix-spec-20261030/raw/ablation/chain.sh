#!/bin/bash
# CPU-only replays for protocol §11 (loop-suffix-spec-20261030): ablation of v3 C.
# NUMA 0, cores 0-7. Run only after the prob-tree wall-clock shards have finished.
set -e
REPO=/mnt/disk2_from_server2/siqizhu4/llm-memory-editability
P=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability
O=$P/loop-suffix-spec-20261030/ablation
PY=/tmp/claude-1015/-mnt-disk2-from-server2-siqizhu4-llm-memory-editability/2389b664-94d8-4d1e-bf70-7b87ac224151/scratchpad/py311/bin/python
LIB=$P/loop-suffix-spec-20261030/arctic-suffix/pkg
SET=sb-chat
cd $REPO
run() { PYTHONPATH=src taskset -c 0-7 numactl --membind=0 $PY scripts/loop_prob_tree_ablation.py "$@"; }
# plotting needs matplotlib, which the py311 replay env lacks
plot() { PYTHONPATH=src taskset -c 0-7 numactl --membind=0 python3 scripts/loop_prob_tree_ablation.py "$@"; }
for m in ${@:-ouro-2.6b huginn-3.5b}; do
  RUNS=$P/loop-suffix-spec-20261030/runs/$SET/$m
  FROZEN=$P/loop-spec-prob-tree-20261031/calib-v3/$SET-$m.json
  echo "=== calibrate $m $(date -u +%T)"
  run calibrate --model $m --set $SET --runs-dir $RUNS --suffix-lib $LIB --out-dir $O
  run check-calib --model $m --set $SET --out-dir $O --calibration $FROZEN
  echo "=== run $m $(date -u +%T)"
  run run --model $m --set $SET --runs-dir $RUNS --suffix-lib $LIB --out-dir $O \
    --calibration $FROZEN --workers 4 --out $O/$SET-$m.json \
    --summary $REPO/results/loop-suffix-spec-20261030/ablation/summary-$SET-$m.json
  plot plot --model $m --set $SET \
    --v3-summary $REPO/results/loop-spec-prob-tree-20261031/sim/summary-v3-$SET-$m.json \
    --summary $REPO/results/loop-suffix-spec-20261030/ablation/summary-$SET-$m.json \
    --png $REPO/results/loop-suffix-spec-20261030/figures/ablation-$m.png
done
echo "=== all done $(date -u +%T)"
