#!/bin/bash
# CPU-only replays for protocol §10 (loop-suffix-spec-20261030); NUMA 0, cores 0-3.
set -e
REPO=/mnt/disk2_from_server2/siqizhu4/llm-memory-editability
P=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability
O=$P/loop-suffix-spec-20261030/schedule
PY=/tmp/claude-1015/-mnt-disk2-from-server2-siqizhu4-llm-memory-editability/2389b664-94d8-4d1e-bf70-7b87ac224151/scratchpad/py311/bin/python
LIB=$P/loop-suffix-spec-20261030/arctic-suffix/pkg
cd $REPO
run() { PYTHONPATH=src taskset -c 0-3 numactl --membind=0 $PY scripts/loop_suffix_schedule_compare.py "$@"; }
for m in ouro-2.6b huginn-3.5b; do
  echo "=== check $m $(date -u +%T)"
  run check --model $m --runs-dir $P/loop-suffix-spec-20261030/runs/sb-chat/$m --suffix-lib $LIB \
    --calibration $P/loop-spec-prob-tree-20261031/calib-v3/sb-chat-$m.json --n 20
done
for set in sb-chat full; do for m in ouro-2.6b huginn-3.5b; do
  echo "=== run $set $m $(date -u +%T)"
  run run --model $m --set $set --runs-dir $P/loop-suffix-spec-20261030/runs/$set/$m --suffix-lib $LIB \
    --calibration $P/loop-spec-prob-tree-20261031/calib-v3/$set-$m.json --workers 4 \
    --out $O/$set-$m.json
done; done
echo "=== all done $(date -u +%T)"
