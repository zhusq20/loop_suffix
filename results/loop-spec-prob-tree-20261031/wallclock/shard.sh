#!/bin/bash
# usage: shard.sh <model> <shard> <gpu> <cpus> <question ids>; bf16 timing of a subset of
# the questions into sb-chat/<model>-s<shard>, then its summary (amendment 2026-11-01)
set -e
REPO=/mnt/disk2_from_server2/siqizhu4/llm-memory-editability
P=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability
O=$P/loop-spec-prob-tree-20261031; W=$O/wallclock
M=$1; S=$2; GPU=$3; CPUS=$4; IDS=$5
ARMS=ar,wavefront,both-path,prob-low,prob-load
run() {  # step, extra args
  echo "=== $1 start $(date -u +%FT%TZ)"
  docker --context lm-memory run --name loop-spec-prob-tree-20261031-wc-$M-$1-s$S \
    --gpus device=$GPU --cpuset-cpus $CPUS --cpuset-mems ${MEMS:-1} --network none --user 1015:1016 \
    -e PYTHONPATH=/repo/src -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf_modules -e WANDB_MODE=offline -e OMP_NUM_THREADS=1 \
    -v $REPO:/repo:ro -v $P/loop-spec-reuse-20261030/hf:/hf:ro \
    -v /mnt/disk2_from_server2/siqizhu4/loop-spec-src/wavefront-decoding:/wfd:ro \
    -v $P/loop-suffix-spec-20261030/arctic-suffix/pkg:/arctic:ro \
    -v $P/loop-suffix-spec-20261030/runs:/runs:ro -v $O/calib-v3:/calib:ro -v $W:/out \
    --entrypoint python lm-memory-sdft:onb-sn9202-20261028 /repo/scripts/loop_suffix_spec_wallclock.py \
    --model $M --runs-dir /runs/sb-chat/$M --out-dir /out/sb-chat/$M-s$S --prompt-format chat \
    --max-prompt-tokens 3000 --arms $ARMS --prob-calibration /calib/sb-chat-$M.json \
    --batch loop-spec-prob-tree-20261031 --question-ids $IDS "${@:2}"
  echo "=== $1 done $(date -u +%FT%TZ)"
}
run bench --max-new-tokens 1024 --repeats 2
run sum --max-new-tokens 1024 --summarize --monitor /out/${MON:-monitor.jsonl} --own-user siqizhu4
echo "=== all done $(date -u +%FT%TZ)"
