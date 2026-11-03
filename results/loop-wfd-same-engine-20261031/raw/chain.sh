#!/bin/bash
# loop-wfd-same-engine-20261031 (user 2026-10-31): WFD's schedule in our engine vs suffix+shallow.
# sb1024 chat, 2 repeats per question. User chose the two models in parallel:
#   Ouro   GPU 6, CPUs 24-31 (SMT siblings 72-79 left free), NUMA 1
#   Huginn GPU 5, CPUs 32-39 (SMT siblings 80-87 left free), NUMA 1
# Per model: fp32 check (arms must equal AR; wavefront compared with WFD's own code), then bf16
# timing; a failing check stops that model. Summaries after both lanes and the monitor end.
set -uo pipefail
REPO=/mnt/disk2_from_server2/siqizhu4/llm-memory-editability
P=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability
OUT=$P/loop-wfd-same-engine-20261031
BATCH=loop-wfd-same-engine-20261031
mkdir -p $OUT && chmod 777 $OUT
taskset -c 47 python3 $REPO/scripts/loop_suffix_spec_env_monitor.py \
  --cpus 24-39,72-87 --interval 2 --out $OUT/monitor.jsonl &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
run() {  # name gpu cpus model extra...
  echo "=== $1 start $(date -u +%FT%TZ)"
  docker --context lm-memory run --name $BATCH-$1 \
    --gpus device=$2 --cpuset-cpus $3 --cpuset-mems 1 --network none --user 1015:1016 \
    -e PYTHONPATH=/repo/src -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf_modules -e WANDB_MODE=offline -e OMP_NUM_THREADS=1 \
    -v $REPO:/repo:ro -v $P/loop-spec-reuse-20261030/hf:/hf:ro \
    -v /mnt/disk2_from_server2/siqizhu4/loop-spec-src/wavefront-decoding:/wfd:ro \
    -v $P/loop-suffix-spec-20261030/arctic-suffix/pkg:/arctic:ro \
    -v $P/loop-suffix-spec-20261030/runs:/runs:ro -v $OUT:/out \
    --entrypoint python lm-memory-sdft:sn8200-20261028-ready /repo/scripts/loop_suffix_spec_wallclock.py \
    --model $4 --runs-dir /runs/sb-chat/$4 --out-dir /out/sb-chat/$4 --prompt-format chat \
    --max-prompt-tokens 3000 --batch $BATCH --own-user siqizhu4 "${@:5}"
  local rc=$?
  echo "=== $1 done rc=$rc $(date -u +%FT%TZ)"
  return $rc
}
CHECK=wfd,ar,wavefront,shallow,suffix,both,both-path
BENCH=ar,wavefront,shallow,suffix,both,both-path
lane() {  # tag gpu cpus model
  run check-$1 $2 $3 $4 --mode check --max-new-tokens 64 --arms $CHECK \
    && run bench-$1 $2 $3 $4 --max-new-tokens 1024 --repeats 2 --arms $BENCH
}
lane ouro 6 24-31 ouro-2.6b > $OUT/lane-ouro.log 2>&1 &
A=$!
sleep 20  # stagger model loading
lane huginn 5 32-39 huginn-3.5b > $OUT/lane-huginn.log 2>&1 &
B=$!
wait $A; RA=$?
wait $B; RB=$?
echo "=== lanes done ouro rc=$RA huginn rc=$RB $(date -u +%FT%TZ)"
kill $MON 2>/dev/null || true
sleep 3
[ $RA -eq 0 ] && run sum-ouro 6 24-31 ouro-2.6b --max-new-tokens 1024 --summarize --monitor /out/monitor.jsonl
[ $RB -eq 0 ] && run sum-huginn 5 32-39 huginn-3.5b --max-new-tokens 1024 --summarize --monitor /out/monitor.jsonl
echo "=== all done $(date -u +%FT%TZ)"
