#!/bin/bash
# loop-wfd-same-engine-20261031 / wfd-only (user 2026-10-31): time only WFD's schedule in our
# engine; the other arms keep their numbers from loop-suffix-spec-wallclock-20261031. Same
# placement as that batch: GPU 6, CPUs 24-31 (SMT siblings 72-79 free), NUMA 1, models in sequence.
# fp32 checks first (wavefront and AR equal; wavefront equals WFD's own code), then bf16 timing,
# then summaries merged with the earlier batch question by question. set -e stops on any failure.
set -euo pipefail
REPO=/mnt/disk2_from_server2/siqizhu4/llm-memory-editability
P=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability
OUT=$P/loop-wfd-same-engine-20261031/wfd-only
BATCH=loop-wfd-same-engine-20261031
mkdir -p $OUT && chmod 777 $OUT
taskset -c 47 python3 $REPO/scripts/loop_suffix_spec_env_monitor.py \
  --cpus 24-31,72-79 --interval 2 --out $OUT/monitor.jsonl &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
run() {  # name model extra...
  echo "=== $1 start $(date -u +%FT%TZ)"
  docker --context lm-memory run --name $BATCH-wfdonly-$1 \
    --gpus device=6 --cpuset-cpus 24-31 --cpuset-mems 1 --network none --user 1015:1016 \
    -e PYTHONPATH=/repo/src -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf_modules -e WANDB_MODE=offline -e OMP_NUM_THREADS=1 \
    -v $REPO:/repo:ro -v $P/loop-spec-reuse-20261030/hf:/hf:ro \
    -v /mnt/disk2_from_server2/siqizhu4/loop-spec-src/wavefront-decoding:/wfd:ro \
    -v $P/loop-suffix-spec-20261030/arctic-suffix/pkg:/arctic:ro \
    -v $P/loop-suffix-spec-20261030/runs:/runs:ro \
    -v $P/loop-suffix-spec-wallclock-20261031:/old:ro -v $OUT:/out \
    --entrypoint python lm-memory-sdft:sn8200-20261028-ready /repo/scripts/loop_suffix_spec_wallclock.py \
    --model $2 --runs-dir /runs/sb-chat/$2 --out-dir /out/sb-chat/$2 --prompt-format chat \
    --max-prompt-tokens 3000 --batch $BATCH-wfdonly --own-user siqizhu4 "${@:3}"
  echo "=== $1 done $(date -u +%FT%TZ)"
}
run check-ouro ouro-2.6b --mode check --max-new-tokens 64 --arms wfd,ar,wavefront
run check-huginn huginn-3.5b --mode check --max-new-tokens 64 --arms wfd,ar,wavefront
run bench-ouro ouro-2.6b --max-new-tokens 1024 --repeats 2 --arms wavefront
run bench-huginn huginn-3.5b --max-new-tokens 1024 --repeats 2 --arms wavefront
kill $MON 2>/dev/null || true
sleep 3
run sum-ouro ouro-2.6b --max-new-tokens 1024 --summarize --monitor /out/monitor.jsonl \
  --merge-old /old/sb-chat/ouro-2.6b/bench-bfloat16.jsonl
run sum-huginn huginn-3.5b --max-new-tokens 1024 --summarize --monitor /out/monitor.jsonl \
  --merge-old /old/sb-chat/huginn-3.5b/bench-bfloat16.jsonl
echo "=== all done $(date -u +%FT%TZ)"
