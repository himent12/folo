#!/usr/bin/env bash
# First-epoch A/B on the row-10 set: main ($BASE), the new build ($NEW) with a cold pipeline cache (deleted before
# the run) and with a warm one (left by the previous run), alternated per round; run it under the GPU lock.
# Prints epoch 1 duration_s, the process wall and the compile timer's last line: FOLO_GPU_TRACE_COMPILE=1 makes
# folo-gpu print `folo-gpu: compiled N pipelines, module X s, total Y s` to stderr (docs/GPU.md, Pipeline cache;
# a build without the timer, such as main before it, prints nothing there).
# Usage: BASE=... NEW=... [ROUNDS=3] bench/trainer/first_epoch.sh "" | "--amp"
BASE=${BASE:?path of the main folo.exe}
NEW=${NEW:?path of the new folo.exe}
C=${CACHE:-runs-bench/pipeline-cache}
T=${OUT:-runs-bench/first-epoch}
amp=$1
one() { # name bin
  rm -rf $T
  local t0=$(date +%s.%N)
  out=$(FOLO_CACHE_DIR=$C FOLO_GPU_TRACE_COMPILE=1 timeout 300 $2 train --data D:/folo-bench/data --model folo-n \
    --epochs 1 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 --device gpu:0 $amp --project $T --name r --json \
    --json-events epoch 2>&1)
  local t1=$(date +%s.%N)
  trace=$(echo "$out" | grep '^folo-gpu: compiled' | tail -1 | sed 's/^folo-gpu: //')
  dur=$(grep '"event":"epoch"' $T/r/metrics.jsonl | grep -o '"duration_s":[0-9.]*' | cut -d: -f2)
  echo "amp=[$amp] $1: epoch1 ${dur}s wall $(awk "BEGIN{print $t1 - $t0}")s $trace"
}
rm -rf $C
one new-warmup $NEW
for r in $(seq 1 ${ROUNDS:-3}); do
  one base $BASE
  rm -rf $C; one new-cold $NEW
  one new-warm $NEW
done
