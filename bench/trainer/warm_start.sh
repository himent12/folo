#!/usr/bin/env bash
# Warm-start A/B of several binaries, alternated run by run; run it under the GPU lock.
# Each binary keeps its own cache folder ($OUT/cache-<name>), filled by one untimed warm-up run, so every timed run
# is a second start: the pipeline cache, and for builds with parallel precompile the kernel list, are on disk.
#   train:   one row-10 epoch (folo-n, batch 16, 640 px); prints epoch 1 duration_s, process wall and the compile
#            timer's lines (FOLO_GPU_TRACE_COMPILE=1; the "first uses" line is the time the threads that needed
#            the kernels spent, only printed by builds with precompile).
#   predict: `folo predict` of one image (process start to exit, which is the first image's latency plus start-up).
# Usage: OUT=D:/... [ROUNDS=3] [AMP=--amp] bench/trainer/warm_start.sh train|predict name=path/folo.exe ...
set -u
mode=$1; shift
OUT=${OUT:?folder for caches and runs}
IMG=${IMG:-D:/folo-bench/data/images/val}
WEIGHTS=${WEIGHTS:-D:/folo-weights/folo-n-coco.folo}
mkdir -p "$OUT"
img=$(ls "$IMG" | head -1)
one() { # name exe
  local t="$OUT/run-$1" c="$OUT/cache-$1" load
  rm -rf -- "$t"
  # Machine load before the run (other processes: this one is idle), percent of all cores.
  load=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average" | tr -d '')
  local t0 t1 out dur=""
  t0=$(date +%s.%N)
  if [ "$mode" = train ]; then
    out=$(FOLO_CACHE_DIR="$c" FOLO_GPU_TRACE_COMPILE=1 timeout 600 "$2" train --data D:/folo-bench/data --model folo-n \
      --epochs 1 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 --device gpu:0 ${AMP:-} --project "$t" --name r \
      --json --json-events epoch 2>&1)
  else
    out=$(FOLO_CACHE_DIR="$c" FOLO_GPU_TRACE_COMPILE=1 timeout 600 "$2" predict --model "$WEIGHTS" --device gpu:0 \
      --json "$IMG/$img" 2>&1)
  fi
  t1=$(date +%s.%N)
  [ "$mode" = train ] && dur="epoch1 $(grep '"event":"epoch"' "$t/r/metrics.jsonl" | grep -o '"duration_s":[0-9.]*' | cut -d: -f2)s "
  local trace first
  trace=$(echo "$out" | grep '^folo-gpu: compiled' | tail -1 | sed 's/^folo-gpu: //')
  first=$(echo "$out" | grep '^folo-gpu: .* first uses' | tail -1 | sed 's/^folo-gpu: //')
  echo "$mode ${AMP:-} $1: load ${load}% ${dur}wall $(awk "BEGIN{print $t1 - $t0}")s | $trace${first:+ | $first}"
}
for b in "$@"; do one "${b%%=*}" "${b#*=}" > /dev/null; done
for r in $(seq 1 "${ROUNDS:-3}"); do
  for b in "$@"; do one "${b%%=*}" "${b#*=}"; done
done
