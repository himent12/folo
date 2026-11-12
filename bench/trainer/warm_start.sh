#!/usr/bin/env bash
# Warm-start A/B of several binaries, alternated run by run; run it under the GPU lock.
# Each binary keeps its own cache folder ($OUT/cache-<name>), filled by one untimed warm-up run, so every timed run
# is a second start: the pipeline cache, and for builds with parallel precompile the kernel list, are on disk.
#   train:   one row-10 epoch (folo-n, batch 16, 640 px); prints epoch 1 duration_s, process wall and the compile
#            timer's lines (FOLO_GPU_TRACE_COMPILE=1; the "first uses" line is the time the threads that needed
#            the kernels spent, only printed by builds with precompile).
#   predict: `folo predict` of one image (process start to exit, which is the first image's latency plus start-up).
# Every run's output is kept in $OUT/log-<name>-<n>.txt. A run that exits nonzero or times out, or whose output
# lacks the epoch record (train), is printed as FAILED (builds before the compile timer, such as
# 647b981f, print no timer line; that is not a failure) with its log, not as a sample.
# Usage: OUT=D:/... [ROUNDS=3] [AMP=--amp] bench/trainer/warm_start.sh train|predict name=path/folo.exe ...
set -u
mode=$1; shift
OUT=${OUT:?folder for caches and runs}
IMG=${IMG:-D:/folo-bench/data/images/val}
WEIGHTS=${WEIGHTS:-D:/folo-weights/folo-n-coco.folo}
mkdir -p "$OUT"
img=$(ls "$IMG" | head -1)
runs=0
one() { # name exe
  local t="$OUT/run-$1" c="$OUT/cache-$1" load
  runs=$((runs + 1))
  local log="$OUT/log-$1-$runs.txt"
  rm -rf -- "$t"
  # Machine load before the run (other processes: this one is idle), percent of all cores.
  load=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average" | tr -d '\r')
  local t0 t1 code dur=""
  t0=$(date +%s.%N)
  if [ "$mode" = train ]; then
    FOLO_CACHE_DIR="$c" FOLO_GPU_TRACE_COMPILE=1 timeout 600 "$2" train --data D:/folo-bench/data --model folo-n \
      --epochs 1 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 --device gpu:0 ${AMP:-} --project "$t" --name r \
      --json --json-events epoch > "$log" 2>&1
  else
    FOLO_CACHE_DIR="$c" FOLO_GPU_TRACE_COMPILE=1 timeout 600 "$2" predict --model "$WEIGHTS" --device gpu:0 \
      --json "$IMG/$img" > "$log" 2>&1
  fi
  code=$?
  t1=$(date +%s.%N)
  local trace first why=""
  trace=$(grep '^folo-gpu: compiled' "$log" | tail -1 | sed 's/^folo-gpu: //')
  first=$(grep '^folo-gpu: .* first uses' "$log" | tail -1 | sed 's/^folo-gpu: //')
  if [ "$mode" = train ]; then
    dur=$(grep -s '"event":"epoch"' "$t/r/metrics.jsonl" | grep -o '"duration_s":[0-9.]*' | cut -d: -f2 | head -1)
    [ -n "$dur" ] || why="no epoch record"
    dur="epoch1 ${dur}s "
  fi
  [ "$code" -eq 124 ] && why="timed out${why:+, $why}"
  [ "$code" -ne 0 ] && [ "$code" -ne 124 ] && why="exit $code${why:+, $why}"
  if [ -n "$why" ]; then
    echo "$mode ${AMP:-} $1: FAILED ($why), log $log"
    return 1
  fi
  echo "$mode ${AMP:-} $1: load ${load}% ${dur}wall $(awk "BEGIN{print $t1 - $t0}")s | $trace${first:+ | $first}"
}
for b in "$@"; do one "${b%%=*}" "${b#*=}" > /dev/null || echo "warm-up of ${b%%=*} FAILED, see $OUT"; done
for r in $(seq 1 "${ROUNDS:-3}"); do
  for b in "$@"; do one "${b%%=*}" "${b#*=}"; done
done
