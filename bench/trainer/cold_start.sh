#!/usr/bin/env bash
# Cold starts of one row-10 epoch (folo-n, batch 16, 640 px), alternated per round; run it under the GPU lock.
# The NVIDIA driver's shader cache is pointed at an empty folder for each run (`__GL_SHADER_DISK_CACHE_PATH`, which
# its Vulkan driver honours too), so the driver compiles every pipeline as on a machine that never ran FOLO, without
# touching the cache other processes use.
# Rows per round: main ($BASE) first-ever (empty FOLO cache); the new build ($NEW) first-ever (no kernel list yet);
# the new build with the kernel list its previous run wrote but no pipeline cache (driver and FOLO pipeline caches
# cold: a driver update, or a list copied from another machine). Prints what bench/trainer/warm_start.sh prints.
# Every run's output is kept in $OUT/log-<n>.txt. A run that exits nonzero or times out, or whose output lacks the
# epoch record or the compile timer's line, is printed as FAILED with its log, not as a sample.
# Usage: BASE=... NEW=... OUT=D:/... [ROUNDS=3] bench/trainer/cold_start.sh "" | "--amp"
set -u
OUT=${OUT:?folder for caches and runs}
amp=$1
mkdir -p "$OUT"
runs=0
one() { # label exe cachedir
  rm -rf -- "$OUT/fe" "$OUT/nv"
  mkdir -p "$OUT/nv"
  runs=$((runs + 1))
  local log="$OUT/log-$runs.txt" t0 t1 code trace first dur why=""
  t0=$(date +%s.%N)
  __GL_SHADER_DISK_CACHE_PATH="$OUT/nv" FOLO_CACHE_DIR="$3" FOLO_GPU_TRACE_COMPILE=1 timeout 600 "$2" train \
    --data D:/folo-bench/data --model folo-n --epochs 1 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 \
    --device gpu:0 $amp --project "$OUT/fe" --name r --json --json-events epoch > "$log" 2>&1
  code=$?
  t1=$(date +%s.%N)
  trace=$(grep '^folo-gpu: compiled' "$log" | tail -1 | sed 's/^folo-gpu: //')
  first=$(grep '^folo-gpu: .* first uses' "$log" | tail -1 | sed 's/^folo-gpu: //')
  dur=$(grep -s '"event":"epoch"' "$OUT/fe/r/metrics.jsonl" | grep -o '"duration_s":[0-9.]*' | cut -d: -f2 | head -1)
  [ -n "$dur" ] || why="no epoch record"
  [ -n "$trace" ] || why="no compile timer line${why:+, $why}"
  [ "$code" -eq 124 ] && why="timed out${why:+, $why}"
  [ "$code" -ne 0 ] && [ "$code" -ne 124 ] && why="exit $code${why:+, $why}"
  if [ -n "$why" ]; then
    echo "amp=[$amp] $1: FAILED ($why), log $log"
    return 1
  fi
  echo "amp=[$amp] $1: epoch1 ${dur}s wall $(awk "BEGIN{print $t1 - $t0}")s | $trace${first:+ | $first} (log $log)"
}
for r in $(seq 1 "${ROUNDS:-3}"); do
  rm -rf -- "$OUT/cache-main" "$OUT/cache-new"
  one "main first-ever" "${BASE:?main folo.exe}" "$OUT/cache-main"
  one "new first-ever (no list)" "${NEW:?new folo.exe}" "$OUT/cache-new"
  rm -f "$OUT"/cache-new/gpu/*.bin
  one "new, driver + pipeline cache cold, list kept" "$NEW" "$OUT/cache-new"
done
