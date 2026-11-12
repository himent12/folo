#!/usr/bin/env bash
# Slow-disk A/B of the checkpoint writer (docs/hill/r2-trainer.md, PR 5) on the row-10 set, 4 epochs: a build with
# bench/trainer/slow_write.patch applied, which reads two environment variables in train/writer:
#   FOLO_TMP_WRITE_DELAY_MS=<ms>  sleep that long at the start of every epoch's checkpoint write (a slow disk)
#   FOLO_TMP_SYNC=1               write on the training thread, as main did (the path taken when no thread starts)
# Modes, alternated per round: sync (1 s per write, inline: main's behaviour), async (1 s per write, writer thread),
# none (no delay). Prints each run's process wall; every run's output stays in $OUT/<mode>-<round>.log and its run
# folder in $OUT/<mode>-<round>/. Run it under the GPU lock.
# Build the binary (then undo the patch):
#   git apply bench/trainer/slow_write.patch && cargo build --release -p folo-cli && \
#     cp "$CARGO_TARGET_DIR/release/folo.exe" D:/folo-build/folo-slow-write.exe && \
#     git apply -R bench/trainer/slow_write.patch
# Usage: BIN=D:/folo-build/folo-slow-write.exe [ROUNDS=3] [DELAY_MS=1000] [OUT=runs-bench/slow-write] \
#   bench/trainer/slow_write.sh
BIN=${BIN:?path of a folo.exe built with bench/trainer/slow_write.patch}
OUT=${OUT:-runs-bench/slow-write}
DELAY=${DELAY_MS:-1000}
mkdir -p -- "$OUT"
one() { # mode round
  local name="$1-$2" delay="" sync=""
  case $1 in
    sync) delay=$DELAY sync=1 ;;
    async) delay=$DELAY ;;
    none) ;;
  esac
  rm -rf -- "${OUT:?}/$name"
  local t0=$(date +%s.%N)
  FOLO_TMP_WRITE_DELAY_MS=$delay FOLO_TMP_SYNC=$sync timeout 600 "$BIN" train --data D:/folo-bench/data \
    --model folo-n --epochs 4 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 --device gpu:0 \
    --project "$OUT" --name "$name" --json --json-events epoch >"$OUT/$name.log" 2>&1
  local code=$? t1=$(date +%s.%N)
  echo "$1 round $2: wall $(awk "BEGIN{printf \"%.2f\", $t1 - $t0}") s (exit $code)"
}
for r in $(seq 1 "${ROUNDS:-3}"); do
  for m in sync async none; do one "$m" "$r"; done
done
