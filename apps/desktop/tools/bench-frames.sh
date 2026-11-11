#!/usr/bin/env bash
# The frame bench (folo-ui `--bench-frames`) over the app's main screens at two window sizes, scale 1.5.
# Usage: tools/bench-frames.sh [EXE] [FRAMES]   (EXE defaults to $CARGO_TARGET_DIR/release/folo-desktop.exe)
# Build with allocation counts: cargo build --release -p folo-desktop --features bench-alloc
set -euo pipefail
exe=${1:-${CARGO_TARGET_DIR:-target}/release/folo-desktop.exe}
frames=${2:-100}
for size in "1280 800" "2560 1440"; do
    set -- $size
    for scenario in overview run-live dataset models runs playground settings; do
        echo "== $scenario"
        "$exe" --scenario "$scenario" --width "$1" --height "$2" --scale 1.5 --bench-frames "$frames"
    done
done
