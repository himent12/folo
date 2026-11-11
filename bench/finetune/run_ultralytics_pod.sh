#!/usr/bin/env bash
# The Ultralytics half of the fine-tuning matrix on a plain PyTorch GPU pod (RunPod's runpod/pytorch images, any
# Linux box with CUDA torch): installs what run_ultralytics.py needs, fetches the datasets, runs the matrix and
# packs the results into one tarball.
#
#   bash run_ultralytics_pod.sh [OUT_DIR]          # default OUT_DIR /workspace/finetune
#
# Environment: SEEDS (0,1,2), EPOCHS (100), DATASETS (all), ULTRALYTICS_VERSION (8.4.198), WORKERS (Ultralytics'
# default), LABEL (where it runs, for the result files; default "pod <GPU name>"), SERVE_PORT (when set, serve
# OUT_DIR over HTTP on that port once done, so the tarball can be downloaded; the pod is terminated from outside).
# It needs only this folder's files (datasets.py, scorer.py, common.py, run_ultralytics.py) next to it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/workspace/finetune}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "start $(date -u +%FT%TZ)"

python3 -m pip install -q "ultralytics==${ULTRALYTICS_VERSION:-8.4.198}" pycocotools requests
# opencv-python needs libGL, which plain CUDA images lack; the headless build has the same cv2.
python3 -m pip uninstall -y -q opencv-python || true
python3 -m pip install -q opencv-python-headless
python3 -m pip freeze > "$OUT/pip-freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 "$HERE/datasets.py" fetch --root "$OUT/data" ${DATASETS:+--only "$DATASETS"}
python3 "$HERE/run_ultralytics.py" \
    --data-root "$OUT/data" --work-dir "$OUT/work" --results "$OUT/results" \
    --weights "$OUT/weights/yolo26n.pt" --download-weights \
    --seeds "${SEEDS:-0,1,2}" --epochs "${EPOCHS:-100}" ${DATASETS:+--datasets "$DATASETS"} \
    ${WORKERS:+--workers "$WORKERS"} --compute-label "${LABEL:-pod $GPU}" || echo "some runs failed (see above)"

tar czf "$OUT/results.tgz" -C "$OUT" results log.txt pip-freeze.txt gpu.txt
echo "done $(date -u +%FT%TZ): $OUT/results.tgz"
touch "$OUT/DONE"
if [ -n "${SERVE_PORT:-}" ]; then
    cd "$OUT" && exec python3 -m http.server "$SERVE_PORT"
fi
