#!/usr/bin/env bash
# The Ultralytics half of the fine-tuning matrix on a plain PyTorch GPU pod (RunPod's runpod/pytorch images, any
# Linux box with CUDA torch): installs what run_ultralytics.py needs, fetches the datasets, runs the matrix and
# packs the results into one tarball.
#
#   bash run_ultralytics_pod.sh [OUT_DIR]          # default OUT_DIR /workspace/finetune
#
# Environment:
#   PASSES      space-separated <seeds>:<jobs> passes, run in order (default "0,1,2:1": every seed, one run at a
#               time). "0:1 1,2:2" trains seed 0 alone (clean training times) and seeds 1 and 2 two at a time on
#               the GPU (cheaper; their times are recorded as shared).
#   ORDER       seed (default) or dataset: run_ultralytics.py --order; "dataset" with DATASETS listing the longest
#               dataset first keeps the long runs from finishing alone at the end of a --jobs pass.
#   EPOCHS (100), DATASETS (all, in this order), WEIGHTS (yolo26n.pt; yolo26s.pt for an s row), ULTRALYTICS_VERSION (8.4.198),
#   WORKERS (Ultralytics' default), LABEL (where it runs, for the result files; default "pod <GPU name>"),
#   SERVE_PORT (when set, OUT_DIR is served over HTTP on that port from the start, so log.txt can be followed and
#               results.tgz downloaded; the pod is terminated from outside).
# It needs only this folder's files (datasets.py, scorer.py, common.py, run_ultralytics.py) next to it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/workspace/finetune}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "start $(date -u +%FT%TZ)"
if [ -n "${SERVE_PORT:-}" ]; then
    (cd "$OUT" && python3 -m http.server "$SERVE_PORT" > /dev/null 2>&1) &
fi

python3 -m pip install -q "ultralytics==${ULTRALYTICS_VERSION:-8.4.198}" pycocotools requests
# opencv-python needs libGL, which plain CUDA images lack; the headless build has the same cv2.
python3 -m pip uninstall -y -q opencv-python || true
python3 -m pip install -q opencv-python-headless
python3 -m pip freeze > "$OUT/pip-freeze.txt"
python3 -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ultralytics', ultralytics.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python3 "$HERE/datasets.py" fetch --root "$OUT/data" ${DATASETS:+--only "$DATASETS"}
for pass in ${PASSES:-0,1,2:1}; do
    echo "pass seeds ${pass%%:*}, ${pass##*:} at a time: $(date -u +%FT%TZ)"
    python3 "$HERE/run_ultralytics.py" \
        --data-root "$OUT/data" --work-dir "$OUT/work" --results "$OUT/results" \
        --weights "$OUT/weights/${WEIGHTS:-yolo26n.pt}" --download-weights --device 0 \
        --seeds "${pass%%:*}" --jobs "${pass##*:}" --order "${ORDER:-seed}" --epochs "${EPOCHS:-100}" \
        ${DATASETS:+--datasets "$DATASETS"} \
        ${WORKERS:+--workers "$WORKERS"} --compute-label "${LABEL:-pod $GPU}" || echo "some runs failed (see above)"
done

tar czf "$OUT/results.tgz" -C "$OUT" results log.txt pip-freeze.txt gpu.txt work/logs
echo "done $(date -u +%FT%TZ): $OUT/results.tgz"
touch "$OUT/DONE"
if [ -n "${SERVE_PORT:-}" ]; then
    wait
fi
