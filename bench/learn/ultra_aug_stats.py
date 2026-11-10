"""The incumbent's side of crates/folo-data/examples/aug_stats.rs: boxes per image and box-size quantiles of
Ultralytics' training transform (default hyperparameters) on a YOLO-layout data file.

    YOLO_CONFIG_DIR=<scratch> python bench/learn/ultra_aug_stats.py --data mini320.yaml --imgsz 320 --samples 4000 [--no-mosaic]
"""

import argparse
import json
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--no-mosaic", action="store_true")
    a = ap.parse_args()
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    cfg = get_cfg(overrides={"imgsz": a.imgsz, "mosaic": 0.0 if a.no_mosaic else 1.0})
    data = check_det_dataset(a.data, autodownload=False)
    ds = build_yolo_dataset(cfg, data["train"], 64, data, mode="train", stride=32)
    random.seed(0)
    np.random.seed(0)
    counts, side, ws, hs = [], [], [], []
    for _ in range(a.samples):
        s = ds[random.randrange(len(ds))]
        b = s["bboxes"].numpy()  # normalised xywh
        counts.append(len(b))
        if len(b):
            w, h = b[:, 2] * a.imgsz, b[:, 3] * a.imgsz
            side.extend(np.sqrt(w * h))
            ws.extend(w)
            hs.extend(h)
    side, ws, hs, counts = map(np.asarray, (side, ws, hs, counts))
    q = lambda v: [round(float(np.quantile(v, p, method="closest_observation")), 2) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
    print(json.dumps({
        "samples": a.samples, "mosaic": not a.no_mosaic, "boxes_per_image": round(float(counts.mean()), 3),
        "empty": round(float((counts == 0).mean()), 4), "boxes": int(len(side)), "side_q05_25_50_75_95": q(side),
        "w_q": q(ws), "h_q": q(hs), "side_lt_8_16_32": [round(float((side < t).mean()), 4) for t in (8, 16, 32)],
    }))


if __name__ == "__main__":
    main()
