"""Signed edge residuals (truth - B* box, outward positive, network pixels) of confident matched boxes, per side, to
compare the label conventions of two box tables (e.g. --aug training views against dev).

    python residuals.py <dump dir> [<dump dir> ...] [--min-score 0.25] [--min-iou 0.5]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import table  # noqa: E402

N = np.array([-1.0, -1.0, 1.0, 1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--min-iou", type=float, default=0.5)
    a = ap.parse_args()
    for d in a.dumps:
        t = table(os.path.join(d, "boxes.npz"))
        m = (np.asarray(t["score"]) >= a.min_score) & (np.asarray(t["gt_iou"]) >= a.min_iou)
        r = (np.asarray(t["gt"])[m] - np.asarray(t["box"])[m]) * N
        b = np.asarray(t["box"])[m]
        size = np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]] * 2, axis=1)[:, [0, 1, 2, 3]]
        print(f"{os.path.basename(d)}: {m.sum()} boxes")
        print("   mean residual px (l t r b):", np.round(r.mean(0), 3), " median:", np.round(np.median(r, 0), 3))
        print("   mean residual / size:      ", np.round((r / size).mean(0), 4))
        print("   mean |residual| px:        ", np.round(np.abs(r).mean(0), 3))
        with open(os.path.join(d, "images.jsonl"), encoding="utf-8") as f:
            ims = [json.loads(l) for l in f if l.strip()]
        if "zoom" in ims[0]:
            img = np.repeat(np.arange(len(ims)), np.asarray(t["img_count"]))[m]
            flip = np.array([im["flip"] for im in ims])[img]
            zoom = np.array([im["zoom"] for im in ims])[img]
            for lab, sel in (("flip", flip), ("no flip", ~flip), ("zoom < 0.9", zoom < 0.9),
                             ("zoom 0.9-1.1", (zoom >= 0.9) & (zoom < 1.1)), ("zoom > 1.1", zoom >= 1.1)):
                print(f"   {lab:<13} n {sel.sum():>6} mean residual px", np.round(r[sel].mean(0), 3))


if __name__ == "__main__":
    main()
