"""Confidence gates on a refiner's per-side correction, from saved distributions (evaluate.py --save-boxes writes
<boxes>-dist.npz: per side the probe scale s, the mean E[t] and the spread std[t] of the 17-bin softmax).

    python gate.py <dump dir> <gt json> <refined boxes .npy> [--z 0.5 1 1.5] [--js 1 2]

Variants: `z<c>` moves a side only where |E[t]| > c * std[t] (hard gate); `js<c>` scales each side's move by the
positive part of 1 - c * std^2 / E^2 (James-Stein-like shrinkage). Prints COCO AP of each against B*.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fusion"))

import coco  # noqa: E402
from data import FeatureDump  # noqa: E402
from evaluate import detections  # noqa: E402

N = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("gt")
    ap.add_argument("refined")
    ap.add_argument("--z", type=float, nargs="*", default=[0.5, 1.0, 1.5])
    ap.add_argument("--js", type=float, nargs="*", default=[1.0, 2.0])
    a = ap.parse_args()
    d = FeatureDump(a.dump, os.path.join(a.dump, "boxes.npz"))
    truth = coco.Truth(a.gt)
    files = [m["file"] for m in d.images]
    box = np.asarray(d.t["box"])
    z = np.load(a.refined[:-4] + "-dist.npz")
    s, mu, sd = z["scale"], z["mean"], z["std"]
    ratio = np.abs(mu) / np.maximum(sd, 1e-6)
    print(f"|E|/std quantiles 10/50/90%: {np.percentile(ratio, [10, 50, 90]).round(3)}")
    variants = {}
    for c in a.z:
        variants[f"z{c}"] = np.where(ratio > c, mu, 0.0)
    for c in a.js:
        variants[f"js{c}"] = mu * np.maximum(0.0, 1 - c * sd ** 2 / np.maximum(mu ** 2, 1e-12))
    for name, r in variants.items():
        b = (box + N * s * r).astype(np.float32)
        st = coco.evaluate(truth, coco.detections_array(truth, files, detections(d, b))).stats
        moved = (r != 0).mean()
        print(f"{name:<8} sides moved {moved:.3f}  AP {st[0]:.4f} AP50 {st[1]:.4f} AP75 {st[2]:.4f} "
              f"S {st[3]:.4f} M {st[4]:.4f} L {st[5]:.4f}", flush=True)


if __name__ == "__main__":
    main()
