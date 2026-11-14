"""COCO AP of B* + lambda * (refined - B*) for a few lambda, from saved refined boxes (CPU only).

    python shrink.py <dump dir> <gt json> <refined boxes .npy> [--lam 0.25 0.5 0.75]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("gt")
    ap.add_argument("refined")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    a = ap.parse_args()
    d = FeatureDump(a.dump, os.path.join(a.dump, "boxes.npz"))
    truth = coco.Truth(a.gt)
    files = [m["file"] for m in d.images]
    box = np.asarray(d.t["box"])
    ref = np.load(a.refined)
    for lam in a.lam:
        b = (box + lam * (ref - box)).astype(np.float32)
        s = coco.evaluate(truth, coco.detections_array(truth, files, detections(d, b))).stats
        print(f"lambda {lam:<5} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S {s[3]:.4f} M {s[4]:.4f} "
              f"L {s[5]:.4f}", flush=True)


if __name__ == "__main__":
    main()
