"""Where a refiner helps and hurts: IoU with the matched truth before and after, by score band and by the IoU before.

    python breakdown.py <dump dir> <refined boxes .npy> [--min-iou 0.3]

Rows are the B* boxes whose same-class truth overlaps at IoU >= --min-iou (the truth of the B* box, as boxes.py
matched it). Per band: boxes, mean IoU before / after, the share improved, the share that crosses 0.5 / 0.75
upward and downward (the thresholds AP50 and AP75 count).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import table  # noqa: E402
from refiner import box_iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("refined")
    ap.add_argument("--min-iou", type=float, default=0.3)
    a = ap.parse_args()
    t = table(os.path.join(a.dump, "boxes.npz"))
    ref = np.load(a.refined)
    m = np.asarray(t["gt_iou"]) >= a.min_iou
    gt = np.asarray(t["gt"])[m]
    i0 = box_iou(torch.from_numpy(np.asarray(t["box"])[m]), torch.from_numpy(gt)).numpy()
    i1 = box_iou(torch.from_numpy(ref[m]), torch.from_numpy(gt)).numpy()
    sc = np.asarray(t["score"])[m]

    def show(label, sel):
        if not sel.any():
            return
        a0, a1 = i0[sel], i1[sel]
        print(f"{label:<16} n {sel.sum():>7}  IoU {a0.mean():.4f} -> {a1.mean():.4f} ({a1.mean() - a0.mean():+.4f})  "
              f"improved {(a1 > a0).mean():.3f}  0.5 up/down {((a0 < .5) & (a1 >= .5)).mean():.3f}/"
              f"{((a0 >= .5) & (a1 < .5)).mean():.3f}  0.75 up/down {((a0 < .75) & (a1 >= .75)).mean():.3f}/"
              f"{((a0 >= .75) & (a1 < .75)).mean():.3f}")

    print("by score")
    for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        show(f"[{lo}, {hi})", (sc >= lo) & (sc < hi))
    print("by IoU before")
    for lo, hi in ((0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        show(f"[{lo}, {hi})", (i0 >= lo) & (i0 < hi))
    print("score >= 0.25 by IoU before")
    for lo, hi in ((0.3, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        show(f"[{lo}, {hi})", (sc >= 0.25) & (i0 >= lo) & (i0 < hi))


if __name__ == "__main__":
    main()
