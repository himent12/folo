"""IoU change by true-IoU band on the first N images of several dumps (e.g. training views against dev), to tell
underfitting (tight boxes hurt on the training views too) from overfitting (only on new images).

    python bandcheck.py <refiner.pt> <dump dir> [<dump dir> ...] [--n 2000] [--min-score 0.25]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import FeatureDump, loader, to_device  # noqa: E402
from refiner import box_iou  # noqa: E402
from train import build  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--min-score", type=float, default=0.25)
    a = ap.parse_args()
    dev = torch.device("cuda")
    st = torch.load(a.model, map_location=dev)
    model = build(st["config"]).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    for path in a.dumps:
        d = FeatureDump(path, os.path.join(path, "boxes.npz"))
        i0, i1, gi = [], [], []
        for idx, pyr, rows, _ in loader(d, list(range(min(a.n, d.n))), 16, 0.5, 4):
            b = to_device(rows, dev)
            keep = b["score"] >= a.min_score
            if not keep.any():
                continue
            pyr = [torch.as_tensor(p).to(dev) for p in pyr]
            proj = model.project(pyr) if model.features else None
            logits, s, _ = model(proj, b)
            ref = model.refine(b["box"], logits, s)
            i0.append(box_iou(b["box"][keep], b["gt"][keep]).cpu().numpy())
            i1.append(box_iou(ref[keep], b["gt"][keep]).cpu().numpy())
        i0, i1 = np.concatenate(i0), np.concatenate(i1)
        print(f"{os.path.basename(path)}: {len(i0)} boxes (score >= {a.min_score}, IoU >= 0.5)")
        for lo, hi in ((0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
            sel = (i0 >= lo) & (i0 < hi)
            print(f"   IoU [{lo}, {hi}) n {sel.sum():>6}  {i0[sel].mean():.4f} -> {i1[sel].mean():.4f} "
                  f"({i1[sel].mean() - i0[sel].mean():+.4f})  improved {(i1[sel] > i0[sel]).mean():.3f}")


if __name__ == "__main__":
    main()
