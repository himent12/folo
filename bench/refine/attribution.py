"""Which refined boxes move AP: score B* with the refinement applied to all kept boxes, only to those with a
same-class truth at IoU >= 0.3 (the boxes training sees), or only to the others, and with each refined box's
change capped.

    python attribution.py <dump dir> <gt json> <refined boxes .npy> [--name <label>]
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
    ap.add_argument("--only", nargs="+", help="score only these variants")
    a = ap.parse_args()
    d = FeatureDump(a.dump, os.path.join(a.dump, "boxes.npz"))
    truth = coco.Truth(a.gt)
    files = [m["file"] for m in d.images]
    box = np.asarray(d.t["box"])
    ref = np.load(a.refined)
    m = np.asarray(d.t["gt_iou"]) >= 0.3
    move = np.abs(ref - box)
    w = np.maximum(box[:, 2] - box[:, 0], 1)
    h = np.maximum(box[:, 3] - box[:, 1], 1)
    rel = (move / np.stack([w, h, w, h], 1)).max(1)
    print(f"boxes {len(box)}, matched {m.sum()}; relative move: matched median {np.median(rel[m]):.4f} "
          f"p99 {np.percentile(rel[m], 99):.4f}; unmatched median {np.median(rel[~m]):.4f} "
          f"p99 {np.percentile(rel[~m], 99):.4f}")
    inv = (ref[:, 2] <= ref[:, 0]) | (ref[:, 3] <= ref[:, 1])
    print(f"inverted or empty refined boxes: {inv.sum()} ({inv[m].sum()} matched)")
    variants = {
        "Bstar": box,
        "all": ref,
        "matched-only": np.where(m[:, None], ref, box),
        "unmatched-only": np.where(m[:, None], box, ref),
    }
    sc = np.asarray(d.t["score"])
    for lo, hi in ((0, 0.05), (0.05, 0.25), (0.25, 1.01)):
        sel = (sc >= lo) & (sc < hi)
        variants[f"score[{lo},{hi})"] = np.where(sel[:, None], ref, box)
    gi = np.asarray(d.t["gt_iou"])
    for lo, hi in ((0.3, 0.6), (0.6, 0.8), (0.8, 1.01)):
        sel = (gi >= lo) & (gi < hi)
        variants[f"gt_iou[{lo},{hi})"] = np.where(sel[:, None], ref, box)
    # Oracle gates (upper bounds for any gating of this refiner): refine only the matched boxes it improves.
    import torch
    from refiner import box_iou
    gt = np.asarray(d.t["gt"])
    i0 = box_iou(torch.from_numpy(box), torch.from_numpy(gt)).numpy()
    i1 = box_iou(torch.from_numpy(ref), torch.from_numpy(gt)).numpy()
    better = m & (i1 > i0)
    variants["oracle-gate"] = np.where(better[:, None], ref, box)
    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only}
    for name, b in variants.items():
        ev = coco.evaluate(truth, coco.detections_array(truth, files, detections(d, b.astype(np.float32))))
        s = ev.stats
        print(f"{name:<16} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S {s[3]:.4f} M {s[4]:.4f} L {s[5]:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
