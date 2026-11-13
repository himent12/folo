"""The user-facing operating point: what `folo predict` returns at conf 0.25 (single-label, NMS IoU 0.7, 300
detections), with and without box fusion, against the val2017 truths.

Per configuration: detections, true positives at IoU 0.5 (each truth matched once, greedily by score, per class,
crowd truths and their matches left out as COCO does), precision at IoU 0.5 and at 0.75 (the share of detections that
match a truth that well), recall at 0.5, and the mean IoU of the true positives at 0.5 (how tight the boxes a user
sees are). Usage:

    python operating.py --dump <val dump> --gt instances_val2017.json --configs <json> [--workers 8]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dump import Dump
from post import postprocess

_D = None
_CFGS = None


def _init(dump_path, cfgs):
    global _D, _CFGS
    _D, _CFGS = Dump(dump_path), cfgs


def _post(i):
    im = _D.image(i)
    return im.file, [postprocess(im, c) for c in _CFGS]


def iou_xyxy(a, b):
    """IoU of every box in a (n x 4) with every box in b (m x 4), f64."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match(boxes, scores, classes, gt_boxes, gt_classes, gt_crowd):
    """Per detection: the IoU with the truth it matches at 0.5 (0 if none), and whether it matched a crowd region
    (then it is neither a true nor a false positive)."""
    n = len(scores)
    best = np.zeros(n)
    ignore = np.zeros(n, dtype=bool)
    for c in np.unique(classes):
        d = np.nonzero(classes == c)[0]
        d = d[np.argsort(-scores[d], kind="stable")]
        g = np.nonzero(gt_classes == c)[0]
        if len(g) == 0:
            continue
        iou = iou_xyxy(boxes[d], gt_boxes[g])
        crowd = gt_crowd[g]
        used = np.zeros(len(g), dtype=bool)
        for r, k in enumerate(d):
            cand = np.where(~used & ~crowd, iou[r], -1.0)
            j = int(np.argmax(cand)) if len(cand) else -1
            if j >= 0 and cand[j] >= 0.5:
                used[j] = True
                best[k] = cand[j]
            elif crowd.any() and np.max(np.where(crowd, iou[r], 0.0)) >= 0.5:
                ignore[k] = True
    return best, ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfgs = json.load(open(a.configs)) if os.path.exists(a.configs) else json.loads(a.configs)
    gt = json.load(open(a.gt, encoding="utf-8"))
    cats = sorted(c["id"] for c in gt["categories"])
    cls_of = {c: i for i, c in enumerate(cats)}
    id_of = {im["file_name"]: im["id"] for im in gt["images"]}
    truths = {}
    for an in gt["annotations"]:
        x, y, w, h = an["bbox"]
        truths.setdefault(an["image_id"], []).append((x, y, x + w, y + h, cls_of[an["category_id"]], an["iscrowd"]))
    n_truth = sum(1 for an in gt["annotations"] if not an["iscrowd"])
    d = Dump(a.dump)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.dump, cfgs)) as ex:
        results = list(ex.map(_post, range(len(d)), chunksize=16))
    rows = []
    for k, c in enumerate(cfgs):
        n = tp = tp75 = ign = 0
        ious = []
        for file, outs in results:
            boxes, scores, classes = outs[k]
            t = np.array(truths.get(id_of[file], []), dtype=np.float64).reshape(-1, 6)
            best, ignore = match(boxes, scores, classes, t[:, :4], t[:, 4].astype(np.int64), t[:, 5] > 0)
            n += len(scores)
            ign += int(ignore.sum())
            tp += int((best >= 0.5).sum())
            tp75 += int((best >= 0.75).sum())
            ious.extend(best[best >= 0.5].tolist())
        scored = n - ign
        row = {"name": c["name"], "detections": n, "crowd_matched": ign, "tp50": tp,
               "precision50": tp / max(scored, 1), "precision75": tp75 / max(scored, 1), "recall50": tp / n_truth,
               "mean_iou_tp": float(np.mean(ious)) if ious else 0.0}
        rows.append(row)
        print(f"{c['name']:<12} dets {n:7d}  P50 {row['precision50']:.4f}  P75 {row['precision75']:.4f}  "
              f"R50 {row['recall50']:.4f}  mean IoU of TPs {row['mean_iou_tp']:.4f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
