"""Part 3 oracles on a candidate dump, against a post-processing configuration (report only).

    python oracles.py --dump <dir> --gt <instances json> --cfg <json> [--workers 8]

(a) recall ceiling: the share of non-crowd truths with any same-class candidate at IoU >= 0.5 / 0.75 (original-image
    pixels), and with any candidate at all of any class;
(b) cluster oracle: each kept detection's box is replaced by the box of the member of its cluster (same-class
    candidates with IoU >= tau to it, tau = the config's fusion tau or its NMS iou) that best overlaps the truth the
    kept box overlaps most; the score stays (the cluster's top score);
(c) box oracle: kept boxes moved 25 / 50 / 100% of the way to the same-class truth they overlap most (IoU >= 0.3);
(d) class oracle: a kept detection whose box overlaps a truth of another class at IoU >= 0.5 more than any truth of
    its own class takes that truth's class.
Each oracle is scored with pycocotools against the configuration itself.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coco  # noqa: E402
from dump import Dump  # noqa: E402
from post import iou_matrix, postprocess  # noqa: E402

_d = None
_gt = None


def _init(dump_path, gt_path):
    global _d, _gt
    _d = Dump(dump_path)
    with open(gt_path, encoding="utf-8") as f:
        g = json.load(f)
    cats = sorted(c["id"] for c in g["categories"])
    cls_of = {c: i for i, c in enumerate(cats)}
    id_of = {im["file_name"]: im["id"] for im in g["images"]}
    by = {}
    for a in g["annotations"]:
        if a.get("iscrowd", 0):
            continue
        x, y, w, h = a["bbox"]
        by.setdefault(a["image_id"], []).append((x, y, x + w, y + h, cls_of[a["category_id"]]))
    _gt = (id_of, by)


def _truth(file):
    id_of, by = _gt
    t = np.array(by.get(id_of[file], []), dtype=np.float64).reshape(-1, 5)
    return t[:, :4].astype(np.float32), t[:, 4].astype(np.int64)


def _work(args):
    cfg, lo, hi = args
    out = {"recall": np.zeros(5), "base": [], "cluster": [], "box25": [], "box50": [], "box100": [], "cls": []}
    tau = np.float32((cfg.get("fuse") or {}).get("tau", cfg.get("iou", 0.7)))
    for i in range(lo, hi):
        im = _d.image(i)
        tb, tc = _truth(im.file)
        cand = im.box_orig[im.pair_anchor]
        # (a) recall ceiling
        if len(tb):
            iou = iou_matrix(tb, cand) if len(cand) else np.zeros((len(tb), 0), np.float32)
            same = tc[:, None] == im.pair_class[None, :]
            best_same = np.where(same, iou, 0).max(axis=1) if iou.shape[1] else np.zeros(len(tb))
            best_any = iou.max(axis=1) if iou.shape[1] else np.zeros(len(tb))
            out["recall"] += [len(tb), (best_same >= 0.5).sum(), (best_same >= 0.75).sum(), (best_any >= 0.5).sum(),
                              (best_any >= 0.75).sum()]
        boxes, scores, classes = postprocess(im, cfg)
        out["base"].append((boxes, scores, classes))
        if len(boxes) == 0 or len(tb) == 0:
            for k in ("cluster", "box25", "box50", "box100", "cls"):
                out[k].append((boxes, scores, classes))
            continue
        kt = iou_matrix(boxes, tb)  # K x T
        same = classes[:, None] == tc[None, :]
        kt_same = np.where(same, kt, -1)
        gi = kt_same.argmax(axis=1)
        gmax = kt_same[np.arange(len(boxes)), gi]
        # (b) cluster oracle
        cb = boxes.copy()
        for k in range(len(boxes)):
            if gmax[k] <= 0:
                continue
            mem = np.nonzero(im.pair_class == classes[k])[0]
            mb = cand[mem]
            inc = iou_matrix(boxes[k:k + 1], mb)[0] >= tau
            if not inc.any():
                continue
            mb = mb[inc]
            mt = iou_matrix(mb, tb[gi[k]:gi[k] + 1])[:, 0]
            j = mt.argmax()
            if mt[j] > gmax[k]:
                cb[k] = mb[j]
        out["cluster"].append((cb, scores, classes))
        # (c) box oracle
        for frac, key in ((0.25, "box25"), (0.5, "box50"), (1.0, "box100")):
            bb = boxes.copy()
            m = gmax >= 0.3
            bb[m] = boxes[m] + np.float32(frac) * (tb[gi[m]] - boxes[m])
            out[key].append((bb, scores, classes))
        # (d) class oracle
        other = np.where(~same, kt, -1)
        oi = other.argmax(axis=1)
        omax = other[np.arange(len(boxes)), oi]
        cc = classes.copy()
        flip = (omax >= 0.5) & (omax > gmax)
        cc[flip] = tc[oi[flip]]
        out["cls"].append((boxes, scores, cc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.loads(a.cfg) if not os.path.exists(a.cfg) else json.load(open(a.cfg, encoding="utf-8"))
    d = Dump(a.dump)
    n = len(d)
    files = [m["file"] for m in d.images]
    jobs = [(cfg, lo, min(lo + 125, n)) for lo in range(0, n, 125)]
    keys = ["base", "cluster", "box25", "box50", "box100", "cls"]
    res = {k: [] for k in keys}
    rec = np.zeros(5)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.dump, a.gt)) as ex:
        for part in ex.map(_work, jobs):
            rec += part["recall"]
            for k in keys:
                res[k].extend(part[k])
    truth = coco.Truth(a.gt)
    print(f"(a) recall ceiling over {int(rec[0])} non-crowd truths: same class IoU>=0.5 {rec[1] / rec[0]:.4f}, "
          f">=0.75 {rec[2] / rec[0]:.4f}; any class >=0.5 {rec[3] / rec[0]:.4f}, >=0.75 {rec[4] / rec[0]:.4f}")
    rows = {"recall": (rec / rec[0]).tolist()}
    for k in keys:
        ev = coco.evaluate(truth, coco.detections_array(truth, files, res[k]))
        s = [float(x) for x in ev.stats]
        rows[k] = s
        print(f"{k:<8} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S {s[3]:.4f} M {s[4]:.4f} L {s[5]:.4f}", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"cfg": cfg, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
