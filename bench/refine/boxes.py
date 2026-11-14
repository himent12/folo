"""Lane refine's box tables: B* (lane fusion's post-process, bench/fusion/post.py) on a candidate dump, plus what the
refiner reads for each kept box.

    python boxes.py --dump <dir> [--gt <instances json>] --out <boxes.npz> [--workers 8]

B* = DFL evidence fusion (tau 0.75, a 1, b 1, eps 0.1, excess variance) after hard NMS at IoU 0.65, max_det 1000,
conf 0.001, exactly bench/fusion's `Bstar` configuration. Per kept box (in each image's B* order, by descending
score) the table holds:

- `img` (position in the dump), `cls`, `score`, `anchor`, `level` (of the kept candidate), `stride`;
- `box` the fused B* box and `box_kept` the kept candidate's own box, network pixels `x1 y1 x2 y2`;
- `probs` the kept candidate's 4 x 16 DFL probabilities (f16);
- `sigma` per side (l, t, r, b) the kept candidate's DFL width in pixels, `stride * sqrt(max(var - f(1 - f), 0) +
  0.1)` (MODEL.md §9.4: `1 / sqrt(precision)`), and `sigma_fused`, `1 / sqrt(sum over the cluster of score_j *
  precision_j / sum score_j)` (the score-weighted mean precision of the members);
- `spread` per side: the members' weighted standard deviation around the fused edge (pixels; the fusion weights),
  and `members` the cluster size;
- `gt` the same-class truth it overlaps most (network pixels, crowd excluded) and `gt_iou` (0 and -1 boxes when
  there is none). Truths come from `--gt` (a COCO json; mapped with the letterbox) or, for an `--aug` dump, from
  the dump's own `targets` (clipped to the part of the image the view shows; see truths_of).

Per image: `img_start` / `img_count` into the box rows, the letterbox (`ratio`, `pad`, `orig`) and `valid`, the
image's area in network pixels (after the view's flip and zoom, clipped to the input).
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fusion"))

from dump import Dump  # noqa: E402
import post  # noqa: E402

BSTAR = {"iou": 0.65, "max_det": 1000, "fuse": {"kind": "var", "tau": 0.75, "a": 1.0, "b": 1.0, "eps": 0.1,
                                                "excess": True}}
EPS = 0.1

_dump = None
_gt = None


def _init(path, gt):
    global _dump, _gt
    _dump = Dump(path)
    _gt = gt


def bstar_kept(im):
    """B*'s kept candidates: (candidate anchors, classes, scores, kept positions) as post.postprocess computes them."""
    conf, max_cand = 0.001, 30000
    ca, cc, cs = im.pair_anchor, im.pair_class, im.pair_score
    ok = cs > np.float32(conf)
    ca, cc, cs = ca[ok], cc[ok], cs[ok]
    if len(cs) > max_cand:
        top = np.sort(np.lexsort((cc, ca, -cs.astype(np.float64)))[:max_cand])
        ca, cc, cs = ca[top], cc[top], cs[top]
    kept = post.batched_nms(im.box_net[ca], cs, cc, BSTAR["iou"], BSTAR["max_det"])
    return ca, cc, cs, kept


def side_moments(logits):
    p, mean, var = post.side_stats(logits)
    fr = mean - np.floor(mean)
    ex = np.maximum(var - fr * (1 - fr), 0.0)
    return p, ex


def table(im, truths):
    """The box rows of one image (dict of arrays)."""
    ca, cc, cs, kept = bstar_kept(im)
    fused = post.fuse(im, im.box_net, kept, ca, cc, cs, BSTAR["fuse"])
    k = len(kept)
    ka = ca[kept]
    p, ex = side_moments(im.logits)
    stride = im.stride.astype(np.float64)
    prec = 1.0 / ((stride ** 2)[:, None] * (ex + EPS))  # per candidate anchor, per side
    out = {
        "cls": cc[kept].astype(np.int16), "score": cs[kept].astype(np.float32), "anchor": im.anchor[ka].astype(np.int32),
        "level": im.level[ka].astype(np.int8), "stride": im.stride[ka].astype(np.float32),
        "box": fused.astype(np.float32), "box_kept": im.box_net[ka].astype(np.float32),
        "probs": p[ka].astype(np.float16),
        "sigma": (stride[ka][:, None] * np.sqrt(ex[ka] + EPS)).astype(np.float32),
    }
    sig_f = np.zeros((k, 4))
    spread = np.zeros((k, 4))
    members = np.zeros(k, dtype=np.int32)
    tau = np.float32(BSTAR["fuse"]["tau"])
    for c in np.unique(cc[kept]):
        kk = np.nonzero(cc[kept] == c)[0]
        mem = np.nonzero(cc == c)[0]
        iou = post.iou_matrix(im.box_net[ca[kept[kk]]], im.box_net[ca[mem]])
        inside = iou >= tau
        s = np.where(inside, cs[mem][None, :].astype(np.float64), 0.0)  # K x M
        pm = prec[ca[mem]]  # M x 4
        w = s[:, :, None] * pm[None, :, :]  # K x M x 4
        sw = w.sum(axis=1)
        sig_f[kk] = 1.0 / np.sqrt(np.maximum(sw / np.maximum(s.sum(axis=1), 1e-12)[:, None], 1e-12))
        x = im.box_net[ca[mem]].astype(np.float64)  # M x 4
        d = x[None, :, :] - fused[kk].astype(np.float64)[:, None, :]
        spread[kk] = np.sqrt((w * d * d).sum(axis=1) / np.maximum(sw, 1e-12))
        members[kk] = inside.sum(axis=1)
    out["sigma_fused"] = sig_f.astype(np.float32)
    out["spread"] = spread.astype(np.float32)
    out["members"] = members
    gt = np.full((k, 4), -1.0, dtype=np.float32)
    gt_iou = np.zeros(k, dtype=np.float32)
    tc, tb = truths
    for c in np.unique(out["cls"]):
        kk = np.nonzero(out["cls"] == c)[0]
        tt = np.nonzero(tc == c)[0]
        if len(tt) == 0:
            continue
        iou = post.iou_matrix(out["box"][kk], tb[tt])
        j = iou.argmax(axis=1)
        gt[kk] = tb[tt[j]]
        gt_iou[kk] = iou[np.arange(len(kk)), j]
    out["gt"] = gt
    out["gt_iou"] = gt_iou
    return out


def view_region(m, size=640.0):
    """The image's area in network pixels for this view (letterbox, then the --aug flip and zoom)."""
    ratio, pad, orig = m["ratio"], m["pad"], m["orig"]
    x1, y1 = pad[0], pad[1]
    x2, y2 = pad[0] + orig[0] * ratio[0], pad[1] + orig[1] * ratio[1]
    if "zoom" in m:
        z = m["zoom"]
        if m["flip"]:
            x1, x2 = size - x2, size - x1
        x1, x2 = z * (x1 - size / 2) + size / 2, z * (x2 - size / 2) + size / 2
        y1, y2 = z * (y1 - size / 2) + size / 2, z * (y2 - size / 2) + size / 2
    return [max(x1, 0.0), max(y1, 0.0), min(x2, size), min(y2, size)]


def truths_of(m, gt):
    """(classes, network-pixel boxes) of one dump image."""
    if "targets" in m:
        # A zoomed-in view crops the image: the truth is the part of the object inside the view (clipped to the
        # visible region), and a truth with less than a fifth of its area visible is dropped.
        t = np.array(m["targets"], dtype=np.float32).reshape(-1, 5)
        v = view_region(m)
        b = t[:, 1:5].copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(v[0], v[2])
        b[:, [1, 3]] = b[:, [1, 3]].clip(v[1], v[3])
        area = lambda x: np.maximum(x[:, 2] - x[:, 0], 0) * np.maximum(x[:, 3] - x[:, 1], 0)  # noqa: E731
        keep = (area(b) >= 0.2 * area(t[:, 1:5])) & (b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)
        return t[keep, 0].astype(np.int64), b[keep]
    if gt is None:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    anns = gt["by_file"].get(m["file"], [])
    rx, ry = m["ratio"]
    px, py = m["pad"]
    cls, boxes = [], []
    for a in anns:
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        cls.append(gt["cat_index"][a["category_id"]])
        boxes.append([x * rx + px, y * ry + py, (x + w) * rx + px, (y + h) * ry + py])
    return np.array(cls, dtype=np.int64), np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _work(rng):
    lo, hi = rng
    rows = []
    for i in range(lo, hi):
        im = _dump.image(i)
        t = table(im, truths_of(_dump.images[i], _gt))
        t["img"] = np.full(len(t["cls"]), i, dtype=np.int32)
        rows.append(t)
    return rows


def load_gt(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    files = {im["id"]: im["file_name"] for im in g["images"]}
    by_file = {}
    for a in g["annotations"]:
        by_file.setdefault(files[a["image_id"]], []).append(a)
    cats = sorted(c["id"] for c in g["categories"])
    return {"by_file": by_file, "cat_index": {c: k for k, c in enumerate(cats)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    gt = load_gt(a.gt) if a.gt else None
    d = Dump(a.dump)
    n = len(d)
    chunks = [(lo, min(lo + 100, n)) for lo in range(0, n, 100)]
    parts = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.dump, gt)) as ex:
        for k, rows in enumerate(ex.map(_work, chunks)):
            parts.extend(rows)
            if k % 20 == 0:
                print(f"{chunks[k][1]} / {n} images", flush=True)
    cols = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    counts = np.array([len(p["cls"]) for p in parts], dtype=np.int64)
    cols["img_count"] = counts
    cols["img_start"] = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    cols["ratio"] = np.array([m["ratio"] for m in d.images], dtype=np.float32)
    cols["pad"] = np.array([m["pad"] for m in d.images], dtype=np.float32)
    cols["orig"] = np.array([m["orig"] for m in d.images], dtype=np.float32)
    cols["valid"] = np.array([view_region(m) for m in d.images], dtype=np.float32)
    np.savez(a.out, **cols)
    print(f"{n} images, {len(cols['cls'])} boxes, {(cols['gt_iou'] >= 0.3).sum()} with a truth at IoU >= 0.3")


if __name__ == "__main__":
    main()
