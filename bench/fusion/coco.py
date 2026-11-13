"""pycocotools scoring of post-processed dumps, and a paired bootstrap over images.

`detections_array` turns per-image (boxes, scores, classes) into loadRes' N x 7 array exactly as folo writes its
results file (crates/folo-cli/src/coco.rs write_results: x, y in f64 of the f32 corner, w = x2 - x1 in f64, score
f64 of the f32). `evaluate` runs COCOeval and returns the 12 stats plus the per-image match records the bootstrap
needs. `bootstrap` recomputes AP (IoU 0.50:0.95, area all, 100 detections) on image resamples, paired across
configurations, from those records, with pycocotools' accumulate arithmetic (a resampled image counts as many
times as it is drawn).
"""

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class Truth:
    def __init__(self, ann_file):
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(ann_file)
        self.cat_ids = sorted(self.coco.getCatIds())
        self.id_of_file = {im["file_name"]: im["id"] for im in self.coco.dataset["images"]}
        self.img_ids = sorted(self.coco.getImgIds())


def detections_array(truth, files, results):
    rows = []
    cats = np.array(truth.cat_ids, dtype=np.float64)
    for file, (boxes, scores, classes) in zip(files, results):
        if len(scores) == 0:
            continue
        img = truth.id_of_file[file]
        b = boxes.astype(np.float64)
        r = np.empty((len(scores), 7))
        r[:, 0] = img
        r[:, 1] = b[:, 0]
        r[:, 2] = b[:, 1]
        r[:, 3] = b[:, 2] - b[:, 0]
        r[:, 4] = b[:, 3] - b[:, 1]
        r[:, 5] = scores.astype(np.float64)
        r[:, 6] = cats[classes]
        rows.append(r)
    return np.concatenate(rows) if rows else np.zeros((0, 7))


def evaluate(truth, dets, quiet=True):
    sink = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        res = truth.coco.loadRes(dets)
        ev = COCOeval(truth.coco, res, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def match_records(ev):
    """Per category, for area 'all' and maxDet 100: the detections (score, image position, matched per IoU
    threshold, ignored per threshold) and the number of non-ignored truths per image."""
    p = ev.params
    n_img = len(p.imgIds)
    n_area = len(p.areaRng)
    a0 = 0  # area 'all'
    md = p.maxDets[-1]
    img_pos = {i: k for k, i in enumerate(p.imgIds)}
    recs = []
    for ci, _ in enumerate(p.catIds):
        E = [ev.evalImgs[ci * n_area * n_img + a0 * n_img + i] for i in range(n_img)]
        scores, imgs, tp, ig = [], [], [], []
        npig = np.zeros(n_img, dtype=np.int64)
        for i, e in enumerate(E):
            if e is None:
                continue
            s = np.asarray(e["dtScores"][:md])
            scores.append(s)
            imgs.append(np.full(len(s), i, dtype=np.int64))
            tp.append(np.asarray(e["dtMatches"])[:, :md] != 0)
            ig.append(np.asarray(e["dtIgnore"])[:, :md].astype(bool))
            npig[i] = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
        if not scores:
            recs.append(None)
            continue
        scores = np.concatenate(scores)
        order = np.argsort(-scores, kind="mergesort")
        recs.append({
            "imgs": np.concatenate(imgs)[order],
            "tp": np.concatenate(tp, axis=1)[:, order],
            "ig": np.concatenate(ig, axis=1)[:, order],
            "npig": npig,
        })
    return recs, img_pos


def ap_from_records(recs, weights, rec_thrs):
    """pycocotools' accumulate (precision at 101 recall points, mean over categories with truths) with image
    weights (counts in a resample)."""
    aps = []
    for r in recs:
        if r is None:
            continue
        npig = float((r["npig"] * weights).sum())
        if npig == 0:
            continue
        w = weights[r["imgs"]].astype(np.float64)
        tps = np.logical_and(r["tp"], ~r["ig"]) * w
        fps = np.logical_and(~r["tp"], ~r["ig"]) * w
        tp_sum = np.cumsum(tps, axis=1)
        fp_sum = np.cumsum(fps, axis=1)
        # Detections of weight 0 must not add precision points: keep only columns with weight > 0.
        keep = w > 0
        tp_sum, fp_sum = tp_sum[:, keep], fp_sum[:, keep]
        T, nd = tp_sum.shape
        prec_all = np.zeros((T, len(rec_thrs)))
        if nd:
            rc = tp_sum / npig
            pr = tp_sum / (fp_sum + tp_sum + np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            for t in range(T):
                inds = np.searchsorted(rc[t], rec_thrs, side="left")
                valid = inds < nd
                prec_all[t, valid] = pr[t, inds[valid]]
        aps.append(prec_all.mean())
    return float(np.mean(aps)) if aps else -1.0


def bootstrap(rec_sets, n_img, n=1000, seed=0):
    """AP of each record set on the same n image resamples. Returns an (n, k) array."""
    rng = np.random.default_rng(seed)
    rec_thrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
    out = np.zeros((n, len(rec_sets)))
    for b in range(n):
        wts = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
        for k, recs in enumerate(rec_sets):
            out[b, k] = ap_from_records(recs, wts, rec_thrs)
    return out
