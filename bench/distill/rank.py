"""Gate step 0 of class-wise rank distillation (hill-climb 5, docs/notes/hill/r5-accuracy.md).

Three CPU measurements on hill-climb 3's val2017 candidate dumps (student folo-n-coco-v3, teacher folo-s-coco), all
on the STUDENT's candidate set with the student's boxes, default post-processing (hard NMS 0.7, 300 detections):

  s        the student as dumped
  vswap    at shared pairs (the same anchor and class scored > 0.001 by both), the teacher's score
  rswap    (a) rank swap: per class over the whole val set, the student's own shared-pair scores, re-assigned in
           the teacher's order of those pairs (the student's values, the teacher's within-class ranking)
  rswap16  the same with the ranking per class within each group of 16 consecutive images (a training batch's scope)
  band_hi  (b) the teacher's score at shared pairs where the student's score is >= 0.1
  band_lo  (b) the teacher's score at shared pairs where the student's score is < 0.1
  rband_hi / rband_lo  the rank swap restricted to the same bands (student values within the band)

(c) The weighted Pearson correlation rho_c of the two models' class logits, as the training term computes it
(docs/MODEL.md §8.9, rank term): per element w = max(p, p~)^gamma above tau = 0.005 (else 0), logits floored at
logit(tau); an element missing from one dump has a score below 0.001 there, so its floored logit is logit(tau).
Per class over each group of 16 images (used iff n >= 8 and var_y > 1e-4) and per class over the whole set.

    python bench/distill/rank.py [--workers 10] [--skip-ap]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fusion"))
import coco  # noqa: E402
from dump import Dump  # noqa: E402
from post import postprocess  # noqa: E402

S = "D:/folo-data/runs-hill3/fusion/dumps/land-val"
T = "D:/folo-data/runs-hill3/fusion/dumps/s-val"
GT = "D:/folo-data/coco/annotations/instances_val2017.json"
NC = 80
GROUP = 16
TAU = 0.005
EPS = 1e-4
RANK_MIN = 8
CFG = {}
_s = _t = None
_new = None  # config name -> list of per-image new student pair scores (only in the post workers)


def _init():
    global _s, _t
    _s, _t = Dump(S), Dump(T)


def _keys(im):
    return im.anchor[im.pair_anchor] * NC + im.pair_class


def _shared(i):
    """Per image: student pair positions, classes, student and teacher scores of the shared pairs, plus the union
    elements (class, student score or 0, teacher score or 0) for the correlation."""
    a, b = _s.image(i), _t.image(i)
    assert a.file == b.file
    ka, kb = _keys(a), _keys(b)
    _, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    # Union of pairs for rho: every key in either dump.
    keys = np.union1d(ka, kb)
    ps = np.zeros(len(keys), np.float64)
    pt = np.zeros(len(keys), np.float64)
    ps[np.searchsorted(keys, ka)] = a.pair_score
    pt[np.searchsorted(keys, kb)] = b.pair_score
    return (ia, a.pair_class[ia], a.pair_score[ia].astype(np.float64), b.pair_score[ib].astype(np.float64),
            (keys % NC).astype(np.int64), ps, pt, a.pair_score.sum(), a.pair_score[ia].sum(), len(ka), len(ia))


def _post_init(new):
    global _new
    _init()
    _new = new


def _post(args):
    name, lo, hi = args
    out = []
    for i in range(lo, hi):
        im = _s.image(i)
        if name != "s":
            im.pair_score = _new[name][i]
        out.append(postprocess(im, CFG))
    return out


def rank_assign(cls, s, t, groups):
    """Per (group, class): the sorted student values placed in the teacher's descending order."""
    new = s.copy()
    key = groups * NC + cls
    order = np.lexsort((-t, key))  # by group-class, then teacher score descending (stable)
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    for a, b in zip(starts, ends):
        idx = order[a:b]
        new[idx] = np.sort(s[idx])[::-1]
    return new


def weighted_rho(cls, x, y, w, groups):
    """Per (group, class) weighted Pearson correlation as the training term: returns rho, W, n, used."""
    key = groups * NC + cls
    uk, inv = np.unique(key, return_inverse=True)
    m = len(uk)
    W = np.bincount(inv, w, m)
    n = np.bincount(inv, (w > 0).astype(np.float64), m)
    Ws = np.where(W > 0, W, 1.0)
    mx = np.bincount(inv, w * x, m) / Ws
    my = np.bincount(inv, w * y, m) / Ws
    dx, dy = x - mx[inv], y - my[inv]
    vx = np.bincount(inv, w * dx * dx, m) / Ws
    vy = np.bincount(inv, w * dy * dy, m) / Ws
    cxy = np.bincount(inv, w * dx * dy, m) / Ws
    rho = cxy / (np.sqrt(vx + EPS) * np.sqrt(vy + EPS))
    used = (n >= RANK_MIN) & (vy > EPS)
    return rho, W, n, used, uk // NC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-ap", action="store_true")
    a = ap.parse_args()
    d = Dump(S)
    n = len(d)
    files = [m["file"] for m in d.images]
    jobs = list(range(n))
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        per = list(ex.map(_shared, jobs, chunksize=50))
    cov = np.array([[p[7], p[8], p[9], p[10]] for p in per]).sum(0)
    print(f"shared pairs: {cov[3]:.0f} of the student's {cov[2]:.0f} ({cov[3] / cov[2]:.3f}); student pair-score "
          f"mass at shared pairs {cov[1] / cov[0]:.3f}", flush=True)

    img = np.concatenate([np.full(len(p[0]), i, np.int64) for i, p in enumerate(per)])
    cls = np.concatenate([p[1] for p in per])
    s = np.concatenate([p[2] for p in per])
    t = np.concatenate([p[3] for p in per])
    splits = np.cumsum([len(p[0]) for p in per])[:-1]

    # (c) weighted rho of the logits.
    l = np.log(TAU / (1 - TAU))
    ucls = np.concatenate([p[4] for p in per])
    uimg = np.concatenate([np.full(len(p[4]), i, np.int64) for i, p in enumerate(per)])
    ps = np.concatenate([p[5] for p in per])
    pt = np.concatenate([p[6] for p in per])
    mval = np.maximum(ps, pt)
    keep = mval > TAU
    ucls, uimg, ps, pt, mval = ucls[keep], uimg[keep], ps[keep], pt[keep], mval[keep]

    def flo(p):
        q = np.clip(p, 1e-12, 1 - 1e-7)
        return np.maximum(np.log(q) - np.log1p(-q), l)

    x, y = flo(ps), flo(pt)
    print(f"rho elements (max(p, p~) > {TAU}): {len(x)}, {len(x) / n:.0f} per image", flush=True)
    for gamma in (0.5, 1.0):
        w = mval ** gamma
        for scope, groups in (("group16", uimg // GROUP), ("whole set", np.zeros_like(uimg))):
            rho, W, cnt, used, _ = weighted_rho(ucls, x, y, w, groups)
            ng = groups.max() + 1
            wm = (W[used] * rho[used]).sum() / W[used].sum()
            # The loss's normalisation is per batch: the mean over groups of each group's W-weighted rho.
            gid = (np.unique(groups * NC + ucls) // NC)
            per_g = []
            for g in range(ng):
                sel = used & (gid == g)
                if sel.any():
                    per_g.append((W[sel] * rho[sel]).sum() / W[sel].sum())
            print(f"rho gamma {gamma} {scope:<9}: W-weighted mean {wm:.4f}, mean over batches {np.mean(per_g):.4f}, "
                  f"plain mean {rho[used].mean():.4f}, median {np.median(rho[used]):.4f}, classes used per group "
                  f"{used.sum() / ng:.1f} (of {len(used) / ng:.1f} present)", flush=True)
        # Same, the top tail only: elements above 0.1 in either model.
        top = mval >= 0.1
        rho, W, cnt, used, _ = weighted_rho(ucls[top], x[top], y[top], w[top], uimg[top] // GROUP)
        print(f"rho gamma {gamma} group16, elements >= 0.1 only: W-weighted mean "
              f"{(W[used] * rho[used]).sum() / W[used].sum():.4f}, classes used per group "
              f"{used.sum() / (uimg.max() // GROUP + 1):.1f}", flush=True)

    if a.skip_ap:
        return
    new = {}
    zeros = np.zeros_like(img)
    variants = {
        "vswap": t.copy(),
        "rswap": rank_assign(cls, s, t, zeros),
        "rswap16": rank_assign(cls, s, t, img // GROUP),
        "band_hi": np.where(s >= 0.1, t, s),
        "band_lo": np.where(s < 0.1, t, s),
    }
    hi = s >= 0.1
    rb = s.copy()
    rb[hi] = rank_assign(cls[hi], s[hi], t[hi], zeros[hi])
    variants["rband_hi"] = rb
    rb = s.copy()
    rb[~hi] = rank_assign(cls[~hi], s[~hi], t[~hi], zeros[~hi])
    variants["rband_lo"] = rb
    print(f"shared pairs with student score >= 0.1: {hi.sum()} ({hi.mean():.4f})", flush=True)
    for name, v in variants.items():
        parts = np.split(v.astype(np.float32), splits)
        full = []
        for i in range(n):
            sc = d.image(i).pair_score.copy()
            sc[per[i][0]] = parts[i]
            full.append(sc)
        new[name] = full
    truth = coco.Truth(GT)
    names = ["s"] + list(variants)
    chunks = [(lo, min(lo + 125, n)) for lo in range(0, n, 125)]
    with ProcessPoolExecutor(a.workers, initializer=_post_init, initargs=(new,)) as ex:
        for name in names:
            res = []
            for part in ex.map(_post, [(name, lo, hi_) for lo, hi_ in chunks]):
                res.extend(part)
            ev = coco.evaluate(truth, coco.detections_array(truth, files, res))
            st = [float(v) for v in ev.stats]
            print(f"{name:<9} AP {st[0]:.4f} AP50 {st[1]:.4f} AP75 {st[2]:.4f} S {st[3]:.4f} M {st[4]:.4f} "
                  f"L {st[5]:.4f}", flush=True)


if __name__ == "__main__":
    main()
