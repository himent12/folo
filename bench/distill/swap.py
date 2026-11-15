"""Counsel A (hill-climb 4): box / score swap between folo-s (teacher) and folo-n v3 (student) on val2017 dumps.

For every image, at anchors that are candidates (class-max score > 0.001) in BOTH dumps:
  sT: student scores and classes, teacher box of the same anchor (what perfect localization distillation gives)
  sH: student scores, box halfway between student and teacher
  tS: teacher scores and classes, student box of the same anchor (what the teacher loses with the student's boxes)
Anchors missing from the other dump keep their own box. Default post-processing (hard NMS 0.7, 300).
Also per-side DFL statistics at shared anchors that lie inside a truth they match.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fusion"))
import coco  # noqa: E402
from dump import Dump  # noqa: E402
from post import postprocess, side_stats  # noqa: E402

S = "D:/folo-data/runs-hill3/fusion/dumps/land-val"
T = "D:/folo-data/runs-hill3/fusion/dumps/s-val"
GT = "D:/folo-data/coco/annotations/instances_val2017.json"
CFG = {}
_s = _t = None


class Im:
    pass


def clone(im):
    o = Im()
    for k in ("ratio", "pad", "orig", "level", "pair_anchor", "pair_class", "pair_score", "logits", "file"):
        setattr(o, k, getattr(im, k))
    o.box_net = im.box_net.copy()
    o.box_orig = im.box_orig.copy()
    return o


def _init():
    global _s, _t
    _s, _t = Dump(S), Dump(T)


def _work(r):
    lo, hi = r
    out = {k: [] for k in ("s", "t", "sT", "sH", "tS")}
    cov = np.zeros(6)
    for i in range(lo, hi):
        a, b = _s.image(i), _t.image(i)
        assert a.file == b.file
        # positions of shared anchors
        common, ia, ib = np.intersect1d(a.anchor, b.anchor, return_indices=True)
        out["s"].append(postprocess(a, CFG))
        out["t"].append(postprocess(b, CFG))
        x = clone(a)
        x.box_net[ia] = b.box_net[ib]
        x.box_orig[ia] = b.box_orig[ib]
        out["sT"].append(postprocess(x, CFG))
        x = clone(a)
        x.box_net[ia] = 0.5 * (a.box_net[ia] + b.box_net[ib])
        x.box_orig[ia] = 0.5 * (a.box_orig[ia] + b.box_orig[ib])
        out["sH"].append(postprocess(x, CFG))
        y = clone(b)
        y.box_net[ib] = a.box_net[ia]
        y.box_orig[ib] = a.box_orig[ia]
        out["tS"].append(postprocess(y, CFG))
        # coverage, weighted by pair score (what AP sees): share of student pair-score mass at shared anchors
        ms = np.zeros(len(a.anchor), bool)
        ms[ia] = True
        mt = np.zeros(len(b.anchor), bool)
        mt[ib] = True
        cov += [a.pair_score.sum(), a.pair_score[ms[a.pair_anchor]].sum(), b.pair_score.sum(),
                b.pair_score[mt[b.pair_anchor]].sum(), len(a.anchor), len(common)]
    return out, cov


def main():
    d = Dump(S)
    n = len(d)
    files = [m["file"] for m in d.images]
    jobs = [(lo, min(lo + 125, n)) for lo in range(0, n, 125)]
    keys = ["s", "t", "sT", "sH", "tS"]
    res = {k: [] for k in keys}
    cov = np.zeros(6)
    with ProcessPoolExecutor(10, initializer=_init) as ex:
        for part, c in ex.map(_work, jobs):
            cov += c
            for k in keys:
                res[k].extend(part[k])
    print(f"coverage: student pair-score mass at shared anchors {cov[1] / cov[0]:.3f}, teacher {cov[3] / cov[2]:.3f}, "
          f"student anchors shared {cov[5] / cov[4]:.3f}")
    truth = coco.Truth(GT)
    for k in keys:
        ev = coco.evaluate(truth, coco.detections_array(truth, files, res[k]))
        s = [float(v) for v in ev.stats]
        print(f"{k:<3} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S {s[3]:.4f} M {s[4]:.4f} L {s[5]:.4f}", flush=True)


if __name__ == "__main__":
    main()
