"""How often the censoring guard would fire: the share of candidate sides whose DFL distribution puts more than
`--thr` of its mass on the last bin (the side may be farther than reg_max - 1 bins: the distance is censored), or
on bin 0, per pyramid level and per COCO size bucket of the candidate's own box (original-image area: small < 32²,
medium < 96², large). Candidates: every anchor whose best class scores above `--conf`. Usage:

    python censor_stats.py --dump <dir> [--conf 0.05] [--thr 0.3 0.5]
"""

import argparse

import numpy as np

from dump import Dump
from post import side_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--thr", type=float, nargs="+", default=[0.3, 0.5])
    ap.add_argument("--images", type=int, default=0, help="first N images only (0: all)")
    a = ap.parse_args()
    d = Dump(a.dump)
    n = len(d) if a.images == 0 else min(a.images, len(d))
    sides = {}  # (level, size) -> [count of sides, far counts per thr, near counts per thr]
    for i in range(n):
        im = d.image(i)
        best = np.zeros(len(im.anchor), dtype=np.float32)
        np.maximum.at(best, im.pair_anchor, im.pair_score)
        sel = best > np.float32(a.conf)
        if not sel.any():
            continue
        p, _, _ = side_stats(im.logits[sel])
        b = im.box_orig[sel].astype(np.float64)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        size = np.where(area < 32 ** 2, 0, np.where(area < 96 ** 2, 1, 2))
        lv = im.level[sel]
        for l in np.unique(lv):
            for s in range(3):
                m = (lv == l) & (size == s)
                if not m.any():
                    continue
                e = sides.setdefault((int(l), s), [0, np.zeros(len(a.thr)), np.zeros(len(a.thr))])
                e[0] += 4 * int(m.sum())
                for k, t in enumerate(a.thr):
                    e[1][k] += np.count_nonzero(p[m][:, :, -1] > t)
                    e[2][k] += np.count_nonzero(p[m][:, :, 0] > t)
    names = ["small", "medium", "large"]
    head = "  ".join(f"last>{t:<4} first>{t:<4}" for t in a.thr)
    print(f"{n} images, candidates above conf {a.conf}")
    print(f"level size      sides     {head}")
    for (l, s) in sorted(sides):
        c, far, near = sides[(l, s)]
        cols = "  ".join(f"{far[k] / c:9.4f} {near[k] / c:10.4f}" for k in range(len(a.thr)))
        print(f"P{l + 3:<4} {names[s]:<8} {c:9d}  {cols}")
    tot = sum(v[0] for v in sides.values())
    far = sum(v[1] for v in sides.values())
    near = sum(v[2] for v in sides.values())
    print(f"all            {tot:9d}  " + "  ".join(f"{far[k] / tot:9.4f} {near[k] / tot:10.4f}" for k in range(len(a.thr))))


if __name__ == "__main__":
    main()
