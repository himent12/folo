"""Paired bootstrap over images of AP (IoU 0.50:0.95, all areas, 100 detections) between configurations.

    python bootstrap.py <records dir> <baseline name> <name> [<name> ...] [--n 1000] [--seed 0] [--workers 8]
                        [--pair A:B ...]

Reads the match records run.py --records wrote. Prints, per configuration, AP with all images and the difference to
the baseline with its 95% percentile interval over the resamples (the same resamples for every configuration, so
the comparison is paired); `--pair A:B` also prints B minus A on the same resamples. The resample weights are drawn
in order from one generator (`--seed`), so the result does not depend on `--workers`.
"""

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coco  # noqa: E402

REC_THRS = np.linspace(0.0, 1.00, 101, endpoint=True)
_sets = None


def _load(records, names):
    sets = []
    for n in names:
        with open(os.path.join(records, f"{n}.pkl"), "rb") as f:
            recs, img_pos = pickle.load(f)
        sets.append(recs)
    return sets, len(img_pos)


def _init(records, names):
    global _sets
    _sets, _ = _load(records, names)


def _aps(weights):
    return [coco.ap_from_records(r, weights, REC_THRS) for r in _sets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("base")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pair", action="append", default=[], help="A:B, also report B - A")
    a = ap.parse_args()
    names = [a.base] + a.names
    sets, n_img = _load(a.records, names)
    full = [coco.ap_from_records(r, np.ones(n_img, dtype=np.int64), REC_THRS) for r in sets]
    del sets
    rng = np.random.default_rng(a.seed)
    weights = [np.bincount(rng.integers(0, n_img, n_img), minlength=n_img) for _ in range(a.n)]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.records, names)) as ex:
        boot = np.array(list(ex.map(_aps, weights, chunksize=8)))
    col = {n: k for k, n in enumerate(names)}

    def show(x, y, label):
        d = boot[:, col[y]] - boot[:, col[x]]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_le0 = float((d <= 0).mean())
        print(f"{label:<32} AP {full[col[y]]:.4f}  diff {full[col[y]] - full[col[x]]:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
              f"P(diff<=0) {p_le0:.3f}  ({a.n} paired image resamples)")

    print(f"{a.base:<32} AP {full[0]:.4f}")
    for n in a.names:
        show(a.base, n, n)
    for p in a.pair:
        x, y = p.split(":")
        show(x, y, f"{y} - {x}")


if __name__ == "__main__":
    main()
