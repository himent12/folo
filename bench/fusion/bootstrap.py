"""Paired bootstrap over images of AP (IoU 0.50:0.95, all areas, 100 detections) between configurations.

    python bootstrap.py <records dir> <baseline name> <name> [<name> ...] [--n 1000] [--seed 0]

Reads the match records run.py --records wrote. Prints, per configuration, AP with all images (which must equal
pycocotools' AP: checked), and the difference to the baseline with its 95% percentile interval over the
resamples (the same resamples for every configuration, so the comparison is paired).
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coco  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("base")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = [a.base] + a.names
    sets = []
    for n in names:
        with open(os.path.join(a.records, f"{n}.pkl"), "rb") as f:
            recs, img_pos = pickle.load(f)
        sets.append(recs)
    n_img = len(img_pos)
    rec_thrs = np.linspace(0.0, 1.00, 101, endpoint=True)
    full = [coco.ap_from_records(r, np.ones(n_img, dtype=np.int64), rec_thrs) for r in sets]
    boot = coco.bootstrap(sets, n_img, a.n, a.seed)
    d = boot[:, 1:] - boot[:, :1]
    print(f"{a.base:<32} AP {full[0]:.4f}")
    for k, n in enumerate(a.names):
        lo, hi = np.percentile(d[:, k], [2.5, 97.5])
        p_le0 = float((d[:, k] <= 0).mean())
        print(f"{n:<32} AP {full[k + 1]:.4f}  diff {full[k + 1] - full[0]:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
              f"P(diff<=0) {p_le0:.3f}  ({a.n} paired image resamples)")


if __name__ == "__main__":
    main()
