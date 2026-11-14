"""Mark each truth's primary box in a box table: of the B* boxes matched to the same truth (boxes.py's `gt`, the
same-class truth a box overlaps most) at IoU >= 0.5, the highest-scoring one (ties: the earlier row). It is the box
COCO's greedy matching counts as that truth's detection at IoU 0.5 in nearly every case; the others are duplicates.

    python primary.py <dump dir> [...]       (adds primary.npy to <dump>/boxes.npy.d)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import table  # noqa: E402


def main():
    for d in sys.argv[1:]:
        t = table(os.path.join(d, "boxes.npz"))
        img = np.asarray(t["img"]).astype(np.int64)
        gt = np.asarray(t["gt"])
        ok = np.asarray(t["gt_iou"]) >= 0.5
        sc = np.asarray(t["score"])
        rows = np.nonzero(ok)[0]
        key = np.concatenate([img[rows, None].astype(np.float64), gt[rows].astype(np.float64)], axis=1)
        order = np.lexsort((rows, -sc[rows].astype(np.float64)))  # by score, then row
        _, first = np.unique(key[order], axis=0, return_index=True)
        prim = np.zeros(len(sc), dtype=bool)
        prim[rows[order[first]]] = True
        np.save(os.path.join(d, "boxes.npy.d", "primary.npy"), prim)
        print(f"{os.path.basename(d)}: {prim.sum()} primary boxes of {ok.sum()} at IoU >= 0.5")


if __name__ == "__main__":
    main()
