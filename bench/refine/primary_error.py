"""Edge error closed on each truth's primary box (primary.py), by COCO size, for saved refined boxes.

    python primary_error.py <dump dir> <name>=<refined boxes .npy> [...]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import table  # noqa: E402


def main():
    d = sys.argv[1]
    t = table(os.path.join(d, "boxes.npz"))
    m = np.asarray(t["primary"])
    gt, box = np.asarray(t["gt"])[m], np.asarray(t["box"])[m]
    img = np.repeat(np.arange(len(t["img_count"])), np.asarray(t["img_count"]))[m]
    r = np.asarray(t["ratio"])[img]
    area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) / (r[:, 0] * r[:, 1])
    size = np.where(area < 32 ** 2, "S", np.where(area < 96 ** 2, "M", "L"))
    e0 = np.abs(box - gt)
    print(f"{m.sum()} primary boxes; B* mean |edge error| px: all {e0.mean():.3f}, "
          + ", ".join(f"{z} {e0[size == z].mean():.3f}" for z in "SML"))
    for spec in sys.argv[2:]:
        name, path = spec.split("=")
        e1 = np.abs(np.load(path)[m] - gt)
        closed = {z: 1 - e1[size == z].mean() / e0[size == z].mean() for z in "SML"}
        print(f"{name:<14} closed: all {1 - e1.mean() / e0.mean():+.4f}, "
              + ", ".join(f"{z} {v:+.4f}" for z, v in closed.items()))
        # Per edge: the share moved closer to its truth (by more than 0.1 px) and farther, and the median change.
        dz = e1 - e0
        print(f"{'':<14} edges closer / farther / median change px: "
              + ", ".join(f"{z} {(dz[size == z] < -0.1).mean():.3f} / {(dz[size == z] > 0.1).mean():.3f} / "
                          f"{np.median(dz[size == z]):+.3f}" for z in "SML"))


if __name__ == "__main__":
    main()
