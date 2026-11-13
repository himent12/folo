"""The configurations of each rung of the ladder (r3-fusion notebook, Part 2), written as run.py config files.

    python ladder.py <out dir>
"""

import json
import os
import sys


def rungs():
    r = {}
    r["r1"] = [{"name": "B0"}] + [{"name": f"iou{t}", "iou": t} for t in (0.6, 0.65, 0.75)] + [
        {"name": "maxdet1000", "max_det": 1000},
        {"name": "maxdet100", "max_det": 100},
    ]
    r["r2"] = [{"name": f"soft{s}", "nms": "soft", "soft_sigma": s} for s in (0.1, 0.3, 0.5, 1.0)] + [
        {"name": f"matrix-g{s}", "nms": "matrix", "matrix_sigma": s} for s in (1.0, 2.0, 4.0)
    ] + [{"name": "matrix-lin", "nms": "matrix", "matrix_kernel": "linear"}]
    r["r3"] = [{"name": f"off{d4:+.2f}{d5:+.2f}", "level_offset": [0.0, d4, d5]}
               for d4 in (-0.3, 0.0, 0.3) for d5 in (-0.3, 0.0, 0.3) if (d4, d5) != (0.0, 0.0)]
    r["r4"] = [{"name": f"T{t}", "dfl_T": t} for t in (0.7, 0.85, 1.15, 1.3)] + [
        {"name": f"win{w}", "dfl_window": w} for w in (1, 2, 3)
    ]
    vote = []
    for tau in (0.55, 0.65, 0.75, 0.85):
        for a in (0.0, 1.0, 2.0):
            vote.append({"name": f"vote-t{tau}-a{a}", "fuse": {"kind": "vote", "tau": tau, "a": a}})
    for s in (0.025, 0.05, 0.1):
        vote.append({"name": f"vote-t0.55-a1.0-s{s}", "fuse": {"kind": "vote", "tau": 0.55, "a": 1.0, "sigma": s}})
    r["r5"] = vote
    return r


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for k, v in rungs().items():
        with open(os.path.join(out, f"{k}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(v, f, indent=1)
        print(k, len(v))
