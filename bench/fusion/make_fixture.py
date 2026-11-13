"""Write crates/folo-vision/tests/fixtures/fusion/dfl.json: a few images of a candidate dump with the boxes this
Python post-process (post.py: hard NMS, then DFL evidence fusion) gives, for the Rust reference test
(crates/folo-vision/tests/fusion_reference.rs).

Each image keeps its candidates above `--conf`: per anchor its grid point and stride, network box (rounded to 1/100
px) and 64 DFL logits (rounded to 1/1000), and the (anchor, class, score) pairs. The expected boxes are computed
from the rounded values, read back as f32 exactly as the Rust test reads them, so both sides start from the same
numbers. Usage:

    python make_fixture.py --dump D:/folo-data/runs-hill3/fusion/dumps/v1-val --images 3 17 --out <json>
"""

import argparse
import json

import numpy as np

from dump import Dump
from post import F32, batched_nms, fuse

CFG = {"conf": 0.02, "iou": 0.65, "max_det": 1000,
       "fuse": {"kind": "var", "tau": 0.75, "a": 1.0, "b": 1.0, "eps": 0.1, "excess": True}}


class Im:
    pass


def reduced(im, conf):
    keep = im.pair_score > F32(conf)
    used = np.unique(im.pair_anchor[keep])
    pos = {int(a): k for k, a in enumerate(used)}
    r = Im()
    r.point = im.point[used]
    r.stride = im.stride[used]
    r.box_net = np.round(im.box_net[used].astype(np.float64), 2).astype(F32)
    r.logits = np.round(im.logits[used].astype(np.float64), 3).astype(F32)
    r.pair_anchor = np.array([pos[int(a)] for a in im.pair_anchor[keep]], dtype=np.int64)
    r.pair_class = im.pair_class[keep]
    r.pair_score = im.pair_score[keep]
    return r


def expected(r, cfg):
    ca, cc, cs = r.pair_anchor, r.pair_class, r.pair_score
    kept = batched_nms(r.box_net[ca], cs, cc, cfg["iou"], cfg["max_det"])
    net = fuse(r, r.box_net, kept, ca, cc, cs, cfg["fuse"])
    return net, cs[kept], cc[kept], kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--images", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Dump(args.dump)
    images = []
    for i in args.images:
        r = reduced(d.image(i), CFG["conf"])
        net, scores, classes, kept = expected(r, CFG)
        moved = int(np.count_nonzero(np.any(net != r.box_net[r.pair_anchor[kept]], axis=1)))
        print(f"image {i} ({d.images[i]['file']}): {len(r.stride)} anchors, {len(r.pair_score)} pairs, "
              f"{len(kept)} kept, {moved} moved")
        images.append({
            "file": d.images[i]["file"],
            "anchors": [[float(p[0]), float(p[1]), float(s)] for p, s in zip(r.point, r.stride)],
            "boxes": [[round(float(v), 2) for v in b] for b in r.box_net],
            "logits": [[round(float(v), 3) for v in row.reshape(-1)] for row in r.logits],
            "pairs": [[int(a), int(c), float(s)] for a, c, s in zip(r.pair_anchor, r.pair_class, r.pair_score)],
            "expected": [[float(v) for v in b] + [float(s), int(c)] for b, s, c in zip(net, scores, classes)],
        })
    out = {"generator": "bench/fusion/make_fixture.py", "dump": d.header["model"], "reg_max": d.reg_max,
           "num_classes": d.num_classes, "cfg": CFG, "images": images}
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
