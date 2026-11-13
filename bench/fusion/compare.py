"""Compare two COCO results files detection by detection (folo val --save-json against run.py --save-json).

    python compare.py <a.json> <b.json>
"""

import json
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        dets = json.load(f)
    by = defaultdict(list)
    for d in dets:
        by[d["image_id"]].append((d["category_id"], d["score"], tuple(d["bbox"])))
    for v in by.values():
        v.sort()
    return dets, by


def main():
    a, ba = load(sys.argv[1])
    b, bb = load(sys.argv[2])
    print(f"{len(a)} vs {len(b)} detections, {len(ba)} vs {len(bb)} images")
    diff_img = 0
    max_box = 0.0
    for img in set(ba) | set(bb):
        x, y = ba.get(img, []), bb.get(img, [])
        if len(x) != len(y) or any(p[0] != q[0] or p[1] != q[1] for p, q in zip(x, y)):
            diff_img += 1
            continue
        for p, q in zip(x, y):
            max_box = max(max_box, max(abs(u - v) for u, v in zip(p[2], q[2])))
    print(f"images whose (class, score) lists differ: {diff_img}; largest box difference elsewhere: {max_box}")


if __name__ == "__main__":
    main()
