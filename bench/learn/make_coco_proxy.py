"""Build the COCO proxy set of docs/hill/learn.md: a fixed, seeded subset of COCO 2017 that FOLO and Ultralytics
both read (COCO JSON for FOLO and pycocotools, YOLO txt labels for Ultralytics, the same image files).

    python bench/learn/make_coco_proxy.py --coco D:/folo-data/coco --out D:/folo-data/runs-hill/learn/mini320 \
        --train 20000 --val 2000 --long-side 320 --seed 0

Layout written under --out:
    images/{train2017,val2017}/*.jpg        originals (hard links) or, with --long-side N, shrunk copies (INTER_AREA,
                                            JPEG quality 95) whose long side is at most N
    labels/{train2017,val2017}/*.txt        YOLO labels (class cx cy w h, normalised; crowd regions skipped)
    annotations/instances_{train,val}2017.json   the COCO subset (boxes and areas scaled with the images, crowd kept)
    ../<name>.yaml                          the Ultralytics data file (beside the folder, so FOLO never sees a YAML)

The subset is the first N ids of a seeded shuffle of the sorted image ids, so a larger --train is a superset of a
smaller one with the same seed. Needs numpy + opencv-python.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path


def shrink(job):
    src, dst, long_side = job
    import cv2

    if os.path.exists(dst):
        im = cv2.imread(dst)
        return im.shape[1], im.shape[0]
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    r = long_side / max(h, w)
    if r < 1:
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return im.shape[1], im.shape[0]


def build_split(coco: Path, out: Path, tag: str, n: int, seed: int, long_side: int, workers: int):
    with open(coco / "annotations" / f"instances_{tag}.json", "r", encoding="utf-8") as f:
        js = json.load(f)
    images = sorted(js["images"], key=lambda im: im["id"])
    random.Random(seed).shuffle(images)
    images = images[:n]
    keep = {im["id"] for im in images}
    anns = [a for a in js["annotations"] if a["image_id"] in keep]
    cats = sorted(js["categories"], key=lambda c: c["id"])
    cls = {c["id"]: i for i, c in enumerate(cats)}

    img_dir, lbl_dir = out / "images" / tag, out / "labels" / tag
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    scale = {}
    if long_side > 0:
        jobs = [(str(coco / tag / im["file_name"]), str(img_dir / im["file_name"]), long_side) for im in images]
        with Pool(workers) as pool:
            sizes = pool.map(shrink, jobs, chunksize=64)
        for im, (nw, nh) in zip(images, sizes):
            scale[im["id"]] = (nw / im["width"], nh / im["height"])
            im["width"], im["height"] = nw, nh
    else:
        for im in images:
            dst = img_dir / im["file_name"]
            if not dst.exists():
                try:
                    os.link(coco / tag / im["file_name"], dst)
                except OSError:
                    shutil.copyfile(coco / tag / im["file_name"], dst)
            scale[im["id"]] = (1.0, 1.0)

    size = {im["id"]: (im["width"], im["height"]) for im in images}
    lines = {im["id"]: [] for im in images}
    for a in anns:
        sx, sy = scale[a["image_id"]]
        x, y, w, h = a["bbox"]
        a["bbox"] = [x * sx, y * sy, w * sx, h * sy]
        a["area"] = a["area"] * sx * sy
        if long_side > 0:
            a["segmentation"] = []  # the polygons are not rescaled; bbox evaluation does not read them
        if a.get("iscrowd", 0):
            continue
        W, H = size[a["image_id"]]
        x, y, w, h = a["bbox"]
        x1, y1, x2, y2 = max(x, 0.0), max(y, 0.0), min(x + w, W), min(y + h, H)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        lines[a["image_id"]].append(
            f"{cls[a['category_id']]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
        )
    for im in images:
        rows = lines[im["id"]]
        p = lbl_dir / (Path(im["file_name"]).stem + ".txt")
        if rows:
            p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        elif p.exists():
            p.unlink()

    ann_dir = out / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    sub = {"info": js.get("info", {}), "licenses": js.get("licenses", []), "images": images, "annotations": anns,
           "categories": js["categories"]}
    with open(ann_dir / f"instances_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(sub, f)
    present = {a["category_id"] for a in anns if not a.get("iscrowd", 0)}
    boxes = sum(len(v) for v in lines.values())
    print(f"{tag}: {len(images)} images, {boxes} boxes, {len(present)} of {len(cats)} classes present")
    return [c["name"] for c in cats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--long-side", type=int, default=0, help="shrink images to this long side (0: hard-link originals)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names = build_split(a.coco, a.out, "val2017", a.val, a.seed, a.long_side, a.workers)
    build_split(a.coco, a.out, "train2017", a.train, a.seed, a.long_side, a.workers)
    yaml = a.out.parent / f"{a.out.name}.yaml"
    body = [f"path: {a.out.as_posix()}", "train: images/train2017", "val: images/val2017", "names:"]
    body += [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    yaml.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {yaml}")


if __name__ == "__main__":
    main()
