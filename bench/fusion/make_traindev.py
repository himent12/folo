"""The fusion lane's train-dev set: a fixed, seeded list of 5000 train2017 images and their COCO ground truth.

    python make_traindev.py <coco root> <out dir> [--n 5000] [--seed 0] [--name traindev]

Writes <out>/<name>.txt (image file names, sorted) and <out>/instances_<name>.json (the subset of
instances_train2017.json: those images, their annotations, every category). Images without annotations are kept
(val2017 has 48 of them too), so the set is a uniform sample of train2017.
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="traindev")
    ap.add_argument("--exclude", help="a list file whose images must not be picked (for a second list)")
    a = ap.parse_args()
    with open(os.path.join(a.coco, "annotations", "instances_train2017.json"), encoding="utf-8") as f:
        gt = json.load(f)
    images = sorted(gt["images"], key=lambda im: im["id"])
    if a.exclude:
        with open(a.exclude, encoding="utf-8") as f:
            skip = {l.strip() for l in f if l.strip()}
        images = [im for im in images if im["file_name"] not in skip]
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(images), size=a.n, replace=False))
    chosen = [images[i] for i in pick]
    ids = {im["id"] for im in chosen}
    sub = {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": chosen,
        "annotations": [an for an in gt["annotations"] if an["image_id"] in ids],
        "categories": gt["categories"],
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.name}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(im["file_name"] + "\n" for im in sorted(chosen, key=lambda im: im["file_name"])))
    with open(os.path.join(a.out, f"instances_{a.name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(sub, f)
    print(f"{len(chosen)} images, {len(sub['annotations'])} annotations")


if __name__ == "__main__":
    main()
