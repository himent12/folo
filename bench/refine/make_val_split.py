"""Lane refine's split of val2017: val-A = the 1000 images with the smallest image ids, val-B = the other 4000.

    python make_val_split.py <coco root> <out dir> [--n-a 1000]

Writes <out>/valA.txt and <out>/valB.txt (image file names, sorted) and <out>/instances_valA.json /
instances_valB.json (the subsets of instances_val2017.json, every category kept). val-A is for design checks and
the gate (at most five reads), val-B is read once for the final configuration (docs/notes/hill/r3-refine.md).
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco")
    ap.add_argument("out")
    ap.add_argument("--n-a", type=int, default=1000)
    a = ap.parse_args()
    with open(os.path.join(a.coco, "annotations", "instances_val2017.json"), encoding="utf-8") as f:
        gt = json.load(f)
    images = sorted(gt["images"], key=lambda im: im["id"])
    os.makedirs(a.out, exist_ok=True)
    for name, chosen in (("valA", images[:a.n_a]), ("valB", images[a.n_a:])):
        ids = {im["id"] for im in chosen}
        sub = {
            "info": gt.get("info", {}),
            "licenses": gt.get("licenses", []),
            "images": chosen,
            "annotations": [an for an in gt["annotations"] if an["image_id"] in ids],
            "categories": gt["categories"],
        }
        with open(os.path.join(a.out, f"{name}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(im["file_name"] + "\n" for im in sorted(chosen, key=lambda im: im["file_name"])))
        with open(os.path.join(a.out, f"instances_{name}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(sub, f)
        print(f"{name}: {len(chosen)} images, {len(sub['annotations'])} annotations")


if __name__ == "__main__":
    main()
