"""One scorer for both tools: COCO detections JSON against the dataset's own ground truth, with pycocotools.

    python bench/finetune/scorer.py <annotations/instances_<split>.json> <detections.json>

Detections must already use the ground truth's image ids and category ids (FOLO's `val --save-json` writes them
when given that file; `ultralytics_to_gt` maps Ultralytics' `save_json` records onto them). Every image of the
ground truth is evaluated, also those without a detection. COCO's protocol: AP over IoU 0.50:0.95, at most 100
detections per image (pycocotools' default `maxDets`), 101-point interpolated precision.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path


def check_detections(gt: dict, dets: list[dict]) -> None:
    """Fail loudly on detections that cannot belong to this ground truth (the usual id-mapping bugs)."""
    image_ids = {im["id"] for im in gt["images"]}
    cat_ids = {c["id"] for c in gt["categories"]}
    bad_img = {d["image_id"] for d in dets} - image_ids
    bad_cat = {d["category_id"] for d in dets} - cat_ids
    if bad_img:
        raise ValueError(f"{len(bad_img)} detection image ids are not in the ground truth, e.g. {sorted(bad_img, key=str)[:5]}")
    if bad_cat:
        raise ValueError(f"detection category ids {sorted(bad_cat)} are not in the ground truth {sorted(cat_ids)}")
    for d in dets:
        if len(d["bbox"]) != 4:
            raise ValueError(f"malformed bbox {d['bbox']} (want [x, y, w, h])")


def degenerate_to_empty(dets: list[dict]) -> tuple[list[dict], int]:
    """Detections with a negative width or height (an inverted box: an under-trained YOLO26 head emits some) as
    zero-size boxes at the same place and score, so they stay what they are, false positives that overlap nothing,
    instead of whatever pycocotools makes of a negative area. Returns the detections and how many were changed."""
    out, n = [], 0
    for d in dets:
        x, y, w, h = d["bbox"]
        if w < 0 or h < 0:
            d = {**d, "bbox": [x, y, max(0.0, w), max(0.0, h)]}
            n += 1
        out.append(d)
    return out, n


def ultralytics_to_gt(records: list[dict], gt: dict, class_map: list[int]) -> list[dict]:
    """Ultralytics `DetectionValidator.jdict` records onto the ground truth's ids.

    Ultralytics writes `image_id` as the file stem (an int when the stem is all digits, so `0001` becomes 1) and
    `category_id = class_map[class]` (`1..nc` for a custom dataset, COCO's 91-class ids for COCO itself). The
    records also carry `file_name`, which is matched against the ground truth's file names; the class index is
    recovered through `class_map` and becomes the ground truth's category of that index (its categories are in
    class order)."""
    by_name = {im["file_name"]: im["id"] for im in gt["images"]}
    if len(by_name) != len(gt["images"]):
        raise ValueError("ground truth file names are not unique")
    cats = [c["id"] for c in gt["categories"]]
    cls_of = {cid: i for i, cid in enumerate(class_map)}
    out = []
    for r in records:
        name = r.get("file_name")
        if name is None or name not in by_name:
            raise ValueError(f"detection of {name!r} (image_id {r.get('image_id')!r}) matches no ground-truth image")
        c = cls_of.get(r["category_id"])
        if c is None or c >= len(cats):
            raise ValueError(f"category_id {r['category_id']} is not in the validator's class map")
        out.append({"image_id": by_name[name], "category_id": cats[c], "bbox": list(r["bbox"]), "score": r["score"]})
    return out


def score(gt: dict, dets: list[dict]) -> dict:
    """pycocotools COCOeval (bbox) of `dets` against `gt`: AP@[.5:.95], AP50, AP75, the size APs and AR@100.
    Zero detections score 0 (pycocotools' loadRes refuses an empty list)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    check_detections(gt, dets)
    dets, inverted = degenerate_to_empty(dets)
    gt = dict(gt)
    gt.setdefault("info", {})
    gt.setdefault("licenses", [])
    keys = ["ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"]
    base = {
        "images": len(gt["images"]),
        "truths": len(gt["annotations"]),
        "detections": len(dets),
        "images_without_detections": len({im["id"] for im in gt["images"]} - {d["image_id"] for d in dets}),
        "inverted_boxes": inverted,  # scored as zero-size false positives (degenerate_to_empty)
    }
    if not dets:
        return {**{k: 0.0 for k in keys}, **base}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes([dict(d) for d in dets])
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # pycocotools reports -1 for an area range without ground truth; keep that as None.
    stats = {k: (float(v) if v >= 0 else None) for k, v in zip(keys, ev.stats)}
    return {**stats, **base}


def main(argv: list[str]) -> None:
    gt = json.loads(Path(argv[0]).read_text())
    dets = json.loads(Path(argv[1]).read_text())
    print(json.dumps(score(gt, dets), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
