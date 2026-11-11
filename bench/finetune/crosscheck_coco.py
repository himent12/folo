"""Check the common scorer against both tools' own COCO numbers on COCO val2017.

    python bench/finetune/crosscheck_coco.py --coco D:/folo-data/coco --work-dir D:/folo-verify/accuracy/coco-check \
        --weights yolo26n.pt --folo folo --folo-weights folo-n-coco-v2 --out bench/finetune/results/crosscheck-coco.json

It lays out val2017 (or its first --limit images) once more in the layout the fine-tuning datasets use: images
hard-linked (copied across drives) under images/val2017, YOLO labels from the non-crowd boxes (what Ultralytics
trains and validates on), data.yaml, and the original annotations (crowd regions and segment areas included) as
annotations/instances_val2017.json. Then:

- FOLO: `folo val --coco --save-json` of --folo-weights on that folder: FOLO's own pycocotools-exact AP and its
  detections, which scorer.py scores again;
- Ultralytics: `YOLO(weights).val(save_json=True)` exactly as run_ultralytics.py evaluates (conf 0.001, max_det 300,
  640 px), its own mAP50-95 (its metric, on the YOLO labels), and its detections mapped by scorer.ultralytics_to_gt
  and scored by scorer.py against the original annotations: with all 5,000 images this is the number Ultralytics
  publishes for the model (YOLO26n: 0.401 end to end, docs/BENCHMARKS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import datasets  # noqa: E402
import run_ultralytics  # noqa: E402


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def lay_out(coco: Path, dest: Path, limit: int | None) -> tuple[dict, Path]:
    gt = json.loads((coco / "annotations" / "instances_val2017.json").read_text())
    images = sorted(gt["images"], key=lambda im: im["id"])
    if limit:
        images = images[:limit]
    keep = {im["id"] for im in images}
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    cls_of = {c["id"]: i for i, c in enumerate(cats)}
    sub = {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": images,
        "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
        "categories": cats,
    }
    (dest / "images" / "val2017").mkdir(parents=True, exist_ok=True)
    (dest / "labels" / "val2017").mkdir(parents=True, exist_ok=True)
    (dest / "annotations").mkdir(parents=True, exist_ok=True)
    by_img: dict[int, list] = {}
    for a in sub["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in images:
        link(coco / "val2017" / im["file_name"], dest / "images" / "val2017" / im["file_name"])
        w, h = im["width"], im["height"]
        rows = []
        for a in by_img.get(im["id"], []):
            if a.get("iscrowd"):
                continue
            x, y, bw, bh = a["bbox"]
            box = datasets.clip_box((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)
            if box:
                rows.append((cls_of[a["category_id"]], *box))
        (dest / "labels" / "val2017" / (Path(im["file_name"]).stem + ".txt")).write_text(
            datasets.format_label(rows), newline="\n"
        )
    # Ultralytics wants a train key; nothing trains here, so it names the same images.
    yaml = "train: images/val2017\nval: images/val2017\n" + datasets.data_yaml([c["name"] for c in cats], [])
    (dest / "data.yaml").write_text(yaml, newline="\n")
    gt_path = dest / "annotations" / "instances_val2017.json"
    gt_path.write_text(json.dumps(sub))
    return sub, gt_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-check the common scorer on COCO val2017")
    ap.add_argument("--coco", required=True, type=Path, help="COCO root: annotations/instances_val2017.json, val2017/")
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images by id (default: all 5,000)")
    ap.add_argument("--weights", required=True, help="yolo26n.pt")
    ap.add_argument("--folo", default="folo")
    ap.add_argument("--folo-weights", default="folo-n-coco-v2")
    ap.add_argument("--device", default="0", help="Ultralytics device")
    ap.add_argument("--folo-device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    work = args.work_dir
    data = work / (f"coco-val-{args.limit}" if args.limit else "coco-val")
    gt, gt_path = lay_out(args.coco, data, args.limit)
    import scorer

    out: dict = {"images": len(gt["images"]), "annotations": len(gt["annotations"]), "created": common.now_iso()}

    # FOLO
    fdet = work / "folo-detections.json"
    p = subprocess.run(
        [args.folo, "val", "--model", args.folo_weights, "--data", str(data), "--split", "val", "--coco",
         "--save-json", str(fdet), "--device", args.folo_device, "--json"],
        capture_output=True, text=True,
    )
    if p.returncode != 0:
        raise SystemExit(f"folo val failed: {p.stderr[-2000:]}")
    fval = json.loads(p.stdout.strip().splitlines()[-1])
    fscore = scorer.score(gt, json.loads(fdet.read_text()))
    out["folo"] = {
        "weights": args.folo_weights,
        "own_coco_ap": fval["coco"]["ap"], "own_coco_ap50": fval["coco"]["ap50"], "own_map50_95": fval["map50_95"],
        "scorer": {k: fscore[k] for k in ("ap", "ap50", "ap75", "detections")},
    }

    # Ultralytics
    weights = Path(args.weights).resolve()
    run_ultralytics.ensure_weights(weights, False)
    run_ultralytics.isolate_ultralytics(work / "ul", weights)
    import torch
    import ultralytics
    from ultralytics import YOLO

    captured = []
    model = YOLO(str(weights))
    model.add_callback("on_val_end", lambda v: captured.append(v))
    m = model.val(data=str(data / "data.yaml"), split="val", imgsz=640, batch=16, conf=0.001, max_det=300,
                  save_json=True, plots=False, project=str(work / "ul" / "runs"), name="val", exist_ok=True,
                  device=args.device)
    v = captured[-1]
    dets = scorer.ultralytics_to_gt(v.jdict, gt, list(v.class_map))
    uscore = scorer.score(gt, dets)
    out["ultralytics"] = {
        "weights": weights.name, "version": ultralytics.__version__, "torch": torch.__version__,
        "own_map50_95": float(m.box.map), "own_map50": float(m.box.map50),
        "scorer": {k: uscore[k] for k in ("ap", "ap50", "ap75", "detections")},
    }
    print(json.dumps(out, indent=2))
    if args.out:
        common.write_json_atomic(args.out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
