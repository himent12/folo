"""Train the incumbent (Ultralytics) from scratch on the proxy set and write its detections for pycocotools.

    python bench/learn/ultra_train.py --model yolo11n.yaml --data D:/folo-data/runs-hill/learn/mini320.yaml \
        --epochs 30 --imgsz 320 --batch 16 --seed 0 --workers 8 --work-dir D:/folo-data/runs-hill/learn/ultra \
        --name y11n-s0 [--set optimizer=SGD lr0=0.01 ...]

Everything not named is the Ultralytics default (`pretrained=False` is forced: the comparison is from scratch). The
run goes to <work-dir>/runs/<name>; after training, the best weights are validated once more with `save_json` and
the detections are rewritten with the dataset's own category ids to <work-dir>/runs/<name>/dets.json, ready for
bench/learn/coco_eval.py. Ultralytics' settings, downloads and telemetry are confined to --work-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"


def isolate(work_dir: Path) -> None:
    assert "ultralytics" not in sys.modules
    os.environ["YOLO_CONFIG_DIR"] = str(work_dir / "ultralytics-config")
    import ultralytics.utils as uu
    from ultralytics.utils import SETTINGS

    overrides = {"weights_dir": str(work_dir / "weights"), "runs_dir": str(work_dir / "runs"),
                 "datasets_dir": str(work_dir / "datasets"), "sync": False}
    for k in ("clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb"):
        overrides[k] = False
    for k, v in overrides.items():
        dict.__setitem__(SETTINGS, k, v)
    uu.WEIGHTS_DIR = work_dir / "weights"
    uu.RUNS_DIR = work_dir / "runs"
    try:
        from ultralytics.utils.events import events

        events.enabled = False
    except Exception:
        pass


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--resume", action="store_true", help="continue <work-dir>/runs/<name>/weights/last.pt")
    ap.add_argument("--stop-after", type=int, default=0, help="stop cleanly after this many epochs (resume later)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="extra Ultralytics overrides, key=value")
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    isolate(a.work_dir)
    from ultralytics import YOLO

    run = a.work_dir / "runs" / a.name
    extra = {k: parse_value(v) for k, v in (s.split("=", 1) for s in a.set)}
    t0 = time.time()
    if not a.eval_only:
        if a.resume:
            model = YOLO(str(run / "weights" / "last.pt"))
            kw = dict(resume=True)
        else:
            model = YOLO(a.model)
            kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, workers=a.workers,
                      pretrained=False, project=str(a.work_dir / "runs"), name=a.name, exist_ok=True, plots=False,
                      device=0, **extra)
        if a.stop_after:
            def stop(trainer):
                if trainer.epoch + 1 >= a.stop_after:
                    trainer.stop = True  # checked at the end of the epoch, after last.pt is written

            model.add_callback("on_fit_epoch_end", stop)
        model.train(**kw)
    train_s = time.time() - t0
    best = run / "weights" / "best.pt"
    model = YOLO(str(best))
    r = model.val(data=a.data, imgsz=a.imgsz, batch=2 * a.batch, workers=a.workers, save_json=True, plots=False,
                  project=str(a.work_dir / "runs"), name=a.name + "-val", exist_ok=True, device=0)
    pred = Path(r.save_dir) / "predictions.json"
    dets = json.loads(pred.read_text())
    # Ultralytics writes class index + 1 for a custom dataset; the ground truth keeps COCO's own category ids.
    import yaml

    data = yaml.safe_load(Path(a.data).read_text())
    gt = json.loads((Path(data["path"]) / "annotations" / "instances_val2017.json").read_text())
    cat_ids = sorted(c["id"] for c in gt["categories"])
    for d in dets:
        d["category_id"] = cat_ids[d["category_id"] - 1]
    (run / "dets.json").write_text(json.dumps(dets))
    summary = {"name": a.name, "model": a.model, "train_seconds": round(train_s, 1),
               "ultralytics_map50_95": float(r.box.map), "ultralytics_map50": float(r.box.map50),
               "detections": len(dets)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
