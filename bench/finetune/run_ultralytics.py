"""Fine-tune an Ultralytics model (YOLO26n by default) on the benchmark datasets and score every run with the common
scorer.

    python bench/finetune/run_ultralytics.py --data-root D:/folo-data/finetune --work-dir D:/folo-data/runs-finetune/ul \
        --results bench/finetune/results --weights yolo26n.pt --device 0 --seeds 0,1,2 --epochs 100

Self-contained: it needs only torch, ultralytics, pycocotools and Pillow, and the files of this folder
(datasets.py, scorer.py, common.py), so it runs on a plain PyTorch pod as on Windows (run_ultralytics_pod.sh).
Each run is one subprocess (a crash or a leak does not take the matrix down), with Ultralytics' defaults:

    YOLO(weights).train(data=..., epochs=E, imgsz=640, batch=16, seed=s, deterministic=True, patience=max(100, E))

(`patience` at least the epoch count, so no run stops early; `plots=False`, which changes no number). Its best
checkpoint (`best.pt`, Ultralytics' fitness) is then validated by Ultralytics on the evaluation split with
`save_json=True` (conf 0.001, max_det 300, 640 px, its default NMS-free end-to-end head), the detections are
mapped onto the ground truth's ids (scorer.ultralytics_to_gt) and scored with pycocotools (scorer.score).
A run whose result file exists is skipped. Ultralytics stays inside --work-dir: its settings, downloads and runs go
there, auto-install and telemetry are off.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import datasets  # noqa: E402

# Ultralytics publishes its COCO checkpoints in this release; --download-weights fetches yolo26<size>.pt from it.
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
# sha256 of the checkpoints measured so far; a file of one of these names must match (others are only recorded).
KNOWN_SHA256 = {"yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"}

os.environ["YOLO_AUTOINSTALL"] = "false"  # never pip-install into the environment


def isolate_ultralytics(work_dir: Path, weights: Path) -> None:
    """Settings, weights, datasets and runs under work_dir; no telemetry or integrations. The AMP check loads
    `yolo26n.pt` by name, so a copy of the weights is put in the weights dir (no download) when they are that file."""
    assert "ultralytics" not in sys.modules, "isolate_ultralytics() must run before ultralytics is imported"
    os.environ["YOLO_CONFIG_DIR"] = str(work_dir / "ultralytics-config")
    (work_dir / "weights").mkdir(parents=True, exist_ok=True)
    if weights.name == "yolo26n.pt" and not (work_dir / "weights" / "yolo26n.pt").is_file():
        shutil.copyfile(weights, work_dir / "weights" / "yolo26n.pt")
    import ultralytics.utils as uu
    from ultralytics.utils import SETTINGS

    overrides = {
        "weights_dir": str(work_dir / "weights"),
        "runs_dir": str(work_dir / "runs"),
        "datasets_dir": str(work_dir / "datasets"),
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": False,
    }
    for k, v in overrides.items():
        dict.__setitem__(SETTINGS, k, v)
    uu.WEIGHTS_DIR = work_dir / "weights"
    uu.RUNS_DIR = work_dir / "runs"
    try:
        from ultralytics.utils.events import events

        events.enabled = False
    except Exception:
        pass


def ensure_weights(path: Path, download: bool) -> str:
    """sha256 of the weights file; with `download`, fetch a published `yolo26<size>.pt` when it is missing."""
    if not path.is_file():
        if not download or not (path.name.startswith("yolo26") and path.suffix == ".pt"):
            raise SystemExit(f"{path} not found (pass --download-weights to fetch a published yolo26<size>.pt)")
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f"{ASSETS_URL}/{path.name}", headers={"User-Agent": datasets.USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    digest = datasets.sha256_file(path)
    want = KNOWN_SHA256.get(path.name)
    if want and digest != want:
        raise SystemExit(f"{path}: sha256 {digest} is not the published {path.name} ({want})")
    return digest


def model_name(weights: Path) -> str:
    """The model a checkpoint names: its stem (`yolo26n.pt` -> `yolo26n`)."""
    return weights.stem


def epochs_in(results_csv: Path) -> int | None:
    if not results_csv.is_file():
        return None
    lines = [l for l in results_csv.read_text().splitlines() if l.strip()]
    return max(0, len(lines) - 1)


def run_one(args) -> None:
    """Train and score one run (in this process); writes its result file."""
    key = common.RunKey("ultralytics", model_name(Path(args.weights)), args.dataset, "pretrained", args.seed)
    spec = datasets.SPECS[args.dataset]
    data_dir = Path(args.data_root) / args.dataset
    yaml = data_dir / "data.yaml"
    gt_path = data_dir / "annotations" / f"instances_{spec.eval_split}.json"
    work = Path(args.work_dir)
    weights = Path(args.weights).resolve()
    weights_sha = ensure_weights(weights, False)
    isolate_ultralytics(work, weights)
    import torch
    import ultralytics
    from ultralytics import YOLO

    import scorer

    project = work / "runs"
    patience = max(100, args.epochs)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        patience=patience,
        project=str(project),
        name=key.stem,
        exist_ok=True,
        plots=False,
        device=args.device,
        **({"workers": args.workers} if args.workers is not None else {}),
    )
    train_wall = time.perf_counter() - t0
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"{best} was not written")

    captured = []
    ev = YOLO(str(best))
    ev.add_callback("on_val_end", lambda v: captured.append(v))
    m = ev.val(
        data=str(yaml),
        split=spec.eval_split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        max_det=300,
        save_json=True,
        plots=False,
        project=str(project),
        name=key.stem + "-eval",
        exist_ok=True,
        device=args.device,
    )
    v = captured[-1]
    gt = json.loads(gt_path.read_text())
    dets = scorer.ultralytics_to_gt(v.jdict, gt, list(v.class_map))
    det_path = run_dir / f"detections-{spec.eval_split}.json"
    det_path.write_text(json.dumps(dets))
    metrics = scorer.score(gt, dets)
    on_gpu = torch.cuda.is_available() and str(args.device) != "cpu"
    gpu = torch.cuda.get_device_name(0) if on_gpu else f"cpu ({platform.processor() or platform.machine()})"
    rec = common.result_record(
        key,
        epochs_planned=args.epochs,
        epochs_run=epochs_in(run_dir / "results.csv"),
        imgsz=args.imgsz,
        batch=args.batch,
        weights={"name": weights.name, "sha256": weights_sha},
        eval_split=spec.eval_split,
        metrics=metrics,
        tool_reported={"map50_95": float(m.box.map), "map50": float(m.box.map50), "source": "ultralytics val of best.pt"},
        train_wall_s=train_wall,
        gpu=gpu,
        versions={
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "dataset_layout_sha256": json.loads((data_dir / "meta.json").read_text())["layout_sha256"],
        },
        compute={
            "where": args.compute_label or platform.node(),
            "device": str(args.device),
            "patience": patience,
            # Ultralytics' default amp=True, which it turns off itself on the CPU or when its AMP check fails.
            "amp": bool(getattr(model.trainer, "amp", False)),
        },
    )
    rec["detections_file"] = str(det_path)
    rec["run_dir"] = str(run_dir)
    common.write_json_atomic(key.result_path(Path(args.results)), rec)
    print(f"{key.stem}: AP {metrics['ap']:.4f} AP50 {metrics['ap50']:.4f} in {train_wall:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune Ultralytics YOLO26n on the benchmark datasets")
    ap.add_argument("--data-root", required=True, help="folder of datasets.py fetch")
    ap.add_argument("--work-dir", required=True, help="Ultralytics settings, weights and runs go here")
    ap.add_argument("--results", required=True, help="one result JSON per run is written here")
    ap.add_argument("--weights", default="yolo26n.pt", help="the pretrained checkpoint (default yolo26n.pt; its stem names the model)")
    ap.add_argument("--download-weights", action="store_true", help="fetch a published yolo26<size>.pt when --weights is missing")
    ap.add_argument("--datasets", default=",".join(datasets.SPECS), help="comma-separated (default: all)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--device", required=True, help="Ultralytics device: 0 (the first GPU, on a pod), cpu (smoke runs on a shared PC)"
    )
    ap.add_argument("--workers", type=int, default=None, help="data loader workers (default: Ultralytics' 8)")
    ap.add_argument("--fetch", action="store_true", help="fetch the datasets into --data-root first")
    ap.add_argument("--compute-label", default=None, help="where this runs, for the result files (e.g. runpod RTX 4090)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dataset", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.one:
        run_one(args)
        return 0
    names = [s.name for s in datasets.pick(args.datasets)]
    seeds = common.parse_seeds(args.seeds)
    weights = Path(args.weights).resolve()
    ensure_weights(weights, args.download_weights)
    if args.fetch:
        for name in names:
            datasets.fetch(datasets.SPECS[name], Path(args.data_root))
    failed = []
    for name in names:
        for seed in seeds:
            key = common.RunKey("ultralytics", model_name(weights), name, "pretrained", seed)
            if key.result_path(Path(args.results)).is_file():
                print(f"{key.stem}: done already, skipped", flush=True)
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "--one", "--dataset", name, "--seed", str(seed)]
            for opt in ("data_root", "work_dir", "results", "epochs", "imgsz", "batch", "device", "workers", "compute_label"):
                val = getattr(args, opt)
                if val is not None:
                    cmd += [f"--{opt.replace('_', '-')}", str(val)]
            cmd += ["--weights", str(weights)]
            print(f"{key.stem}: training", flush=True)
            rc = subprocess.call(cmd)
            if rc != 0:
                print(f"{key.stem}: failed (exit {rc})", flush=True)
                failed.append(key.stem)
    if failed:
        print(f"failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
