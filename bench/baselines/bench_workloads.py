"""What ultralytics_bench.py measures: the models (names, image size, loading, the Ultralytics predictor end to
end) and the synthetic dataset it trains and predicts on.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from bench_timing import time_host

MODELS = ["yolo26n", "yolo26s", "yolo26m"]
TRAIN_MODELS = ["yolo26n", "yolo26s"]
IMGSZ = 640


def load_detection_model(weights: Path, device: str, half: bool):
    """The raw nn.Module Ultralytics runs at inference: fused (Conv+BN folded), eval mode, and, like AutoBackend does
    on x86 CPUs with oneDNN, channels_last weights on the CPU (inputs stay NCHW-contiguous, as the predictor sends)."""
    import torch
    from ultralytics import YOLO

    m = YOLO(str(weights)).model
    m = m.fuse(verbose=False).eval().to(device)
    if device == "cpu" and torch.backends.mkldnn.is_available():
        m = m.to(memory_format=torch.channels_last)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.half() if half else m.float()


def sample_images(data_dir: Path, n: int):
    """Decoded 640x640 BGR uint8 images from the synthetic val set (decode cost is not part of e2e timing)."""
    import cv2

    files = sorted((data_dir / "images" / "val").glob("*.jpg"))
    return [cv2.imread(str(files[i % len(files)])) for i in range(n)]


def predict_e2e(
    weights: Path, device: str, half: bool, batch: int, images, warmup: int, iters: int, threads: int | None = None
) -> dict:
    """Ultralytics predictor: letterbox + to-tensor + forward + postprocess + Results, boxes copied to the host.

    threads: torch intra-op threads to force before every call. The predictor's first call runs select_device(), which
    on the CPU resets torch to min(8, cores - 1) threads; None keeps whatever Ultralytics chooses.
    """
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = images[:batch]
    speeds = []
    used_threads = set()

    def fn():
        if threads:
            torch.set_num_threads(threads)
        res = model.predict(
            imgs, imgsz=IMGSZ, device=device, quantize=16 if half else None, verbose=False, batch=batch
        )
        for r in res:
            r.boxes.data.cpu()  # results on the host: forces the device sync
        speeds.append(res[0].speed)
        used_threads.add(torch.get_num_threads())

    r = time_host(fn, batch, warmup=warmup, iters=iters)
    speeds = speeds[warmup:]
    r["torch_threads_seen"] = sorted(used_threads)
    # Ultralytics' own per-image breakdown (ms per image, averaged over the batch)
    r["ultralytics_speed_ms_per_image"] = {
        k: round(statistics.median(s[k] for s in speeds), 4) for k in ("preprocess", "inference", "postprocess")
    }
    return r


def make_dataset(data_dir: Path, n_train: int, n_val: int, seed: int = 0) -> Path:
    """Shapes (rectangle / ellipse / triangle) on coloured noise, YOLO format, in a directory used for this
    (size, seed) only. Reuses a complete dataset; regenerates an interrupted one from scratch."""
    import cv2
    import numpy as np

    yaml_path = data_dir / "data.yaml"
    marker = data_dir / ".complete"
    if marker.exists():
        return yaml_path
    if data_dir.exists():  # an interrupted generation: start over, including Ultralytics' label caches
        import shutil

        shutil.rmtree(data_dir)
    rng = np.random.default_rng(seed)
    for split, count in [("train", n_train), ("val", n_val)]:
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = rng.integers(0, 256, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
            img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 3.0)))
            lines = []
            for _ in range(int(rng.integers(1, 9))):
                cls = int(rng.integers(0, 3))
                w, h = rng.uniform(0.05, 0.35, 2) * IMGSZ
                cx, cy = rng.uniform(w / 2, IMGSZ - w / 2), rng.uniform(h / 2, IMGSZ - h / 2)
                x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                color = tuple(int(c) for c in rng.integers(0, 256, 3))
                if cls == 0:
                    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, -1)
                elif cls == 1:
                    cv2.ellipse(img, (int(cx), int(cy)), (int(w / 2), int(h / 2)), 0, 0, 360, color, -1)
                else:
                    pts = np.array([[cx, y0], [x0, y1], [x1, y1]], dtype=np.int32)
                    cv2.fillPoly(img, [pts], color)
                lines.append(f"{cls} {cx / IMGSZ:.6f} {cy / IMGSZ:.6f} {w / IMGSZ:.6f} {h / IMGSZ:.6f}")
            cv2.imwrite(str(data_dir / "images" / split / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (data_dir / "labels" / split / f"{i:05d}.txt").write_text("\n".join(lines) + "\n")
    yaml_path.write_text(
        f"path: {data_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        "  0: rectangle\n  1: ellipse\n  2: triangle\n"
    )
    marker.write_text("ok\n")
    return yaml_path
