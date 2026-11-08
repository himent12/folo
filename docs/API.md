# FOLO Studio HTTP API (v1)

Served by `folo serve` (default `http://127.0.0.1:7878`). JSON over HTTP, live updates over Server-Sent Events.
All paths are under `/api`. Times are RFC 3339 strings; durations are seconds (`*_s`) or milliseconds (`*_ms`).
Boxes in responses are **pixel `xyxy`** in the original image unless a field says otherwise.

This file is the contract between `folo-server` and `web/`. Change it in the same PR as the code that changes the
behaviour, and keep `web/src/lib/api/types.ts` in sync.

## Errors
Non-2xx responses: `{ "error": { "code": "dataset_not_found", "message": "...", "hint": "..." } }`.

## System
- `GET /api/health` → `{ "status": "ok", "version": "0.1.0", "mock": false }`
- `GET /api/system` → `{ "cpu": { "name", "threads", "util" }, "memory": { "used_bytes", "total_bytes" },
  "gpus": [{ "ordinal", "name", "backend": "vulkan|dx12|metal", "memory_bytes", "util" | null }] }`

## Datasets
- `GET /api/datasets` → `{ "datasets": [Dataset] }`
- `POST /api/datasets` body `{ "name", "path", "format": "auto|yolo|coco" }` → `Dataset` (registers a folder on disk)
- `POST /api/datasets/synthetic` body `{ "name", "images": 1000, "classes": ["circle","square","triangle"], "imgsz": 640, "seed": 0 }` → `Dataset`
- `GET /api/datasets/{id}` → `Dataset` with `stats`
- `GET /api/datasets/{id}/images?split=train|val|test&offset=0&limit=100&class=<id>` →
  `{ "total", "images": [{ "id", "file", "split", "width", "height", "boxes": [{ "class", "x1", "y1", "x2", "y2" }] }] }`
- `GET /api/datasets/{id}/images/{image_id}` → original image bytes; `?thumb=256` returns a JPEG thumbnail of that long side.
- `DELETE /api/datasets/{id}` → `204` (unregisters; never deletes files)

```jsonc
// Dataset
{ "id": "ds_8f2c", "name": "shapes", "path": "D:/data/shapes", "format": "yolo",
  "classes": ["circle", "square", "triangle"], "created": "2026-11-08T06:00:00Z",
  "splits": { "train": 800, "val": 200, "test": 0 },
  "stats": {                                   // present on GET /api/datasets/{id}
    "boxes": 4210, "empty_images": 12, "broken_labels": 0,
    "class_counts": [1400, 1390, 1420],
    "box_size_hist": { "edges": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], "counts": [5, 80, 600, 1500, 1400, 600, 25] },
    "boxes_per_image_hist": { "edges": [0,1,2,4,8,16,32], "counts": [12, 100, 300, 350, 200, 38] }
  } }
```

## Models
- `GET /api/models` → `{ "models": [Model] }` (`.folo` files under the models directory and run outputs)
- `GET /api/models/{id}` → `Model`
- `POST /api/models/{id}/export` body `{ "format": "folo-f16|onnx" }` → `{ "path", "bytes" }`

```jsonc
// Model
{ "id": "m_31aa", "name": "shapes-n-best", "arch": "folo-n", "imgsz": 640, "classes": ["circle","square","triangle"],
  "params": 2630000, "flops_g": 6.6, "bytes": 5400000, "created": "...", "run_id": "run_7d1e",
  "metrics": { "map50": 0.91, "map50_95": 0.74, "precision": 0.9, "recall": 0.86 } }
```

## Training runs
- `GET /api/runs` → `{ "runs": [Run] }` (newest first)
- `POST /api/runs` body `TrainConfig` → `Run` (status `queued`, then `running`)
- `GET /api/runs/{id}` → `Run`
- `POST /api/runs/{id}/stop` → `Run` (status `stopping` → `stopped`; the last checkpoint is kept)
- `DELETE /api/runs/{id}` → `204` (only when not running)
- `GET /api/runs/{id}/metrics` → `{ "steps": [Step], "epochs": [Epoch] }` (full history, for charts on page load)
- `GET /api/runs/{id}/samples?epoch=<n>` → `{ "epoch", "samples": [{ "image_id", "dataset_id", "width", "height",
  "truth": [Box], "pred": [ScoredBox] }] }` (validation predictions on a fixed set of 16 images)
- `GET /api/runs/{id}/events` → SSE stream. Each SSE `event:` name is the payload type:
  - `status` `{ "status": "queued|running|stopping|stopped|completed|failed", "error": null | "..." }`
  - `step` `Step` (at most ~10 per second)
  - `epoch` `Epoch` (after each validation)
  - `log` `{ "t", "level": "debug|info|warn|error", "message" }`
  The stream starts with a `status` event and replays nothing else (use `/metrics` for history).

```jsonc
// TrainConfig
{ "name": "shapes-n", "dataset_id": "ds_8f2c", "arch": "folo-n", "epochs": 100, "batch": 16, "imgsz": 640,
  "device": "auto|cpu|gpu:0", "optimizer": "auto|sgd|adamw", "lr0": 0.01, "weights": null | "m_31aa",
  "augment": { "mosaic": 1.0, "mixup": 0.0, "hsv": [0.015, 0.7, 0.4], "fliplr": 0.5, "scale": 0.5, "translate": 0.1,
               "close_mosaic": 10 },
  "seed": 0 }
// Run
{ "id": "run_7d1e", "name": "shapes-n", "status": "running", "config": TrainConfig, "created": "...",
  "started": "...", "finished": null, "epoch": 12, "epochs": 100, "step": 340, "steps_per_epoch": 50,
  "eta_s": 1840, "best": { "epoch": 11, "map50_95": 0.61 } | null, "device": "NVIDIA GeForce RTX 4080 SUPER (vulkan)",
  "model_id": null | "m_31aa", "error": null }
// Step
{ "epoch": 12, "step": 340, "lr": 0.0098, "loss": { "box": 1.21, "cls": 0.83, "dfl": 1.02, "total": 3.06 },
  "imgs_per_s": 412.5, "eta_s": 1840 }
// Epoch
{ "epoch": 12, "train_loss": { "box": 1.18, "cls": 0.80, "dfl": 1.00, "total": 2.98 },
  "val": { "map50": 0.82, "map50_95": 0.61, "precision": 0.84, "recall": 0.77,
           "per_class": [{ "class": 0, "map50": 0.85, "map50_95": 0.64, "precision": 0.86, "recall": 0.80 }] },
  "duration_s": 31.2 }
```

## Inference
- `POST /api/predict` — `multipart/form-data` with `image` (file), `model_id`, optional `conf` (0.25), `iou` (0.7),
  `max_det` (300); or JSON `{ "model_id", "dataset_id", "image_id", "conf", "iou" }` →
  ```jsonc
  { "model_id": "m_31aa", "image": { "width": 1280, "height": 720 },
    "boxes": [{ "class": 1, "name": "square", "score": 0.93, "x1": 10, "y1": 20, "x2": 200, "y2": 180 }],
    "timing": { "preprocess_ms": 0.8, "inference_ms": 3.1, "postprocess_ms": 0.2, "total_ms": 4.1 },
    "device": "gpu:0" }
  ```
- `GET /api/predict/stream?model_id=..&conf=..` — WebSocket: the client sends JPEG frames (binary), the server
  answers each with the same JSON as `/api/predict` (for webcam / screen demos). Later milestone.

## Mock mode
`folo serve --mock` answers every endpoint with deterministic, realistic data: three datasets, a handful of models,
one finished run with a full metric history, and one live run whose SSE stream advances in real time (a 100-epoch run
compressed into ~10 minutes, then it loops). Images come from the synthetic shapes generator.
