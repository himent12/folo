# FOLO Studio HTTP API (v1)

Served by `folo serve` (default `http://127.0.0.1:7878`). JSON over HTTP, live updates over Server-Sent Events.
All paths are under `/api`. Times are RFC 3339 strings; durations are seconds (`*_s`) or milliseconds (`*_ms`).
Boxes in responses are **pixel `xyxy`** in the original image unless a field says otherwise. Ids (`ds_…`, `m_…`,
`run_…`, image ids) are opaque strings.

This file is the contract between `folo-server` and its clients. Change it in the same PR as the code that changes the
behaviour, and keep the Rust types in `crates/folo-api` (used by the server, the CLI and FOLO Desktop) and
`web/src/lib/api/types.ts` in sync.

## Conventions
- Creating endpoints answer `201 Created` with the new object; deletes answer `204 No Content`; everything else `200`.
- Request bodies are JSON (`Content-Type: application/json`) except multipart uploads to `/api/predict`. Unknown
  fields are ignored; missing optional fields take the defaults listed here.
- Responses are gzip-compressed when the client sends `Accept-Encoding: gzip` (never images or SSE).
- CORS: by default only browser origins on `localhost`, `127.0.0.1` or `[::1]` (any port, e.g. the Vite dev server
  on :5173) may call the API; `folo serve --cors-any` allows every origin. Because browsers send simple form POSTs
  without a preflight, state-changing requests (`POST`, `DELETE`) whose `Origin` header is neither such a loopback
  origin nor the server itself are refused with `403 forbidden_origin`. Requests without `Origin` (curl, agents,
  FOLO Desktop) are unaffected.
- Outside `/api`, the server serves the built web app (`web/dist`, `--web-dir`) with an SPA fallback to
  `index.html`; without a build it answers 404 with a hint.

## Errors
Non-2xx responses: `{ "error": { "code": "dataset_not_found", "message": "...", "hint": "..." | null } }`.
`hint` is always present (`null` when there is nothing useful to add). Codes are stable:

| status | code | when |
|---|---|---|
| 400 | `bad_request`, `invalid_path`, `invalid_query`, `invalid_body`, `invalid_config`, `unknown_format`, `invalid_dataset` | malformed input (`invalid_config`: a `TrainConfig` field is out of range, or the dataset has no classes / training images; the message names it. `invalid_dataset`: e.g. more than 4096 classes) |
| 403 | `forbidden_origin` | a state-changing request from a disallowed browser origin (see Conventions) |
| 404 | `dataset_not_found`, `image_not_found`, `model_not_found`, `run_not_found`, `path_not_found`, `epoch_not_available`, `route_not_found` | something named does not exist |
| 405 | `method_not_allowed` | wrong HTTP method on a known path |
| 409 | `run_active`, `run_not_active` | deleting a run that is still active; stopping one that is not |
| 413 | `payload_too_large` | body over the limit (64 MiB for `/api/predict`, 2 MiB elsewhere) |
| 415 | `unsupported_media_type` | JSON endpoint called without `Content-Type: application/json` |
| 422 | `invalid_body`, `image_decode_failed` | well-formed JSON with wrong types / missing required fields; an upload that is not an image |
| 500 | `internal` | a bug or an IO failure |
| 501 | `not_implemented` | the feature exists in the API but not yet in the engine (real mode: training, inference, export, synthetic datasets, COCO) |

## System
- `GET /api/health` → `{ "status": "ok", "version": "0.1.0", "api": "v1", "mock": false }`
- `GET /api/system` → `{ "cpu": { "name", "threads", "util" }, "memory": { "used_bytes", "total_bytes" },
  "gpus": [{ "ordinal", "name", "backend": "vulkan|dx12|metal", "memory_bytes", "util" | null }] }`
  (`util` in `0..1`; CPU utilisation is measured since the previous call. `gpus` is empty until GPU detection lands.)

## Datasets
- `GET /api/datasets` → `{ "datasets": [Dataset] }` (oldest first, without `stats`)
- `POST /api/datasets` body `{ "name"?, "path", "format": "auto|yolo|coco" }` → `201 Dataset` with `stats`.
  Registers a folder on disk; `name` defaults to the folder name, `format` to `auto`. YOLO folders are read from
  `data.yaml` (`path`, `train`, `val`, `test`, `names`) or `images/{train,val,test}` + `labels/…`. Errors:
  `path_not_found`, `unknown_format`, `invalid_dataset`, `not_implemented` (COCO for now). Without `names` / `nc`,
  classes are named `class0..` up to the largest label index; indices of 4096 or more count as broken labels.
- `POST /api/datasets/synthetic` body `{ "name", "images": 1000, "classes": ["circle","square","triangle"], "imgsz": 640, "seed": 0 }`
  → `201 Dataset` with `stats`. Only `name` is required. Images are split 80 / 20 into train / val.
- `GET /api/datasets/{id}` → `Dataset` with `stats`
- `GET /api/datasets/{id}/images?split=train|val|test&offset=0&limit=100&class=<id>` →
  `{ "total", "images": [{ "id", "file", "split", "width", "height", "boxes": [Box] }] }`.
  Every parameter is optional: no `split` means all splits; `limit` defaults to 100 and is capped at 1000; `class`
  keeps images with at least one box of that class. `total` counts all matches before `offset` / `limit`. `file` is
  relative to the dataset root.
- `GET /api/datasets/{id}/images/{image_id}` → original image bytes (`Content-Type` from the file); `?thumb=256`
  returns a JPEG whose long side is 256 (clamped to 16..2048, never upscaled). Mock images are immutable and sent
  with a long `Cache-Control`.
- `DELETE /api/datasets/{id}` → `204` (unregisters; never deletes files)

```jsonc
// Dataset
{ "id": "ds_8f2c", "name": "shapes", "path": "D:/data/shapes", "format": "yolo|coco",
  "classes": ["circle", "square", "triangle"], "created": "2026-11-08T06:00:00Z",
  "splits": { "train": 800, "val": 200, "test": 0 },
  "stats": {                                   // present on GET /api/datasets/{id} and on creation
    "boxes": 4210, "empty_images": 12, "broken_labels": 0,
    "class_counts": [1400, 1390, 1420],
    // relative box size = sqrt(box area / image area)
    "box_size_hist": { "edges": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], "counts": [5, 80, 600, 1500, 1400, 600, 25] },
    "boxes_per_image_hist": { "edges": [0,1,2,4,8,16,32], "counts": [12, 100, 300, 350, 200, 38] }
  } }
// Histograms: counts[i] counts values in [edges[i], edges[i+1]); the last bin also takes values above its upper edge.
// Box (ground truth)
{ "class": 0, "x1": 10, "y1": 20, "x2": 200, "y2": 180 }
// ScoredBox (a prediction)
{ "class": 1, "name": "square", "score": 0.93, "x1": 10, "y1": 20, "x2": 200, "y2": 180 }
```

## Models
- `GET /api/models` → `{ "models": [Model] }` (newest first; `.folo` files under the models directory and run outputs)
- `GET /api/models/{id}` → `Model`
- `GET /api/models/{id}/eval` → validation results of the model on its dataset's `val` split (`train` when there is
  none), for the PR curves and confusion matrix of the Models view:
  ```jsonc
  { "pr": [{ "class": 0, "recall": [0.0, 0.01, ..., 1.0], "precision": [1.0, ..., 0.0], "ap50": 0.91 }],
    "confusion": [[412, 3, 1, 21], [2, 398, 4, 30], [0, 6, 405, 25], [9, 14, 7, 0]],
    "per_class": [ClassMetrics],   // as in Epoch.val.per_class; map50 equals pr[class].ap50
    "images": 200 }
  ```
  `pr` has one curve per class, in class order, sampled at 101 recall points (`0.00` to `1.00`, IoU 0.5).
  `precision` is the COCO-interpolated envelope (it never rises with recall), and `ap50` is its mean.
  `confusion` is `(classes + 1) × (classes + 1)`: rows are the true class, columns the predicted class, and the
  last row and column are background. `confusion[c][n]` counts missed objects of class `c`, `confusion[n][c]`
  counts false positives predicted as `c`, and `confusion[n][n]` is always 0.
- `POST /api/models/{id}/export` body `{ "format": "folo-f16|onnx" }` → `{ "path", "bytes" }`

```jsonc
// Model
{ "id": "m_31aa", "name": "shapes-n-best", "arch": "folo-n|folo-s|folo-m|folo-l|folo-x", "imgsz": 640,
  "classes": ["circle","square","triangle"], "params": 2630000, "flops_g": 6.6, "bytes": 5400000,
  "created": "...", "run_id": "run_7d1e" | null,
  "metrics": { "map50": 0.91, "map50_95": 0.74, "precision": 0.9, "recall": 0.86 } | null }
```

## Training runs
- `GET /api/runs` → `{ "runs": [Run] }` (newest first)
- `POST /api/runs` body `TrainConfig` → `201 Run` (status `queued`, then `running`). Errors: `dataset_not_found`,
  `model_not_found` (bad `weights`), `invalid_config`.
- `GET /api/runs/{id}` → `Run`
- `POST /api/runs/{id}/stop` → `Run` (status `stopping` → `stopped`; the last checkpoint is kept). `409 run_not_active`
  when the run already finished.
- `DELETE /api/runs/{id}` → `204` (`409 run_active` while `queued`, `running` or `stopping`)
- `GET /api/runs/{id}/metrics?every=1` → `{ "steps": [Step], "epochs": [Epoch] }` (full history, for charts on page
  load; `every=N` keeps every N-th step plus the latest one, to thin long runs)
- `GET /api/runs/{id}/samples?epoch=<n>` → `{ "epoch", "samples": [{ "image_id", "dataset_id", "width", "height",
  "truth": [Box], "pred": [ScoredBox] }] }` (validation predictions at `conf >= 0.25` on a fixed set of 16 images;
  `epoch` defaults to the latest finished epoch; `404 epoch_not_available` before the first validation or for a
  future epoch)
- `GET /api/runs/{id}/logs?limit=500` → `{ "lines": [{ "t", "level": "debug|info|warn|error", "message" }] }`:
  the run's log history, most recent `limit` lines (default 500, clamped to 1..5000), oldest first. The lines are
  the same as the SSE `log` events, so a dashboard opened mid-run loads this and then appends live events. The
  shape is stable (the MCP proxy uses it, docs/MCP.md).
- `GET /api/runs/{id}/events` → SSE stream. Each SSE `event:` name is the payload type, `data:` is its JSON:
  - `status` `{ "status": "queued|running|stopping|stopped|completed|failed", "error": null | "..." }`
  - `step` `Step` (at most ~10 per second; steps in between are skipped, `/metrics` has them all)
  - `epoch` `Epoch` (after each validation)
  - `log` `{ "t", "level": "debug|info|warn|error", "message" }`

  The stream starts with a `status` event and replays nothing else (use `/metrics` and `/logs` for history). A `: keep-alive`
  comment is sent every 15 s. The stream stays open after the run finishes (so `EventSource` does not reconnect in a
  loop) and ends when the run is deleted or the server shuts down. Any number of clients may subscribe. A client too
  slow to keep up gets a `warn` `log` event saying how many events it missed. Clients must ignore unknown event
  names (new types may be added).

```jsonc
// TrainConfig (only dataset_id is required; the rest shows the defaults)
{ "name": "", "dataset_id": "ds_8f2c", "arch": "folo-n", "epochs": 100, "batch": 16, "imgsz": 640,
  "device": "auto",            // auto | cpu | gpu:<n>
  "optimizer": "auto",         // auto | sgd | adamw
  "lr0": 0.01, "weights": null, // or a model id to fine-tune from
  "augment": { "mosaic": 1.0, "mixup": 0.0, "hsv": [0.015, 0.7, 0.4], "fliplr": 0.5, "scale": 0.5, "translate": 0.1,
               "close_mosaic": 10 },
  "seed": 0 }
// An empty name becomes "<dataset name>-<arch suffix>", e.g. "shapes-1k-n". Limits: epochs 1..=10000,
// batch 1..=1024, imgsz a multiple of 32 in 32..=4096, 0 < lr0 < 10.
// Run
{ "id": "run_7d1e", "name": "shapes-n", "status": "running", "config": TrainConfig, "created": "...",
  "started": "..." | null, "finished": null | "...",
  "epoch": 12,             // epoch in progress, 1-based (0 before the first step; the last epoch once finished)
  "epochs": 100,
  "step": 575,             // optimizer steps completed over all epochs
  "steps_per_epoch": 50,
  "eta_s": 1840 | null,    // null unless queued / running / stopping
  "best": { "epoch": 11, "map50_95": 0.61 } | null, "device": "NVIDIA GeForce RTX 4080 SUPER (vulkan)",
  "model_id": null | "m_31aa",  // set once the run completed, or stopped after at least one validated epoch
  "error": null | "..." }
// Step
{ "epoch": 12, "step": 575, "lr": 0.0098, "loss": { "box": 1.21, "cls": 0.83, "dfl": 1.02, "total": 3.06 },
  "imgs_per_s": 412.5, "eta_s": 1840 }      // step is the global, 1-based step number
// Epoch
{ "epoch": 12, "train_loss": { "box": 1.18, "cls": 0.80, "dfl": 1.00, "total": 2.98 },
  "val": { "map50": 0.82, "map50_95": 0.61, "precision": 0.84, "recall": 0.77,
           "per_class": [{ "class": 0, "map50": 0.85, "map50_95": 0.64, "precision": 0.86, "recall": 0.80 }] },
  "duration_s": 31.2 }
```

## Inference
- `POST /api/predict` — `multipart/form-data` with `image` (file), `model_id`, optional `conf` (0.25), `iou` (0.7),
  `max_det` (300) (instead of `image`, the form may name `dataset_id` + `image_id`); or JSON
  `{ "model_id", "dataset_id", "image_id", "conf"?, "iou"?, "max_det"? }` →
  ```jsonc
  { "model_id": "m_31aa", "image": { "width": 1280, "height": 720 },
    "boxes": [ScoredBox],                       // highest score first
    "timing": { "preprocess_ms": 0.8, "inference_ms": 3.1, "postprocess_ms": 0.2, "total_ms": 4.1 },
    "device": "gpu:0" }
  ```
  `conf` and `iou` must be in `0..=1`, `max_det >= 1`. Uploads up to 64 MiB; JPEG, PNG, BMP and WebP are accepted.
  Boxes only use the model's classes: on a dataset image, ground-truth classes the model does not know (by name) are
  never predicted.
- `GET /api/predict/stream?model_id=..&conf=..` — WebSocket: the client sends JPEG frames (binary), the server
  answers each with the same JSON as `/api/predict` (for webcam / screen demos). Later milestone; answers
  `501 not_implemented` for now.

## Mock mode
`folo serve --mock` answers every endpoint with deterministic, realistic data (the same ids and values on every
start, except live-run timestamps):
- Datasets: `ds_5a1e` **shapes-1k** (circle / square / triangle, 640², 800 / 200), `ds_7c2f` **traffic-cams** (car /
  truck / bus / person / bicycle, 1280×720, 1900 / 400 / 100, 3 broken labels), `ds_3b9d` **r6-operators** (body /
  Head, 1920×1080, 700 / 160). Images are rendered procedurally on request; their boxes match the labels exactly.
- Models: six (`m_31aa`, `m_31ab`, `m_42c1`, `m_42c2`, `m_53d1`, `m_53d2`), each linked to its run, with
  deterministic `/eval` results consistent with its metrics (smooth PR curves, a mostly diagonal confusion matrix).
- Runs: four completed, one stopped (`run_2b02`), one failed with an out-of-memory error (`run_2b03`), all with full
  metric histories; and one live run (`run_3c02`, r6-operators / folo-s) whose SSE stream advances in real time: a
  100-epoch run compressed into ~10 minutes (6 s per epoch, ~7 steps per second), starting around epoch 12, then it
  loops (`status` back to `queued`, metrics restart).
- `POST /api/runs` starts a simulated run (6 s per epoch) that can be stopped; when it completes, its model appears
  in `/api/models`. `POST /api/datasets` scans real folders; `POST /api/datasets/synthetic` makes a procedural
  dataset. Predictions are deterministic in the image bytes (or dataset image) and the model.
