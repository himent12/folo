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
- Outside `/api`, the server serves the built web app (`web/dist`, `--web-dir`, or the copy embedded in `embed-web`
  builds) with an SPA fallback to `index.html`; without a build it answers 404 with a hint. Embedded files carry an
  ETag; Vite's content-hashed `assets/` are `Cache-Control: immutable`, everything else `no-cache`.

## Errors
Non-2xx responses: `{ "error": { "code": "dataset_not_found", "message": "...", "hint": "..." | null } }`.
`hint` is always present (`null` when there is nothing useful to add). Codes are stable:

| status | code | when |
|---|---|---|
| 400 | `bad_request`, `invalid_path`, `invalid_query`, `invalid_body`, `invalid_config`, `unknown_format`, `invalid_dataset` | malformed input (`invalid_config`: a `TrainConfig` field is out of range, or the dataset has no classes / training images; the message names it. `invalid_dataset`: e.g. more than 4096 classes) |
| 403 | `forbidden_origin` | a state-changing request from a disallowed browser origin (see Conventions) |
| 404 | `dataset_not_found`, `image_not_found`, `model_not_found`, `run_not_found`, `path_not_found`, `epoch_not_available`, `route_not_found` | something named does not exist |
| 405 | `method_not_allowed` | wrong HTTP method on a known path |
| 409 | `run_active`, `run_not_active`, `run_not_resumable` | deleting (or resuming) a run that is still active; stopping one that is not; resuming one that completed |
| 413 | `payload_too_large` | body over the limit (64 MiB for `/api/predict`, 2 MiB elsewhere) |
| 415 | `unsupported_media_type` | JSON endpoint called without `Content-Type: application/json` |
| 422 | `invalid_body`, `image_decode_failed`, `invalid_weights` | well-formed JSON with wrong types / missing required fields; an upload that is not an image; a `.folo` file that cannot be read |
| 500 | `internal` | a bug or an IO failure |
| 501 | `not_implemented` | the feature exists in the API but not yet in the engine (ONNX export, streaming inference, a `gpu:<n>` device this build cannot train or run models on; in mock mode, resuming runs) |

## System
- `GET /api/health` → `{ "status": "ok", "version": "0.1.0", "api": "v1", "mock": false }`
- `GET /api/system` → `{ "cpu": { "name", "threads", "util" }, "memory": { "used_bytes", "total_bytes" },
  "gpus": [{ "ordinal", "name", "backend": "vulkan|dx12|metal", "memory_bytes", "util" | null }] }`
  (`util` in `0..1`; CPU utilisation is measured since the previous call.) In real mode `gpus` comes from folo-gpu
  (wgpu): every physical adapter once, through its preferred backend (software adapters are left out), best first,
  so ordinal 0 is the adapter `gpu:0` and `device: auto` use. wgpu reports neither memory size nor load, so
  `memory_bytes` is `0` and `util` `null`.

## Datasets
- `GET /api/datasets` → `{ "datasets": [Dataset] }` (oldest first, without `stats`)
- `POST /api/datasets` body `{ "name"?, "path", "format": "auto|yolo|coco" }` → `201 Dataset` with `stats`.
  Registers a dataset on disk: a folder, a `data.yaml` or a COCO instances `.json` (relative paths are resolved
  against the server's working directory and stored absolute in `path`); `name` defaults to the folder name,
  `format` to `auto`. Datasets are read with folo-data, exactly as for training (formats and detection rules:
  `docs/DATA.md`): YOLO from `data.yaml` (`path`, `train`, `val`, `test`, `names`; a YAML with only `names` uses the
  folder layout) or `images/{train,val,test}` + `labels/…`; COCO from `annotations/instances_*.json` (Roboflow
  layouts too). Corrupt or missing images and images with broken labels are left out (`stats.broken_labels` counts
  the latter; `folo dataset check` lists them all). Without `names` / `nc`, classes are named `class0..` up to the
  largest label index. Errors: `path_not_found`, `unknown_format` (nothing there looks like a dataset),
  `invalid_dataset` (malformed `data.yaml` / JSON, `nc` disagreeing with `names`, more than 4096 classes). The
  request returns once the dataset is indexed (every image header is read, in parallel).
- `POST /api/datasets/synthetic` body `{ "name", "images": 1000, "classes": ["circle","square","triangle"], "imgsz": 640, "seed": 0 }`
  → `201 Dataset` with `stats`. Only `name` is required. Generates square JPEG images (folo-data's shapes
  generator, deterministic in `images`, `classes`, `imgsz` and `seed`) into `<data dir>/datasets/<name>` (the name
  lowercased to `a-z0-9-`, with `-2`, `-3`, ... when taken) and registers them. Images are split 80 / 20 into
  train / val. `classes`: 1 to 8 distinct names; `circle`, `square`, `triangle`, `star`, `ring`, `cross`,
  `hexagon` and `heart` get that shape, every other name the next shape no name has claimed. `400 invalid_body`
  when `name` is blank, `images` is outside `1..=100000`, `imgsz` outside `32..=4096`, or `classes` breaks those
  rules (the message says which).
- `GET /api/datasets/{id}` → `Dataset` with `stats`
- `GET /api/datasets/{id}/images?split=train|val|test&offset=0&limit=100&class=<id>` →
  `{ "total", "images": [{ "id", "file", "split", "width", "height", "boxes": [Box] }] }`.
  Every parameter is optional: no `split` means all splits; `limit` defaults to 100 and is capped at 1000; `class`
  keeps images with at least one box of that class. `total` counts all matches before `offset` / `limit`. `file` is
  relative to the dataset root (the registered folder, or the root folo-data resolves from it: a `data.yaml`'s
  `path:`, the folder above `annotations/`), absolute for images outside it. Image ids are the position in the dataset (train, then val, then test, each
  in path order): stable while the files do not change.
- `GET /api/datasets/{id}/issues?kind=<kind>&offset=0&limit=100` →
  `{ "total", "issues": [{ "kind", "severity": "error|warning|info", "file", "line": null | 1.., "message" }] }`:
  what `folo dataset check` finds in the dataset (the same objects as its `--json` `issues`, without `fix`; kinds
  and severities are listed in docs/CLI.md), in dataset order. Computed together with the stats and cached with
  them (re-computed when the files change); in real mode image headers are read but images are not fully decoded
  (`folo dataset check` without `--headers-only` also catches truncated image data). `kind` keeps one kind
  (`broken_label`, `class_overflow`, `out_of_range_box`, `clipped_box`, `zero_size_box`, `duplicate_label`,
  `corrupt_image`, `missing_image`, `duplicate_image`, `empty_image`, `other`; anything else is `400
  invalid_query`); `limit` defaults to 100 and is capped at 1000; `total` counts all matches before `offset` /
  `limit`. `file` is relative to the dataset root like image files (and exactly as `folo dataset check --json` names it). Issues of images the loader skipped
  name files that `/images` does not list.
- `GET /api/datasets/{id}/images/{image_id}` → original image bytes (`Content-Type` from the file); `?thumb=256`
  returns a JPEG whose long side is 256 (clamped to 16..2048, never upscaled; EXIF orientation applied, so boxes
  line up). Mock images are immutable and sent with a long `Cache-Control`; real ones with `no-cache`.
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
- `GET /api/models` → `{ "models": [Model] }` (newest first by `created`). In real mode: the weights of every run that
  ended `completed` or `stopped` with a checkpoint (`weights/best.folo` as `m_<run id suffix>`, named
  `<run name>-best`, with the run's best metrics; `weights/last.folo` as `m_<suffix>-last`, `<run name>-last`, with
  the last validated epoch's metrics), plus every `.folo` file under `<data dir>/models/` (recursively; id `m_` and 8
  hex digits of the path's hash, stable across restarts; name = file stem; `metrics` from `folo.metrics` when the
  file has them). `params` counts the stored learnable tensors, `flops_g` is the architecture's GFLOPs at `imgsz`
  (docs/MODEL.md §3's 640 px figure scaled by the pixel count), `created` is `folo.created` or the file's
  modification time.
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
  In real mode the model's dataset is its run's (the registered copy while it is registered, else the path the run
  trained on); a model from the models folder is evaluated on the first registered dataset with exactly its classes
  (`404 dataset_not_found` when there is none). Computed with folo-vision's validator (docs/METRICS.md; MODEL.md
  §8.7 settings: `conf 0.001`, `iou 0.7`, the model's `imgsz`) on the server's `--device`, then cached (in memory and
  in `cache/eval/<model id>.json`) until the weights file or the dataset's files change. `pr` has curves for the
  classes with ground truth.
- `POST /api/models/{id}/export` body `{ "format": "folo-f16|onnx" }` → `{ "path", "bytes" }`. Real mode writes the
  fused EMA weights as f16 (as `folo export` does, docs/FORMAT.md) to `<data dir>/exports/<model name>-f16.folo`
  (`-f16-2`, `-f16-3`, ... when taken; never overwriting). `onnx` answers `501 not_implemented`.

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
  `model_not_found` (bad `weights`), `invalid_config` (checked before anything is queued: the limits below, the
  trainer's (real mode: `imgsz >= 64`), a dataset without classes or training images), `not_implemented` (a
  `gpu:<n>` device the engine cannot train on). Runs train one at a time, first come first served; the others wait
  `queued` (their log says how many runs are ahead).
  With `"resume_run_id": "<run id>"` in the body, the request **resumes** that `stopped` or `failed` run instead →
  `200 Run` (same id, status `queued` again): from its `weights/last.folo` with the full checkpoint state
  (docs/FORMAT.md: epoch, optimizer, schedule, EMA, RNG), or from scratch when no epoch had finished. The run keeps
  its config: every other field in the body must equal it or be omitted (`dataset_id` too; an empty `name` counts
  as omitted), except `device`, which may change. Errors: `run_not_found`, `run_active` (still queued / running /
  stopping), `run_not_resumable` (it completed; start a new run with `weights` set to its model instead),
  `invalid_config` (a field differs; the message names it), `dataset_not_found` (its dataset folder is gone).
  `weights: <model id>` without `resume_run_id` starts a **new** run initialised from those weights (matching
  tensors load; the class outputs are re-initialised when the class lists differ; no optimizer state).
- `GET /api/runs/{id}` → `Run`
- `POST /api/runs/{id}/stop` → `Run` (status `stopping` → `stopped`; the last checkpoint is kept). `409 run_not_active`
  when the run already finished. A queued run stops at once; a running one after its current batch (the epoch in
  progress is discarded; `weights/last.folo` is the last finished epoch's). `model_id` is set when at least one epoch
  was validated. The final `status` event is sent once the run can be resumed or deleted.
- `DELETE /api/runs/{id}` → `204` (`409 run_active` while `queued`, `running` or `stopping`). Real mode removes the run
  folder, its weights included, so its models disappear too.
- `GET /api/runs/{id}/metrics?every=1` → `{ "steps": [Step], "epochs": [Epoch] }` (full history, for charts on page
  load; `every=N` keeps every N-th step plus the latest one, to thin long runs)
- `GET /api/runs/{id}/samples?epoch=<n>` → `{ "epoch", "samples": [{ "image_id", "dataset_id", "width", "height",
  "truth": [Box], "pred": [ScoredBox] }] }` (validation predictions at `conf >= 0.25` on a fixed set of 16 images,
  in original-image pixels: in real mode the first 16 of the validation split, or of the training split when there
  is none;
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
// batch 1..=1024, imgsz a multiple of 32 in 32..=4096 (real mode trains at 64 or more), 0 < lr0 < 10.
// POST /api/runs also takes "resume_run_id": "run_…" (see above); it is not part of Run.config.
// Real mode maps TrainConfig onto the trainer's config (docs/MODEL.md §8.1): the other recipe keys keep their
// defaults, and "auto" resolves to the GPU when a model can be placed on it, else the CPU.
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
  Real mode runs folo-vision's `Predictor` (docs/MODEL.md §9: EMA weights, fused, letterboxed `rect` at the model's
  `imgsz`) on the server's `--device` (`device` is `cpu` or `gpu:<n>`). Loaded models stay in an LRU of 4 (keyed by
  the weights file's path, size and modification time, so retrained weights are reloaded); `timing` is measured:
  `preprocess_ms` includes decoding the image, `postprocess_ms` covers decoding the head outputs, NMS and rescaling.
- `GET /api/predict/stream?model_id=..&conf=..` — WebSocket: the client sends JPEG frames (binary), the server
  answers each with the same JSON as `/api/predict` (for webcam / screen demos). Later milestone; answers
  `501 not_implemented` for now.

## Real mode
`folo serve` without `--mock` keeps its state in the data dir (`--data-dir`, default `$FOLO_HOME` or `~/.folo`):
- `datasets.json`: the registry (`id`, `name`, `path`, `format`, `created`, and a snapshot of `classes` and
  `splits`), rewritten atomically (temporary file + rename) on every change, so registrations survive restarts.
  A malformed registry is never overwritten: registering and unregistering fail with `500 internal` until it is
  fixed.
- `cache/datasets/<id>.json`: each dataset's index (images, sizes, boxes, stats, issues) plus a fingerprint of the files it
  was built from (relative path, size and modification time of every file under the dataset folder and under
  everything outside it that folo-data read: split folders and image lists a `data.yaml` points to, folders of
  every candidate image and label, also skipped ones, and split paths or a `path:` that do not exist yet; the data dir's own `cache/` and
  `datasets.json` are left out, so a data dir inside a dataset folder does not invalidate it). On start the
  registry is listed immediately and every dataset is checked on a background thread: a fresh cache is loaded, a
  stale or missing one rebuilt; requests for a dataset that is still being indexed wait for it. While running,
  `GET /api/datasets/{id}` and image listings re-check the fingerprint at most every 5 s and re-index when files
  were added, removed or edited. Datasets whose folder no longer exists are left out of the list (but stay in the
  registry).
- `cache/thumbs/<id>/`: thumbnails, keyed by the image's path, size, modification time and the requested side.
- `datasets/<name>/`: synthetic datasets. `DELETE /api/datasets/{id}` removes the dataset's caches, never its files.

- `runs/<id>/`: one folder per training run: `run.json` (the `Run` plus the dataset reference, rewritten atomically
  on every status change and after every epoch), `log.jsonl` (every log line, as `/logs` returns them),
  `config.json` (the trainer's resolved config), `metrics.jsonl` (every trainer event, every step; `/metrics` reads
  it, and a resumed run's repeated steps replace the earlier ones), `weights/last.folo` and `weights/best.folo`
  (full checkpoints, docs/FORMAT.md) and `samples/epoch_NNN.json`. On start every run is listed again; runs a
  previous process left `queued`, `running` or `stopping` become `stopped` (with a `warn` log line) and can be
  resumed with `resume_run_id`. On a graceful shutdown the active run stops after its current batch.
- `models/`: `.folo` files listed as models (scanned on every listing; metadata read once per file version).
- `exports/`: exported models. `cache/eval/`: cached `/eval` results.

## Validation
- `POST /api/val` body `{ "model_id", "dataset_id", "split": "val", "imgsz": null, "conf": 0.001, "iou": 0.7 }` →
  `{ "model_id", "dataset_id", "split", "imgsz", "images", "val": Epoch.val, "timing": { "total_s", "per_image_ms" } }`.
  Only `model_id` and `dataset_id` are required; `imgsz: null` uses the model's size. Errors: `model_not_found`,
  `dataset_not_found`, `invalid_body` (an empty split, `conf` / `iou` outside `0..=1`, a bad `imgsz`),
  `not_implemented` (real mode, until the evaluator is wired in). The MCP `val` tool uses it in proxy mode.
- Planned: `POST /api/predict/batch` (many uploads in one request). Until it exists the MCP proxy sends one
  `POST /api/predict` per image.

## Mock mode
`folo serve --mock` answers every endpoint with deterministic, realistic data (the same ids and values on every
start, except live-run timestamps):
- Datasets: `ds_5a1e` **shapes-1k** (circle / square / triangle, 640², 800 / 200), `ds_7c2f` **traffic-cams** (car /
  truck / bus / person / bicycle, 1280×720, 1900 / 400 / 100, 3 broken labels), `ds_3b9d` **r6-operators** (body /
  Head, 1920×1080, 700 / 160). Images are rendered procedurally on request; their boxes match the labels exactly.
  `/issues` is consistent with the stats: one error per broken label file (`broken_label`, `class_overflow` or
  `out_of_range_box`, for the 3 of traffic-cams), an `empty_image` per image without boxes, and a few
  deterministic `clipped_box` / `duplicate_image` warnings.
- Models: six (`m_31aa`, `m_31ab`, `m_42c1`, `m_42c2`, `m_53d1`, `m_53d2`), each linked to its run, with
  deterministic `/eval` results consistent with its metrics (smooth PR curves, a mostly diagonal confusion matrix).
- Runs: four completed, one stopped (`run_2b02`), one failed with an out-of-memory error (`run_2b03`), all with full
  metric histories; and one live run (`run_3c02`, r6-operators / folo-s) whose SSE stream advances in real time: a
  100-epoch run compressed into ~10 minutes (6 s per epoch, ~7 steps per second), starting around epoch 12, then it
  loops (`status` back to `queued`, metrics restart).
- `POST /api/runs` starts a simulated run (6 s per epoch) that can be stopped; when it completes, its model appears
  in `/api/models`. `POST /api/datasets` scans real folders; `POST /api/datasets/synthetic` makes a procedural
  dataset. Predictions are deterministic in the image bytes (or dataset image) and the model. `POST /api/val` derives
  per-class metrics from the model's metrics (deterministic per model, dataset and split).
