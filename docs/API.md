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
- Request bodies are JSON (`Content-Type: application/json`) except multipart uploads (`/api/predict`,
  `/api/label/jobs/{id}/images`). Unknown fields are ignored; missing optional fields take the defaults listed here.
- Responses are gzip-compressed when the client sends `Accept-Encoding: gzip` (never images, SSE or weights
  files: `application/octet-stream`).
- CORS: by default only browser origins on `localhost`, `127.0.0.1` or `[::1]` (any port, e.g. the Vite dev server
  on :5173) may call the API; `folo serve --cors-any` allows every origin. Because browsers send simple form POSTs
  without a preflight, state-changing requests (`POST`, `PUT`, `DELETE`) whose `Origin` header is neither such a loopback
  origin nor the server itself are refused with `403 forbidden_origin`. Requests without `Origin` (curl, agents,
  FOLO Desktop) are unaffected.
- Token: a server started with `folo serve --token <secret>` (or `FOLO_TOKEN`) answers `401 unauthorized` to every
  `/api` request without `Authorization: Bearer <secret>`, except `GET /api/health` (so a client can tell a server
  that is up from one it may not use). The comparison is constant-time and the token is never logged. Without
  `--token` no header is needed (the default: the server listens on loopback). docs/CLOUD.md, *Securing a server*.
- Outside `/api`, the server serves the built web app (`web/dist`, `--web-dir`, or the copy embedded in `embed-web`
  builds) with an SPA fallback to `index.html`; without a build it answers 404 with a hint. Embedded files carry an
  ETag; Vite's content-hashed `assets/` are `Cache-Control: immutable`, everything else `no-cache`.

## Errors
Non-2xx responses: `{ "error": { "code": "dataset_not_found", "message": "...", "hint": "..." | null } }`.
`hint` is always present (`null` when there is nothing useful to add). Codes are stable, and they are one
vocabulary with the CLI and MCP (`folo_api::codes` lists every code with its status and the CLI's exit code; a
test keeps the sources in line with it). Messages that wrap a driver report are cut to 500 characters:

| status | code | when |
|---|---|---|
| 400 | `bad_request`, `invalid_path`, `invalid_query`, `invalid_body`, `invalid_config`, `unknown_format`, `invalid_dataset` | malformed input (`invalid_config`: a `TrainConfig` field is out of range, or the dataset has no classes / training images; the message names it. `invalid_dataset`: e.g. more than 4096 classes) |
| 401 | `unauthorized` | a request without the server's token (see Conventions) |
| 403 | `forbidden_origin` | a state-changing request from a disallowed browser origin (see Conventions) |
| 404 | `dataset_not_found`, `image_not_found`, `model_not_found`, `run_not_found`, `label_job_not_found`, `path_not_found`, `device_not_found`, `epoch_not_available`, `route_not_found`, `upload_not_found`, `provider_not_found`, `connection_not_found`, `offer_not_found`, `instance_not_found` | something named does not exist (`device_not_found`: a `gpu:<n>` this machine has no usable adapter for; the hint lists the devices it has). The `*_not_found` of listed things hint at the list that has them |
| 405 | `method_not_allowed` | wrong HTTP method on a known path |
| 409 | `run_active`, `run_not_active`, `run_not_resumable`, `run_not_queued`, `label_job_active`, `label_job_not_active`, `upload_conflict`, `connection_in_use`, `cloud_refused` | deleting (or resuming) a run that is still active; stopping one that is not; resuming one that completed; moving one that does not wait in the queue; changing a labeling job while its agent run is active, or an image while it is in an agent call; stopping a job without a run; an upload part out of order or an upload completed before it is whole; removing a cloud connection with live instances or active runs; a cloud provider that says no (no capacity for the offer, no balance, a quota: its message is passed on) |
| 413 | `payload_too_large` | body over the limit (64 MiB for `/api/predict` and image uploads to a labeling job, 2 MiB elsewhere) |
| 415 | `unsupported_media_type` | JSON endpoint called without `Content-Type: application/json` |
| 422 | `invalid_body`, `image_decode_failed`, `invalid_weights`, `invalid_upload`, `cloud_auth_failed` | well-formed JSON with wrong types / missing required fields; an upload that is not an image; a `.folo` file that cannot be read; an uploaded archive that is not a readable tar or holds something unsafe; credentials a cloud provider rejects |
| 500 | `internal` | a bug or an IO failure |
| 501 | `not_implemented`, `not_supported` | the feature exists in the API but not yet in the engine (streaming inference; in mock mode, resuming runs); `not_supported`: valid, but this machine cannot run it (`POST /api/runs` with `"amp": true` where `capabilities.amp.available` is false: refused before anything is queued, with the same reason and a hint) |
| 502 | `cloud_unavailable`, `server_unreachable`, `bad_gateway` | a cloud provider's API does not answer or fails; a worker (another FOLO server) does not answer, or answers with something that is not this API. As `Run.error_code` of a cloud run also `cloud_boot_failed`, `cloud_sync_failed` and `cloud_instance_lost` (docs/CLOUD.md) |
| 507 | `gpu_out_of_memory` | the GPU ran out of memory (inference; a training run fails with this message in `Run.error`); the hint says to lower `batch` or `imgsz` or use a smaller `arch` |

## System
- `GET /api/health` → `{ "status": "ok", "version": "0.1.0", "api": "v1", "mock": false }`
- `GET /api/system` → `{ "cpu": { "name", "threads", "util" }, "memory": { "used_bytes", "total_bytes" },
  "gpus": [{ "ordinal", "name", "backend": "vulkan|dx12|metal", "memory_bytes", "util" | null }],
  "data_dir"?: "D:/folo-data", "features": ["export_onnx_switches", "queue_control", "weights_onnx"],
  "capabilities"?: Capabilities, "gpu_pool"?: { "device", "live_bytes", "cached_bytes", "peak_live_bytes" } }`
  (`util` in `0..1`; CPU utilisation is measured since the previous call. `data_dir` is the real server's
  `--data-dir`, with `/` separators, where its runs, models and exports live; the mock server leaves it out. `features` lists optional API features
  the server implements, so a client can check before relying on one: `export_onnx_switches` (the export switches
  below; older servers accept and ignore them), `queue_control` (`POST /api/runs/{id}/queue`, `Run.queue_position`
  and `queue` events), `weights_onnx` (`/weights?format=onnx`; older servers answer `400 invalid_query`), `run_cli` (`GET /api/runs/{id}/cli`
  and `RunEstimate.cli`), `web_wasm` (the Studio this server serves includes the in-browser runtime, folo-wasm,
  so its Playground can run models with WebGPU or the wasm CPU; left out for a build without it, or without a
  web app), `agent_labeling` (the `/api/label` endpoints, Agent labeling below), `uploads` (`/api/uploads`),
  `run_cursors` (`after_step` / `after_epoch` on `/metrics`, `after` and `total` on `/logs`, `Range` on `/weights`),
  `run_checkpoint` (`checkpoint` on `POST /api/runs`; these three are what a cloud worker needs, docs/CLOUD.md),
  `cloud` (`/api/cloud`, `TrainConfig.compute`, `Run.cloud` and `cloud` events). `folo mcp --server` refuses the
  export switches without `export_onnx_switches`.) In real mode `gpus` comes from folo-gpu
  (wgpu): every physical adapter once, through its preferred backend (software adapters are left out), best first,
  so ordinal 0 is the adapter `gpu:0` and `device: auto` use. wgpu reports neither memory size nor load, so
  `memory_bytes` is `0` and `util` `null`.
- `Capabilities` (`/api/system`'s `capabilities`, also in `folo doctor --json` and MCP's `folo_info`) says what a
  run with `device: auto` can use on this server, so a client can disable a switch, showing the reason, before it
  submits a run:
  ```json
  { "device": "gpu:0", "shader_f16": true,
    "tensor_cores": { "available": true, "enabled": false,
                      "reason": "off (not requested: FOLO_GPU_TENSOR_CORES=1 or --tensor-cores)" },
    "amp": { "available": false,
             "reason": "mixed-precision training (amp) needs a GPU with tensor cores (f16 cooperative matrices): gpu:0 has tensor cores off (not requested: ...); train without amp (f32), or ..." } }
  ```
  `device` is what `auto` resolves to (`gpu:0` or `cpu`; the real server creates its GPU context on the first
  request, as a run would). `shader_f16`: the device computes in f16, so `precision: f16` inference runs in half
  precision (else it falls back to f32). `tensor_cores.available`: the adapter reports the 16×16×16 f16 → f32
  cooperative matrices the tensor-core kernels need (docs/GPU.md, Tensor cores); `enabled`: they are on for this
  server (`folo serve --tensor-cores` / `FOLO_GPU_TENSOR_CORES=1`, and the start-up self-test passed); `reason`:
  why not, exactly as `folo doctor` prints it (`off (not requested: ...)`, `off: <adapter> (<backend>) has no
  cooperative matrices in wgpu`, `off: the tensor-core self-test failed (...)`, ...), `null` when enabled.
  `amp.available`: a run with `"amp": true` can start, the trainer's own check (`folo_vision::train::amp_support`, without its side effect)
  for that device; `amp.reason` is the `not_supported` message such a run would fail with, `null` when available.
  Every field has a default, so clients parse objects from older and newer servers; servers older than this field
  leave out `capabilities`. Studio and Desktop can disable the Mixed precision switch when `amp.available` is
  false and show `amp.reason`. `POST /api/runs` (and MCP `train`) refuse such a run with `501 not_supported`
  (`amp.reason` as the message) before anything is queued.
- `gpu_pool` (real mode, once the server has used the GPU): FOLO's GPU buffer pool. `live_bytes` are buffers in
  use (loaded predictors, a run's tensors), `cached_bytes` free buffers kept for reuse, `peak_live_bytes` the most
  in use since the last run or benchmark probe started. A finished run gives its cached buffers back to the driver
  before its final `status` event, and inference (`/api/predict`, `/api/val`, `/eval`, probes) does after 30 s
  without a request while no run is queued or training; the cached predictors (loaded models) are dropped after
  5 minutes of that. So an idle server holds no pooled GPU memory (`cached_bytes` 0, `live_bytes` 0 once the
  predictors are gone), and other processes on the GPU get it back. Absent in mock mode, on a CPU-only server and
  from servers older than the field.

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
  request returns once the dataset is indexed (every image header is read, in parallel). Registering a path that
  is already registered (the same folder by its canonical path, whatever the spelling, with `format` `auto` or
  the same format) returns the existing dataset (still `201`) instead of a duplicate; its `name` is kept.
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
- `GET /api/datasets/{id}/issues?kind=<kind>&severity=error|warning|info&offset=0&limit=100` →
  `{ "total", "issues": [{ "kind", "severity": "error|warning|info", "file", "line": null | 1.., "message",
  "fix"? }], "counts": { "error", "warning", "info", "by_kind": { kind: n } } }`:
  what `folo dataset check --fix-dry-run` finds in the dataset (the same objects as its `--json` `issues`; `fix`
  says what a cleanup would do and is absent for infos; kinds and severities are listed in docs/CLI.md), in
  dataset order. `counts` covers every issue of the dataset whatever the filter (clients of older servers
  without it read zeros). Computed together with the stats and cached with
  them (re-computed when the files change); in real mode image headers are read but images are not fully decoded
  (`folo dataset check` without `--headers-only` also catches truncated image data). `kind` keeps one kind
  (`broken_label`, `class_overflow`, `out_of_range_box`, `clipped_box`, `zero_size_box`, `duplicate_label`,
  `corrupt_image`, `missing_image`, `duplicate_image`, `empty_image`, `other`; anything else is `400
  invalid_query`), `severity` one severity (anything else is `400 invalid_query`); `limit` defaults to 100 and is
  capped at 1000; `total` counts all matches before `offset` /
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
  modification time. `head` is `folo.config`'s head type (`o2m` for files from before the o2o head), so a front end
  shows a model's head (and that it predicts without NMS) without looking up its run. `trained_with` comes from the
  file's `folo.train` (docs/FORMAT.md: training checkpoints, so run models and checkpoints copied into `models/`):
  its `amp`, `imgsz` and `epochs` (the planned epochs; `amp` is `false` for files from before mixed precision);
  `null` for files without it (`folo export` / `folo-f16` exports, imported weights) or with one that does not parse.
  `precision` is the stored weight dtype, `folo.dtype` (docs/FORMAT.md): `f32` (run checkpoints, f32 exports) or
  `f16` (`folo-f16` exports); `fused` is `folo.fused` (`true` for exports, whose Conv+BN are fused and
  which load for inference only; writers leave the key out of unfused files, so a missing key is `false`). Both are
  left out of the object (not `null`) when unknown: a file without `folo.dtype` (FOLO's writer always sets it), a
  dtype or `folo.fused` value this server has no name for (such a file still lists, and loading it says what is
  wrong). Clients read a `precision` value they do not know as unknown too, so a later dtype is additive. Metadata
  comes from the file's header alone (the first `8 + N` bytes, docs/FORMAT.md), never the tensor data, read once
  per file version; a file whose header is invalid is skipped with a warning in the server log.
  **Published weights:** `folo serve` and `folo mcp` also list the published weights downloaded with `folo weights
  pull` (the `folo weights` cache: `$FOLO_WEIGHTS_DIR`, else `<FOLO_HOME or ~/.folo>/weights`, whatever the
  `--data-dir`; docs/CLI.md, Weights). Their id and name are the published name (`folo-n-coco`), `source` is
  `published` and `about` is the line `folo weights list` prints, so `TrainConfig.weights: "folo-n-coco"`, predict,
  val, eval and export take the name everywhere a model id goes (Studio's "Start from", MCP `train { weights }`).
  A file pulled while the server runs lists at once. A published name that is not downloaded is `404
  model_not_found` with `folo weights pull <name>` as the hint.
  `source` (additive, left out by older servers): `run` (a run's `best` / `last`), `file` (`<data dir>/models`) or
  `published`; `about` only for published weights. `path` (additive): the weights file on the server's machine, `/`
  separators, what `folo train --weights`, `val --model` and `export --model` take there (left out by the mock).
- `GET /api/models/{id}` → `Model`. The weights of a run in progress are not listed, but its `m_<run suffix>-last`
  (and `m_<run suffix>`, once it has a `best.folo`) answer here and on `/weights` as soon as the file exists: how a
  cloud control plane copies a worker run's checkpoint while it trains (docs/CLOUD.md).
- `GET /api/models/{id}/eval` → validation results of the model on its dataset's `val` split (`train` when there is
  none, see Split fallback below), for the PR curves and confusion matrix of the Models view:
  ```jsonc
  { "pr": [{ "class": 0, "recall": [0.0, 0.01, ..., 1.0], "precision": [1.0, ..., 0.0], "ap50": 0.91 }],
    "confusion": [[412, 3, 1, 21], [2, 398, 4, 30], [0, 6, 405, 25], [9, 14, 7, 0]],
    "per_class": [ClassMetrics],   // as in Epoch.val.per_class; map50 equals pr[class].ap50
    "images": 200,                 // images of `split` evaluated
    "best_f1_conf": 0.312,         // the confidence per_class precision / recall are taken at (null: unknown)
    // What was evaluated (provenance):
    "dataset_id": "ds_5a1e", "dataset_name": "shapes-1k",
    "split": "val",                // the split actually evaluated: "val", or "train" on a fallback
    "split_fallback": null,        // or { "requested": "val", "reason": "empty_split", "message": "..." }
    "confusion_conf": 0.25,        // the confusion matrix counts detections with score > confusion_conf
    "confusion_iou": 0.45 }        // ... matched to a ground truth at IoU > confusion_iou (docs/METRICS.md §7)
  ```
  `pr` has one curve per class, in class order, sampled at 101 recall points (`0.00` to `1.00`, IoU 0.5).
  `precision` is the COCO-interpolated envelope (it never rises with recall), and `ap50` is its mean.
  `confusion` is `(classes + 1) × (classes + 1)`: rows are the true class, columns the predicted class, and the
  last row and column are background. `confusion[c][n]` counts missed objects of class `c`, `confusion[n][c]`
  counts false positives predicted as `c`, and `confusion[n][n]` is always 0.
  In real mode the model's dataset is its run's (the registered copy while it is registered, else the path the run
  trained on); a model from the models folder is evaluated on the first registered dataset with exactly its classes
  (`404 dataset_not_found` when there is none); `400 invalid_dataset` when the dataset's classes (names, in order)
  are no longer the model's. Computed with folo-vision's validator (docs/METRICS.md; MODEL.md
  §8.7 settings: `conf 0.001`, `iou 0.7`, the model's `imgsz`) on the server's `--device`, then cached (in memory and
  in `cache/eval/<model id>.json`) until the weights file or the dataset's files change (results cached by servers
  from before `best_f1_conf` or `split` are computed again once). `pr` has curves for the classes with ground truth.
- **Provenance.** `dataset_id` / `dataset_name` are the dataset evaluated (above: the run's, or for a models-folder
  model the first registered one with its classes; a run whose dataset is no longer registered reports the id and
  name it had). `confusion_conf` / `confusion_iou` are the evaluator's confusion-matrix settings (`score > 0.25`,
  `IoU > 0.45`, class-agnostic matching; docs/METRICS.md §7), not the NMS `conf 0.001` / `iou 0.7` and not
  `best_f1_conf`: the matrix and the precision / recall columns are at different thresholds, so a view that shows
  both says so ("matrix at conf > 0.25, IoU > 0.45"). Every field is `null` when unknown (older servers leave them
  out: read missing as `null`, and a missing `split` as unknown, not as `val`).
- **Split fallback.** `/eval` is defined on the validation split. A dataset whose `val` split has no images is
  evaluated on its `train` split instead (rather than failing, so a model of a train-only dataset still gets curves),
  and the response says so: `split: "train"` and `split_fallback: { "requested": "val", "reason": "empty_split",
  "message": "dataset 'tiny' has no val images, so this is its train split (10 images): training-set numbers, not
  held-out ones" }` (the server also logs a warning). These numbers measure fit on training images, not
  generalisation, so clients must not label them "validation": show the split (or `message`) next to them.
  `reason` is a stable code (`empty_split`, the only one today); treat an unknown code as a fallback too.
  `split_fallback` is `null` whenever `split` is `val`. Training does the same: an epoch's `val` (`Epoch.val`,
  model `metrics`) is on the training split when the dataset has no validation images, and the
  run's log says so; `POST /api/val` never falls back (an empty split is `400 invalid_dataset`).
- **Operating point (`best_f1_conf`).** Precision and recall (overall and per class) are taken at one confidence
  threshold: the one on the `0.000, 0.001, ..., 1.000` grid that maximises the class-mean F1 at IoU 0.5 (the highest
  on ties; detections with `score >= best_f1_conf` count; docs/METRICS.md). Every object with precision and recall
  says which threshold that was, as `best_f1_conf` next to them: `/eval` (top level), `Model.metrics`,
  `Epoch.val` of every validated epoch, not only the best one's (so `/metrics`, SSE `epoch` events and
  `metrics.jsonl`, the same value in all three) and `POST /api/val`'s `val`. It is `null`
  when unknown: no class has ground truth, an epoch recorded before the field, and `.folo` files whose
  `folo.metrics` predate it (docs/FORMAT.md; newer checkpoints store it there). Additive: clients of older servers
  read a missing field as `null`. Studio and Desktop can show it as "P 0.84 / R 0.77 at conf 0.31".
- **Run, model and `/eval` numbers.** A run model's `metrics` are the validation of the epoch that wrote its
  weights, unchanged (`<run>-best`: the epoch `Run.best` names, whose fitness is mAP@0.5:0.95 in the trainer too;
  `<run>-last`: the last validated epoch), so they equal that epoch's `val` in `/metrics`, `best_f1_conf`
  included. Overall precision, recall and mAP are the means of `per_class` (docs/METRICS.md). `/eval` and `POST
  /api/val` evaluate the weights again: the fused EMA weights, with the validator's batch (16) on the server's
  `--device`, against the dataset's files as they are now. Fused BatchNorm and another batch or device change
  float rounding, so a detection close to a threshold can flip and the per-class numbers can differ from the
  training epoch's in the last digits (more if the dataset changed since). The mock follows the same rules: its
  `/eval` per-class numbers are within 0.01 of the epoch's, with the same `best_f1_conf`.
- `POST /api/models/{id}/export` body `{ "format": "folo-f16|onnx", "decode"?: false, "dynamic"?: false,
  "half"?: false }` → `{ "path", "bytes" }`. The three switches are `folo export`'s `--decode`, `--dynamic` and
  `--half` and only apply to `onnx` (with `folo-f16` any of them is `400 invalid_body`). Real mode writes
  the fused EMA weights to `<data dir>/exports/`: `folo-f16` as f16 `.folo` (as `folo export` does, docs/FORMAT.md)
  to `<model name>-f16.folo`; `onnx` as an ONNX graph (opset 17) to `<model name>.onnx`, with the defaults of
  `folo export --format onnx` (docs/CLI.md, ONNX export): f32, a fixed batch of 1, the model's training imgsz
  (square), and the raw head outputs `box_raw` / `cls_raw`, unless a switch says otherwise (`decode` writes
  `<model name>-decoded.onnx`, `half` `<model name>-f16.onnx`, both `-decoded-f16.onnx`, as `folo export` names
  them). In the model name, control characters and the characters a Windows file name cannot hold (`/ \ : < > " |
  ? *`) become `-`, and it is cut to 200 bytes. Names get `-2`, `-3`, ... when taken (exports never
  overwrite). `path` uses `/` separators. A model the ONNX exporter cannot map is `501 not_implemented` naming the
  part (no FOLO v1 model hits this).
- `GET /api/models/{id}/weights` (optional `?format=f16|onnx`) → the model's `.folo` file (docs/FORMAT.md), for loading
  it elsewhere, e.g. into the browser detector (docs/WASM.md), or its ONNX graph. Without `format` it is the file as
  stored: a run model is its full training checkpoint (`weights/best.folo` / `last.folo`, with EMA and optimizer
  state); `format=f16` is the `folo-f16` export of `POST /api/models/{id}/export` (fused EMA weights in F16, a
  fraction of a checkpoint's size); `format=onnx` is its `onnx` export, with the same defaults (f32, batch 1, the
  model's training imgsz, raw head outputs) and the same switches as query parameters: `&decode=true`,
  `&dynamic=true`, `&half=true` (each `true` or `false`, default `false`). Both exports are made on the first
  request and cached in `cache/weights/` (one file per switch combination) until the model's file changes; the
  first ONNX request of a model takes about as long as `POST /api/models/{id}/export`. Any other `format`, or a
  switch with a `format` other than `onnx`, is `400 invalid_query`; a model the ONNX exporter cannot map is `501
  not_implemented` naming the part (no FOLO v1 model hits this); `404 model_not_found` as usual (also when the file
  disappears). Errors are the JSON error object, never a partial file. Studio downloads exports this way, so a
  remote browser gets the file instead of a server path.
  - `Content-Type: application/octet-stream` (also for ONNX, which has no registered media type), `Content-Length`,
    `Content-Disposition: attachment; filename="<model name>.folo"` (`<model name>-f16.folo`; ONNX as the export
    names it: `<model name>.onnx`, `<model name>-f16.onnx` with `half`; characters outside `A-Z a-z 0-9 . _ -`
    become `_`, and the name is cut to 200 characters).
    Real files are streamed from disk (never read into memory whole), and never gzip-compressed.
  - Validators: a strong `ETag` (real mode: the file's size and modification time in nanoseconds, cheap and exact
    for FOLO's writers, which replace files whole; a replacement keeping both the size and the exact modification
    time, e.g. with `touch -r`, is not noticed; mock mode: a hash of the bytes) and `Last-Modified`, with `Cache-Control: no-cache` (a run's `last.folo` changes every
    epoch). `If-None-Match` (weak comparison, `*` too) and, without it, `If-Modified-Since` answer
    `304 Not Modified` with the validators and no body. `HEAD` answers the headers alone.
  - Ranges (feature `run_cursors`): the answer carries `Accept-Ranges: bytes`, and a request with one range,
    `Range: bytes=<first>-<last>` or `bytes=<first>-`, gets `206 Partial Content` with `Content-Range: bytes
    <first>-<last>/<length>` and those bytes (`<last>` is clamped to the file). A range that starts at or past the
    end is `416 Range Not Satisfiable` with `Content-Range: bytes */<length>`; a header this server does not
    understand (several ranges, a suffix range) is ignored and the whole file is sent with `200`. `If-Match:
    <etag>` on any request answers `412 Precondition Failed` (no body) when the file's ETag differs, so a download
    in parts never mixes two versions of a file that is replaced meanwhile (a run's `last.folo`): start over with
    the new ETag. A cloud control plane copies checkpoints this way (docs/CLOUD.md).
  - CORS exposes `ETag` and `Content-Disposition` to page scripts, so Studio (also from the Vite dev server) can
    cache weights by ETag.

```jsonc
// Model
{ "id": "m_31aa", "name": "shapes-n-best", "arch": "folo-n|folo-s|folo-m|folo-l|folo-x", "imgsz": 640,
  "classes": ["circle","square","triangle"], "params": 2630000, "flops_g": 6.6, "bytes": 5400000,
  "created": "...", "run_id": "run_7d1e" | null,
  "metrics": { "map50": 0.91, "map50_95": 0.74, "precision": 0.9, "recall": 0.86 } | null,
  "head": "o2m|o2o",                                        // additive: missing (older servers) = "o2m"
  "trained_with": { "amp": false, "imgsz": 640, "epochs": 100 } | null,  // additive: missing = null
  "precision"?: "f32|f16",                                  // additive: missing (older servers) = unknown
  "fused"?: false }                                         // additive: missing (older servers) = unknown
```

## Training runs
- `GET /api/runs` → `{ "runs": [Run] }` (newest first)
- `POST /api/runs` body `TrainConfig` → `201 Run` (status `queued`, then `running`). Errors: `dataset_not_found`,
  `model_not_found` (bad `weights`), `invalid_config` (checked before anything is queued: the limits below, the
  trainer's (real mode: `imgsz >= 64`), a dataset without classes or training images), `404 device_not_found` (a
  `gpu:<n>` this machine has no usable adapter for), `503 shutting_down` (real mode: the server is shutting down;
  also for resuming). Runs train one at a time, first come first served; the others wait
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
  `"checkpoint": "<model id>"` (feature `run_checkpoint`; not part of `Run.config`) starts a new run that
  **continues** the training stored in that model's file, which must be a full checkpoint (a run's `last.folo`,
  docs/FORMAT.md): epoch, optimizer, schedule, EMA and RNG, as `folo train --resume` does. The file is copied
  into the new run as its `weights/last.folo`; the body's other fields are the run's `config` as usual and should
  be those of the run that wrote the checkpoint (the trainer takes its settings from the checkpoint). Errors:
  `model_not_found`, `422 invalid_weights` (the file has no training state), `400 invalid_body` (together with
  `resume_run_id` or `weights`). This is how a cloud control plane resumes a run on a fresh worker: it uploads
  the checkpoint (`/api/uploads`) and names it here.
  **Cloud runs** (feature `cloud`, docs/CLOUD.md): with `"compute": { "connection_id", "offer_id", "max_hours"? ,
  "disk_gb"? }` the run trains on that cloud connection instead of this machine. It is checked like any run plus:
  `connection_not_found`, `offer_not_found`, `400 invalid_config` (`max_hours` outside `0.25..=720`, `disk_gb`
  outside `10..=4096`), `422 cloud_auth_failed` / `502 cloud_unavailable` (the offer could not be read). `device`
  must be `auto` (the worker picks its GPU; anything else is `400 invalid_config`), and `amp` is checked on the
  worker, not here. The run does not enter the training queue (`queue_position` stays `null`); `Run.cloud` and
  the `cloud` event say where it is. Resuming one (`resume_run_id`) launches a new instance and continues from the
  checkpoint copied to this machine, with a new `Run.cloud` (its `cost_usd` is the new instance's); `compute` may
  change like `device`, and `"compute": null` resumes it here, as a run of this machine from then on (in the
  training queue, without `cloud`; what its instances cost stays in its log). A cloud run another FOLO process on the data dir follows (it
  holds the run's `cloud.lock`) is shown as that process last persisted it, and stopping it here is `409
  run_active`.
- `GET /api/runs/{id}` → `Run`
- `POST /api/runs/{id}/stop` → `Run` (status `stopping` → `stopped`; the last checkpoint is kept). `409 run_not_active`
  when the run already finished. A queued run stops at once; a running one after its current batch (the epoch in
  progress is discarded; `weights/last.folo` is the last finished epoch's). `model_id` is set when at least one epoch
  was validated. The final `status` event is sent once the run can be resumed or deleted.
- `POST /api/runs/{id}/queue` body `{ "action": "top|up|down|bottom|start" }` or `{ "position": 1 }` → `Run` (with
  its new `queue_position`). Moves a run that waits in the training queue. The queue holds the runs that are
  `queued` and not yet taken by the worker, front first; `Run.queue_position` is a run's 1-based place in it (`1`
  starts next) and `null` for every other run. Exactly one of `action` and `position` (both or neither, or
  `position: 0`, are `400 invalid_body`):
  - `top` / `bottom`: to the front / back; `up` / `down`: one place (a no-op at the end it moves towards);
    `position: n`: to place `n` (a larger `n` than the queue's length means the back). The other runs keep their
    order.
  - `start`: "start now". FOLO trains one run at a time and never preempts it: when nothing is training the run
    starts at once (the worker takes it, as it would any front run); otherwise it moves to the front and starts
    when the active run ends (completes, fails or is stopped).
  - Races: the worker takes the front run under the queue's lock, the same lock the move holds, so a move either
    happens before the run is taken (and counts) or finds it gone. A run the worker has already taken (still
    `queued` while its trainer is built, then `running`), and every finished or stopping run, is `409
    run_not_queued` (its message says whether it is starting, running or finished); `404 run_not_found` as usual.
    Moves never change a run's status.
  - Every change of the queue (a run queued, resumed, moved, taken by the worker, stopped or deleted while it
    waits) sends each waiting run a `queue` SSE event with its place (the mock: each run whose place changed, see
    the event), and the moved run gets a `log` line (also in `/logs`). Mock
    mode: runs created through the API share one training slot the same way (the seeded looping run `run_3c02` is
    outside the queue).
- `DELETE /api/runs/{id}` → `204` (`409 run_active` while `queued`, `running` or `stopping`). Real mode removes the run
  folder, its weights included, so its models disappear too.
- `GET /api/runs/{id}/metrics?every=1` → `{ "steps": [Step], "epochs": [Epoch] }` (full history, for charts on page
  load; `every=N` keeps every N-th step plus the latest one, to thin long runs). `after_step=<n>` keeps the steps
  with `step > n` and `after_epoch=<m>` the epochs with `epoch > m` (feature `run_cursors`; either or both, applied
  before `every`), so a client that has the history up to a step and an epoch polls for what is new. With
  `after_step`, the steps of the epoch in progress stop before the first one the real server has not recorded yet
  (it records them in batches), so the answer never skips a step and a poller's cursor never passes one. After a
  resume the run repeats the steps of the epoch it restarts (they replace the earlier ones in the history), so a
  poller that sees `Run.step` below its cursor starts again from that step.
- `GET /api/runs/{id}/samples?epoch=<n>` → `{ "epoch", "samples": [{ "image_id", "dataset_id", "width", "height",
  "truth": [Box], "pred": [ScoredBox] }] }` (validation predictions at `conf >= 0.25` on a fixed set of 16 images,
  in original-image pixels: in real mode the first 16 of the validation split, or of the training split when there
  is none;
  `epoch` defaults to the latest finished epoch; `404 epoch_not_available` before the first validation or for a
  future epoch)
- `GET /api/runs/{id}/logs?limit=500` → `{ "lines": [{ "t", "level": "debug|info|warn|error", "message" }],
  "total": 812 }`:
  the run's log history, most recent `limit` lines (default 500, clamped to 1..5000), oldest first. `total` counts
  every line of the log (feature `run_cursors`; left out when it is 0, so a missing `total` reads as 0, as
  from older servers), and `after=<n>` answers the lines after
  the first `n` instead (the oldest `limit` of them), so `after=<total of the last answer>` polls for new lines
  without missing or repeating one. The lines are
  the same as the SSE `log` events, so a dashboard opened mid-run loads this and then appends live events. The
  shape is stable (the MCP proxy uses it, docs/MCP.md).
- `GET /api/runs/{id}/events` → SSE stream. Each SSE `event:` name is the payload type, `data:` is its JSON:
  - `status` `{ "status": "queued|running|stopping|stopped|completed|failed", "error": null | "...",
    "steps_per_epoch": 50, "epochs": 100 }` (the run's length, as in `Run`); a `failed` one also has `"error_code"`
    (`Run.error_code`)
  - `step` `Step` (at most ~10 per second; steps in between are skipped, `/metrics` has them all)
  - `epoch` `Epoch` (after each validation)
  - `log` `{ "t", "level": "debug|info|warn|error", "message" }`
  - `queue` `{ "position": 2, "waiting": 3 }`: the run's place in the training queue (`Run.queue_position`) and
    how many runs wait there, it included; sent to every waiting run after each change of the queue (see `POST
    /api/runs/{id}/queue`), in order with its other events (a run's last `queue` event comes before its `running`
    status). The events carry state, not changes: a client keeps the latest one (or re-reads the run). The mock
    derives them from its schedule every 100 ms, so it sends one when a waiting run's place or the count changed,
    and changes within one tick coalesce. Servers older than the event never send it.
  - `cloud` `RunCloud`: a cloud run's `Run.cloud` after it changed (a new phase, transfer progress at most about
    once a second, the cost about once a minute). State, not a change: a client keeps the latest one. Only cloud
    runs send it.

  The stream starts with a `status` event and replays nothing else (use `/metrics` and `/logs` for history). A `: keep-alive`
  comment is sent every 15 s. The stream stays open after the run finishes (so `EventSource` does not reconnect in a
  loop) and ends when the run is deleted or the server shuts down. Any number of clients may subscribe. A client too
  slow to keep up gets a `warn` `log` event saying how many events it missed. Clients must ignore unknown event
  names (new types may be added).
  `status` and `step` events carry `steps_per_epoch` and `epochs`, so progress can be drawn from the stream alone.
  `steps_per_epoch` is only known once the trainer is built: a run's snapshot (`GET /api/runs/{id}`) and its
  `status` events say `0` until then (typically the initial `queued` status of a new run), and the trainer's
  `running` status and every `step` carry the real value. Both fields are missing from servers that predate them
  (clients should treat that as `0`).

```jsonc
// TrainConfig (only dataset_id is required; the rest shows the defaults)
{ "name": "", "dataset_id": "ds_8f2c", "arch": "folo-n", "epochs": 100, "batch": 16, "imgsz": 640,
  "device": "auto",            // auto | cpu | gpu:<n>
  "optimizer": "auto",         // auto | sgd | adamw
  "lr0": 0.01, "weights": null, // or a model id to fine-tune from (a published name such as "folo-n-coco" too);
                               // the run then trains the weights' architecture, whatever `arch` says
  "augment": { "mosaic": 1.0, "mixup": 0.0, "hsv": [0.015, 0.7, 0.4], "fliplr": 0.5, "scale": 0.5, "translate": 0.1,
               "close_mosaic": 10 },
  "seed": 0,
  "amp": false,                // mixed-precision training (docs/MODEL.md §8.8): f16 on GPU tensor cores, f32 master
                               // weights; needs `folo serve --tensor-cores` on a GPU with them, else POST /api/runs
                               // answers `501 not_supported` (nothing is queued)
  "head": "o2m",               // o2m | o2o: o2o adds the NMS-free one-to-one head (docs/MODEL.md §6.6), trained with
                               // dual assignment; the run's models then predict and validate without NMS (predict
                               // and val pick the head from the model, no request field needed). Additive: missing
                               // means o2m
  "compute": null }            // or { "connection_id": "cc_1f2e", "offer_id": "NVIDIA GeForce RTX 4090@secure",
                               // "max_hours": 12, "disk_gb": null }: train on a cloud connection (Cloud runs,
                               // above). Left out of the object when null; missing means this machine
// An empty name becomes "<dataset name>-<arch suffix>", e.g. "shapes-1k-n". Limits: epochs 1..=10000,
// batch 1..=1024, imgsz a multiple of 32 in 32..=4096 (real mode trains at 64 or more), 0 < lr0 < 10.
// POST /api/runs also takes "resume_run_id": "run_…" (see above); it is not part of Run.config.
// Real mode maps TrainConfig onto the trainer's config (docs/MODEL.md §8.1): the other recipe keys keep their
// defaults, and "auto" resolves to the GPU when a model can be placed on it, else the CPU.
// Run
{ "id": "run_7d1e", "name": "shapes-n", "status": "running", "config": TrainConfig, "created": "...",
  // + "error_code": "not_supported" | "training_failed" | ... when failed (folo_api::codes; absent otherwise and from
  //   older servers), next to "error" (the message); the final `status` event carries it too
  "started": "..." | null, "finished": null | "...",
  "epoch": 12,             // epoch in progress, 1-based (0 before the first step; the last epoch once finished)
  "epochs": 100,
  "step": 575,             // optimizer steps completed over all epochs
  "steps_per_epoch": 50,  // 0 until the trainer is built (before the first `running` status; events carry it)
  "eta_s": 1840 | null,    // null unless queued / running / stopping
  "best": { "epoch": 11, "map50_95": 0.61 } | null, "device": "NVIDIA GeForce RTX 4080 SUPER (vulkan)",
  "model_id": null | "m_31aa",  // set once the run completed, or stopped after at least one validated epoch
  "error": null | "...",
  "queue_position": null | 2,   // 1-based place in the training queue while the run waits there (1 starts next)
  "resolved": null | ResolvedTrain,   // what the trainer resolved `config` to (below); null until it is built
  "cloud"?: RunCloud }          // only for cloud runs (config.compute): where it executes and how far it is
// RunCloud (docs/CLOUD.md has the lifecycle). `device` of a cloud run is the worker's GPU once it trains.
{ "connection_id": "cc_1f2e", "provider": "runpod", "offer": CloudOffer,   // the offer as it was at the start
  "instance_id": "ci_8a2f" | null,  // GET /api/cloud/instances; null before the launch and for `remote`
  "phase": "provisioning|booting|syncing|training|collecting|releasing|released",
  "progress": { "done_bytes": 1048576, "total_bytes": 52428800 } | null,   // syncing / collecting (total 0: unknown)
  "max_hours": 12, "deadline": "..." | null,   // when the time limit ends; null until the instance exists
  "cost_usd": 0.42 | null,          // estimate so far, final once released; null without a price
  "synced": "..." | null }          // when weights/last.folo was last copied to this machine
// While phase is provisioning, booting or syncing the status is `queued` (nothing trains yet): show the phase.
// ResolvedTrain: the settings the run trains with. `config` stays what was requested ("optimizer": "auto",
// "lr0": 0.01); these are what that became (docs/MODEL.md §8.2, §8.4), as in the run folder's config.json.
{ "optimizer": "sgd|adamw",     // never auto
  "lr0": 0.001429,              // base LR of every parameter group: auto's pick, else config.lr0
  "lrf": 0.01, "cos_lr": false, // the last step runs at lr0 · lrf; linear or cosine decay
  "momentum": 0.9,              // SGD momentum, or AdamW β1, after warmup
  "betas": [0.9, 0.999] | null, // AdamW only
  "weight_decay": 0.0005,       // of the decay group, after the batch scaling weight_decay · batch · accumulate / nbs
  "batch": 16, "accumulate": 4, "nbs": 64,  // gradients of `accumulate` batches per optimizer step (≈ nbs images)
  "optimizer_steps_per_epoch": 13,
  "warmup_iters": 100, "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.0,
  "close_mosaic": 7 }           // close-mosaic epochs after the cap at half the run (docs/MODEL.md §8.4): mosaic is
                               // off for the last `close_mosaic` epochs; left out for runs older than the field
                               // whose config.json lacks `close_mosaic_epochs`
// Run.resolved is set when the trainer is built (by the time the `running` status is sent: fetch the run then),
// recomputed on resume, and read from the run folder's config.json for runs recorded before the field; it stays
// null for runs whose trainer never started (stopped while queued, failed to build). Mock runs fill it once they
// start, with the real trainer's resolution of their config. Additive: missing reads as null.
// Step
{ "epoch": 12, "step": 575, "lr": 0.0098, "loss": { "box": 1.21, "cls": 0.83, "dfl": 1.02, "total": 3.06 },
  "imgs_per_s": 412.5, "eta_s": 1840,
  "steps_per_epoch": 50, "epochs": 100 }  // epoch is 1-based; step is the global, 1-based step number
// Epoch (epoch is 1-based, as everywhere in this API: Run.epoch, Run.best.epoch, Step.epoch, ?epoch= queries)
{ "epoch": 12, "train_loss": { "box": 1.18, "cls": 0.80, "dfl": 1.00, "total": 2.98 },
  "val": { "map50": 0.82, "map50_95": 0.61, "precision": 0.84, "recall": 0.77,
           "per_class": [{ "class": 0, "map50": 0.85, "map50_95": 0.64, "precision": 0.86, "recall": 0.80 }],
           "best_f1_conf": 0.312 | null },   // precision / recall are at this confidence (Operating point, above)
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
  `timing` values (here, in `folo predict --json` and in MCP) are milliseconds rounded to 0.1 µs (4 decimals).
  Boxes only use the model's classes: on a dataset image, ground-truth classes the model does not know (by name) are
  never predicted.
  Real mode runs folo-vision's `Predictor` (docs/MODEL.md §9: EMA weights, fused, letterboxed `rect` at the model's
  `imgsz`) on the server's `--device` (`device` is `cpu` or `gpu:<n>`) at the server's `--precision` (`f32`, the
  default; `f16` runs the model in half precision, f16 weights and activations with f32 accumulation, on a GPU with
  `shader-f16`, and falls back to f32 with a logged warning elsewhere; so `device: gpu:0` plus `precision: f16`; with
  `folo serve --tensor-cores` the f16 convolutions and matrix products run on the GPU's tensor cores where it has
  them).
  Loaded models stay in an LRU of 4 (keyed by the weights file's path, size and modification time, and the precision,
  so retrained weights are reloaded); `timing` is measured wall time, and on a GPU each stage ends only once the
  device has finished the work it queued (the predictor waits for the device after the letterbox and after the
  forward pass), so no stage is billed for another's GPU time:
  - `preprocess_ms`: decoding the image file, the letterbox and the upload (on a GPU the letterbox runs there);
  - `inference_ms`: the forward pass, until the device has finished it;
  - `postprocess_ms`: decoding the head outputs into candidates (on a GPU, a small kernel), reading them back, NMS
    and mapping the boxes to image pixels;
  - `total_ms`: their sum.
  The browser runtime (docs/WASM.md) cannot wait for WebGPU mid-pipeline, so there `inference_ms` also holds the
  on-device head decode and its readback. FOLO Desktop's Local GPU uses the same predictor as the server.
- `GET /api/predict/stream?model_id=..&conf=..` — WebSocket: the client sends JPEG frames (binary), the server
  answers each with the same JSON as `/api/predict` (for webcam / screen demos). Later milestone; answers
  `501 not_implemented` for now.

## Throughput and estimates
Speed measured on this server, so clients can show this machine's numbers instead of static ones, and say where each
number comes from. Servers with these endpoints list `throughput`, `run_estimate` and `benchmark` in `/api/system`'s
`features`. Every number carries a `source`: `measured` (this server: training runs, `/api/predict` calls, benchmark
probes) or `reference` (docs/BENCHMARKS.md's FOLO measurements on the dev machine, an RTX 4080 SUPER and an
i7-14700KF, scaled by compute to other sizes; `note` says which and how).

- `GET /api/system/throughput` → `ThroughputTable`: `training` rows keyed by (`device`, `arch`, `imgsz`, `batch`,
  `amp`) and `inference` rows keyed by (`device`, `arch`, `imgsz`, `precision`, `tensor_cores`; `batch` is always 1).
  Every row has a flat `imgs_per_s` (training: the mosaic epochs' median, else the close-mosaic epochs'; inference:
  `1000 / latency_ms`) and `measured_on`, the adapter or CPU the number was measured on (a reference's is the dev
  machine's, whatever this server has; `device` is the device of this server the row is for). Measured rows come
  first, newest first; then a `reference` row for every arch at 640 and 320 px (batch 16), f32 and (GPU) amp / f16,
  on the server's `device: auto` device and on the CPU, unless a measured row of that very adapter or CPU (`device`
  and `measured_on`) has the same arch, imgsz, batch and amp / precision. Rows measured on another adapter under the
  same ordinal (a data dir moved to another machine, a new GPU) stay listed next to the reference; estimates never
  use them.
- `POST /api/runs/estimate` body `TrainConfig` (as for `POST /api/runs`, with the same checks and errors:
  `dataset_not_found`, `model_not_found`, `invalid_config`, `device_not_found`, `422 invalid_body`; nothing is
  created) → `RunEstimate`: wall time per epoch and in total with a range, from the training row of the device the run
  would use. The basis is the measured row with the run's arch, imgsz, batch and amp; else the nearest measured row
  on that device by compute and batch (the same amp preferred), carried over by the reference's ratio between the two
  settings (`basis.exact: false`; the reference has no batch dependence, so another batch keeps the rate); else the
  reference itself. The headline fields are `est_epoch_s` (mean seconds per epoch, validation included),
  `est_total_s` with `est_total_s_low` / `est_total_s_high`, and `est_peak_vram_bytes`.
  - Each epoch trains `train_images` at the rate of its phase (mosaic epochs, then the last `close_mosaic` epochs,
    capped at half the run as the trainer does; every epoch when `augment.mosaic` is 0) and validates `val_images`
    (the val split, else the training split) at the validation rate; the first epoch adds `startup_s`.
    `est_total_s = startup_s + Σ phases (epochs × (train_images / train_imgs_per_s + val_images / val_imgs_per_s))`
    and `est_epoch_s = (est_total_s - startup_s) / epochs`.
  - `confidence`: `high` for an exact measured row with at least 3 samples (range: the row's 10th–90th percentile
    spread, at least ±10%), `medium` for an exact row with fewer samples (at least -20% / +30%) or one carried from
    within 4x the compute with the same amp (-25% / +40%), `low` otherwise: a row further away or with the other
    precision (-40% / +80%), or the reference (-30% / +60% on the reference machine, -60% / +200% elsewhere; `note`
    then says it was not measured on this device).
  - `est_peak_vram_bytes` (GPU only, `null` on the CPU): the measured peak of a row with the run's device, arch,
    imgsz and amp at the nearest batch, scaled to the run's batch by the reference model; else the reference model (16
    bytes per parameter plus the activations per image, docs/BENCHMARKS.md). `vram_source` says which.
  - `close_mosaic`: the close-mosaic epochs after the cap (the `close_mosaic` phase's epochs; `Run.resolved` has the
    same number once the run is built).
  - `cli`: the `folo train` command that trains this config on the server's machine (see `GET /api/runs/{id}/cli`).
  - `cost` (a config with `compute`, feature `cloud`): `{ "price_per_hour": 0.69, "est_usd": 0.31, "est_usd_low":
    0.12, "est_usd_high": 0.93, "max_usd": 8.28 }`: the offer's price times `est_total_s` and its range (training
    only: booting the instance and uploading the dataset come on top), and `max_usd`, the price times
    `compute.max_hours`, which the run cannot exceed. Left out for an offer without a price. The time estimate of a
    cloud config never uses this server's measurements: it is the reference (`source: reference`, `confidence:
    low`, `device` `cloud`, `device_name` the offer's GPU, `est_peak_vram_bytes` from the reference model), and
    `basis.note` says the reference machine may be faster or slower than that GPU. Errors as `POST /api/runs`.
- `GET /api/runs/{id}/cli` → `{ "cli": "folo train --data D:/data/r6 --weights folo-n-coco --head o2m --epochs 15
  --amp", "argv": ["folo", "train", ...] }`: the CLI command of a run's config, built from the same `TrainConfig`
  mapping the server trains with (a test parses it with the real `folo` arguments and compares the configs).
  `--data` is the dataset's folder on the server's machine, with `--format yolo|coco` when it was registered with
  one (so a folder holding both reads the same); `--weights` is the published name for published weights,
  else the `.folo` file the run started from (so no `--model`: the CLI takes the architecture from the weights,
  and `--head` is always given, since the CLI's default is the weights file's head); settings equal to the
  defaults are left out, and a value starting with `-` is attached (`--name=--amp`). `cli` is quoted for PowerShell,
  as Studio shows commands: arguments with spaces or shell characters are single-quoted, a quote inside doubled
  (`'O''Brien'`, also for the typographic quotes PowerShell reads as quotes); POSIX shells read the same words
  unless an argument holds a quote. `argv` is the same command unquoted, for spawning without a shell (the
  folo-cli tests split `cli` by PowerShell's rules, compare it with `argv` and parse it with the real arguments). `run_not_found` for an unknown run;
  the mock server answers too, with made-up weights paths.
- `POST /api/system/benchmark` body `BenchmarkRequest` (all optional: `{ "arch": "folo-n", "imgsz": 320, "batch":
  16, "amp": false, "budget_s": 4 }`) → `BenchmarkReport`. An explicit quick probe on the server's inference device
  (`--device`): the trainer on a synthetic shapes split for 60% of `budget_s` (the first 30% of that is warm-up and
  not counted; it never validates), then predictions of a random-init model on one JPEG, decoded each time as an
  upload, for the rest. It takes the budget plus setup (building the model, compiling GPU pipelines): about 8 s for
  the defaults on the RTX 4080 SUPER. Both results are added to the table as measured samples (the training row's
  `probes` counts them); the loader renders synthetic images instead of decoding files, so a probe's training rate
  is an approximation of a real dataset's. `409 run_active` while a run is queued or running (or another probe
  runs); runs created during a probe wait for it. `400 invalid_body`: `imgsz` not a multiple of 32 in `64..=1280`,
  `batch` outside `1..=64`, `budget_s` outside `1..=10`. `amp` without tensor cores is `501 not_supported`, a GPU out
  of memory `507 gpu_out_of_memory`, as for a run.

What real mode measures, in `<data dir>/throughput.json` (rewritten atomically; prediction samples at most every
10 s and on shutdown; a malformed file is replaced by a new table). Each series keeps its last 32 values; `samples`
counts all of them:
- **Training**: every validated epoch of a run after the first of its session (the first, of a new or a resumed
  run, compiles pipelines) whose last batch was finite (only then does its final `step` event cover the epoch). The rate is the epoch's last `step` event's `imgs_per_s` (its images over its training
  time; loader waits included, validation excluded); the rest of `Epoch.duration_s` (validation and checkpoint
  writes) gives the validation rate; the first epoch's extra time over the second gives `startup_s`;
  `peak_vram_bytes` is the GPU buffer pool's peak since the run started (inference served meanwhile counts too).
- **Inference**: every `/api/predict` call's `timing` (`total_ms`, image decode included, and `inference_ms`),
  except a model's first after it is loaded and calls that overlap a run (one queued or training when the call
  starts or ends, or the worker taking up or letting go of a run during it).
- **ETA of running runs**: `Run.eta_s` (and `Step.eta_s`) is the run's own measured rate, per mosaic phase: the
  batches left in mosaic epochs at the run's mean mosaic batch time, loader waits included; those in close-mosaic
  epochs at their own mean once they have started, before that at the mosaic batches' step time without the loader
  waits (without mosaic the loader keeps up); plus the validations left at their mean. On R6 (1080p JPEGs) a mosaic
  epoch of folo-n took 24 s and a close-mosaic one 10 s, so one rate for both is off by up to 2x.

The mock (`folo serve --mock`) has measured rows on its GPU for folo-n and folo-s at 640 (f32, from its runs), a
folo-n 320 px amp probe and folo-n 640 predictions; everything else is the reference, so both sources show. Its
estimates use them the same way; its benchmark answers at once with numbers near the reference (the scripted live
run does not block it, runs created through the API do).

```jsonc
// ThroughputTable
{ "device": "gpu:0", "device_name": "NVIDIA GeForce RTX 4080 SUPER",
  "training": [TrainThroughput], "inference": [InferThroughput] }
// TrainThroughput
{ "device": "gpu:0", "measured_on": "NVIDIA GeForce RTX 4080 SUPER", "arch": "folo-n", "imgsz": 640, "batch": 16,
  "amp": false,
  "imgs_per_s": 212.4,           // the headline rate: mosaic's median, else close_mosaic's
  "mosaic": Rate | null,         // epochs with mosaic
  "close_mosaic": Rate | null,   // epochs without (the close-mosaic phase, or mosaic 0)
  "val": Rate | null,            // validation images per second of an epoch's validation (and checkpoint) time
  "startup_s": 5.4 | null,       // a session's first epoch over a steady one
  "peak_vram_bytes": 3652000000 | null,  // GPU only; the largest seen
  "source": "measured|reference", "samples": 99, "probes": 0,
  "updated": "2026-11-07T21:14:09Z" | null,   // null for a reference
  "note": null | "docs/BENCHMARKS.md: FOLO trainer, COCO train2017 (50 epochs, batch 16) on an ..." }
// Rate: the median of the recent samples, their 10th / 90th percentiles (min / max below 10 samples); a
// reference carries its expected range and 0 samples
{ "imgs_per_s": 212.4, "low": 205.1, "high": 219.8, "samples": 81 }
// InferThroughput (batch 1)
{ "device": "gpu:0", "measured_on": "...", "arch": "folo-n", "imgsz": 640, "batch": 1, "precision": "f32|f16",
  "tensor_cores": false, "latency_ms": 5.9, "latency_ms_low": 5.5, "latency_ms_high": 6.4,
  "inference_ms": 3.1 | null, "imgs_per_s": 169.5, "source": "measured|reference", "samples": 214,
  "updated": "..." | null, "note": null | "..." }
// RunEstimate
{ "device": "gpu:0", "device_name": "...", "train_images": 800, "val_images": 200, "batches_per_epoch": 50,
  "epochs": 100,
  "phases": [{ "phase": "mosaic", "epochs": 90, "train_imgs_per_s": 212.7, "train_s": 3.76, "val_s": 0.71,
               "epoch_s": 4.47 },
             { "phase": "close_mosaic", "epochs": 10, "train_imgs_per_s": 221.2, "train_s": 3.62, "val_s": 0.71,
               "epoch_s": 4.33 }],
  "val_imgs_per_s": 281.9, "startup_s": 5.5, "est_epoch_s": 4.5, "est_total_s": 451.2, "est_total_s_low": 406.1,
  "est_total_s_high": 496.3,
  "source": "measured|reference", "confidence": "high|medium|low",
  "basis": { "source": "measured", "arch": "folo-n", "imgsz": 640, "batch": 16, "amp": false,
             "measured_on": "NVIDIA GeForce RTX 4080 SUPER", "samples": 99, "updated": "..." | null,
             "exact": true, "note": "measured on this device with these settings: 99 epochs of runs" },
  "val_source": "measured|reference",
  "est_peak_vram_bytes": 3652000000 | null, "vram_source": "measured|reference" | null }
// BenchmarkReport
{ "device": "gpu:0", "device_name": "...", "train_imgs_per_s": 344.7, "train_batches": 37, "latency_ms": 6.0,
  "predictions": 247, "elapsed_s": 8.1,
  "training": TrainThroughput, "inference": InferThroughput }   // the rows after the probe
```

## Agent labeling
Turns a set of unlabeled images into a detection dataset: a coding-agent CLI on the server's machine (**Claude
Code**, `claude`, or **Codex**, `codex`) looks at the images and draws the boxes, a person reviews and corrects
them, and the labeled images become a YOLO dataset that trains like any other. Servers with these endpoints list
`agent_labeling` in `/api/system`'s `features` (real mode only: the mock answers `501 not_implemented`).

A **labeling job** holds the images, the class list, the instructions for the agent and the labels so far. It is
long-lived: images can be added, labeled in several agent runs (also by different agents), corrected by hand and
exported more than once.

- `GET /api/label/agents` → `{ "agents": [LabelAgent] }`: Claude Code, then Codex, each with whether its CLI is
  installed. The server looks for `claude` / `codex` on its `PATH` (`FOLO_CLAUDE_BIN` / `FOLO_CODEX_BIN` name
  another executable) and runs `--version`; the result is cached for 60 s. `available` says nothing about being
  signed in: a run finds that out on its first call and fails with the CLI's own message. `models` are Claude
  Code's model aliases (`haiku`, `sonnet`, `opus`), and for Codex the models of its own catalog that take images
  (`codex debug models`). `default_model` and `default_effort` are what a run uses when it names none: Claude
  Code `haiku` at `medium`, Codex its own default model at `low` (docs/LABELING.md has the measurements behind
  them).
- `GET /api/label/jobs` → `{ "jobs": [LabelJob] }` (newest first)
- `POST /api/label/jobs` body `{ "name", "classes": ["car", "person"], "instructions"?: "", "path"?: "D:/photos" }`
  → `201 LabelJob`. `name` must not be blank (one line of at most 100 characters); `classes`: 1 to 80 distinct,
  non-blank names of at most 64 characters (no `,`, `:`, line breaks or other control characters; they are
  compared without case); `instructions`:
  at most 4000 characters. With `path`, the images of that folder are added as by `POST …/images`
  (`path_not_found` creates nothing). Anything else wrong is `400 invalid_body`, naming the field.
- `GET /api/label/jobs/{id}` → `LabelJob`
- `PUT /api/label/jobs/{id}` body `{ "name"?, "classes"?, "instructions"? }` → `LabelJob`. Absent fields stay.
  Boxes refer to classes by position, so once the job has boxes `classes` may only rename or append (a shorter
  list is `400 invalid_body`). `409 label_job_active` while an agent run is active.
- `DELETE /api/label/jobs/{id}` → `204`: removes the job, its uploaded images and its labels (never files added
  from a folder, and never a dataset made from it). `409 label_job_active` while a run is active.

**Images**
- `POST /api/label/jobs/{id}/images` → `{ "added": [LabelImage], "skipped": [{ "file", "reason" }], "job": LabelJob }`.
  Either `multipart/form-data` with one `image` file part per image (JPEG, PNG, BMP, WebP; up to 64 MiB per
  request, so a client sends a large set in several requests; the files are stored in the job's folder), or JSON
  `{ "path": "D:/photos" }` for a folder on the server's machine (searched recursively, in path order; or one
  image file), whose images are read in place, not copied. A file that is not a readable image, an upload with
  the bytes of an image the job already has, or a folder image it already has (the same file) is listed in
  `skipped` with the reason instead of failing the request; so is everything past the job's limit of 10000
  images. File names are unique within a job: a second `a.jpg` becomes `a-2.jpg`. `404 path_not_found` for a
  missing folder. Images may be added while a run is active (they wait for the next run).
- `GET /api/label/jobs/{id}/images?status=pending|labeling|labeled|failed&offset=0&limit=100` →
  `{ "total", "images": [LabelImage] }`, in the order the images were added. `limit` defaults to 100 and is capped
  at 1000; `total` counts all matches before `offset` / `limit`; any other `status` is `400 invalid_query`.
- `GET /api/label/jobs/{id}/images/{image_id}` → the image bytes; `?thumb=256` as for dataset images.
- `PUT /api/label/jobs/{id}/images/{image_id}` body `{ "boxes": [Box] }` → `LabelImage`: labels drawn by hand.
  They replace the image's boxes; the image becomes `labeled` with `source: "manual"` (an empty list labels it
  as a background image). Boxes are clipped to the image; one with a `class` outside the job's classes, a
  non-finite coordinate, or less than a pixel left after clipping is `400 invalid_body`, as are more than 1000
  boxes. `409 label_job_active` while the image is in an agent call (`labeling`). Every other image of a running
  job can be edited: one that still waited for the run leaves it (`run.total` drops by one), so labels drawn by
  hand are never replaced by a call that had not started.
- `DELETE /api/label/jobs/{id}/images/{image_id}` → `204` (an uploaded image's file is deleted too). `409
  label_job_active` while the image is in an agent call, as for `PUT`.

**Asking an agent**
- `POST /api/label/jobs/{id}/start` body `StartLabeling` → `LabelJob` (`status: "running"`, with the new `run`).
  Only `agent` is required; `model` and `effort` default to the agent's `default_model` and `default_effort`.
  The run labels the `scope` (`unlabeled`, the default: `pending` and `failed`
  images; `all`: every image, replacing existing labels, hand-drawn ones too), or exactly `image_ids`. `note` (at
  most 2000 characters) is an extra instruction for this run only: the way to tell the agent what its last
  labels got wrong. Errors: `422 invalid_body` (an unknown `agent`), `400 invalid_body` (a `model` with characters
  outside `A-Z a-z 0-9 . _ : / - [ ]` or over 100 of them, `batch` outside `1..=16`, `parallel` outside `1..=8`,
  an unknown image id, nothing to label), `501 not_supported` (the agent's CLI is not installed; the hint says
  how to get it), `409 label_job_active` (a run is already active).
- `POST /api/label/jobs/{id}/stop` → `LabelJob` (`stopping`, then `idle` with `run.outcome: "stopped"`). The agent
  calls in flight are killed and their images keep the state they had. `409 label_job_not_active` when no run
  is active.
- `GET /api/label/jobs/{id}/events` → SSE stream, as for runs (a `: keep-alive` comment every 15 s; open until
  the job is deleted or the server shuts down; unknown event names must be ignored):
  - `job` `LabelJob`: first, then after every change of the job (status, counts, run progress, usage);
  - `image` `LabelImage`: an image was added, sent to the agent (`labeling`), labeled, failed or edited;
  - `removed` `{ "id" }`: an image was deleted;
  - `reload` `{}`: more changed than the stream carries, so load the images again (sent instead of one `image`
    event each when more than 100 images are added at once, and to a client that fell behind the stream);
  - `log` `{ "t", "level", "message" }`: the job's log (calls made, tokens used, boxes dropped, failures).

  The stream replays nothing: load the job's images with `GET …/images`, then apply events.
- `GET /api/label/jobs/{id}/logs?limit=500` → `{ "lines": [{ "t", "level", "message" }] }`: the job's log history
  (the `log` events), as `GET /api/runs/{id}/logs`.

**How a run works** (and what it costs)
- The run splits its images into batches (`batch` images per agent call: Claude Code 4, Codex 8 by default) and
  keeps `parallel` calls (default 3) in flight. Each call is **one model request with no tools**: the images are
  attached to the prompt (downscaled to at most 1024 px on the long side and re-encoded as JPEG, so a call costs
  about 1000 input tokens per image whatever the files' size), and the reply is one JSON object with a box list
  per image. Coordinates in the reply are relative (`0..1000` of the image's width and height), so they are
  independent of the downscaling; the server converts them to pixels of the original image.
- Claude Code is started as `claude -p` with its system prompt replaced by FOLO's labeling prompt, no tools, no
  MCP servers, no user or project settings, hooks, skills or `CLAUDE.md`, and no session file; the images go in
  on stdin as a `stream-json` user message. The fixed cost of a call is a few hundred tokens. Codex is started
  as `codex exec` with `--ephemeral --sandbox read-only --ignore-user-config`, FOLO's instructions as the model
  instructions, web search and the features that add tools switched off, the images as `--image` attachments
  and the reply shape as `--output-schema` (with `label` limited to the job's classes); that takes a call's fixed
  cost from about 20000 tokens to about 8000, which are Codex's own and stay, so its default batch is larger.
  Both run in an empty scratch folder and use the account the CLI is signed in with. Measured with
  `crates/folo-server/tests/label_live.rs` (Claude Code 2.1.302, codex-cli 0.167.0), eight 320 px images: Claude
  Code 2 calls and 3500 input tokens in all, Codex 1 call and 9200.
- A reply's boxes are checked before they are stored: a label that is not one of the job's classes (compared
  without case), a malformed box, or one that is empty after clipping is dropped (the log says how many), and an
  image the reply does not mention is `failed`. One bad box costs only itself, and the JSON is found with text
  around it or split into one object per image. A reply with no such JSON at all is asked for once more; then
  its images are `failed` with the reason.
- A run never burns tokens on a broken setup: when the CLI cannot be started or exits with an error (not signed
  in, an unknown model, a rate limit), or three calls in a row time out (5 minutes each) or return garbage
  (every call counts, a retry too), the run ends with `outcome: "failed"`, the CLI's message in `run.error` and
  the images it did not get to unchanged. Once a run is stopped or has failed, no further call is started and
  the ones in flight are killed. `POST …/start` again continues with what is left.
- `usage` adds up what the CLIs report per call. `cost_usd` is Claude Code's own figure (list prices, also on a
  subscription); Codex reports none.

**Making the dataset**
- `POST /api/label/jobs/{id}/dataset` body `{ "name"?, "val_fraction"?: 0.2, "seed"?: 0 }` → `201 Dataset` with
  `stats`. Writes the job's `labeled` images as a YOLO dataset into `<data dir>/datasets/<name>` (the name
  lowercased to `a-z0-9-`, with `-2`, `-3`, ... when taken, as for synthetic datasets) and registers it:
  `images/{train,val}/` (the files copied byte for byte), `labels/{train,val}/<stem>.txt` and a `data.yaml` with
  the job's classes. `pending` and `failed` images are left out; a labeled image without boxes is a background
  image (an empty label file). The images are shuffled with `seed` and `round(n · val_fraction)` of them go to
  `val` (`val_fraction` in `0..=0.5`, else `400 invalid_body`; at least one image always stays in `train`).
  `name` defaults to the job's. `400 invalid_dataset` when no image is labeled, `409 label_job_active` while a
  run is active. While the dataset is written the job cannot be changed, deleted or given to an agent, and its
  images cannot be edited or removed (`409 label_job_active`); images can still be added (they are not part of
  that dataset). The job keeps existing and remembers the dataset as `dataset_id`; exporting again makes
  another dataset.

```jsonc
// LabelAgent
{ "id": "claude|codex", "name": "Claude Code", "available": true, "version": "2.1.302" | null,
  "models": [{ "id": "haiku", "label": "Haiku" }],     // suggestions; `model` takes any name the CLI accepts
  "default_model": "haiku" | null,                     // null: the CLI's own default
  "default_effort": "low|medium|high",                 // what a run without `effort` uses
  "hint": null | "Install Claude Code (https://claude.com/claude-code) and sign in with `claude`, or ..." }
// LabelJob
{ "id": "lb_3f9a", "name": "street", "classes": ["car", "person"], "instructions": "Label every ...",
  "created": "...", "status": "idle|running|stopping",
  "counts": { "images": 120, "pending": 20, "labeled": 96, "failed": 4, "boxes": 513 },
  "run": LabelRun | null,        // the active run, else the most recent one
  "usage": LabelUsage,           // every run of the job
  "dataset_id": "ds_8f2c" | null }
// LabelRun
{ "agent": "claude", "model": "sonnet" | null, "effort": "low|medium|high", "started": "...",
  "finished": "..." | null, "total": 24, "done": 16, "failed": 0,   // images of this run
  "outcome": null | "completed|stopped|failed", "error": null | "...", "hint"?: "...", "usage": LabelUsage }
// LabelUsage
{ "calls": 4, "input_tokens": 18400, "cached_input_tokens": 3252, "output_tokens": 5100, "cost_usd": 0.021 | null }
// LabelImage
{ "id": "17", "file": "IMG_0042.jpg", "width": 1920, "height": 1080,
  "status": "pending|labeling|labeled|failed",
  "boxes": [Box],                       // pixel xyxy; kept while the image is labeled again
  "source": null | "agent|manual",      // who drew `boxes`
  "error": null | "..." }               // why the last agent call failed
// StartLabeling (only agent is required; the rest shows the defaults)
{ "agent": "claude", "model": null, "effort": null, "scope": "unlabeled", "image_ids": null, "note": "",
  "batch": null, "parallel": null }
```

## Uploads
Sending a dataset or a weights file to a server that cannot read the sender's disk (feature `uploads`): a cloud
worker, a server on another machine. An upload is a byte stream cut into parts, so every request is small,
retryable and independent of proxy limits; the control plane of a cloud run uses exactly this (docs/CLOUD.md).

- `POST /api/uploads` body `{ "kind": "dataset|model", "name", "format"?: "auto|yolo|coco", "fingerprint"?,
  "bytes"? }` → `201 { "upload": Upload, "existing": null }`. `fingerprint` (at most 128 characters) names the
  content: when this server already holds a finished upload of that kind with it (and the dataset is still
  registered, the model still listed), the answer is `200 { "upload": null, "existing": { "dataset": Dataset } }`
  (or `{ "model": Model }`) and there is nothing to send. `bytes` is the stream's length when known (reported
  back, not checked). `400 invalid_body`: a blank `name`, a `format` with `kind: model`, a longer fingerprint.
- `PUT /api/uploads/{id}/parts/{n}` body: the next bytes of the stream (`Content-Type:
  application/octet-stream`, at most `max_part_bytes`, else `413 payload_too_large`) → `Upload`. `n` must be
  `Upload.part`. Sending part `n - 1` again (a retry after a lost answer) is accepted when it has the length of the
  part received and changes nothing; any other `n` is `409 upload_conflict`, and the hint names the part to send.
  One part at a time per upload.
- `GET /api/uploads/{id}` → `Upload` (where to continue after a failure).
- `POST /api/uploads/{id}/complete` → `201 { "dataset": Dataset }` (with `stats`) or `201 { "model": Model }`.
  A dataset's archive is whole once its end-of-archive marker has arrived: completing it earlier (inside a file,
  or between files) is `409 upload_conflict`, and the session stays open for the rest. Content that is not usable
  drops the session with its files: `422 invalid_upload` / `invalid_weights`, or what registering the dataset
  answers (`400 unknown_format` / `invalid_dataset`).
- `DELETE /api/uploads/{id}` → `204` (drops the session and what it received).

```jsonc
// Upload
{ "id": "up_5c1d", "kind": "dataset", "name": "shapes-1k",
  "part": 3,                    // index of the next part
  "received_bytes": 25165824, "max_part_bytes": 33554432,
  "expires": "..." }            // dropped then unless a part arrives first (one hour after the last one)
```
- **Datasets** (`kind: dataset`): the stream is a tar archive (ustar; GNU and pax long names are read) of the
  dataset's folder, as `POST /api/datasets` would read it from disk: a `data.yaml` with `images/` and `labels/`,
  or `annotations/instances_*.json` with the images. It is unpacked as it arrives (a part is answered once its
  bytes are on disk, so memory stays at about one part) and, on `complete`, moved to `<data dir>/datasets/<name>`
  (the name lowercased to `a-z0-9-`, with `-2`, `-3`, ... when taken) and registered with `format` (default
  `auto`). Only regular files and directories with relative paths inside the folder are accepted: absolute paths,
  `..`, links and devices are `422 invalid_upload` on the part that carries them, and so is a stream that is not a
  tar archive; the session is then dropped.
  FOLO itself sends datasets in a canonical YOLO layout (docs/CLOUD.md, *Dataset*).
- **Models** (`kind: model`): the stream is one `.folo` file, stored as `<data dir>/models/<name>.folo` (the name
  reduced as for the `Content-Disposition` of `/weights`, `-2`, ... when taken) and listed like any file there. `422 invalid_weights` when its
  header does not parse.
- Sessions live in memory: a restart drops them (and removes what they received), and a session without a part
  for an hour expires. A real server only; the mock answers `501 not_implemented`.

## Cloud compute
Training runs on compute that is not this machine (feature `cloud`): GPU instances rented from a provider for one
run, or another FOLO server. docs/CLOUD.md describes the model, the lifecycle of a cloud run and each provider;
this section is the wire contract. A cloud run is created with `TrainConfig.compute` (Training runs, above).

- `GET /api/cloud/providers` → `{ "providers": [CloudProvider] }`: the kinds of compute this server knows, with the
  fields a connection to each takes (a client builds its form from them).
- `GET /api/cloud/connections` → `{ "connections": [CloudConnection] }` (oldest first). Nothing is checked on a
  listing: `status`, `account` and `balance_usd` are from the last check.
- `POST /api/cloud/connections` body `{ "provider", "name"?, "values": { "<field key>": "..." } }` → `201
  CloudConnection`. The values are checked with the provider before anything is stored: `404 provider_not_found`,
  `400 invalid_body` (a required field is missing or malformed; the message names it), `422 cloud_auth_failed`
  (the provider rejects them), `502 cloud_unavailable` (it does not answer); a key that is not one of the
  provider's fields is `400 invalid_body` too. `name` defaults to the provider's name (` 2`, ` 3`, ... when taken).
- `POST /api/cloud/connections/{id}/check` → `CloudConnection`: checks the stored credentials again and
  refreshes `status`, `account`, `balance_usd`, `error` and `checked`. A failed check is still `200`, with
  `status: "error"`; only an unknown id is an error (`404 connection_not_found`).
- `DELETE /api/cloud/connections/{id}` → `204`: forgets the credentials. `409 connection_in_use` while it has live
  instances or active runs.
- `GET /api/cloud/connections/{id}/offers` → `{ "offers": [CloudOffer], "fetched": "..." }`: what the connection
  can run on now, cheapest first (offers without a price first). Answers are cached for 60 s (`fetched` is when
  the provider was asked; `?refresh=true` asks again). `422 cloud_auth_failed`, `502 cloud_unavailable`.
- `GET /api/cloud/instances` → `{ "instances": [CloudInstance] }`: the instances this server launched: those alive
  first, then those that ended in the last 7 days, newest first.
- `DELETE /api/cloud/instances/{id}` → `CloudInstance` (`terminating`, or `terminated` when the provider confirmed
  at once): terminates it now. A run still using it fails with `cloud_instance_lost`. Terminating one that is
  already gone answers its record; `404 instance_not_found`.

Secrets never leave the server: `values` in answers holds the fields that are not secret as entered and a hint
(`••••` plus the last 4 characters; just `••••` for a value shorter than 8 characters, so a short secret is not
mostly shown) for the others.

```jsonc
// CloudProvider
{ "id": "runpod", "name": "RunPod", "about": "GPU pods rented by the hour, one per run.",
  "docs_url": "https://www.runpod.io/console/user/settings" | null,
  "fields": [{ "key": "api_key", "label": "API key", "secret": true, "required": true,
               "placeholder": "rpa_..." | null, "help": "..." | null }],
  "provisions": true,        // launches an instance per run and terminates it (false: a machine that already runs)
  "self_terminates": true }  // its instances terminate themselves after a run's deadline, also while this server is off
// CloudConnection
{ "id": "cc_1f2e", "provider": "runpod", "name": "RunPod", "created": "...",
  "status": "ok|error", "account": "you@example.com" | null, "balance_usd": 42.17 | null,
  "error": null | "...",     // why the last check failed
  "note": null | "...",      // something to know about a working connection (an unencrypted http:// URL)
  "checked": "..." | null, "values": { "api_key": "••••a1b2" } }
// CloudOffer
{ "id": "NVIDIA GeForce RTX 4090@secure",   // opaque, unique within the connection
  "gpu": "RTX 4090", "gpu_count": 1, "vram_bytes": 25769803776,
  "price_per_hour": 0.69 | null,            // US dollars; null for compute that is not billed (your own machine)
  "region": null | "EU-RO-1", "detail": null | "Secure Cloud",
  "availability": "high|medium|low|none|unknown" }   // none: launching would be refused right now
// CloudInstance
{ "id": "ci_8a2f", "connection_id": "cc_1f2e", "provider": "runpod",
  "provider_ref": "abc123xyz" | null,       // the provider's own id (to find it in its console); null while launching
  "name": "folo-run_7d1e-4c1a", "run_id": "run_7d1e" | null, "offer": CloudOffer,
  "status": "launching|starting|running|terminating|terminated|lost",
  "created": "...", "ended": "..." | null, "deadline": "..." | null,
  "cost_usd": 0.42 | null,                  // price_per_hour times its lifetime: an estimate, the provider's bill counts
  "error": null | "..." }                   // the last failed provider request about it (a termination being retried)
```

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
- `datasets/<name>/`: synthetic datasets and the datasets made from labeling jobs. `DELETE /api/datasets/{id}`
  removes the dataset's caches, never its files.
- `label/<job id>/`: one folder per labeling job: `job.json` (the job, its images and their boxes, rewritten
  atomically after every change; during an agent run at most every 5 seconds and when it ends, so a crash can
  lose the labels of the run's last few seconds), `images/` (uploaded files; folder imports stay where they are)
  and `log.jsonl`.
  Its thumbnails are in `cache/thumbs/<job id>/`. On start every job is listed again, `idle`: a run a previous
  process left active is closed as `stopped` (its finished images keep their labels), and files in `images/`
  that no image of the job refers to (an upload a crash interrupted) are removed. On shutdown the active runs'
  agent calls are killed.

- `runs/<id>/`: one folder per training run: `run.json` (the `Run` plus the dataset reference, rewritten atomically
  on every status change and after every epoch), `log.jsonl` (every log line, as `/logs` returns them),
  `config.json` (the trainer's resolved config), `metrics.jsonl` (every trainer event, every step; `/metrics` reads
  it, and a resumed run's repeated steps replace the earlier ones), `weights/last.folo` and `weights/best.folo`
  (full checkpoints, docs/FORMAT.md) and `samples/epoch_NNN.json`. On start every run is listed again; runs a
  previous process left `queued`, `running` or `stopping` become `stopped` (with a `warn` log line) and can be
  resumed with `resume_run_id`. On a graceful shutdown the active run stops after its current batch.
- `models/`: `.folo` files listed as models (scanned on every listing; metadata read once per file version).
- `exports/`: exported models. `cache/eval/`: cached `/eval` results. `cache/weights/<model id>.<hash>.<variant>`:
  the exports of `/weights?format=…`: `f16.folo`, and `onnx` / `decode.onnx` / `decode-dynamic-half.onnx` / … (the
  ONNX switches that are on). The hash is of the source file's path, size and modification time; copies made from
  an older version of the file are removed when a new one is made, and a run's when the run is deleted.

- `cloud/connections.json`: the cloud connections with their credentials (created readable by the owner only; never
  served). `cloud/instances/<id>.json`: the ledger of launched instances, one file each, written before the
  provider is asked (docs/CLOUD.md, *Instances are never left behind*); records of instances that ended more than
  7 days ago are removed. A cloud run's folder also has `cloud.json` (its worker's address and token, the ids it
  has there and its cursors) and `cloud.lock`, which the process that follows the run holds locked (the file
  stays). `cloud/.lock` is held across every change of `connections.json` and of a ledger record, so a server and
  `folo cloud` on one data dir never lose each other's changes.
- `uploads/<id>/`: the files of upload sessions in progress (a dataset is unpacked here and moved to
  `datasets/<name>/` when it completes, so a crash never leaves half a dataset there), removed when a session
  completes, is deleted or expires, and on start. `uploads.json`: the fingerprints of finished uploads with the
  dataset or model each became, so the same content is not sent twice.

## Validation
- `POST /api/val` body `{ "model_id", "dataset_id", "split": "val", "imgsz": null, "conf": 0.001, "iou": 0.7 }` →
  `{ "model_id", "dataset_id", "dataset_name", "split", "imgsz", "images", "val": Epoch.val, "timing": { "total_s", "per_image_ms" } }`.
  `split` is the requested one (no fallback); `dataset_name` is the dataset's name (`null`, or missing, from older
  servers).
  Only `model_id` and `dataset_id` are required; `imgsz: null` uses the model's size. Errors: `model_not_found`,
  `dataset_not_found`, `invalid_body` (`conf` / `iou` outside `0..=1`, an `imgsz` that is not a multiple of 32
  in `32..=4096`), `invalid_dataset` (an empty split, as `folo val` reports it; real mode: the dataset's classes
  are not the model's, in the same order). Real mode runs folo-vision's evaluator (docs/METRICS.md) on the server's `--device`
  (at its `--precision`, as `/api/predict`; the cached `/eval` metrics are always f32) with the
  request's `conf` / `iou` (the other NMS settings as in validation during training: multi-label, `max_det` 300), so
  the defaults on the validation split reproduce `/eval`'s per-class numbers; it is computed on every request (not
  cached). `timing.total_s` includes loading the model and the dataset; `per_image_ms` is the evaluation loop
  divided by `images`. The MCP `val` tool calls it (through the store, or over HTTP in proxy mode).
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
- Models: six (`m_31aa`, `m_31ab`, `m_42c1`, `m_42c2`, `m_53d1`, `m_53d2`), each linked to its run (`head` and
  `trained_with` match the run's config: `m_31ab` / `run_1a02` is an `o2o` model, `m_53d1` / `run_3c01` was trained
  with `amp`, and `m_53d2`, its f16 export, has `trained_with: null`; `m_53d2` is `precision: "f16"`,
  `fused: true`, every other model `"f32"` and `false`), with
  deterministic `/eval` results consistent with its metrics (smooth PR curves, a mostly diagonal confusion matrix),
  with the provenance of a real one: the run's dataset id and name, `split` (`val`; `train` with a `split_fallback`
  for a dataset without validation images: a user-made synthetic dataset of one or two images) and the
  real evaluator's `confusion_conf` / `confusion_iou`.
  `/weights` serves a real, loadable `.folo` file: always the **folo-n** architecture (whatever the model's `arch`)
  with the model's classes, `imgsz` and `head` and deterministic random weights (about 10 MB, 5 MB as `f16`; an
  `f16` model's file as stored is its fused f16 export, matching its `precision` and `fused`), generated on
  the first request and kept in memory, so the in-browser path works in mock mode (it detects nothing useful).
  `Last-Modified` is the server's start time.
- Runs: four completed, one stopped (`run_2b02`), one failed with an out-of-memory error (`run_2b03`), all with full
  metric histories; and one live run (`run_3c02`, r6-operators / folo-s) whose SSE stream advances in real time: a
  100-epoch run compressed into ~10 minutes (6 s per epoch, ~7 steps per second), starting around epoch 12, then it
  loops (`status` back to `queued`, metrics restart).
- `/api/system`'s `capabilities` reports the RTX 4080 SUPER with tensor cores on and `amp` available. To test the
  other states, start the mock with `FOLO_MOCK_TENSOR_CORES=off` (tensor cores available but not requested, no
  `amp`) or `FOLO_MOCK_TENSOR_CORES=none` (also `0` / `false`: the adapter through DX12, `gpus[0].backend` `dx12`,
  no tensor cores, no `amp`; `shader_f16` stays true). The reasons are worded as the real server's.
- `POST /api/runs` queues a simulated run (6 s per epoch) that can be stopped and moved in the queue; runs created
  this way train one at a time (the first after 2 s in `queued`), like the real server's. When one completes, its
  model appears in `/api/models`. Every mock run has `resolved` (the real trainer's resolution of its config) once
  it started, and every `val` / `/eval` / model metric has a deterministic `best_f1_conf`. `/weights?format=onnx`
  is the real exporter's graph of the generated `.folo` file. `POST /api/datasets` scans real folders; `POST /api/datasets/synthetic` makes a procedural
  dataset. Predictions are deterministic in the image bytes (or dataset image) and the model. `POST /api/val` derives
  per-class metrics from the model's metrics (deterministic per model, dataset and split).
- Agent labeling (`/api/label/…`) answers `501 not_implemented` and `agent_labeling` is not in `features`: the
  mock never starts an agent CLI. Studio's own in-browser mock simulates a job for UI work.
- Cloud: the real providers are listed; one RunPod connection exists (`cc_4d2a`, `you@example.com`, balance 42.17)
  with a fixed list of offers (one of them with `availability: none`). `POST /api/cloud/connections` accepts any
  values except a secret containing `bad` (`422 cloud_auth_failed`). A run created with `compute` is simulated
  through every phase (about 2 s each of `provisioning`, `booting` and `syncing` with progress, then training like
  any mock run, then `collecting` and `releasing`), with an instance in `/api/cloud/instances` whose cost grows
  with its price; `DELETE` on it fails the run with `cloud_instance_lost`. `run_3c01` is a finished cloud run
  (`cloud.phase: released`, with its cost). `FOLO_MOCK_CLOUD=none` starts without a connection, `off` leaves the
  `cloud` feature out. `/api/uploads` answers `501 not_implemented`.
