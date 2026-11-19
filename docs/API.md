# FOLO Studio HTTP API (v1)

Served by `folo serve` (default `http://127.0.0.1:7878`). JSON over HTTP, live updates over Server-Sent Events.
All paths are under `/api`. Times are RFC 3339 strings; durations are seconds (`*_s`) or milliseconds (`*_ms`).
Boxes in responses are **pixel `xyxy`** in the original image unless a field says otherwise. Ids (`ds_…`, `m_…`,
`run_…`, image ids) are opaque strings.

This file is the contract between `folo-server` and its clients. Change it in the same PR as the code that changes the
behaviour, and keep the Rust types in `crates/folo-api` (used by the server, the CLI and FOLO Desktop) and
`web/src/lib/api/types/` (a file per section, all re-exported by `types.ts`) in sync.

## Conventions
- Creating endpoints answer `201 Created` with the new object; deletes answer `204 No Content`; everything else `200`.
- Request bodies are JSON (`Content-Type: application/json`) except multipart uploads (`/api/predict`,
  `/api/label/jobs/{id}/images`). Unknown fields are ignored; missing optional fields take the defaults listed here.
- Enum values grow: a client reads a value it does not know as unknown rather than failing the whole record (the Rust
  types decode `TrainConfig.optimizer` and `ResolvedTrain.optimizer` into an `Unknown` variant, which a server refuses
  with `invalid_config` where a run would train with it). Version skew: clients built before `musgd` (FOLO Desktop,
  `folo-client`, a `folo serve` polling a remote server or a RunPod worker, before the release that added it) decode
  `Run` strictly and fail on a run whose `config.optimizer` or `resolved.optimizer` is `musgd` (`unknown variant
  'musgd'`), which fails their whole runs list: update those clients together with a server that can train MuSGD.
- Responses are gzip-compressed when the client sends `Accept-Encoding: gzip` (never images, SSE or weights
  files: `application/octet-stream`).
- CORS: by default only browser origins on `localhost`, `127.0.0.1` or `[::1]` (any port, e.g. the Vite dev server
  on :5173) may call the API; `folo serve --cors-any` allows every origin. Because browsers send simple form POSTs
  without a preflight, state-changing requests (`POST`, `PUT`, `DELETE`) whose `Origin` header is neither such a loopback
  origin nor the server itself are refused with `403 forbidden_origin`. Requests without `Origin` (curl, agents,
  FOLO Desktop) are unaffected.
- Host: a server listening on loopback (`folo serve`'s default `--host 127.0.0.1`, also `--mock`) answers only
  requests whose `Host` header is `localhost`, `127.x.y.z` or `[::1]` (with or without a port, any case, `localhost.`
  too) or a name given with `folo serve --allow-host <name>` (repeatable; for a tunnel or reverse proxy that keeps
  its own `Host`). Anything else, reads, SSE and Studio's files included, is `403 forbidden_host` before any handler
  runs: a website whose name re-resolves to 127.0.0.1 (DNS rebinding) sends its own name as `Host`. A request
  without `Host` is served. A server bound beyond loopback (`--host 0.0.0.0`, a LAN address, a cloud worker) checks
  no `Host`: protect it with `--token`.
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
| 400 | `bad_request`, `invalid_path`, `invalid_query`, `invalid_body`, `invalid_config`, `unknown_format`, `invalid_dataset`, `dataset_unfinished`, `bench_class_mismatch` | malformed input (`invalid_config`: a `TrainConfig` field is out of range, or the dataset has no classes / training images; the message names it. `invalid_dataset`: e.g. more than 4096 classes, a benchmark source without a labeled image. `bench_class_mismatch`: a model entry whose model knows none of the benchmark's classes) |
| 401 | `unauthorized` | a request without the server's token (see Conventions) |
| 403 | `forbidden_origin` | a state-changing request from a disallowed browser origin (see Conventions) |
| 403 | `forbidden_host` | any request to a server on loopback whose `Host` is not a loopback name nor an `--allow-host` one (see Conventions); the hint names `--allow-host` and `--token` |
| 404 | `dataset_not_found`, `image_not_found`, `model_not_found`, `run_not_found`, `label_job_not_found`, `path_not_found`, `device_not_found`, `no_matching_dataset` (`/eval` of a model no registered dataset has the classes of), `epoch_not_available`, `route_not_found`, `upload_not_found`, `provider_not_found`, `connection_not_found`, `offer_not_found`, `instance_not_found`, `bench_not_found`, `bench_entry_not_found`, `bench_loop_not_found` | something named does not exist (`device_not_found`: a `gpu:<n>` this machine has no usable adapter for; the hint lists the devices it has). The `*_not_found` of listed things hint at the list that has them |
| 405 | `method_not_allowed` | wrong HTTP method on a known path |
| 409 | `run_active`, `run_not_active`, `dataset_in_use`, `run_not_resumable`, `run_not_queued`, `label_job_active`, `label_job_not_active`, `label_boxes_changed`, `upload_conflict`, `output_exists`, `connection_in_use`, `cloud_refused`, `bench_frozen`, `bench_active`, `bench_entry_active`, `bench_entry_not_active`, `bench_entry_not_done`, `bench_entry_not_open`, `bench_entry_in_loop`, `bench_loop_active`, `bench_loop_not_active` | deleting (or resuming) a run that is still active; stopping one that is not; deleting a dataset's files while a run is active or another dataset reads them; resuming one that completed; moving one that does not wait in the queue; changing a labeling job while its agent run is active, or an image while it is in an agent call; stopping a job without a run; a box edit or review made on a revision of an image's boxes that is no longer theirs; an upload part out of order or an upload completed before it is whole; removing a cloud connection with live instances or active runs; a cloud provider that says no (no capacity for the offer, no balance, a quota: its message is passed on); changing the plan of a benchmark that has entries; deleting a benchmark with an active entry or loop, an active entry, an entry a running loop uses, or a running loop; stopping an entry or a loop that has ended; the result or images of an entry that is not done; sending predictions to (or finishing) an entry that is not an open external entry |
| 413 | `payload_too_large` | body over the limit (64 MiB for `/api/predict`, image uploads to a labeling job, new benchmark entries and parts of an open one, 2 MiB elsewhere) |
| 415 | `unsupported_media_type` | JSON endpoint called without `Content-Type: application/json` |
| 422 | `invalid_body`, `image_decode_failed`, `invalid_weights`, `invalid_upload`, `cloud_auth_failed` | well-formed JSON with wrong types / missing required fields; an upload that is not an image; a `.folo` file that cannot be read; an uploaded archive that is not a readable tar or holds something unsafe; credentials a cloud provider rejects |
| 500 | `internal` | a bug or an IO failure |
| 501 | `not_implemented`, `not_supported` | the feature exists in the API but not yet in the engine (streaming inference; in mock mode, resuming runs); `not_supported`: valid, but this machine cannot run it (`POST /api/runs` with `"amp": true` where `capabilities.amp.available` is false: refused before anything is queued, with the same reason and a hint) |
| 502 | `cloud_unavailable`, `server_unreachable`, `bad_gateway`, `agent_failed` | a cloud provider's API does not answer or fails; a worker (another FOLO server) does not answer, or answers with something that is not this API; a coding agent's call fails or its answer is not usable (designing a benchmark plan). As `Run.error_code` of a cloud run also `cloud_boot_failed`, `cloud_sync_failed` and `cloud_instance_lost` (docs/CLOUD.md) |
| 507 | `gpu_out_of_memory` | the GPU ran out of memory, or one tensor is larger than the GPU allows a buffer to be (inference; a training run fails with this message in `Run.error`); the hint says to lower `batch` or `imgsz` or use a smaller `arch` |

## System
- `GET /api/health` → `{ "status": "ok", "version": "0.1.0", "api": "v1", "mock": false, "features": [...] }`
  (`features` as in `/api/system`, without its hardware probe, so a client asks here; absent from older servers)
- `GET /api/system` → `{ "cpu": { "name", "threads", "util" }, "memory": { "used_bytes", "total_bytes" },
  "gpus": [{ "ordinal", "name", "backend": "vulkan|dx12|metal", "memory_bytes", "util" | null }],
  "data_dir"?: "/home/me/.folo", "features": ["export_onnx_switches", "queue_control", "weights_onnx"],
  "capabilities"?: Capabilities, "gpu_pool"?: { "device", "live_bytes", "cached_bytes", "peak_live_bytes" } }`
  (`util` in `0..1`; CPU utilisation is measured since the previous reading, over at least 200 ms (sysinfo's minimum interval): a call sooner than 200 ms after the last reading returns that reading's CPU numbers again, the first reading after the server started is measured over that interval, and memory is read fresh on every call. `data_dir` is the real server's
  `--data-dir`, with `/` separators, where its runs, models and exports live; the mock server leaves it out. `features` lists optional API features
  the server implements, so a client can check before relying on one: `export_onnx_switches` (the export switches
  below; older servers accept and ignore them), `queue_control` (`POST /api/runs/{id}/queue`, `Run.queue_position`
  and `queue` events), `weights_onnx` (`/weights?format=onnx`; older servers answer `400 invalid_query`), `run_cli` (`GET /api/runs/{id}/cli`
  and `RunEstimate.cli`), `web_wasm` (the Studio this server serves includes the in-browser runtime, folo-wasm,
  so its Playground can run models with WebGPU or the wasm CPU; left out for a build without it, or without a
  web app), `agent_labeling` (the `/api/label` endpoints, Agent labeling below), `label_image_filters`
  (`reviewed` and `ids` on `GET /api/label/jobs/{id}/images`; older servers ignore them), `label_revisions`
  (`expect` on `PUT /api/label/jobs/{id}/images/{image_id}` and `POST /api/label/jobs/{id}/review`), `uploads` (`/api/uploads`),
  `run_cursors` (`after_step` / `after_epoch` on `/metrics`, `after` and `total` on `/logs`, `Range` on `/weights`),
  `run_checkpoint` (`checkpoint` on `POST /api/runs`; these three are what a cloud worker needs, docs/CLOUD.md),
  `cloud` (`/api/cloud`, `TrainConfig.compute`, `Run.cloud` and `cloud` events), `vision_bench` (the `/api/bench` endpoints, Vision benchmarks below). `folo mcp --server` refuses the
  export switches without `export_onnx_switches`.) In real mode `gpus` comes from folo-gpu
  (wgpu): every physical adapter once, through its preferred backend (software adapters are left out), best first,
  so ordinal 0 is the adapter `gpu:0` and `device: auto` use. `memory_bytes` is the adapter's largest device-local
  memory heap on Vulkan (an RTX 4080 SUPER reports 15.7 GiB), and `0` through DX12 or Metal, where wgpu cannot say; wgpu
  reports no load, so `util` is `null`.
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
- `GET /api/datasets` → `{ "datasets": [Dataset] }` (oldest first, without `stats`). When `datasets.json` could not be
  read at start (malformed, unreadable), no dataset loaded and the list answers `500 internal` "… datasets.json is
  malformed …" with a hint, not an empty list; once the file is fixed, `500 internal` "… lists N datasets this server
  did not load when it started" with the hint to restart `folo serve`, until a restart loads them. Registering
  (`POST /api/datasets`, uploads, sample sets) and unregistering answer the same errors meanwhile, so nothing is
  listed without the datasets the file has; a file fixed to list no dataset lifts them.
- `POST /api/datasets` body `{ "name"?, "path", "format": "auto|yolo|coco" }` → `201 Dataset` with `stats`.
  Registers a dataset on disk: a folder, a `data.yaml` or a COCO instances `.json`. `path` must be absolute
  (surrounding whitespace and one pair of quotes, as Windows' "Copy as path" adds, are ignored; a relative path is
  `400 invalid_body`, never resolved against the server's working directory); `name` defaults to the folder's name
  (for a COCO file in `annotations/`, the folder above it; for a `data.yaml`, its folder), numbered `-2`, `-3`, ...
  when a dataset already has that name; `format` defaults to `auto`. Datasets are read with folo-data, exactly as for training (formats and detection rules:
  `docs/DATA.md`): YOLO from `data.yaml` (`path`, `train`, `val`, `test`, `names`; a YAML with only `names` uses the
  folder layout) or `images/{train,val,test}` + `labels/…`; COCO from `annotations/instances_*.json` (Roboflow
  layouts too). Corrupt or missing images and images with broken labels are left out. `stats.broken_labels` counts
  the labels that cannot be read, not images: one per YOLO label file with a broken line, however many it has (its
  image is left out); for COCO one per image left out because annotations of it cannot be read (a bad `bbox` or
  category), however many it has, and one per category,
  image or annotation entry that cannot be read and per annotation file that is not COCO JSON (dropped, the other
  images kept). `/issues` and `folo dataset check` list every broken line and entry. Without `names` / `nc`, classes are named `class0..` up to the
  largest label index. Errors: `path_not_found`, `unknown_format` (nothing there looks like a dataset; the folder above
  a dataset says "'<path>' is not a dataset folder, but '<path>/<sub>' inside it holds a dataset" (looked for one and
  two levels down), with a hint to register that one; a folder with images but no labels says "'<path>' has N images
  but no labels", with a hint to label them; a folder that is not a dataset is said so without walking it: these
  hints look at 20,000 entries at most, also inside the folders they look for a dataset in (there by the names
  detection uses: a `data.yaml`, a YAML with a `train:` line read to 64 KiB, `images/` + `labels/`, an
  `annotations/` JSON; only regular files are read, and nothing the `folo mcp --confine` read guard refuses), and
  nothing waits for them),
  `invalid_dataset` (malformed `data.yaml` / JSON, `nc` disagreeing with `names`, more than 4096 classes),
  `dataset_unfinished` (the dataset's folder holds a `.folo-unfinished` file: a `folo dataset synth` / `convert`
  is writing it or was stopped while writing it; delete the folder and run the command again, docs/DATA.md). The
  request returns once the dataset is indexed (every image header is read, in parallel). Registering a path that
  is already registered (the same folder by its canonical path, whatever the spelling, with `format` `auto` or
  the same format) returns the existing dataset (still `201`) instead of a duplicate; its `name` is kept.
- `POST /api/datasets/synthetic` body `{ "name", "images": 1000, "classes": ["circle","square","triangle"], "imgsz": 640, "seed": 0 }`
  → `201 Dataset` with `stats`. Only `name` is required. Generates square JPEG images (folo-data's shapes
  generator, deterministic in `images`, `classes`, `imgsz` and `seed`; a twentieth of the objects extra-small, so
  under 10% of the boxes are under 5% of the image side and a sample set passes Studio's health checks) into
  `<data dir>/datasets/<name>` (the name
  lowercased to `a-z0-9-`, at most 48 characters, a Windows device name such as `nul` or `com1` with `-set` appended,
  with `-2`, `-3`, ... when taken) and registers them, named `name`, or `name-2`, `name-3`,
  ... when a dataset already has that name (case aside; decided as the set is registered, so two generations of one
  name at once get two names). Images are split 80 / 20 into
  train / val. `classes`: 1 to 8 distinct names; `circle`, `square`, `triangle`, `star`, `ring`, `cross`,
  `hexagon` and `heart` get that shape, every other name the next shape no name has claimed. `400 invalid_body`
  when `name` is blank, `images` is outside `1..=100000`, `imgsz` outside `32..=4096`, or `classes` breaks those
  rules (the message says which). `409 output_exists` when another writer (such as `folo dataset synth`) took the
  new folder first; that folder is left as it is.
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
- `DELETE /api/datasets/{id}` → `204` (unregisters; its files stay, unless `?files=true`, below). Runs that used the dataset keep their
  metrics and models; their sample images (`/images` of this id) are gone with it. Real mode keeps the unregistered
  dataset's identity (its canonical path and format) and id as a tombstone while a run the server has loaded
  references the id: registering that identity again gets the same id back, and those runs find their dataset again
  (also for a dataset registered by an older version, once this one unregisters it). A dataset registered as new
  gets a fresh id that no registered dataset, tombstone or loaded run has, so an id is never handed to another
  dataset while a run names it. Unregistering waits for a run being created or resumed, so that run counts. A
  tombstone is dropped lazily: at the next registry write (registering or unregistering a dataset), when no run the
  server has loaded references its id then (registering that dataset again afterwards gives it a fresh id). `?files=true` also deletes the dataset's folder, only for a dataset this
  server wrote into its data dir (the folder right under `<data dir>/datasets`: a generated sample set, a labeling
  job's export, an upload); any other is `400 invalid_query`, and nothing is unregistered or deleted. It is `409
  run_active` while any run on this server is queued, running or stopping, whatever dataset it trains on. It is `409
  dataset_in_use` while another registered dataset, or a run of an unregistered dataset that keeps its id, reads
  images or labels in that folder by its description as it reads at that moment (its root, split sources and the
  folders of its images and labels), and also while what one of them reads cannot be checked (its description
  cannot be read or parsed, or one of those folders cannot be listed or resolved): the message names that dataset,
  nothing is unregistered or deleted, and unregistering without `files=true` still works. A dataset whose path is not
  there at all (its folder was moved away, or it was deleted with its files and a run keeps its id) reads nothing and
  refuses nothing. The check covers the
  folders of other registrations and retained runs, resolved through directory links (a folder of another dataset
  that is a junction or symlink into this one counts), but not a link to a single image file from elsewhere.
  Registering a dataset or creating or resuming a run waits until the checks and the deletion are done. Mock mode keeps
  the ids of unregistered datasets in memory the same way (for as long as it runs); there a generated set goes with
  `files=true` (also `409 run_active` while a run created through the API is active) and a folder registered from
  disk is refused.

```jsonc
// Dataset
{ "id": "ds_8f2c", "name": "shapes", "path": "/data/shapes", "format": "yolo|coco",
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
  val, eval and export take the name everywhere a model id goes (Studio's and Desktop's "Start from", MCP `train { weights }`).
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
  `pr` has one curve per class that has ground truth in the split (a class without any has none; each curve names
  its `class`), in class order, sampled at 101 recall points (`0.00` to `1.00`, IoU 0.5).
  `precision` is the COCO-interpolated envelope (it never rises with recall), and `ap50` is its mean.
  `confusion` is `(classes + 1) × (classes + 1)`: rows are the true class, columns the predicted class, and the
  last row and column are background. `confusion[c][n]` counts missed objects of class `c`, `confusion[n][c]`
  counts false positives predicted as `c`, and `confusion[n][n]` is always 0.
  In real mode the model's dataset is its run's (the registered copy while it is registered, else the path the run
  trained on); a model from the models folder is evaluated on the first registered dataset with exactly its classes
  (`404 no_matching_dataset` when there is none: the message gives the class count and the first four names); `400 invalid_dataset` when the dataset's classes (names, in order)
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
    names it: `<model name>.onnx`, `-decoded.onnx` with `decode`, `-f16.onnx` with `half`, `-decoded-f16.onnx` with
    both; characters outside `A-Z a-z 0-9 . _ -`
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
  `queued` (their log says how many runs are ahead). A body with a `distill` key (distillation from a teacher,
  docs/MODEL.md §8.9) is `400 invalid_config`, for a new run and a resume alike: distillation is a setting of
  `folo train --teacher` only, refused here so that a request for a teacher never trains without one. A run that
  continues a distillation run's checkpoint keeps its teacher (it is part of the checkpoint's `folo.train`), and
  `GET /api/runs/{id}/cli` of such a run adds `--teacher` and its `--set distill.*` values.
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
- `POST /api/runs/{id}/stop` → `Run` (status `stopping` → `stopped`; the last finished epoch's checkpoint is kept,
  none before the first epoch ends). `409 run_not_active` when the run already finished. A queued run stops at once
  (one that never started is better deleted: `DELETE /api/runs/{id}` below); a running one after its current batch (the epoch in
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
- `DELETE /api/runs/{id}` → `204` (`409 run_active` while `running` or `stopping`, or `queued` after it had started: a
  resumed run waiting to continue; and for every active cloud run, `queued` while its instance starts included, and
  for any cloud run another FOLO process on the data dir follows). A local run waiting in the queue that never
  started leaves the queue and is deleted (the runs behind it move up; Remove from queue in Studio and FOLO Desktop). Real mode removes the run folder, its weights included, so its
  models disappear too.
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
  the run's log history (the trainer's lines, the server's, and one per finished epoch: `epoch 3/5: loss 3.214,
  mAP50-95 0.073, 15.2 s`), most recent `limit` lines (default 500, clamped to 1..5000), oldest first. `total` counts
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
  "optimizer": "auto",         // auto | sgd | adamw | musgd (docs/MODEL.md §8.2)
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
// An empty name becomes "<dataset name>-<arch suffix>", e.g. "shapes-1k-n", the dataset name cut so the whole fits in
// 64 characters. Limits: name at most 64 characters (as a dataset's; longer is `400 invalid_config` for a new run;
// resuming a run named longer by an older version keeps its name), epochs 1..=10000,
// batch 1..=1024, imgsz a multiple of 32 in 32..=4096 (real mode trains at 64 or more), 0 < lr0 < 10.
// POST /api/runs also takes "resume_run_id": "run_…" (see above); it is not part of Run.config.
// Real mode maps TrainConfig onto the trainer's config (docs/MODEL.md §8.1): the other recipe keys keep their
// defaults, and "auto" resolves to the GPU when a model can be placed on it, else the CPU.
// Run
{ "id": "run_7d1e", "name": "shapes-n", "status": "running", "config": TrainConfig, "created": "...",
  // + "error_code": "not_supported" | "training_failed" | ... when failed (folo_api::codes; absent otherwise and from
  //   older servers), next to "error" (the message); the final `status` event carries it too. A `stopped` run has
  //   "interrupted" when the server stopped or restarted while it trained or waited (not a client's stop)
  "started": "..." | null, "finished": null | "...",
  // "finished" of a run "interrupted" by a server that was killed (not stopped) is when it last reported (a step, an
  //   epoch, a log line; recorded at least every 10 s while those come), not when the server started again: the time
  //   it was down is not training time. A run killed during a long phase that reports nothing (a long validation
  //   pass) ends at its last report, before that phase
  // + "paused_s": 52800.0 for a resumed run: the seconds it spent not training since "started", from each end (stop,
  //   failure) until it trained again; while it waits in the queue after a resume, the wait so far is included (and
  //   a resumed run stopped there keeps it). Its training time is "finished" (or now, while it is active) minus
  //   "started" minus "paused_s", which does not grow while it waits. Absent for a run never resumed and from older
  //   servers
  "epoch": 12,             // epoch in progress, 1-based (0 before the first step); once ended, the last epoch it
                           // finished (completed, stopped) or the one it failed in (failed)
  "epochs": 100,
  "step": 575,             // optimizer steps completed over all epochs
  "steps_per_epoch": 50,  // 0 until the trainer is built (before the first `running` status; events carry it)
  "eta_s": 1840 | null,    // null unless queued / running / stopping
  "best": { "epoch": 11, "map50_95": 0.61 } | null, "device": "NVIDIA GeForce RTX 4080 SUPER (vulkan)",
  "model_id": null | "m_31aa",  // set once the run completed, or stopped after at least one validated epoch
  "dataset": { "name": "shapes-1k", "classes": ["circle", "square"] },  // the dataset as it was when the run was
                           // created, so the run still names it once the dataset is removed (`classes` empty for runs
                           // from before they were recorded; absent in mock mode and from older servers)
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
  "synced": "..." | null,           // when weights/last.folo was last copied to this machine
  "ended_in": "syncing" | null,     // the phase it was in when the run ended (set at releasing): where a failed run
                                    // failed, and the phases after it it never reached; null before, and from older servers
  "unanswered_since": "..." | null } // while the run trains, since when its worker has not answered (it may still train
                                    // there); null while it answers, once the run ends, and from older servers
// While phase is provisioning, booting or syncing the status is `queued` (nothing trains yet): show the phase.
// ResolvedTrain: the settings the run trains with. `config` stays what was requested ("optimizer": "auto",
// "lr0": 0.01); these are what that became (docs/MODEL.md §8.2, §8.4), as in the run folder's config.json.
{ "optimizer": "sgd|adamw|musgd", // never auto
  "lr0": 0.01,                  // base LR of every parameter group: auto's pick, else config.lr0
  "lrf": 0.01, "cos_lr": false, // the last step runs at lr0 · lrf; linear or cosine decay
  "momentum": 0.937,            // SGD / MuSGD momentum, or AdamW β1, after warmup
  "betas": null,                // AdamW only: [β1, β2], e.g. [0.9, 0.999]
  "weight_decay": 0.0005,       // of the decay group, after the batch scaling weight_decay · batch · accumulate / nbs
  "batch": 16, "accumulate": 4, "nbs": 64,  // gradients of `accumulate` batches per optimizer step (≈ nbs images)
  "optimizer_steps_per_epoch": 13,
  "warmup_iters": 39, "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
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
  "duration_s": 31.2 }   // training + validation + the checkpoint's snapshot, from the epoch's start to the
                         // snapshot. Its files are written while the next epoch trains; if the previous epoch's
                         // write is still running at this epoch's end (a disk slower than an epoch), the wait for
                         // it is included too (one write at a time).
// When the Epoch event arrives, that epoch's weights/last.folo (and best.folo when it improved) and
// samples/epoch_NNN.json are on disk, each written to a temporary name, synced and renamed over the previous one.
// The trainer writes them on a writer thread and holds the events of the next epoch behind the Epoch event until
// they are, so the order of events is that of the run; a write that fails fails the run with no Epoch event for
// that epoch.
```

## Inference
- `POST /api/predict` — `multipart/form-data` with `image` (file), `model_id`, optional `conf` (0.25), `iou` (0.7),
  `max_det` (300) (instead of `image`, the form may name `dataset_id` + `image_id`); or JSON
  `{ "model_id", "dataset_id", "image_id", "conf"?, "iou"?, "max_det"? }` →
  ```jsonc
  { "model_id": "m_31aa", "image": { "width": 1280, "height": 720 },
    "boxes": [ScoredBox],                       // highest score first
    "timing": { "preprocess_ms": 0.8, "inference_ms": 3.1, "postprocess_ms": 0.2, "total_ms": 4.1 },
    "device": "gpu:0",
    "cold": true }                              // only on a model's first prediction since it was loaded
  ```
  `cold` (absent when false) marks the model's first prediction since the server loaded it: its `timing` includes the
  warm-up (GPU kernels compiled, buffers allocated), so it is not what the next call takes. Such a call is never a
  throughput sample (`GET /api/system/throughput` measures warm calls only).
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
  - with a fused model, single-label, the head is evaluated candidate-first (docs/MODEL.md §9.5: the box branch
    runs only at the candidates, so it cannot run before they are known), and `inference_ms` holds the whole
    network including the candidate extraction, the candidates' box decode and, on a GPU, their readback;
    `postprocess_ms` NMS and the mapping to image pixels;
  - `total_ms`: their sum.
  `/api/predict` (and MCP's `predict_batch`, which calls it per image) times one image per call, so these are that
  call's stages. A batch through the library (`Predictor::predict_batch`) gives each image its pass's stage times
  divided by the images in it; on the CPU the batch runs as chunks of two images side by side, each image getting
  half its chunk's times, so the images' `total_ms` add up to more than the batch's wall time (by about the number
  of chunks running at once): per image they are a share of the work, not the time a caller waited.
  The browser runtime (docs/WASM.md) cannot wait for WebGPU mid-pipeline, so there `inference_ms` also holds the
  on-device head decode and its readback. FOLO Desktop's Local GPU uses the same predictor as the server.
- `GET /api/predict/stream?model_id=..&conf=..` — WebSocket: the client sends JPEG frames (binary), the server
  answers each with the same JSON as `/api/predict` (for webcam / screen demos). Later milestone; answers
  `501 not_implemented` for now.

## Throughput and estimates
Speed measured on this server, so clients can show this machine's numbers instead of static ones, and say where each
number comes from. Servers with these endpoints list `throughput`, `run_estimate` and `benchmark` in `/api/system`'s
`features`. Every number carries a `source`: `measured` (this server: training runs, `/api/predict` calls, benchmark
probes) or `reference` (docs/BENCHMARKS.md's FOLO measurements on our test machine, an RTX 4080 SUPER and an
i7-14700KF, scaled by compute to other sizes; `note` says which and how).

- `GET /api/system/throughput` → `ThroughputTable`: `training` rows keyed by (`device`, `arch`, `imgsz`, `batch`,
  `amp`) and `inference` rows keyed by (`device`, `arch`, `imgsz`, `precision`, `tensor_cores`; `batch` is always 1).
  Every row has a flat `imgs_per_s` (training: the mosaic epochs' median, else the close-mosaic epochs'; inference:
  `1000 / latency_ms`) and `measured_on`, the adapter or CPU the number was measured on (a reference's is our test
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
- `GET /api/runs/{id}/cli` → `{ "cli": "folo train --data /data/fasteners --weights folo-n-coco --head o2m --epochs 15
  --amp", "argv": ["folo", "train", ...] }`: the CLI command of a run's config, built from the same `TrainConfig`
  mapping the server trains with (a test parses it with the real `folo` arguments and compares the configs).
  `--data` is the dataset's folder on the server's machine, with `--format yolo|coco` when it was registered with
  one (so a folder holding both reads the same); `--weights` is the published name for published weights,
  else the `.folo` file the run started from (so no `--model`: the CLI takes the architecture from the weights,
  and `--head` is always given, since the CLI's default is the weights file's head); settings equal to the
  defaults are left out, and a value starting with `-` is attached (`--name=--amp`). The recipe a run trained with is read
  from its folder's `config.json` `train` object or, when the folder has none (a cloud run: the server mirrors its
  history and copies its weights), from the `folo.train` of its `weights/last.folo` (else `best.folo`). An `auto`
  run that trained under an older `auto` rule than a new run takes (`auto_rule`, rule 1 when absent) ends with
  `--set auto_rule=N`, so a long run made before rule 3 repeats with SGD, not MuSGD, and a short one made before
  rule 4 with AdamW; a run made before `warmup_floor` existed (no `warmup_floor`) ends with
  `--set warmup_floor=100`, the warmup floor it trained with; an `auto` rule-5 run that trained from pretrained
  weights (`transfer`) but whose command has no `--weights` (a run continued from an imported checkpoint) ends with
  `--set auto_rule=4` (and `--set musgd.muon=.. --set musgd.sgd=..` when its weights were not the defaults), which
  resolves the same optimizer with the MuSGD step weights it used (docs/MODEL.md §8.2).
  A run whose optimizer this
  server does not know (made by a newer FOLO) gets `400 invalid_config` instead of a command that would train `auto`. `cli` is quoted for PowerShell,
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
  time; loader waits included, validation excluded); the rest of `Epoch.duration_s` (validation, the
  checkpoint's snapshot, and a wait for the previous epoch's checkpoint write if it was still running) gives the
  validation rate, so a disk slower than an epoch shows as slower validation; the first epoch's extra time over the second gives `startup_s`;
  `peak_vram_bytes` is the GPU buffer pool's peak since the run started (inference served meanwhile counts too).
- **Inference**: every `/api/predict` call's `timing` (`total_ms`, image decode included, and `inference_ms`),
  except a model's first after it is loaded and calls that overlap a run (one queued or training when the call
  starts or ends, or the worker taking up or letting go of a run during it).
- **ETA of running runs**: `Run.eta_s` (and `Step.eta_s`) is the run's own measured rate, per mosaic phase: the
  batches left in mosaic epochs at the run's mean mosaic batch time, loader waits included; those in close-mosaic
  epochs at their own mean once they have started, before that at the mosaic batches' step time without the loader
  waits (without mosaic the loader keeps up); plus, for each epoch left, a validation and the epoch's other work
  (starting the loader, writing checkpoints and samples) at their recent time, the latest epoch weighing half. On
  1080p JPEGs a mosaic epoch of folo-n took 24 s and a close-mosaic one 10 s, so one rate for both is off by up to 2x;
  on a small dataset the other work was more than half of each epoch and validation grows as the model starts to
  detect, so leaving the first out or taking the mean since the start said about half the time left.

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
  Code `haiku` at `low`, Codex its own default model at `low` (docs/LABELING.md has the measurements behind
  them). `fast` says the agent's default model has a faster service tier a run can ask for
  (`StartLabeling.fast`): Codex's Fast (priority processing, which uses more of the plan's usage); a model's own
  `fast` says whether it offers it (Codex: its catalog entry lists the `priority` tier). No Codex model offers it
  when the user's Codex `config.toml` sets `model_catalog_json`, `profile`, `openai_base_url` or
  `chatgpt_base_url`, or exists but cannot be read (the catalog FOLO reads may then not be its calls'; `start`
  with `fast` says so). Claude Code
  has none for `claude -p`.
- `GET /api/label/jobs` → `{ "jobs": [LabelJob] }` (newest first)
- `POST /api/label/jobs` body `{ "name", "classes": ["car", "person"], "instructions"?: "", "path"?: "/data/photos" }`
  → `201 LabelJob`. `name` must not be blank (one line of at most 100 characters); `classes`: 1 to 80 distinct,
  non-blank names of at most 64 characters (no `,`, `:`, line breaks or other control characters; they are
  compared without case); `instructions`:
  at most 4000 characters. With `path`, the images of that folder are added as by `POST …/images`
  (`path_not_found` and a relative path create nothing). Anything else wrong is `400 invalid_body`, naming the field.
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
  `{ "path": "/data/photos" }` for a folder on the server's machine (searched recursively, in path order; or one
  image file), whose images are read in place, not copied. A file that is not a readable image, an upload with
  the bytes of an image the job already has, or a folder image it already has (the same file) is listed in
  `skipped` with the reason instead of failing the request; so is everything past the job's limit of 10000
  images, and a folder without a single image (one entry for the folder, naming a few of the other files'
  extensions) or a `path` to a file that is not an image. File names are unique within a job: a second `a.jpg` becomes `a-2.jpg`. The path is taken without the
  whitespace and one pair of matching quotes around it (Windows' "Copy as path"); a relative path is
  `400 invalid_body` (it is never read from the server's working directory), a blank one too. `404 path_not_found`
  for a missing folder. Images may be added while a run is active (they wait for the next run).
- `GET /api/label/jobs/{id}/images?status=pending|labeling|labeled|failed&reviewed=true|false&ids=a,b&offset=0&limit=100`
  → `{ "total", "images": [LabelImage] }`, in the order the images were added. `limit` defaults to 100 and is capped
  at 1000; `total` counts all matches before `offset` / `limit`; any other `status` or `reviewed` is
  `400 invalid_query`. `reviewed` keeps the images whose `reviewed` is that value (`status=labeled&reviewed=false`
  is what is left to review), `ids` (comma-separated) only those images (ids it does not know match nothing); both
  need `label_image_filters` in `features`.
- `GET /api/label/jobs/{id}/images/{image_id}` → the image bytes; `?thumb=256` as for dataset images.
- `PUT /api/label/jobs/{id}/images/{image_id}` body `{ "boxes"?: [Box], "reviewed"?: bool, "expect"?: string }` → `LabelImage`:
  labels drawn by hand, and whether a person checked them. `boxes` replace the image's boxes; the image becomes
  `labeled` with `source: "manual"` and `reviewed: true` (an empty list labels it as a background image).
  `reviewed` alone marks the labels it has as checked (or not) and leaves boxes and `source` alone; only a
  `labeled` image can be marked reviewed (`400 invalid_body` otherwise, and for a body with neither field). Boxes are clipped to the image; one with a `class` outside the job's classes, a
  non-finite coordinate, or less than a pixel left after clipping is `400 invalid_body`, as are more than 1000
  boxes. `409 label_job_active` while the image is in an agent call (`labeling`). Every other image of a running
  job can be edited: one that still waited for the run leaves it (`run.total` drops by one), so labels drawn by
  hand are never replaced by a call that had not started. `expect` (feature `label_revisions`) is the revision of
  the boxes the change was made on: `"<count>-<hash>"`, the number of boxes and the FNV-1a 32-bit hash (8 hex
  digits) of each box's `class` (u32) and `x1`, `y1`, `x2`, `y2` (f32 bits), little-endian, in order
  (`folo_api::label_box_revision`; JSON keeps f32 exact, so a client computes it from the `LabelImage` it read).
  When the image's boxes are another revision now, nothing changes: `409 label_boxes_changed`. The check and the
  save are made under the same lock, so a run (or another edit) cannot land in between.
- `POST /api/label/jobs/{id}/review` body `{ "image_ids": [string] (at most 1000), "reviewed"?: true, "expect"?: {
  image_id: revision } }` → `{ "updated", "failed"?: [{ "id", "code", "message" }] (the first 20), "failed_total",
  "to_review" }` (feature `label_revisions`): marks many images reviewed (or not) in one save, each as `reviewed`
  alone does in `PUT`. An image that is not there (`image_not_found`), is not `labeled`, is in an agent call
  (`label_job_active`) or whose boxes are not its `expect` revision (`label_boxes_changed`) is counted in `failed`
  and the rest are marked. `to_review` is the job's labeled images that are not reviewed, after the change. An
  `expect` key that is not in `image_ids` is `400 invalid_body`, before anything is marked. More than 100 marks
  send one `reload` event instead of an `image` event each.
- `DELETE /api/label/jobs/{id}/images/{image_id}` → `204` (an uploaded image's file is deleted too). `409
  label_job_active` while the image is in an agent call, as for `PUT`.

**Asking an agent**
- `POST /api/label/jobs/{id}/start` body `StartLabeling` → `LabelJob` (`status: "running"`, with the new `run`).
  Only `agent` is required; `model` and `effort` default to the agent's `default_model` and `default_effort`.
  `fast: true` asks for the agent's faster service tier (only with a model that offers it: the named model's
  `fast`, or the agent's `fast` for its default model; the run's `fast` says it did).
  The run labels the `scope` (`unlabeled`, the default: `pending` and `failed`
  images; `all`: every image, replacing existing labels, hand-drawn ones too), or exactly `image_ids`. `note` (at
  most 2000 characters) is an extra instruction for this run only: the way to tell the agent what its last
  labels got wrong. Errors: `422 invalid_body` (an unknown `agent`), `400 invalid_body` (a `model` with characters
  outside `A-Z a-z 0-9 . _ : / - [ ]` or over 100 of them, `batch` outside `1..=16`, `parallel` outside `1..=8`,
  `fast` with a model that has no faster tier, an unknown image id, nothing to label), `501 not_supported` (the agent's CLI is not installed; the hint says
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
    event each when more than 100 images are added or marked at once, and to a client that fell behind the
    stream);
  - `log` `{ "t", "level", "message" }`: the job's log (calls made, tokens used, boxes dropped, failures).

  The stream replays nothing: load the job's images with `GET …/images`, then apply events.
- `GET /api/label/jobs/{id}/logs?limit=500` → `{ "lines": [{ "t", "level", "message" }] }`: the job's log history
  (the `log` events), as `GET /api/runs/{id}/logs`.

**How a run works** (and what it costs)
- The run splits its images into batches (`batch` images per agent call: Claude Code 4, Codex 8 by default) and
  keeps `parallel` calls (default 3) in flight. Each call is **one model request with no tools**: the images are
  attached to the prompt (downscaled to at most 1024 px on the long side for Claude Code and 768 px for Codex, and
  re-encoded as JPEG, so an image costs about 1050 and 650 input tokens whatever the file's size), the prompt
  gives each image's size as sent, and the reply is one JSON object with a box list per image, each box
  `["<class>", x1, y1, x2, y2]` in pixels of the image as sent (the older `{"label", "box"}` objects are read
  too, also as pixels). The server clips each box to the image and scales it to pixels of the original.
- Claude Code is started as `claude -p` with its system prompt replaced by FOLO's labeling prompt, no tools, no
  MCP servers, no user or project settings, hooks, skills or `CLAUDE.md`, and no session file; the images go in
  on stdin as a `stream-json` user message. Prompt caching, the session-title request and the advisor tool are
  switched off by environment variables. The fixed cost of a call is about 900 tokens. Codex is started as `codex
  exec` with `--ephemeral --sandbox read-only --ignore-user-config`, FOLO's instructions as the model
  instructions, web search, the features and config keys that add tools or text switched off, Codex's own model
  catalog handed back without the models' tools (`model_catalog_json`, read with `codex debug models` for the
  built-in provider a call uses, and left out, with a warning in the run's log, when it cannot be read, the user's
  `config.toml` sets one of the keys above or cannot be read, or Codex refuses it as it starts: that call, which
  used no tokens, is made again without it, and that Codex version's calls go without it), the images as
  `--image` attachments and
  the reply shape as `--output-schema` (class names limited to the job's classes), and with `fast` its
  `service_tier` set to `priority`; that takes a call's fixed cost from about 20000 tokens to about 700. Both run
  in an empty scratch folder and use the account the CLI is signed in with. The eval
  `crates/folo-server/examples/label_eval.rs` measures both on COCO photos (docs/LABELING.md).
- A reply's boxes are checked before they are stored: a label that is not one of the job's classes (compared
  without case), a malformed box, or one that is empty after clipping is dropped (the log says how many), and the
  call has failed (below) for an image the reply does not mention or whose every box was dropped. One bad box costs only itself, and the JSON is found with text
  around it or split into one object per image. A reply with no such JSON at all is asked for once more; then
  the call has failed for all its images. A batch makes two calls at most, whatever made the first fail (an
  unreadable reply or a passing fault, below).
- A call that has failed for an image puts the reason in the image's `error`. An image without labels (`pending`,
  or `failed` before) becomes `failed`. An image with labels stays `labeled`: its `boxes`, `source` and `reviewed`
  stay as they were, and so does its place in a dataset; `error` holds the reason until a call labels it or its
  boxes are saved (`PUT …/images/{image_id}` with `boxes`). The job's `counts.failed` counts the images whose
  status is `failed`, so such an image does not add to it; the run's own `failed` counts every image of the run
  whose call failed, labeled ones too.
- A run never burns tokens on a broken setup: when the CLI cannot be started or exits with an error (not signed
  in, an unknown model, a usage limit or a rate limit that asks for a wait of more than a minute, with when to
  start again in `run.hint`, in the CLI's words), or three calls in a row
  time out (5 minutes each) or return garbage (every call counts, a retry too), the run ends with `outcome:
  "failed"`, the CLI's message in `run.error` and the images it did not get to unchanged. A passing fault of the
  provider (overloaded, a 5xx status, a dropped stream, a rate limit that asks for a minute or less) is not such an
  error: the call is made once more after 15 s (or the wait asked for, if longer), and if it fails that way again
  it has failed for its images and counts towards the three in a row (`run.hint` then says the provider is
  busy). A 4xx status other than 429 is never a passing fault, and a fault that asks for a wait of more than a
  minute ends the run as a limit does. Claude Code's error results are read from their structured fields
  (`errors`, `api_error`, and `api_error_status`, which stands for any number in the text) as well as their
  text. Once a run is stopped or has failed, no further call is started and
  the ones in flight are killed. `POST …/start` again continues with what is left.
- `usage` adds up what the CLIs report per call, a failed call's too (Claude Code reports what a call that failed
  used), whatever became of the batch (stopped during a wait, timed out; a call killed after the CLI printed
  what it used counts that, and a Codex turn that completed without a reply counts its tokens). `cost_usd` is Claude Code's own figure (list prices, also on a subscription); Codex reports none.

**Making the dataset**
- `POST /api/label/jobs/{id}/dataset` body `{ "name"?, "val_fraction"?: 0.2, "seed"?: 0 }` → `201 Dataset` with
  `stats`. Writes the job's `labeled` images as a YOLO dataset into `<data dir>/datasets/<name>` (the name
  lowercased to `a-z0-9-`, with `-2`, `-3`, ... when taken, as for synthetic datasets) and registers it:
  `images/{train,val}/` (the files copied byte for byte), `labels/{train,val}/<stem>.txt` and a `data.yaml` with
  the job's classes. `pending` and `failed` images are left out; a labeled image without boxes is a background
  image (an empty label file). The images are shuffled with `seed` and `round(n · val_fraction)` of them go to
  `val` (`val_fraction` in `0..=0.5`, else `400 invalid_body`; at least one image always stays in `train`).
  `name` defaults to the job's. `400 invalid_dataset` when no image is labeled, `404 image_not_found` when a
  labeled image added from a folder is no longer on disk (the message counts them and names one; nothing is
  written), `409 label_job_active` while a run is active. While the dataset is written the job cannot be changed, deleted or given to an agent, and its
  images cannot be edited or removed (`409 label_job_active`); images can still be added (they are not part of
  that dataset). The job keeps existing and remembers the dataset as `dataset_id`; exporting again makes
  another dataset.

```jsonc
// LabelAgent
{ "id": "claude|codex", "name": "Claude Code", "available": true, "version": "2.1.302" | null,
  "models": [{ "id": "haiku", "label": "Haiku", "fast"?: true }], // suggestions (`model` takes any name the CLI
                                                       // accepts); `fast`: it offers the faster tier
  "default_model": "haiku" | null,                     // null: the CLI's own default
  "default_effort": "low|medium|high",                 // what a run without `effort` uses
  "fast": false,                                       // its default model has a faster service tier (Codex)
  "hint": null | "Install Claude Code (https://claude.com/claude-code) and sign in with `claude`, or ..." }
// LabelJob
{ "id": "lb_3f9a", "name": "street", "classes": ["car", "person"], "instructions": "Label every ...",
  "created": "...", "status": "idle|running|stopping",
  "counts": { "images": 120, "pending": 20, "labeled": 96, "failed": 4, "boxes": 513 },  // images by status
  "run": LabelRun | null,        // the active run, else the most recent one
  "usage": LabelUsage,           // every run of the job
  "dataset_id": "ds_8f2c" | null }
// LabelRun
{ "id": "lr_9f1c...", // unique per run (absent from older servers): `started` has one-second resolution
  "agent": "claude", "model": "sonnet" | null, "effort": "low|medium|high", "fast"?: true, "started": "...",
  "finished": "..." | null, "total": 24, "done": 16, "failed": 0,   // images of this run (`failed`: its call
                                                                     // failed, labeled images too)
  "outcome": null | "completed|stopped|failed", "error": null | "...", "hint"?: "...", "usage": LabelUsage }
// LabelUsage
{ "calls": 4, "input_tokens": 18400, "cached_input_tokens": 3252, "output_tokens": 5100, "cost_usd": 0.021 | null }
// LabelImage
{ "id": "17", "file": "IMG_0042.jpg", "width": 1920, "height": 1080,
  "status": "pending|labeling|labeled|failed",
  "boxes": [Box],                       // pixel xyxy; kept while the image is labeled again
  "source": null | "agent|manual",      // who drew `boxes`
  "error": null | "...",                // why the last agent call failed; a labeled image it failed for stays labeled,
                                        // boxes, review and all (new labels or saved boxes clear it)
  "reviewed": false }                   // a person checked its labels (labeled images only); new agent labels clear it
// StartLabeling (only agent is required; the rest shows the defaults)
{ "agent": "claude", "model": null, "effort": null, "fast": false, "scope": "unlabeled", "image_ids": null,
  "note": "", "batch": null, "parallel": null }
```

## Vision benchmarks
Grades what a vision model sees ([BENCH.md](BENCH.md) is the guide and defines every number): a **benchmark** is
a frozen set of images with their true boxes plus a plan of stress conditions; an **entry** is one contestant's
run of it (a model of this server, a coding agent, or predictions made anywhere else) with its score; a **loop**
tests whether a model's own labels can train its successor. Servers with these endpoints list `vision_bench` in
`/api/system`'s `features` (real mode only: the mock answers `501 not_implemented` to every endpoint below). Not
the speed probe of `POST /api/system/benchmark`.

The types are `folo_api::bench` (`crates/folo-api/src/bench/`), field for field. A benchmark lives in
`<data dir>/bench/<id>/` (`bench.json`, its copied `images/`, `entries/<id>.json`, `loops/<id>.json`).

**Catalog**
- `GET /api/bench/catalog` → `BenchCatalog`: `{ "conditions": [{ "id", "name", "group", "description" }],
  "severities": 5, "default_plan": BenchPlan, "axes": [{ "axis", "name", "weight", "description" }] }`. The
  conditions a plan can name (`group` is `light`, `blur`, `noise`, `medium`, `digital` or `geometry`), the plan a
  benchmark gets when it names none (every condition at severities 1, 3 and 5, weight 1), and the five axes of
  the score with their weights.

**Benchmarks**
- `GET /api/bench/benchmarks` → `{ "benchmarks": [Bench] }` (newest first)
- `POST /api/bench/benchmarks` body `CreateBench` → `201 Bench`:
  `{ "name", "description"?: "", "source": { "kind": "dataset", "dataset_id", "split"?: "val" } | { "kind":
  "label_job", "job_id" }, "max_images"?: 500, "seed"?: 0, "plan"?: BenchPlan }`. The new benchmark's `id` is
  `bn_` and 12 random hex digits, never reused (its image URLs are served as immutable). Copies the source's images
  (byte for byte) and true boxes into the benchmark: a dataset split's images (every image of the split; one
  without boxes is a background image), or a labeling job's `labeled` images with their boxes. The classes are
  the source's (1 to 80). A source with more than `max_images` (1 to 5000) is sampled with `seed`: the
  benchmark's order is a seeded shuffle of the source cut to `max_images`, so the sample and the order are one
  draw and a prefix of the order is a fair sample; the same source and seed give the same benchmark. Image ids
  are `1`, `2`, ... in that order; file names are the source's (made unique by stem). True boxes are clipped to
  the image and a box with less than a pixel of width or height left is dropped, so `boxes` counts exactly what
  is scored. `source` comes back with the split filled in. `name` as a labeling job's; `description` at most
  4000 characters; `seed` at most 2^53 − 1 (9007199254740991, the largest integer a browser holds exactly). `400
  invalid_body` for a plan that is not valid (below) and for a bad `name`, `description`, `max_images` or `seed`, `404 dataset_not_found` / `label_job_not_found`, `400 invalid_dataset` when no labeled image
  comes out of the source (an empty split, a job with nothing labeled) or it has no classes or more than 80.
- `fingerprint`: SHA-256 (hex) over the engine's version (`folo_bench::ENGINE_VERSION`, bumped whenever a
  condition's pixels or a definition of the score changes), the seed, the classes, every image's content hash,
  size and true boxes in order, and the plan's conditions, severities and weights (not `rationale`, `why` or
  `designed_by`). A benchmark's fingerprint changes when the server's engine does: entries graded by different
  engines never compare as equals.
- `GET /api/bench/benchmarks/{id}` → `Bench`; `404 bench_not_found`. `entries` counts every entry, `best` is the
  first entry of the leaderboard when it is `done` (`{ "entry_id", "name", "score", "partial" }`, `partial` its
  `score.partial`; a partial entry leads only when no complete one is done), and `active` is true while an entry
  of the benchmark is queued, running or stopping (an open external entry is running) or a loop of it is running
  or stopping (exactly when `DELETE` is `409 bench_active`).
- `PUT /api/bench/benchmarks/{id}` body `{ "name"?, "description"?, "plan"? }` → `Bench`. Absent fields stay; the
  fingerprint follows a new plan. `409 bench_frozen` for a `plan` while the benchmark has a `done` or an active
  entry (failed and stopped entries do not freeze it: they compare with nothing; delete the done ones and stop the
  active ones, or make a new benchmark); the name and description can change any time.
- `DELETE /api/bench/benchmarks/{id}` → `204`: the benchmark, its images, thumbnails, entries and loops (never a
  model, a run or a dataset a loop made). `409 bench_active` while an entry or a loop of it is active.

A **plan** is `{ "conditions": [{ "condition": "haze", "severities": [1, 3, 5], "weight"?: 1, "why"?: null }],
"rationale"?: null, "designed_by"?: null }`. It is valid when every `condition` is in the catalog and named at
most once, `severities` are 1 to 5 distinct values in `1..=5` in ascending order, and `0 < weight <= 10`. An
empty `conditions` runs the unchanged images only. Text is trimmed (empty becomes `null`).

- `POST /api/bench/benchmarks/{id}/design` body `{ "agent": "claude" | "codex", "model"?, "effort"?, "note"?: "" }`
  → `BenchDesign` `{ "plan": BenchPlan, "usage": LabelUsage }` (a design that fails is a `502 agent_failed`
  without its usage). One call to the agent (as agent labeling makes
  them: no tools, one request; `model` and `effort` default to the agent's, `model` checked as for labeling,
  `note` at most 2000 characters) with the benchmark's description and the note, its classes with their box
  counts, the catalog and the first four of its images (scaled as labeling scales them). The answer is read as
  labeling replies are (a JSON object anywhere in the text) and then cleaned: conditions the catalog does not
  have and repeats are dropped, severities rounded, clamped to `1..=5`, deduplicated and sorted (none: 1, 3, 5),
  weights clamped to `0.1..=10` (none: 1), `why` cut to 300 and `rationale` to 2000 characters. What is left
  must be a valid plan with at least one condition; it comes back with `designed_by` and `rationale` set,
  **not saved**: `PUT` it to adopt it. An answer that cannot be read is asked for once more (never more);
  `usage` adds up both calls. The request is answered when the agent is (at most two minutes). `501
  not_supported` when the agent's CLI is not installed, `502 agent_failed` when the CLI fails (its message, and
  the agent's hint when it has one), does not answer in two minutes, or answers twice with something that is not
  a plan (the reason).

**Images**
- `GET /api/bench/benchmarks/{id}/images?offset=0&limit=100&condition=&severity=` → `{ "total", "images":
  [BenchImage] }` in the benchmark's order (`limit` capped at 1000). `boxes` are the true boxes; with `condition`
  and `severity` they are the true boxes under that condition (`distance` moves them; the others do not).
- `GET /api/bench/benchmarks/{id}/images/{image_id}` → the image file as stored. With `?condition=haze&severity=3`:
  the image under that condition, a JPEG at quality 95 of the same size, byte for byte what every contestant is
  graded on (rendered on request; recent ones are kept in memory). `?thumb=256` as for dataset images, for both
  (variant thumbnails are cached under the data dir too). These URLs never change content: `Cache-Control:
  public, max-age=31536000, immutable`. `400 invalid_query` for an unknown condition, a severity outside
  `1..=5`, or one without the other (empty values count as absent); `404 image_not_found` for an unknown image.
  Any condition of the catalog is served, in the plan or not.

**Entries**
- `GET /api/bench/benchmarks/{id}/entries` → `{ "entries": [BenchEntry] }`: the leaderboard. First the `done`
  entries with the benchmark's `fingerprint` that are complete (`score.partial` false) by `score.total`, highest
  first; then the `done` partial ones by `score.total`; then `done` entries with another `fingerprint` (graded
  before an upgrade changed the engine: they do not compare with the others until they are run again) by
  `score.total`; equal scores keep the older entry first (by `created`, which is to the millisecond and
  strictly later for each new entry of a benchmark; then by `id`); then every other entry (queued, running, stopping,
  failed, stopped), newest first. `Bench.best` is the first of this order when it is `done` and has the
  benchmark's fingerprint.
- `POST /api/bench/benchmarks/{id}/entries` body `StartBenchEntry` → `201 BenchEntry`:
  `{ "contestant": { "kind": "model", "model_id" } | { "kind": "agent", "agent", "model"?, "effort"? } |
  { "kind": "external", "scored"?: true, "open"?: false, "predictions"?: [BenchPrediction] }, "name"?,
  "max_images"?, "conditions"? }`. `name` (one line of at most 100 characters) defaults to the model's name, or the agent's and
  its model's (`Claude Code (haiku)`). `max_images` runs the first so many images (at least 1, `400
  invalid_body` for 0; more than the benchmark has runs them all). `progress.total` is the entry's images times
  one plus the plan's (condition, severity) pairs it runs. `409 bench_frozen` when the benchmark's plan changed
  while the request was checked (send it again).
  - `model`: FOLO runs the model on every image under every condition of the plan (`conditions: false` runs the
    unchanged images only; `max_images` the first so many). Its classes are matched to the benchmark's by name
    without case: `400 bench_class_mismatch` when none matches; benchmark classes it lacks are the entry's
    `missing_classes`, and detections of classes the benchmark lacks are dropped. `404 model_not_found`. Model
    entries run one at a time in the order they were started (`queued` until their turn, then `running`), on the
    server's inference device (`folo serve --device`, as `/api/predict`), in f32, with validation's settings (square
    letterbox at the model's size, NMS at confidence 0.001 and IoU 0.7, multi-label, 300 detections; boxes not
    clipped to the image, as validation scores them), so the unchanged images' mAP is what `POST /api/val` (and
    `folo val`) gives on the same images in the same order. Under a condition the model gets exactly the bytes
    of `GET …/images/{image_id}?condition=…`. `speed.ms_per_image` is the wall time of the unchanged pass (decode
    to boxes; images are prepared on several threads while the previous batch of 16 runs) per image, counted
    from the end of the first batch, so loading the model and warming up the device are left out: a throughput,
    not the latency of one image. Each condition's severity is graded right after its pass (its mAP pair is all
    the result needs), so only the unchanged images' detections are kept until the end.
  - `agent`: the agent is asked for boxes as agent labeling asks (the same prompt, with the benchmark's classes
    and its description as the instructions; batches of the agent's default size, three calls in flight, every
    image scaled as labeling scales it; under a condition, the variant's JPEG). Defaults: the first 48 images,
    `conditions: false` (each condition is another call per batch). The entry is `running` from the start, on a
    thread of its own (never behind the model queue). `501 not_supported` when its CLI is not installed. `scored`
    is false: agents give no confidences. `usage` adds up the calls. Failures follow labeling runs: a CLI that
    cannot be started or reports an error ends the entry at once as `failed` with the CLI's message and the
    agent's hint, and so do three calls in a row that time out or return garbage; a reply that cannot be read, or
    a passing fault, is asked once more (two calls a batch at most). An image whose call still failed, that a
    reply left out, or whose every box was dropped has no answer: when at most
    2% of the entry's image answers (at least one) are missing they count as images where the agent found
    nothing; with more the entry is `failed` ("N of M images got no usable answer from the agent"), because that
    score would not be honest.
  - `external`: `{ "kind": "external", "scored"?: true, "open"?: false, "predictions"?: [] }`. `name` is
    required; `max_images` and `conditions` are `400 invalid_body` (the predictions say what ran). Each
    prediction is `{ "image": "<image id>", "condition"?, "severity"?, "boxes": [{ "class", "score"?: 1, "x1",
    "y1", "x2", "y2" }] }`, boxes in pixels of the image as served, at most 1000 boxes per record. Every image of
    the benchmark must have a record without `condition`; a condition of the plan is either answered for every
    image at every planned severity or absent (then robustness is over the conditions present, and the entry is
    partial unless all are). `400 invalid_body` names the first record that is wrong, in order: an image the
    benchmark does not have, a condition that is not in the catalog or a (condition, severity) the plan does not
    run, a severity without its condition or the other way round, a record that repeats another with other
    boxes (of the same request, or received before), more than 1000 boxes, a class outside the benchmark's, a
    coordinate that is not a finite 32-bit float, corners the wrong way round (`x2 < x1` or `y2 < y1`: most often
    `x, y, width, height` sent as corners; a box of no area is taken), a score outside `0..=1` (when `scored`); then, when the entry is finished, an image without its unchanged
    record, and a condition answered for some images or severities but not all. A record that repeats one with
    exactly the same boxes (and scores, when `scored`) is accepted and ignored: a client that retries a request
    whose answer it lost gets no `400`. With `scored: false` every score is taken as 1.
    - `open: false` (the default): `predictions` is the whole entry, scored in the request and answered `done`
      (`conditions` true when a condition was answered). Up to 64 MiB.
    - `open: true`: the entry is created to be filled in parts, which is what a benchmark of any size needs (a
      model at validation's settings gives up to 300 boxes per image, and a 300-image benchmark with 19 planned
      severities is then far over 64 MiB in one body). The answer is `201 BenchEntry` with `status: "running"`;
      `predictions`, if any, is its first part. `progress.total` is the most records it can have (images times
      one plus the plan's (condition, severity) pairs), `progress.done` the records received. An open entry is
      active (`Bench.active`, `409 bench_active`, `409 bench_entry_active`); `stop` discards it (`stopped`,
      nothing scored), and one the server stops with (or a crash leaves) is `failed` at the next start, as other
      active entries are (its records were in memory). One that gets no part (a request to `…/predictions`, a
      retry included) for 30 minutes expires: it is `failed` ("no predictions arrived for 30 minutes", with a hint
      to start it again), what it held is dropped, and a `bench` event says the benchmark is idle again. What an
      open entry holds in memory is capped at 20 million boxes (the unchanged images' and those of conditions'
      severities still incomplete): a part that would pass it is `400 invalid_body`, saying to send pass by
      pass.
  `progress.total` and `progress.done` of a finished external entry are the records it got.
- `POST /api/bench/benchmarks/{id}/entries/{entry_id}/predictions` body `BenchPredictions` `{ "predictions":
  [BenchPrediction] }` → `BenchEntry` (with the new `progress.done`): a part of an open external entry. Up to 64
  MiB per request, any number of requests, records in any order. A part is checked whole as above before any of
  it is taken: a `400 invalid_body` adds none of its records, so it can be fixed and sent again (a record that
  was already received with other boxes is named: "… was received in an earlier part with other boxes"; one
  received with the same boxes is ignored, so a part can be sent again after a lost answer). **Send pass by
  pass** (the unchanged
  images, then each condition and severity): a (condition, severity) is graded as soon as its last image has
  arrived (its mAP pair is all the result needs) and its boxes are dropped, so the server holds about one pass
  at a time; the unchanged images' detections are kept until `finish` (the evaluation and the per-image outcomes
  need them).
- `POST /api/bench/benchmarks/{id}/entries/{entry_id}/finish` → `BenchEntry` `done`: checks that the entry is
  whole (every unchanged image answered; a condition answered for every image at every planned severity, or not
  at all) and scores it, exactly as the same records in one request would be. A `400 invalid_body` naming what is
  missing leaves the entry open, to be completed and finished again.
- `409 bench_entry_not_open` for `…/predictions` and `…/finish` on an entry that is not an open external entry
  (a model or agent entry, or an external entry that is done, stopped, failed or being finished).
  A finished entry has `score` (`total`, `partial`, the five `axes`), `map50_95` with its 95% interval `ci95`
  (as in the result's `accuracy`), `conditions_run` (how many of the plan's conditions it ran completely, every
  planned severity and every image it runs; for an external entry, the conditions it answered; 0 until `done`),
  and `speed` (a model) or `usage` (an agent). An entry is complete (not `partial` on that count) when it ran
  every image of the benchmark and every condition of its plan; `partial` is also set when an axis could not be
  measured: calibration without confidences, and also for a scored contestant with no detection at confidence
  0.1 or above.
- `GET /api/bench/benchmarks/{id}/entries/{entry_id}` → `BenchEntry`; `404 bench_entry_not_found`.
- `GET /api/bench/benchmarks/{id}/entries/{entry_id}/result` → `BenchResult`: `accuracy` (with the 95% interval,
  by size and per class), `robustness` (per condition and severity; `null` when no condition ran), `calibration`
  (`null` without confidences, or when no detection reaches confidence 0.1), `localization` and `self_labeling`
  (`threshold`: the lowest confidence on the grid `0.00 … 1.00` where at least 10 labels are kept and at least 90%
  of them are right, `null` when there is none; `labels`: the labels kept at `threshold`, all detections when
  there is none; a contestant without confidences reaches the target only with at least 10 detections). `409 bench_entry_not_done` until the entry is
  `done`.
- `GET /api/bench/benchmarks/{id}/entries/{entry_id}/images?sort=order|worst&offset=0&limit=100` →
  `{ "total", "images": [BenchEntryImage] }`: per image of the entry (unchanged) the truth, the detections at or
  above the entry's best-F1 confidence, and how many true boxes were `missed` and detections `wrong`. `worst`
  puts the images with the most `missed + wrong` first (ties in the benchmark's order); any other `sort` is `400
  invalid_query`. `409 bench_entry_not_done` as above.
- `POST /api/bench/benchmarks/{id}/entries/{entry_id}/stop` → `BenchEntry`: a `queued` entry, and an open external
  one (what it received is dropped), is `stopped` at once; a `running` one is `stopping` until its work ends (checked between batches; agent calls in flight are killed),
  then `stopped`. `409 bench_entry_not_active` when it has ended.
- `DELETE /api/bench/benchmarks/{id}/entries/{entry_id}` → `204`. `409 bench_entry_active` while it is active,
  `409 bench_entry_in_loop` while a loop that is still running uses it.

Entries do not survive the server: one that is queued or running when the server stops (or that a crashed
process left so) is `failed` ("the server stopped before the entry finished") with a hint to start it again; one
that was `stopping` is `stopped`.

**Self-improvement loops**
- `GET /api/bench/benchmarks/{id}/loops` → `{ "loops": [BenchLoop] }` (newest first)
- `POST /api/bench/benchmarks/{id}/loops` body `StartBenchLoop` → `201 BenchLoop`:
  `{ "teacher": "<model id>", "pool": BenchSource, "generations"?: 1, "min_confidence"?, "student"?: { "arch"?,
  "epochs"?: 30, "imgsz"?, "batch"?, "device"?, "weights"?, "seed"?: 0 }, "name"? }` (`name` defaults to
  `<teacher's name> loop`; `config` comes back with the defaults filled in: the pool's split, `generations`, the
  student's `arch` and `imgsz` (the first teacher's), `epochs` and `device` (`auto`); `batch: null` is
  `TrainConfig`'s; `student.seed` at most 2^53 − 1). Per generation (1 to 5): the teacher's entry on the benchmark (a `done` entry of the model on
  this fingerprint that ran every image and every condition is reused, else a model entry is started: it is
  queued like any other and listed with the others), the teacher's labels of the pool (its detections from
  `min_confidence` up, else from its own entry's self-labeling threshold: generation 1 from the first teacher's,
  generation 2 from the first student's, and so on; clipped to the image),
  written as a YOLO dataset `<data dir>/datasets/<name>-gen-<n>` named `<name> gen <n>` and registered (the
  pool images the teacher kept at least one box on, `labels.labeled_images` of the `labels.images` it looked
  at: an image it kept nothing on is left out, since its silence is not a label; train and val both with the
  teacher's labels, 10% val rounded; pool images whose bytes are in the benchmark are left out and counted in
  `labels.excluded`), a training run of the student
  on it named `<name> gen <n>` (an ordinary run, through the same path as `POST /api/runs`: it is in
  `GET /api/runs`, queued like any other), and the student's entry; the student is the next generation's
  teacher. The pool's own labels are never read. `400 invalid_body` when the teacher's classes are not the
  benchmark's (same names without case, same order: labels are written by class index), for `generations`
  outside `1..=5`, a `min_confidence` outside `0..=1`, an unknown `student.arch`, `student.epochs` 0, a
  `student.imgsz` that is not a multiple of 32 in `32..=4096` or a `student.seed` over 2^53 − 1; the students'
  training settings are checked as `POST /api/runs` checks them (all but the dataset, which does not exist
  yet: `device`, `weights`, `batch` and the rest, with its codes: `404 device_not_found`, `404 model_not_found`,
  `400 invalid_config`, ...); `404 model_not_found`, `dataset_not_found`, `label_job_not_found`; `400
  invalid_dataset` when the pool has no image outside the benchmark (a pool split defaults to `train`; a labeling
  job gives all its images). A loop fails (`error` and `hint` say why) when a teacher has no confidence to label
  from (no `min_confidence`, and its entry has no self-labeling threshold: "'<model>' never reaches 90% precision
  on at least 10 labels on this benchmark, so there is no confidence to label the pool from", before anything is
  labeled; the hint says to set `min_confidence`), when a teacher keeps boxes on fewer than 10 pool images, when
  its run fails or is stopped by someone else, or when an entry fails.
- `GET /api/bench/benchmarks/{id}/loops/{loop_id}` → `BenchLoop`: `generations[0]` is the first teacher, then one
  per student, each with its `labels`, `run_id`, `model_id`, `entry_id`, `score`, `map50_95` and `ci95` as far
  as the loop has come; `verdict` once `done` (the last generation's `map50_95` against generation 0's `ci95`:
  `improved` above it, `regressed` below, `held` inside, its bounds included). `404 bench_loop_not_found`.
- `POST /api/bench/benchmarks/{id}/loops/{loop_id}/stop` → `BenchLoop` (`stopping`, then `stopped`): stops what the
  loop is doing (its entry, its run). `409 bench_loop_not_active` when it has ended.
- `DELETE /api/bench/benchmarks/{id}/loops/{loop_id}` → `204` (its datasets, runs, models and entries stay).
  `409 bench_loop_active` while it runs.

A loop does not survive the server: one that was running when the server stopped is `failed` at the next start
("the server stopped"), with its finished generations kept; one that was `stopping` is `stopped`.

**Events**
- `GET /api/bench/benchmarks/{id}/events` (SSE; a `: keep-alive` comment every 15 s): a `bench` event (`Bench`),
  then `entry` (`BenchEntry`, on every change; progress at most twice a second), `loop` (`BenchLoop`, on every
  change), `bench` again when the benchmark changes (name, plan, entry count, best entry), and `removed`
  (`{ "id" }`, an entry or a loop was deleted), until the client disconnects, the benchmark is deleted or the
  server shuts down. A subscriber that fell behind gets the current `bench` and should load the lists again.
  A new entry sends `entry`, then `bench`; a finished one too; a deleted one `removed`, then `bench`; a `bench`
  event is also sent whenever `active` changes.

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
  answers (`400 unknown_format` / `invalid_dataset`). An archive that wraps the dataset in a folder (a zip of the
  dataset folder) says "the archive is not a dataset at its top, but its folder '<sub>' holds one", with a hint to
  pack that folder's contents at the archive's top level and upload again.
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
  `..`, links and devices are `422 invalid_upload` on the part that carries them, and so are two files whose paths
  differ only by letter case or a trailing dot or space (`A.jpg` and `a.jpg`: one would overwrite the other on
  Windows; the message names both) and a stream that is not a tar archive; the session is then dropped.
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
- `datasets.json`: the registry, a plain list of registrations (`id`, `name`, `path`, `format`, `created`, and a
  snapshot of `classes` and `splits`), as every version writes it, so an older build opening the data dir lists
  the same datasets. `datasets.removed.json`: the tombstones of unregistered datasets (`DELETE
  /api/datasets/{id}`, above); a missing file has none. Both are rewritten atomically (temporary file + rename) on
  every change, the one that gains an id first (a crash between the two leaves the id in both, and the
  registration counts). A `datasets.json` holding an object with `datasets` and `tombstones` (written by pre-release
  builds of this version for a few hours) is read, and written as a list at the next change. A malformed file is
  never overwritten: registering and unregistering fail with `500 internal` until it is fixed.
- `cache/datasets/<id>.json`: each dataset's index (images, sizes, boxes, stats, issues) plus a fingerprint of the files it
  was built from (relative path, size and modification time of every file under the dataset folder and under
  everything outside it that folo-data read: split folders and image lists a `data.yaml` points to, folders of
  every candidate image and label, also skipped ones, and split paths or a `path:` that do not exist yet; the data dir's own `cache/`, `label/`, `bench/` and
  `datasets.json` / `datasets.removed.json` are left out, so a data dir inside a dataset folder does not invalidate it). On start the
  registry is listed immediately and every dataset is checked on a background thread: a fresh cache is loaded, a
  stale or missing one rebuilt; requests for a dataset that is still being indexed wait for it. While running,
  `GET /api/datasets/{id}` and image listings re-check the fingerprint at most every 5 s and re-index when files
  were added, removed or edited. A dataset whose folder no longer exists (moved, renamed, deleted) stays in the list
  with its last classes and splits; `GET /api/datasets/{id}` and its images answer `404 path_not_found` ("the folder
  of dataset '<name>' is gone: ...") until the folder is back (it is re-checked like any change), and `DELETE`
  unregisters it. `/eval` of a model trained on it answers the same; once the dataset is unregistered too, it answers
  `404 path_not_found` "the folder of dataset '<name>' this model trained on is gone: ..." while the run's recorded
  path does not exist.
- `cache/thumbs/<id>/`: thumbnails, keyed by the image's path, size, modification time and the requested side.
- `datasets/<name>/`: synthetic datasets and the datasets made from labeling jobs. `DELETE /api/datasets/{id}`
  removes the dataset's caches, and its files only with `?files=true`.
- `label/<job id>/`: one folder per labeling job: `job.json` (the job, its images and their boxes, rewritten
  atomically after every change; during an agent run at most every 5 seconds and when it ends, so a crash can
  lose the labels of the run's last few seconds), `images/` (uploaded files; folder imports stay where they are)
  and `log.jsonl`.
  Its thumbnails are in `cache/thumbs/<job id>/`. On start every job is listed again, `idle`: a run a previous
  process left active is closed as `stopped` (its finished images keep their labels), and files in `images/`
  that no image of the job refers to (an upload a crash interrupted) are removed. On shutdown the active runs'
  agent calls are killed.
- `bench/<bench id>/`: one folder per vision benchmark: `bench.json` (its definition: classes, source, seed,
  plan, and every image with its file name, size, SHA-256 and true boxes), `images/` (the copied files),
  `entries/<entry id>.json` (an entry, and once it is done its result and per-image outcomes) and
  `loops/<loop id>.json`, all written atomically. Variant thumbnails are in `cache/thumbs/<bench id>/`. On start
  every benchmark is listed again (one whose `bench.json` cannot be read is left out and logged, never
  overwritten; so is an unreadable entry or loop file; a folder without `bench.json`, a benchmark whose making a
  crash cut short, is removed and logged); entries left `queued` or `running` and loops left
  running become `failed` ("the server stopped ...", written back). On shutdown the same happens to the active
  ones: agent calls are killed, a loop stops the run it follows.

- `runs/<id>/`: one folder per training run: `run.json` (the `Run` plus the dataset reference, rewritten atomically
  on every status change and after every epoch), `log.jsonl` (every log line, as `/logs` returns them),
  `config.json` (the trainer's resolved config), `metrics.jsonl` (every trainer event, every step; `/metrics` reads
  it, and a resumed run's repeated steps replace the earlier ones), `weights/last.folo` and `weights/best.folo`
  (full checkpoints, docs/FORMAT.md) and `samples/epoch_NNN.json`. On start every run is listed again; runs a
  previous process left `queued`, `running` or `stopping` become `stopped` (with a `warn` log line; the ones that
  were not stopping get `error_code: "interrupted"`) and can be resumed with `resume_run_id`. On a graceful shutdown
  the active run stops after its current batch, and it and the queued runs end `stopped` with `interrupted` too.
- `models/`: `.folo` files listed as models (scanned on every listing; metadata read once per file version).
- `exports/`: exported models. `cache/eval/`: cached `/eval` results. `cache/weights/<model id>.<hash>.<variant>`:
  the exports of `/weights?format=…`: `f16.folo`, and `onnx` / `decode.onnx` / `decode-dynamic-half.onnx` / … (the
  ONNX switches that are on). The hash is of the source file's path, size and modification time; copies made from
  an older version of the file are removed when a new one is made, and a run's when the run is deleted.

- `cloud/connections.json`: the cloud connections with their credentials (created readable by the owner only; never
  served). `cloud/instances/<id>.json`: the ledger of launched instances, one file each, written before the
  provider is asked (docs/CLOUD.md, *Instances are never left behind*); records of instances that ended more than
  7 days ago are removed (not while the reaper still looks for an instance that was never found). A cloud run's folder also has `cloud.json` (its worker's address and token, the ids it
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
  truck / bus / person / bicycle, 1280×720, 1900 / 400 / 100, 3 broken labels), `ds_3b9d` **fasteners** (bolt /
  nut, 1920×1080, 700 / 160). Images are rendered procedurally on request; their boxes match the labels exactly.
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
  metric histories; and one live run (`run_3c02`, fasteners / folo-s) whose SSE stream advances in real time: a
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
