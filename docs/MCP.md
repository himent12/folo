# `folo mcp`: FOLO for agents

`folo mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio. With it, an agent
(Claude Code, Codex, any MCP client) can label images, build datasets, train, evaluate, run and export FOLO models without screen
scraping. It exposes the same operations as the CLI and `docs/API.md`, with the same error codes.

```jsonc
// e.g. .mcp.json / claude_desktop_config.json (data dir: ~/.folo; add "--data-dir", "<dir>" to the args for another)
{ "mcpServers": { "folo": { "command": "folo", "args": ["mcp"] } } }
```

`folo mcp [--data-dir DIR] [--mock] [--read-only] [--confine] [--server http://127.0.0.1:7878 [--token SECRET]]
[--device auto] [--tensor-cores]`.
`--device` (env `FOLO_DEVICE`: `auto`, `cpu`, `gpu:<n>`) is where the embedded real store runs `predict`,
`predict_batch` and `val`; each training run has its own `device`. `--tensor-cores` (env `FOLO_GPU_TENSOR_CORES=1`)
asks for the GPU's tensor cores in embedded mode, as `folo serve --tensor-cores` does: `train { amp: true }` needs
them (`folo_info`'s `capabilities.amp` says whether they are on). They have to be asked for before the GPU is
first used, so it is a start-up flag, not a runtime switch; with `--server`, the `folo serve` decides.
One process owns a data dir at a time (docs/CLI.md, `serve`): when a `folo serve` already owns `--data-dir` (FOLO
Desktop starts one), the embedded mode goes through that server, as `--server` would; with `--confine` it fails
with `data_dir_in_use` instead. Setup snippets for
Claude Code and Codex are in the README ([Use FOLO from an agent](../README.md#use-folo-from-an-agent)). The implementation is the
`folo-mcp` crate; the end-to-end tests are `crates/folo-cli/tests/mcp/`.

## Status
Implemented: both protocol revisions, all 40 tools with generated input and output schemas, every resource and
both subscription flows, the three prompts, paging and size limits, `--read-only`, `--confine`, and proxy mode. Every
tool goes through the [`Store`](../crates/folo-server/src/store.rs) behind the server, so what works depends on
it:

| | embedded real store (default) | `--mock` | proxy (`--server`) |
|---|---|---|---|
| datasets (register, synthetic, stats, issues, images) | real files under `--data-dir` | 3 fixed mock datasets, plus real folders / synthetic ones | the `folo serve`'s |
| `train` | trains on the FOLO engine, one run at a time (FIFO queue) | simulated, 6 s per epoch | `POST /api/runs` |
| `train` with `resume_run_id` | resumes from `weights/last.folo` (same id) | `not_implemented` | `POST /api/runs` with `resume_run_id` |
| `folo://system/throughput`, `folo://estimate/...` | the throughput measured by this data dir's runs and predictions, else docs/BENCHMARKS.md's reference | the mock's table (measured and reference rows) | `GET /api/system/throughput`, `POST /api/runs/estimate` |
| `run_status`, `list_runs`, `stop_run`, `reorder_run`, `compare_runs` | persisted runs (`runs/<id>/`), also those of earlier sessions | mock runs (API-created ones share one training slot) | `/api/runs/...` |
| `list_models` | `best` / `last` weights of finished runs, `.folo` files under `models/`, and published weights pulled into the `folo weights` cache (id = the name, e.g. `folo-n-coco`) | mock models | `/api/models` |
| `val` | folo-vision's evaluator on the requested split, `conf` / `iou` as given | derived from the model's metrics | `POST /api/val` |
| `predict`, `predict_batch` | the engine (on `--device`), predictors cached | deterministic mock boxes | `POST /api/predict` |
| `export` | `folo-f16` or `onnx` (f32, batch 1, the model's imgsz, raw head outputs; `decode`, `dynamic` and `half` switch those) to `exports/` | a `mock://` path (nothing is written) | `POST /api/models/{id}/export` |
| run subscriptions | fired by the run's live events | fired by the mock runs' events | polled every second |
| `folo://runs/{id}/samples/{epoch}` | the run's validation samples (first 16 val images) | rendered mock samples | `/api/runs/{id}/samples` |
| `cloud_*`, `train` with `compute` | the data dir's connections and instance ledger; cloud runs on their worker (docs/CLOUD.md) | the mock's RunPod connection `cc_4d2a`, simulated cloud runs | `/api/cloud/...`, `POST /api/runs` |
| labeling (`*_label_*`, `start_labeling`, `stop_labeling`, `label_status`) | the data dir's labeling jobs; runs call the `claude` / `codex` CLI of this machine (docs/LABELING.md) | `not_implemented` (the mock has no labeler) | `/api/label/...` |
| benchmarks (`bench_*`, `*_benchmark*`, `*_bench_entry`) | the data dir's benchmarks; a model entry runs in this process on `--device` and stops with the session | `not_implemented` (the mock has no benchmarks) | `/api/bench/...` (a server that lists `vision_bench`, else `not_implemented`); entries run in the server and outlive the session |

`get_dataset`'s `issues` and `list_dataset_issues` come from `Store::list_issues` (`/api/datasets/{id}/issues`
in proxy mode). Real mode has no `not_implemented` tool left. `export` takes the ONNX switches `half`, `dynamic` and
`decode`; a custom imgsz or a fixed batch above 1 are `folo export --format onnx` flags only (docs/CLI.md).

## Protocol
- **Revisions.** The server supports MCP **2026-07-28** and the previous stable revision **2025-11-25**. The two
  wire contracts differ, and both are implemented as the spec defines them:
  - *2026-07-28*: there is no connection-wide negotiation. Every request carries its protocol version and client
    capabilities in its metadata; `server/discover` is optional, and clients may call tools directly. Requests with an
    unsupported version get the spec's unsupported-version error. Resource subscriptions use `subscriptions/listen`
    (acknowledged, with update notifications correlated by subscription id).
  - *2025-11-25*: the `initialize` handshake fixes the revision for the connection; subscriptions use
    `resources/subscribe` / `resources/unsubscribe` with `notifications/resources/updated`.
  Built on the official Rust SDK, `rmcp` pinned to `=3.9.0` in the root Cargo.toml, which implements both
  revisions (the `initialize` handshake, `server/discover`, per-request `_meta` with
  `io.modelcontextprotocol/protocolVersion` and `.../clientCapabilities`, `subscriptions/listen`). Both pass the
  integration tests. A 2026-07-28 session starts with the first request that carries complete metadata; requests
  naming an unsupported version get error `-32022` with the supported versions in `data`. `initialize` with any
  other version is answered with 2025-11-25. Methods of the other revision (`resources/subscribe` on 2026-07-28,
  `subscriptions/listen` on 2025-11-25) are `-32601`.
- **Capabilities.** `tools` (with `listChanged: false`), `resources` (with subscriptions), `prompts`, and `logging`.
  `logging/setLevel` is accepted; the server currently sends no logging notifications (everything goes to stderr).
- **stdio framing.** UTF-8, one JSON-RPC message per line (newline-delimited, never pretty-printed), writes serialised
  through a single writer so concurrent responses and notifications never interleave. **stdout carries protocol messages
  only.** Background work (training after `train` has returned) logs to stderr and to the run's persisted log
  (`run_status.log`). MCP logging notifications are only used for messages scoped to an in-flight request, per the rules
  of the negotiated revision.
- **Progress.** Only when the request carries `_meta.progressToken`: the server echoes it in
  `notifications/progress`, with monotonically increasing `progress` (and `total` when known), at most ~4 per second,
  and stops when the request completes. `val`, `predict_batch` and `create_synthetic_dataset` report progress
  (`predict_batch` per image; the other two at the start and the end, since the store call is one step). `train`
  returns immediately, so training progress is observed through `run_status` or the run resource, not through progress
  notifications.

## Results and errors
Every tool declares an `inputSchema` and an `outputSchema`. The schemas are generated from the same Rust types as the
HTTP API (`schemars`), so `folo describe --json` prints identical schemas, and docs/API.md is the source of field
meanings. They are compacted for `tools/list` (about 91 KB for all 40 tools, 13.6 KB of it the ten labeling
tools and 10.3 KB the eight benchmark tools; a test keeps the whole under 92 KiB, the labeling tools under 14
KiB and the benchmark tools under 11 KiB): no `$schema`,
no numeric `format` annotations, no `minimum: 0` of unsigned types (also optional ones), defaults without f32 noise
(`0.01`, not `0.009999999776482582`); output schemas also leave out defaults and the field descriptions of the shared
types, in `$defs` and when the result is one (`Run`, `TrainConfig`, `Epoch`, `Dataset`, `CloudInstance`, ...;
docs/API.md documents them), and write string enums as `enum`. Every input schema,
nested objects included, has `additionalProperties: false`: **an argument the tool does not know is an
`invalid_arguments` error**, never silently ignored, and the message suggests the likely name
(``unknown argument `lr`; did you mean `lr0`?``; also `model` → `arch`, `augment.mosiac` → `mosaic`).
- **Success:** `structuredContent` validates against the tool's `outputSchema`, plus one short `text` content item
  summarising it (one or two lines, for transcripts).
- **Tool failure** (tool arguments that fail the `inputSchema` (missing or wrongly typed fields), bad values, missing
  dataset, GPU unavailable, ...): a normal result with `isError: true`, no
  `structuredContent`, and a single `text` item containing the JSON error object
  `{ "code": "dataset_not_found", "message": "...", "hint": "..." | null }`. Codes are one vocabulary with the
  CLI and docs/API.md (`folo_api::codes`; the same failure has the same code on every front end): store errors
  pass through unchanged, except the HTTP layer's names for bad input (`invalid_body`, `invalid_query`,
  `invalid_path`, `bad_request`), which are `invalid_arguments`. Added by MCP: `invalid_arguments` (arguments
  that fail the `inputSchema`, unknown ones included, or a documented range, e.g. a run name over 64 characters,
  as `POST /api/runs` refuses it, or other names over 128),
  `read_only`, `path_outside_data_dir`, `path_not_found`, `too_many_inputs`, `payload_too_large` (images over 64
  MiB), `result_too_large` (see below) and `server_unreachable` (proxy mode, `folo serve` not answering). A
  `TrainConfig` value the trainer cannot use (`arch: "folo-q"`, out of range) is `invalid_config` as in the CLI;
  an empty split for `val` is `invalid_dataset`; a `device: "gpu:7"` this machine lacks is `device_not_found`
  (the hint lists the devices it has); a GPU out of memory is `gpu_out_of_memory` (hint: lower `batch` or
  `imgsz`, or a smaller `arch`). Messages are cut to 1000 characters and hints to 500; engine errors (also a
  failed run's `error`) are already cut to 500, so a driver report never floods the context. The `*_not_found`
  codes of listed things hint at the list tool (`list_datasets`, `list_models`, `list_runs`, ...).
- **Protocol errors** (malformed JSON-RPC, a malformed `tools/call` envelope, unknown tool or method, unsupported
  protocol version): JSON-RPC errors per the spec, never `isError` results: unknown tool `-32602`, unknown method
  `-32601`, unsupported version `-32022`, a prompt without its required argument `-32602`. Well-formed JSON that is
  not a JSON-RPC message gets `-32600` (with `id: null`). A line that is not JSON at all is ignored and logged, as by
  every official MCP SDK (answering unparsable input can start an error loop between two peers).
- Resource reads for unknown URIs (or ids) return the spec's resource-not-found error: `-32002` on 2025-11-25,
  `-32602` on 2026-07-28.

## Paging and size limits
- Every list tool takes `{ cursor?: string, limit?: integer (default 50, max 200) }` (cursors are opaque; a bad
  one is an `invalid_arguments` error) and returns
  `{ items: [...], next_cursor: string | null, total: integer }`.
- List items are **summaries** (id, name, status and key numbers). Detail comes from the matching `get`/status tool
  or resource. Summaries are bounded: names are cut to 128 characters, at most 100 class names are listed (with
  `num_classes`), and run errors are cut to 500 characters, so a page always makes progress.
- No tool result exceeds ~32 KB of JSON, and every result still validates against its `outputSchema`. Limits are
  part of each schema: list pages shrink below `limit` to fit (the `next_cursor` continues from the last item
  returned), `predict` returns at most `max_det` boxes, `get_dataset` lists at most 50 issues (errors, then
  warnings; `issue_counts` and `issues_total` count them all, `list_dataset_issues` pages through them), and
  `predict_batch` always returns summaries plus a resource link. Bulk data with no bounded inline form (per-image
  batch predictions, full metric histories, sample grids) exists only as resources, and the outputs that refer to it
  carry a `details_uri` field plus a `resource_link` content item. Every result is measured before it is returned;
  one that still does not fit (say, `get_dataset` on a dataset with thousands of long class names) is the
  `result_too_large` tool error, whose hint names the resource to read instead. Text summaries are at most 300
  characters (except `view_label_image`, whose text lists the image's boxes up to 16,000 bytes).

## Tools
| tool | input | output (`structuredContent`) |
|---|---|---|
| `folo_info` | `{}` | `version`, `mode` (`embedded` / `mock` / `proxy`), `server`, `data_dir` (`/` separators; in proxy mode the server's, from `/api/system`), `read_only`, `confine`, `devices` (`cpu`, `gpus` with backend and memory; `memory_bytes` 0 when unknown), `capabilities` (what `device: auto` runs can use: `shader_f16`, `tensor_cores`, `amp` with the reason when unavailable; `/api/system`'s object, docs/API.md; `null` from an older server), `scales` with params and GFLOPs at 640 and 320 |
| `list_datasets` | paging | page of `{ id, name, format, classes, splits }` |
| `get_dataset` | `{ dataset_id }` | `Dataset` with `stats`, plus `issues`: errors, then warnings, each `{ kind, severity, file, line, message, fix }` (first 50; infos such as `empty_image` are only counted), `issue_counts: { error, warning, info, by_kind }` and `issues_total` |
| `list_dataset_issues` | `{ dataset_id, severity?: "error"\|"warning"\|"info", kind?: string }` + paging | page of `{ kind, severity, file, line, message, fix? }` in dataset order (the `folo dataset check --fix-dry-run` objects), plus `counts` of all issues |
| `list_dataset_images` | `{ dataset_id, split?: "train"\|"val"\|"test", class?: integer, include_boxes?: boolean (default false) }` + paging | page of `{ image_id, file, split, width, height, boxes? }` (`boxes` only with `include_boxes: true`) |
| `register_dataset` | `{ name?, path, format?: "auto"\|"yolo"\|"coco" }` | `Dataset`; a folder registered before (same canonical path) returns that dataset (the summary says "already registered") |
| `create_synthetic_dataset` | `{ name, images?: 1000 (max 100000), classes?: string[], imgsz?: 640, seed?: 0 }` | `Dataset` |
| `list_models` | paging | page of `{ id, name, arch, imgsz, num_classes, classes, metrics, head, trained_with, precision?, fused?, source?, about? }` (`head`, `trained_with`, `precision`, `fused`, `source` and `about` as on the API's Model; `precision` / `fused` are left out when unknown; `source: "published"` marks weights from `folo weights pull`, whose id is the published name) |
| `train` | `TrainConfig` (docs/API.md; `dataset_id` required), or `{ resume_run_id }` (other fields optional and must match the run; `device` and `compute` may change, and an empty `name` counts as left out). `weights` takes any model id, including a published name (`"folo-n-coco"`, once `folo weights pull folo-n-coco` has run on that machine); the run trains the weights' architecture, and an explicit `arch` naming another one is `invalid_config`. `amp: true` (mixed precision) needs GPU tensor cores: `folo mcp --tensor-cores` (or `FOLO_GPU_TENSOR_CORES=1` in the MCP client's `env`), `--tensor-cores` for the `folo serve` it proxies. Without them `train` answers `not_supported` at once (nothing is started), with a hint for the client: `amp: false`, or how to restart this server with tensor cores. `head: "o2o"` trains the NMS-free one-to-one head too (docs/MODEL.md §6.6); `val`, `predict` and `export` then use it (no NMS) because the model says so, no extra argument | `{ run_id, status }` |
| `run_status` | `{ run_id, log_lines?: 20 (max 200) }` | `Run`, plus `last_step`, `last_epoch` and `log` (the last `log_lines` log lines; older ones are dropped first if the result would exceed 32 KB) |
| `list_runs` | `{ status?: string }` + paging | page of `Run` summaries |
| `stop_run` | `{ run_id }` | `Run` |
| `reorder_run` | `{ run_id, action?: "top"\|"up"\|"down"\|"bottom"\|"start", position?: integer (1-based) }`, exactly one of `action` and `position` | `{ run_id, status, queue_position }` (the new place; `null` once it started). Moves a run waiting in the training queue (`POST /api/runs/{id}/queue`, docs/API.md: `start` starts it now when nothing trains, else next; the active run is never preempted). A run that is not waiting (starting, running, finished) is `run_not_queued`; neither or both arguments `invalid_arguments` |
| `val` | `{ model_id, dataset_id, split?: "val", imgsz?, conf?: 0.001, iou?: 0.7 }` | the `POST /api/val` report: `{ model_id, dataset_id, split, imgsz, images, val: Epoch.val, timing: { total_s, per_image_ms } }` |
| `predict` | `{ model_id, image_path \| image_base64 \| (dataset_id + image_id), conf?: 0.25, iou?: 0.7, max_det?: 300 }` (flat fields; exactly one source) | the `/api/predict` response |
| `predict_batch` | `{ model_id, paths: string[] \| glob: string, conf?, iou? }`, at most 256 images **after** glob expansion (more is a `too_many_inputs` error) | `{ batch_id, processed, failed: [{ path, code, message }] (first 50), failed_total, boxes_total, per_class_counts (by class name), timing: { total_ms, mean_ms }, details_uri }` plus a `resource_link` to `folo://predictions/{batch_id}` with the per-image boxes |
| `export` | `{ model_id, format: "folo-f16"\|"onnx", decode?, dynamic?, half? }` (the switches: ONNX only; in proxy mode the server must list `export_onnx_switches` in `/api/system` `features`) | `{ path, bytes }` |
| `compare_runs` | `{ run_ids: string[] (2–8) }` | `runs`: per run `run_id, name, status, arch, epochs_done, epochs, best, best_metrics, final_train_loss, duration_s`; `config_diff`: dotted TrainConfig paths that differ, each with one value per run |
| `cloud_connections` | `{}` | `{ connections: [CloudConnection] }` (docs/API.md, *Cloud compute*): provider, account, balance and the last check, secrets only as a hint. No tool makes a connection or takes credentials: that is Studio's, Desktop's or `folo cloud connect`'s (docs/CLI.md, Cloud) |
| `cloud_offers` | `{ connection_id, refresh?: false }` | `{ offers: [CloudOffer], fetched }`, cheapest first; a run trains on one with `train { compute: { connection_id, offer_id } }` |
| `cloud_instances` | `{}` | `{ instances: [CloudInstance] }`: the instances FOLO launched, alive first, with their run, status and cost so far |
| `cloud_terminate` | `{ instance_id }` | `CloudInstance` (`terminated`, or `terminating` with the error FOLO retries): terminates the instance now; a run still using it fails with `cloud_instance_lost` |

`train`'s `compute` (TrainConfig, docs/API.md *Cloud runs*) trains the run on a cloud connection; `run_status` then
has the run's `cloud` (phase, instance, cost).

### Labeling
Ten tools take unlabeled images to a dataset with a Claude Code or Codex agent (docs/LABELING.md), and let the
calling agent check and correct the boxes itself; the `label_folder` prompt walks through them. A box is
`[class, x1, y1, x2, y2]`: the class index into the job's `classes`, the corners in **pixels of a view of the
image**: the image scaled so its long side is the view's `size` (default 1024, never upscaled), the frame vision
models answer in when they look at it. Every result that lists boxes states that frame as `size: [width, height]`
(a 1920×1080 photo's default view is `[1024, 576]`), and `set_label_boxes` takes boxes with the `size` of the view
they were read on (required with coordinates: boxes of a `size: 512` view read as 1024 ones would land at half
scale). The server stores pixels of the original; a box sent back as a view shows it keeps its stored pixels
exactly.

Each image's boxes have a **revision** (`"<count>-<hash>"`, `label_box_revision` in folo-api), which every view and
`list_label_images` with `include_boxes` return. Every `set_label_boxes` call needs it as `expect` (box numbers
are that revision's, and a full `boxes` list replaces what that revision had, including boxes a view listed past
its first 300), and `review_label_images` takes `expect: { image_id: revision }` (a key not in `image_ids` is
`invalid_arguments`): when the boxes changed since (a run
relabeled the image, someone saved it), the edit or mark is refused with `409 label_boxes_changed` instead of
applying to other boxes. A server with the `label_revisions` feature checks it under the lock it saves under; an
older one gets a check made just before the request (a change in between is not caught, and a review there is one
request per image).

Results carry ids and counts, not records; no tool returns image bytes except `view_label_image`. Typical sizes
(`structuredContent`): `create_label_job` 60 bytes, `label_status` 600 with 5 log lines, `list_label_images` 110
per image (170 with `include_boxes`), `set_label_boxes` 200, `review_label_images` 30. The labeling tools' output
schemas carry no descriptions or `required` lists (this table documents them), and their input schemas leave out
empty defaults and `null` in optional types.

| tool | input | output (`structuredContent`) |
|---|---|---|
| `list_label_jobs` | paging | page of `{ id, name, status: "idle"\|"running"\|"stopping", classes, counts: { images, pending, labeled, failed, boxes }, dataset_id }`, newest first |
| `create_label_job` | `{ name, classes: string[] (1–80), instructions?: "", paths?: string[] (max 100) }`, or `{ job_id, paths }` to add images to a job | `{ job_id, images, added, skipped, skipped_first? }` (`skipped_first`: the first 5 files left out, with the reason). `paths` are folders (searched recursively) or image files, read in place. A path that fails before any image was added leaves no job behind; a later one fails with a hint naming the job to add the rest to |
| `start_labeling` | `{ job_id, agent: "claude"\|"codex", model?, effort?: "low"\|"medium"\|"high", fast?: false, scope?: "unlabeled"\|"all", image_ids?: string[], note?: "", batch?: 1–16, parallel?: 1–8 }` | `{ job_id, total, model }` (`model`: `null` for the CLI's default); returns at once. `image_ids` relabels exactly those images (hand edits too), e.g. a sample first, or "ask again" with a `note` saying what was wrong. An agent whose CLI is not installed is `not_supported`, with how to install it in the hint (`folo label agents` lists them). `fast` asks for Codex's Fast tier (more of the plan's usage), only with a model that offers it; through a server that does not list it for that model (one from before the Fast tier, which would ignore it) it is refused before the start (`invalid_arguments`) |
| `stop_labeling` | `{ job_id }` | `{ job_id, status }`; labeled images stay |
| `label_status` | `{ job_id, log_lines?: 5 (max 50) }` | `{ id, name, status, classes, counts, reviewed, run: { agent, model, effort, total, done, failed, outcome }, calls, input_tokens, cached_input_tokens, output_tokens, cost_usd, dataset_id, log }` (the tokens of every run of the job; `log`: the last lines, `warn: ` / `error: ` before those levels, where a failed run's error is) |
| `list_label_images` | `{ job_id, status?: "pending"\|"labeling"\|"labeled"\|"failed", reviewed?: boolean, include_boxes?: false }` + paging (limit up to 200) | page of `{ id, file, status, num_boxes, source: "agent"\|"manual"\|null, reviewed, error?, size?, revision?, boxes? }` in the order the images were added; `include_boxes` adds the default view's `size`, the `revision` and the `boxes` in that view |
| `view_label_image` | `{ job_id, image_id?, size?: 1024 (256–1568), from?: 1 }` | `{ job_id, classes, to_review, image: { id, file, size, status, source, reviewed, error?, revision, num_boxes, from?, boxes } }` plus an **`image` content item** (`image/jpeg`): the view (`size`), every box outlined in its class colour with its number (1, 2, ...) in a tag, and pixel ticks on all four edges. The `text` item states the frame and the revision, lists the boxes (`2 person 120,80,340,900`; class indices instead of names on every line when any class name is longer than 12 bytes) and the classes. An image with more than 300 boxes lists 300 from number `from` and says how to list the next; the text stays under 16,000 bytes by listing fewer boxes, never by losing that line or the classes. A `from` past the last box lists none and says to start again at 1 (the boxes changed since that paging started). Without `image_id` it shows the next image to review (labeled, not reviewed); `image: null` when none is left |
| `set_label_boxes` | `{ job_id, image_id, expect, size: [w, h], boxes: Box[] }` or `{ job_id, image_id, expect, size?, relabel?: [number, class][], remove?: number[], add?: Box[] }` (numbers as the view shows them) | the image as in a view of the edit's frame. `size` is the view's `size` as it returned it, required with `boxes` (other than `[]`) and `add`; one that is not the size of a view of the image is `invalid_arguments`. Saved as a hand edit (`source: "manual"`) and **reviewed**, like Studio's Save. Boxes the edits leave alone keep their exact pixels. `boxes: []` makes it a background image. A `boxes` list that names a look at least once but fewer times than the image has boxes that differ but look that way in that frame is `invalid_arguments` (which to keep is unknown: remove them by number, or send a larger view's boxes). Both forms at once, no `expect`, coordinates without `size`, a number the image does not have, a box relabeled twice or relabeled and removed, a class out of range, or a box outside the frame or less than a pixel of the image is `invalid_arguments`, named by its place (`box 2`, `add 1`, `relabel 3`) |
| `review_label_images` | `{ job_id, image_ids: string[] (1–500), reviewed?: true, expect?: { id: revision } }` | `{ updated, failed?: [{ id, code, message }] (first 20), failed_total?, to_review }`: Studio's "Looks right" (the boxes stay as they are, and stay the agent's); `reviewed: false` takes the mark back. One save on a server with `label_revisions`. An image without labels cannot be marked. `to_review` is `null` when it could not be counted after the marks were saved |
| `export_label_dataset` | `{ job_id, name?, val_fraction?: 0.2 (0–0.5), seed?: 0 }` | `{ dataset_id, name, path, train, val }`: the labeled images as a registered YOLO dataset, ready for `train` |

A view of a 1920×1080 photo at the default size is a 1024×576 JPEG, which a Claude model reads as about 800 input
tokens; `size: 512` quarters that for a quick look (corrections then pass `size` as that view returned it). The usual loop settles each image before the next view (the
next view shows the same image until it is): `view_label_image` (no `image_id`), then either `review_label_images`
with that id and `expect: { id: revision }`, or `set_label_boxes` with `expect` and the fixes; again until `image`
is `null`. Errors carry hints that name these tools (`label_job_active`: wait or `stop_labeling`;
`label_boxes_changed`: view the image again).

### Benchmarks
Eight tools make and run vision benchmarks (docs/BENCH.md defines every number): frozen sets of labeled images with
a plan of stress conditions, and the entries that ran them. They share their operations with `folo bench`
(docs/CLI.md, Vision benchmarks; `folo_mcp::bench`). A contestant is a model of `list_models` (trained, under
`models/`, or published weights pulled with `folo weights pull`) or a file of predictions made by any other
detector. Agent contestants and agent-designed plans are not tools: they spend tokens, and are Studio's, Desktop's
or the HTTP API's (`/api/bench`, docs/API.md).

Results are small: counts, the plan, and an entry's score with the numbers behind it, never image lists, bins,
curves or histograms (those are on the HTTP API). Typical sizes (`structuredContent`): `bench_catalog` 2.8 KB,
`get_benchmark` 1.7 KB for 80 classes and three entries (about 150 bytes an entry), `start_bench_entry` 120 bytes,
`bench_entry` 150 while it runs and 1 KB once done with two conditions. Their output schemas carry no descriptions or `required` lists, and a benchmark's `source`
and an entry's `result` are plain objects there (this table documents them).

| tool | input | output (`structuredContent`) |
|---|---|---|
| `bench_catalog` | `{}` | `{ conditions: [{ id, group, description }], severities: 5, default_plan: [{ condition, severities, weight }], axes: [{ axis, weight }] }` |
| `list_benchmarks` | paging | page of `{ id, name, source_name, images, num_classes, conditions, entries, best: { entry_id, name, score, partial }?, active }`, newest first |
| `get_benchmark` | `{ benchmark_id }` | `{ id, name, description, source, source_name, classes, images, boxes, plan, seed, fingerprint, active, leaderboard }`: `leaderboard` is the entries best first (the first 50). A result that would pass the size limit has its description cut first, then its names (its own, its classes', its entries'), then rows dropped from the end, and the summary says so (`GET /api/bench/benchmarks/{id}` and `/entries` have them whole). Each row is `{ id, name, kind: "model"\|"agent"\|"external", status, score, partial, map50_95, ci95, stale }` (`stale`: graded on an earlier version of the benchmark) |
| `create_benchmark` | `{ name, dataset_id + split?: "val" \| label_job_id, description?, max_images?: 500 (1–5000), seed?: 0, plan?: { conditions: [{ condition, severities, weight? }] } }` | the benchmark as `list_benchmarks` lists it (`get_benchmark` has the rest). The images are sampled with `seed` and frozen; the plan defaults to `bench_catalog`'s `default_plan`, and one naming an unknown condition or a severity outside 1–5 is `invalid_arguments` |
| `delete_benchmark` | `{ benchmark_id }` | `{ benchmark_id, deleted: true }`. Removes the benchmark's folder for good: its frozen images and its entries' results (a running entry is `bench_active`). The one tool with `destructiveHint: true` |
| `start_bench_entry` | `{ benchmark_id, model_id, name?, max_images?, conditions?: true }` or `{ benchmark_id, predictions_file, name, scored?: true }` | `{ benchmark_id, entry_id, name, status, done, total, score? }` (`score` once done; `bench_entry` has the rest). A model entry runs in the background on the server's device (`--device`), the unchanged images then each condition and severity of the plan; follow it with `bench_entry`. A predictions file is read by this `folo mcp` process, on its machine, also in proxy mode (a path absolute or relative to `--data-dir`); it is sent to the server's benchmark in parts: JSON Lines of `{ image, condition?, severity?, boxes: [{ class, score, x1, y1, x2, y2 }] }` records, `examples/bench_external.py`'s format, one record per line of at most 1 MiB and 1,000 boxes, pixel corners in the served image, `class` the index of one of the benchmark's `classes` (a number, not a name). It is scored at once (the answer is `done` with its score): every image of each pass it has must be answered (an empty `boxes` for none), a pass of the plan it leaves out is not measured (the score is `partial`). After a failed part, or a finish the server refuses with a 4xx other than `409`, the tool reads the entry and, if the server reports it still open, tries to stop and remove it. After a finish with no answer, a 5xx or a `409` it stops and removes nothing (the entry may be being scored): it reads the entry once and returns the score if that read succeeds and says scored. In every other case the error is returned and an entry may be left on the server: `get_benchmark` lists it, `bench_entry` shows it and `stop_bench_entry` ends it. `scored: false` for a detector that gives no confidences |
| `bench_entry` | `{ benchmark_id, entry_id, wait?: 0 (0–50) }` | `{ benchmark_id, entry_id, name, status, done, total, condition?, severity?, images, conditions, missing_classes?, error?, result? }`. `wait` holds the call until the entry ends (or the seconds pass); through a server that stops answering, the call ends 5 s after `wait` with `server_unreachable`. A state that would pass the size limit has its names, error and hint cut, and the summary says so. `result` once it is done: `{ score (0–100), partial, axes: [{ axis, value, weight }] (accuracy, robustness, self_labeling, calibration, localization; value `null` when not measured), accuracy: { map50_95, ci95, map50, map75, precision, recall, best_conf }, conditions: [{ condition, weight, retention, points: [severity, map50_95, retention][] }], ece, mean_iou, self_labeling: { threshold, precision, recall, readiness }, ms_per_image? }` |
| `stop_bench_entry` | `{ benchmark_id, entry_id }` | as `start_bench_entry` (`stopping`, then `stopped`); a finished one is `bench_entry_not_active` |

A model entry belongs to the process that runs it: in embedded mode this `folo mcp` (the entry stops when the
session ends), through a `folo serve` the server (it goes on after the session, and a later session follows it by
id). The usual loop: `bench_catalog` once, `create_benchmark`, `start_bench_entry`, then `bench_entry` with `wait:
50` until `result` is there, then `get_benchmark` for the leaderboard.

Read-only mode (`--read-only`) exposes only `folo_info`, `list_*`,
`get_dataset`, `run_status`, `val`, `predict`, `predict_batch`, `compare_runs`, `cloud_connections`, `cloud_offers`,
`cloud_instances`, `label_status`, `view_label_image`, `bench_catalog`, `get_benchmark` and `bench_entry` (`list_*`
includes `list_label_jobs`, `list_label_images` and `list_benchmarks`). Mutating tools are not listed, and they are **also rejected at dispatch** (`read_only` tool
error) if called anyway.

## Resources
Each resource read returns a JSON snapshot, unless stated otherwise.
- `folo://datasets/{id}`: dataset with stats.
- `folo://datasets/{id}/images/{image_id}{?size}`: the image as a blob, so a multimodal agent can look at it. By
  default a JPEG whose long side is at most 512 px (EXIF orientation applied, never upscaled; a read of a 1920×1080 frame is
  about 55 KB instead of 430 KB); `?size=<px>` picks another long side (clamped to 16..2048), and `?size=original`
  returns the file itself (`image/jpeg` / `image/png`). Box coordinates (`list_dataset_images`) are in original
  pixels, so scale them by the thumbnail's size. Any other query is an invalid URI.
- `folo://runs/{id}`: `{ run: Run, last_step, last_epoch, revision }`. `revision` increases on every change.
  **Subscribable**: the server sends the spec's resource-updated notification for this URI (coalesced, at most one
  per second), and the client re-reads the snapshot. The server follows the run's live events (the store's
  per-run event stream, the same one behind SSE `/events`): a step, an epoch, a status change or a log line
  wakes it, and it checks the snapshot at most once per second (it re-reads runs with events every 10 s as well,
  which is how a deleted run is noticed). Runs without an event stream (proxy mode) are polled every second. On
  2026-07-28, `subscriptions/listen` is acknowledged with the accepted filter (URIs that are not subscribable are
  left out) and each update carries the subscription id in `_meta`; cancelling the listen request ends it. On
  2025-11-25, subscribing to an unknown run is a resource-not-found error. Missed intermediate steps are not an error: epoch history is in
  the metrics resource.
- `folo://runs/{id}/metrics`: the full history (docs/API.md `/metrics`), every step included (about 150 KB for
  12 epochs of 50 batches). Subscribable, and updated after each epoch (and on status changes).
- `folo://runs/{id}/epochs`: `{ epochs: [Epoch], last_step }`, the epoch history without per-step records (about
  1 KB per epoch); what an agent wants for the mAP and loss curves. Subscribable like `/metrics`.
- `folo://runs/{id}/samples/{epoch}{?tile}`: validation samples rendered as an annotated JPEG grid (truth outlined
  white, predictions in class colours), for quick visual checks: 4 cells per row of 160 px (640 px across for the 16
  samples; about 110 KB for 1080p frames, against 250 KB at the old 256 px); `?tile=<px>` picks the cell size
  (clamped to 64..512).
- `folo://predictions/{batch_id}`: per-image boxes of a `predict_batch` call (kept for the server's lifetime).
- `folo://system/throughput`: the measured-speed table of `GET /api/system/throughput` (docs/API.md, *Throughput
  and estimates*): training `imgs_per_s` per (device, arch, imgsz, batch, amp) with mosaic / close-mosaic /
  validation rates, startup and `peak_vram_bytes`, and predict latency per (device, arch, imgsz, precision); every
  row says `source` (`measured` on this server, or a docs/BENCHMARKS.md `reference`), `measured_on`, `samples` and
  `updated`.
- `folo://estimate/{dataset_id}{?arch,epochs,batch,imgsz,device,amp}`: how long a training run would take, before
  starting it: `POST /api/runs/estimate` for that `TrainConfig` (fields left out take their defaults; any other query
  parameter, a repeated one or a value of the wrong type is an invalid URI). The `RunEstimate` has `est_epoch_s`,
  `est_total_s` with `est_total_s_low` / `est_total_s_high`, `est_peak_vram_bytes`, the mosaic and close-mosaic
  `phases`, and where the numbers come from: `source`, `confidence` and `basis`. A resource rather than a tool: a
  tool taking a whole `TrainConfig` would add about 7 KB to every session's `tools/list`.
- `folo://docs/{name}`: repository docs (`ARCHITECTURE`, `API`, `CLI`, `MODEL`, `FORMAT`, `METRICS`, `MCP`, `AGENT_COOKBOOK`) as text/markdown
  (compiled into the binary).

`resources/list` lists datasets, runs (with their metrics and epochs), the throughput table and docs, 100 per page; images,
samples, predictions and estimates are reached through `resources/templates/list`.

## Prompts
Prompt retrieval (`prompts/get`) **only returns messages**. It never starts jobs or touches files; the agent then
calls tools itself.
- `train_on_folder` (arguments: `folder_path` required; `goal` optional, free text, e.g. "fast on CPU"): returns
  instructions to call `register_dataset`, inspect `get_dataset` stats and issues, choose a scale and image size from
  the stats and `folo_info` devices, call `train`, and explain the choices to the user.
- `label_folder` (arguments: `folder_path` and `classes` (comma-separated) required; `instructions` optional):
  returns the labeling walk-through: `create_label_job` on the folder, label a sample of about 20 images first
  (`start_labeling` with `image_ids` and agent `claude`, else `codex`) and follow it with `label_status`, review the
  sample one image at a time (`view_label_image`, then `review_label_images` or `set_label_boxes` with the
  revision, before the next view), ask again with a `note` when the agent got something wrong systematically,
  label the rest, review, `export_label_dataset`, then offer `train`.
- `diagnose_run` (arguments: `run_id` required): returns the run's metric summary and last log lines embedded in the
  message, plus instructions to identify over/underfitting, class imbalance or LR problems and propose a concrete
  TrainConfig diff.
In read-only mode, `train_on_folder` and `label_folder` are not listed.

## Files and safety
- No tool overwrites existing files, and only `delete_benchmark` deletes any (a benchmark's own folder under
  `--data-dir`). Generated outputs (synthetic datasets, run directories, exports,
  prediction dumps) go to new, collision-free paths under `--data-dir` (`<name>`, then `<name>-2`, ... if taken).
  Writes go to a temp file first and are then renamed into place.
- Input paths (`register_dataset.path`, `predict.image_path`, `predict_batch.paths/glob`, the `paths` of
  `create_label_job`, `start_bench_entry.predictions_file`) may be absolute or relative
  to `--data-dir`, and are resolved on the machine running `folo mcp` (in proxy mode the label paths are the
  server's: an absolute path, of the server's OS, is passed on as given, and a relative one is
  `invalid_arguments`). `--data-dir` is where FOLO **writes**; it is not
  a read sandbox. `--confine` restricts reads to `--data-dir` as well, using canonicalised paths with symlinks and
  junctions resolved, and rejects anything outside it with `path_outside_data_dir`. Datasets are confined where they are read:
  a dataset root inside the data dir can still reference files elsewhere (a `data.yaml` `path:`, image lists, linked
  folders, YOLO labels derived from image paths), so `folo mcp --confine` installs folo-data's process-wide read
  guard (`folo_data::guard`). Every manifest, image list, split folder, COCO JSON, image and label the dataset
  reader opens, and every image the server sends, is resolved canonically first and refused outside the data dir,
  whatever the manifest syntax, at registration and at every later re-index (with the guard on, the real store
  rebuilds every index at startup instead of trusting caches built without it; refused folders are not descended). A refused read fails like an
  unreadable file and shows up as a dataset issue or error that names it. On top of that, before any tool or
  resource reads a dataset, its root, every image the store reports, each image's label, and its issues are
  checked, so the agent gets a clear `path_outside_data_dir` error (results cached per dataset; the guard keeps
  enforcing if the dataset changes later). `register_dataset` unregisters a dataset that fails the check, and
  datasets registered earlier without `--confine` that fail it are left out of `list_datasets` and
  `resources/list` and rejected everywhere else. Labeling jobs are checked the same way: label paths outside the
  data dir are refused when they are added, and a job (made earlier without `--confine`) that has images outside it
  is left out of `list_label_jobs` and refused by every other labeling tool (`path_outside_data_dir`; results cached
  per job until its image files change). In proxy mode `folo serve` reads the datasets and the label images, so
  `--confine` covers only the files the MCP process reads itself (`predict.image_path`, `predict_batch`); the
  labeling tools that would have the server read image files (`create_label_job` with paths, `start_labeling`,
  `view_label_image`, `export_label_dataset`) are refused there with `not_supported` (run `folo mcp --confine` on
  the data dir itself to label confined); a job without paths reads nothing and can be made there. All eight
  benchmark tools are refused with `not_supported` under `--confine`, embedded and through a server (with
  `--read-only` too, the mutating ones answer `read_only` first): the store
  reads, writes and deletes a benchmark's files (frozen images, entries, a model entry's weights) without these
  checks. Run `folo mcp` without `--confine` to use them. The checks guard
  what is reachable through FOLO's tools; they do not defend against another process re-pointing links inside the
  data dir between a check and the open (FOLO never creates links).

## Run lifetime
- **Progress fields.** `run_status.epoch` is the epoch in progress (1-based), so `epoch: 2` while running means one
  epoch has finished; `last_epoch.epoch` is the last validated one.
- **Embedded mode** (default): runs execute inside the `folo mcp` process. Run state is persisted under
  `--data-dir/runs/<id>/` (config, metrics, checkpoints at every epoch). When the MCP client disconnects (stdin
  closes) or the process receives a shutdown signal, the runs this session started are stopped (`stop_run` on
  each; not cloud runs, which train on their worker without this process, and which the next `folo mcp` or `folo
  serve` on the data dir follows again, docs/CLOUD.md), then the store is shut down (`Store::shutdown`: queued runs become `stopped`, and the process waits, at
  most a minute, for the active run to stop after its current batch), then the process exits. A `train` still
  in flight at that point fails with `shutting_down` instead of queueing a run nobody would train. The current epoch
  is abandoned, the last epoch's checkpoint is kept, and the status becomes `stopped`. In proxy mode runs belong
  to `folo serve` and are left running. On the next start, persisted runs are listed again (a run a crashed
  process left active is `stopped`, with a warning in its log). `train` with `resume_run_id` resumes that
  `stopped` or `failed` run (same id, status `queued` again) from its `last` checkpoint, with the full state of
  docs/FORMAT.md (epoch, optimizer, scheduler, EMA, RNG), or from scratch when no epoch had finished. The other
  TrainConfig fields must match the run's config or be omitted (`dataset_id` too; `device` and `compute` may change); a field
  that differs is `invalid_config`, a completed run `run_not_resumable` (start a new run from its model instead),
  an active one `run_active`. `train` with
  `weights: <model_id>` instead starts a **new** run initialised from those weights (no optimizer state). Cancelling the `train` request itself (MCP cancellation) after it has returned
  has no effect on the run. Use `stop_run`.
- **Proxy mode** (`folo mcp --server http://127.0.0.1:7878`): every tool maps to the HTTP API of a running
  `folo serve`, so Studio and the agent share runs that outlive the MCP process:
  `list_*`/`get_dataset` → `GET` list/detail endpoints (paging is applied by the MCP layer), `list_dataset_images` →
  `GET /api/datasets/{id}/images`, `train` → `POST /api/runs` (with `resume_run_id` when given), `run_status` → `GET /api/runs/{id}`, `stop_run` → `POST /api/runs/{id}/stop`, `reorder_run` → `POST /api/runs/{id}/queue`, `predict` → `POST /api/predict` (local `image_path` files are
  read by the MCP process and uploaded as multipart), `export` → `POST /api/models/{id}/export`. `run_status`
  takes its `log` from `GET /api/runs/{id}/logs` and its `last_step` / `last_epoch` from `/metrics`, `val` →
  `POST /api/val`, `predict_batch` → one `POST /api/predict` upload per image (until the planned
  `POST /api/predict/batch` exists), `compare_runs` → `GET /api/runs/{id}` + `/metrics` per run, the `cloud_*`
  tools → `/api/cloud/*`, the labeling tools → `/api/label/*` (`view_label_image` draws on the server's
  `?thumb=` image; the `reviewed` filter and the lookups of one image use `GET /api/label/jobs/{id}/images` with
  `reviewed` / `ids`; a server without the `label_image_filters` feature has no lookup of one image, so its images
  are read page by page (1000 a request) and filtered here, stopping once the ids asked for are found: a lookup
  or a count then costs one request per 1000 images of the job; a server without `label_revisions` gets
  revisions checked here and one `PUT` per reviewed image; label paths name files on the server's machine), dataset
  images and sample grids → `GET /api/datasets/{id}/images/{image_id}`. Run resources are polled the same way as in
  embedded mode, so subscriptions work through the proxy too. Every tool works in proxy mode; errors from the
  server keep their codes and statuses (an answer that is not FOLO's, such as a proxy's error page, is
  `bad_gateway`). A server that does not answer is the `server_unreachable` tool error. The URL may be `http://` or
  `https://` (a server behind a TLS proxy or tunnel). A server started with `--token` needs `--token` here too (env
  `FOLO_TOKEN`), sent as `Authorization: Bearer` on every request and never logged; without it every tool is the
  `unauthorized` tool error. The transport is the `folo-client` crate, the same one a cloud control plane uses.

## Example agent session
```
→ create_synthetic_dataset { "name": "shapes", "images": 2000 }      ← { "id": "ds_1", ... }
→ train { "name": "shapes-n", "dataset_id": "ds_1", "arch": "folo-n", "epochs": 30, "imgsz": 320 }
                                                                      ← { "run_id": "run_1", "status": "queued" }
→ (subscribe folo://runs/run_1, or call run_status every ~30 s)       ← { "epoch": 12, "best": { "map50_95": 0.71 }, ... }
→ predict { "model_id": "m_1", "image_path": "test.jpg" }             ← { "boxes": [...], "timing": {...} }
```

## Tests
`crates/folo-cli/tests/mcp/` spawns `folo mcp` and, for **each supported revision**, runs the handshake or discovery (plus,
for 2026-07-28, a direct tool call without discovery and an unsupported-version request), both subscription flows, `tools/list`
(checking the schemas, their size and `additionalProperties: false`), `resources/list`, `prompts/list`, a
`create_synthetic_dataset` → `train` → `run_status` flow and a subscription to the scripted live run on the mock
store, `predict` / `val` / `export` with a seeded model, issues errors-first and paged, a repeated `register_dataset`, unknown and misspelled arguments, the
shared error codes, the `/epochs` resource, the cloud tools on the mock's connection and seeded cloud run, an
`isError` case, a JSON-RPC error case, and a read-only-mode dispatch rejection. It asserts that nothing but valid JSON-RPC lines appears on stdout. Every
structured tool result is validated against the tool's `outputSchema`. `real_store_end_to_end` drives the **real**
store (`--device cpu`, 64 px, a few seconds): a synthetic dataset, a 1-epoch run trained until it completes,
`list_models`, `val` on two splits, `predict` on a dataset image and the same file by path (boxes returned,
identical), the sample grid resource, `export`, then a second, 200-epoch run followed through its resource
subscription while it trains (a subscription sees only later changes, so it follows a run that keeps changing),
stopped after its first epoch, resumed (same id) and left training when the client disconnects; a new session finds it
`stopped` by the graceful shutdown. Further tests cover `--confine` (with `FOLO_DATASET_RECHECK_MS=0`, test
support: the server looks at a dataset's files again on every request instead of after 5 s, so the test can edit a
registered dataset to read outside the data dir and ask again at once) and proxy mode against `folo serve --mock`
(also with a token); folo-mcp's own tests cover an unreachable server. The labeling tools are tested in
`crates/folo-mcp/src/tools/label_tests.rs` with a scripted agent (no agent CLI): the whole flow in process and
through a loopback `folo serve` (also an older one without `label_image_filters` and `label_revisions`, read two
images a request so the fallback crosses pages), the `image/jpeg` content of a view, box edits in pixels and their
errors, revisions against a run (a gated agent holds an image in its call while an edit is tried, then changes
its boxes), `--confine` (in process and through a server) and a job that a missing path leaves behind.
The benchmark tools are tested in `crates/folo-mcp/src/tools/bench_tests.rs`, in process and through a loopback
`folo serve`: a benchmark of a synthetic dataset, a file of predictions (the truth) scored at once, the refusals
and their hints, and the mock without benchmarks; `crates/folo-cli/tests/mcp/bench.rs` enters a published model
over stdio and follows it with `bench_entry`'s `wait` to its score.
