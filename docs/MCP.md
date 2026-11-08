# `folo mcp`: FOLO for agents

`folo mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio. With it, an agent
(Claude Code, Codex, any MCP client) can build datasets, train, evaluate, run and export FOLO models without screen
scraping. It exposes the same operations as the CLI and `docs/API.md`, with the same error codes.

```jsonc
// e.g. .mcp.json / claude_desktop_config.json
{ "mcpServers": { "folo": { "command": "folo", "args": ["mcp", "--data-dir", "D:/folo-data"] } } }
```

## Protocol
- **Revisions.** The server supports MCP revision **2026-07-28** and the previous stable revision **2025-11-25**, and
  negotiates per the spec: the client's discovery or handshake (`server/discover` in 2026-07-28, `initialize` in
  2025-11-25) selects the revision. Use the official Rust SDK (`rmcp`), pinned in Cargo.toml. If the SDK lacks one of
  these revisions, implement the one it supports and document that here. Never claim a revision the server does not pass
  the tests for.
- **Capabilities.** `tools` (with `listChanged: false`), `resources` (with subscriptions), `prompts`, and `logging`.
- **stdio framing.** UTF-8, one JSON-RPC message per line (newline-delimited, never pretty-printed), writes serialised
  through a single writer so concurrent responses and notifications never interleave. **stdout carries protocol messages
  only.** Every log line, including from embedded training, goes to stderr (or to MCP `logging` notifications).
- **Progress.** Only when the request carries `_meta.progressToken`: the server echoes it in
  `notifications/progress`, with monotonically increasing `progress` (and `total` when known), at most ~4 per second,
  and stops when the request completes. `val`, `predict_batch` and `create_synthetic_dataset` report progress. `train`
  returns immediately, so training progress is observed through `run_status` or the run resource, not through progress
  notifications.

## Results and errors
Every tool declares an `inputSchema` and an `outputSchema`. The schemas are generated from the same Rust types as the
HTTP API (`schemars`), so `folo describe --json` prints identical schemas, and docs/API.md is the source of field
meanings.
- **Success:** `structuredContent` validates against the tool's `outputSchema`, plus one short `text` content item
  summarising it (one or two lines, for transcripts).
- **Tool failure** (bad input values, missing dataset, GPU unavailable, ...): a normal result with `isError: true`, no
  `structuredContent`, and a single `text` item containing the JSON error object
  `{ "code": "dataset_not_found", "message": "...", "hint": "..." }`. Codes are the ones in docs/CLI.md.
- **Protocol errors** (unknown tool, malformed JSON-RPC, schema-invalid arguments the SDK rejects): JSON-RPC errors per
  the spec, never `isError` results.
- Resource reads for unknown URIs return the spec's resource-not-found JSON-RPC error.

## Paging and size limits
- Every list tool takes `{ cursor?: string, limit?: integer (default 50, max 200) }` and returns
  `{ items: [...], next_cursor: string | null, total: integer }`.
- List items are **summaries** (id, name, status and key numbers). Detail comes from the matching `get`/status tool
  or resource.
- No tool result exceeds ~32 KB of JSON. Anything bigger (per-image predictions for a batch, full metric histories,
  sample grids) is written as a resource and returned as a `resource_link` content item with its URI.

## Tools
| tool | input | output (`structuredContent`) |
|---|---|---|
| `folo_info` | `{}` | version, devices (CPU, GPUs with backend and memory), data dir, read-only flag, model scales with params / GFLOPs |
| `list_datasets` | paging | page of `{ id, name, format, classes, splits }` |
| `get_dataset` | `{ dataset_id }` | `Dataset` with `stats`, plus `issues: [{ kind, file, message }]` (first 50, with `issues_total`) |
| `list_dataset_images` | `{ dataset_id, split?: "train"\|"val"\|"test", class?: integer }` + paging | page of `{ image_id, file, split, width, height, boxes? }` (`boxes` only with `include_boxes: true`) |
| `register_dataset` | `{ name, path, format?: "auto"\|"yolo"\|"coco" }` | `Dataset` |
| `create_synthetic_dataset` | `{ name, images?: 1000 (max 100000), classes?: string[], imgsz?: 640, seed?: 0 }` | `Dataset` |
| `list_models` | paging | page of `{ id, name, arch, imgsz, classes, metrics }` |
| `train` | `TrainConfig` (docs/API.md) | `{ run_id, status }` |
| `run_status` | `{ run_id, log_lines?: 20 (max 200) }` | `Run`, plus `last_step`, `last_epoch` and the last `log_lines` log lines |
| `list_runs` | `{ status?: string }` + paging | page of `Run` summaries |
| `stop_run` | `{ run_id }` | `Run` |
| `val` | `{ model_id, dataset_id, split?: "val", imgsz?, conf?: 0.001, iou?: 0.7 }` | the `val` report from docs/API.md `Epoch.val`, plus `timing` |
| `predict` | `{ model_id, image_path \| image_base64 \| { dataset_id, image_id }, conf?: 0.25, iou?: 0.7, max_det?: 300 }` | the `/api/predict` response |
| `predict_batch` | `{ model_id, paths: string[] \| glob: string, conf?, iou? }`, at most 256 images **after** glob expansion (more is a `too_many_inputs` error) | `{ processed, failed: [{ path, code, message }], boxes_total, per_class_counts, timing }` plus a `resource_link` to `folo://predictions/{batch_id}` with the per-image boxes |
| `export` | `{ model_id, format: "folo-f16"\|"onnx" }` | `{ path, bytes }` |
| `compare_runs` | `{ run_ids: string[] (2–8) }` | per-run best metrics, final losses, epochs and duration, and the config diff |

Read-only mode (`--read-only`) exposes only `folo_info`, `list_*`, `get_dataset`, `run_status`, `val`, `predict`,
`predict_batch` and `compare_runs`. Mutating tools are not listed, and they are **also rejected at dispatch**
(`read_only` tool error) if called anyway.

## Resources
Each resource read returns a JSON snapshot, unless stated otherwise.
- `folo://datasets/{id}`: dataset with stats.
- `folo://datasets/{id}/images/{image_id}`: the image itself (`image/jpeg` / `image/png`, as a blob), so a multimodal
  agent can look at it.
- `folo://runs/{id}`: `{ run: Run, last_step, last_epoch, revision }`. `revision` increases on every change.
  **Subscribable**: the server sends the spec's resource-updated notification for this URI (coalesced, at most one
  per second), and the client re-reads the snapshot. Missed intermediate steps are not an error: epoch history is in
  the metrics resource.
- `folo://runs/{id}/metrics`: the full history (docs/API.md `/metrics`). Subscribable, and updated after each epoch.
- `folo://runs/{id}/samples/{epoch}`: validation samples rendered as an annotated JPEG grid (truth outlined white,
  predictions in class colours), for quick visual checks.
- `folo://predictions/{batch_id}`: per-image boxes of a `predict_batch` call (kept for the server's lifetime).
- `folo://docs/{name}`: repository docs (`ARCHITECTURE`, `API`, `CLI`, `MODEL`, `FORMAT`, `METRICS`, `MCP`) as text/markdown.

## Prompts
Prompt retrieval (`prompts/get`) **only returns messages**. It never starts jobs or touches files; the agent then
calls tools itself.
- `train_on_folder` (arguments: `folder_path` required; `goal` optional, free text, e.g. "fast on CPU"): returns
  instructions to call `register_dataset`, inspect `get_dataset` stats and issues, choose a scale and image size from
  the stats and `folo_info` devices, call `train`, and explain the choices to the user.
- `diagnose_run` (arguments: `run_id` required): returns the run's metric summary and last log lines embedded in the
  message, plus instructions to identify over/underfitting, class imbalance or LR problems and propose a concrete
  TrainConfig diff.
In read-only mode, `train_on_folder` is not listed.

## Files and safety
- No tool deletes or overwrites existing files. Generated outputs (synthetic datasets, run directories, exports,
  prediction dumps) go to new, collision-free paths under `--data-dir` (`<name>`, then `<name>-2`, ... if taken).
  Writes go to a temp file first and are then renamed into place.
- Input paths (`register_dataset.path`, `predict.image_path`, `predict_batch.paths/glob`) may be absolute or relative
  to `--data-dir`, and are resolved on the machine running `folo mcp`. `--data-dir` is where FOLO **writes**; it is not
  a read sandbox. `--confine` restricts reads to `--data-dir` as well, using canonicalised paths with symlinks and
  junctions resolved, and rejects anything outside it with `path_outside_data_dir`.

## Run lifetime
- **Embedded mode** (default): runs execute inside the `folo mcp` process. Run state is persisted under
  `--data-dir/runs/<id>/` (config, metrics, checkpoints at every epoch). When the MCP client disconnects or the process
  receives a shutdown signal, running jobs are stopped gracefully: the current epoch is abandoned, the last checkpoint
  is kept, and the status becomes `stopped`. On the next start, persisted runs are listed again, and `train` with
  `weights` set to a checkpoint resumes. Cancelling the `train` request itself (MCP cancellation) after it has returned
  has no effect on the run. Use `stop_run`.
- **Proxy mode** (`folo mcp --server http://127.0.0.1:7878`): every tool maps to the HTTP API of a running
  `folo serve`, so Studio and the agent share runs that outlive the MCP process:
  `list_*`/`get_dataset` → `GET` list/detail endpoints (paging is applied by the MCP layer), `list_dataset_images` →
  `GET /api/datasets/{id}/images`, `train` → `POST /api/runs`, `run_status` → `GET /api/runs/{id}` (plus the server's
  log history), `stop_run` → `POST /api/runs/{id}/stop`, `predict` → `POST /api/predict` (local `image_path` files are
  read by the MCP process and uploaded as multipart), `export` → `POST /api/models/{id}/export`. `val`, `predict_batch`
  and `compare_runs` need server endpoints that docs/API.md does not have yet (`POST /api/val`,
  `POST /api/predict/batch`, a run log history endpoint). Until they exist, those tools return `not_implemented` in
  proxy mode, and the PR that implements proxy mode adds them to docs/API.md.

## Example agent session
```
→ create_synthetic_dataset { "name": "shapes", "images": 2000 }      ← { "id": "ds_1", ... }
→ train { "name": "shapes-n", "dataset_id": "ds_1", "arch": "folo-n", "epochs": 30, "imgsz": 320 }
                                                                      ← { "run_id": "run_1", "status": "queued" }
→ (subscribe folo://runs/run_1, or call run_status every ~30 s)       ← { "epoch": 12, "best": { "map50_95": 0.71 }, ... }
→ predict { "model_id": "m_1", "image_path": "test.jpg" }             ← { "boxes": [...], "timing": {...} }
```

## Tests
An integration test spawns `folo mcp` and, for **each supported revision**, runs the handshake/discovery, `tools/list`
(checking the schemas), `resources/list`, `prompts/list`, a `create_synthetic_dataset` → `train` (2 epochs on a tiny
set) → subscribe / `run_status` → `predict` flow, an `isError` case, a JSON-RPC error case, and a read-only-mode
dispatch rejection. It asserts that nothing but valid JSON-RPC lines appears on stdout.
