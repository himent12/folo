# `folo mcp`: FOLO for agents

`folo mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio. With it, an agent
(Claude Code, Codex, any MCP client) can build datasets, train, evaluate, run and export FOLO models without screen
scraping. It exposes the same operations as the CLI and `docs/API.md`, with the same error codes.

```jsonc
// e.g. .mcp.json / claude_desktop_config.json
{ "mcpServers": { "folo": { "command": "folo", "args": ["mcp", "--data-dir", "D:/folo-data"] } } }
```

## Principles
- **Same core as the CLI and the server.** Tools call the same library functions; there is no separate logic to drift.
- **Long jobs are asynchronous.** `train` returns a `run_id` at once. Agents poll `run_status` (cheap, includes the
  latest metrics) or read the `folo://runs/{id}/events` resource. No tool call blocks for more than ~30 s, except
  `val` and `predict` on large inputs, which report progress notifications.
- **Compact, structured results.** Every tool returns `structuredContent` (JSON matching the tool's `outputSchema`)
  plus a short human-readable text summary. Large arrays (per-image predictions, full metric histories) are paged
  or offered as resources instead of being inlined.
- **Errors are actionable.** A failure sets `isError: true` with `{ code, message, hint }`, using the same codes as
  `docs/CLI.md` (e.g. `dataset_not_found`, `invalid_config`, `gpu_unavailable`, `run_not_running`, `not_implemented`).
- **Safe by default.** Tools never delete files on disk. `delete_*` tools only unregister. Paths are resolved against
  `--data-dir` unless they are absolute. `--read-only` exposes only the non-mutating tools.

## Tools
| tool | input (JSON Schema summary) | output |
|---|---|---|
| `folo_info` | `{}` | version, devices (CPU, GPUs with backend and memory), data dir, model scales with params / GFLOPs |
| `list_datasets` | `{}` | `[Dataset]` (docs/API.md) without stats |
| `register_dataset` | `{ name, path, format?: "auto"\|"yolo"\|"coco" }` | `Dataset` |
| `create_synthetic_dataset` | `{ name, images?: 1000, classes?: string[], imgsz?: 640, seed?: 0 }` | `Dataset` |
| `dataset_stats` | `{ dataset_id }` | `Dataset.stats` plus `issues: [{ kind, file, message }]` (first 50) |
| `list_models` | `{}` | `[Model]` |
| `train` | `TrainConfig` (docs/API.md) | `{ run_id, status }` |
| `run_status` | `{ run_id }` | `Run` plus `last_step`, `last_epoch`, and the last 20 log lines |
| `list_runs` | `{ status?: string }` | `[Run]` (summary fields) |
| `stop_run` | `{ run_id }` | `Run` |
| `val` | `{ model_id, dataset_id, split?: "val", imgsz?, conf?: 0.001, iou?: 0.7 }` | the `val` report from docs/API.md `Epoch.val`, plus timing |
| `predict` | `{ model_id, image_path \| image_base64 \| (dataset_id, image_id), conf?: 0.25, iou?: 0.7, max_det?: 300 }` | the `/api/predict` response |
| `predict_batch` | `{ model_id, paths: string[] (≤ 256) \| glob, conf?, iou? }` | `{ results: [{ path, boxes }], timing }` |
| `export` | `{ model_id, format: "folo-f16"\|"onnx" }` | `{ path, bytes }` |
| `compare_runs` | `{ run_ids: string[] (2–8) }` | per-run best metrics, final losses, epochs, duration, and the config diff |

`train` inputs are validated against the TrainConfig JSON Schema, which the server publishes as `outputSchema` /
`inputSchema`. `folo describe --json` prints the same schemas.

## Resources
- `folo://datasets/{id}`: dataset JSON with stats
- `folo://datasets/{id}/images/{image_id}`: the image (`image/jpeg` / `image/png`), so a multimodal agent can look at it
- `folo://runs/{id}/metrics`: full history (docs/API.md `/metrics`)
- `folo://runs/{id}/events`: subscribable; each update is the latest `step` / `epoch` / `status` event
- `folo://runs/{id}/samples/{epoch}`: validation samples rendered as an annotated JPEG grid (truth green, prediction
  in class colours), for quick visual checks by agents
- `folo://docs/{name}`: the repository docs (`ARCHITECTURE`, `API`, `CLI`, `MODEL`, `FORMAT`, `METRICS`)

## Prompts
- `train_on_folder`: given a folder path, check it, register it, pick a sensible scale and image size from the
  dataset stats and GPU, start training, and explain the choices.
- `diagnose_run`: read a run's metrics and logs and suggest concrete config changes (over/underfitting,
  imbalance, LR).

## Example agent session
```
→ create_synthetic_dataset { "name": "shapes", "images": 2000 }      ← { "id": "ds_1", ... }
→ train { "name": "shapes-n", "dataset_id": "ds_1", "arch": "folo-n", "epochs": 30, "imgsz": 320 }
                                                                      ← { "run_id": "run_1", "status": "queued" }
→ run_status { "run_id": "run_1" }   (every ~30 s)                    ← { "epoch": 12, "best": { "map50_95": 0.71 }, ... }
→ predict { "model_id": "m_1", "image_path": "test.jpg" }             ← { "boxes": [...], "timing": {...} }
```

## Implementation notes
- Use the official Rust MCP SDK (`rmcp`) or a small hand-written JSON-RPC 2.0 stdio loop. Protocol version: the
  latest stable that the SDK supports. Never write anything but protocol messages to stdout; logs go to stderr.
- The MCP server embeds the same `Store` as `folo serve` (in-process). `folo mcp --server http://127.0.0.1:7878`
  proxies to a running server instead, so Studio and an agent can share runs.
- Integration test: spawn `folo mcp` and drive `initialize`, `tools/list`, then a `create_synthetic_dataset` → `train`
  (2 epochs on a tiny set) → `run_status` → `predict` flow.
