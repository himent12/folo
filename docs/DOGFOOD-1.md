# Dogfood report 1: FOLO as a new agent sees it

An AI agent that had never seen FOLO tried its five jobs, first with the CLI (`--json`) and then through
`folo mcp`. It read only README.md, llms.txt, docs/AGENT_COOKBOOK.md, docs/MCP.md and docs/CLI.md, and opened
the source only to make the small fixes listed at the end. The MCP client was a 100-line Python stdio client
that used the 2025-11-25 `initialize` handshake. One 2026-07-28 direct call was also tried.

- Build: `folo 0.1.0 (d78dffadc16c 2026-11-08)`, `cargo build --release -p folo-cli`.
- Machine: Windows 11, i7-14700KF, RTX 4080 SUPER (Vulkan). Data was under `D:/folo-dogfood/`.
- **The GPU was shared.** Two other agents' `folo.exe` processes kept it at 100% utilization during most of
  the runs, so the training throughput figures here are upper bounds on wall time, not benchmarks.

## Verdict
All five jobs were completed through both front ends. Nothing blocked the work. The error envelope, exit codes
and hints make FOLO easy to script against. `dataset check` explained every label error injected for this test,
and resume is bit-exact. The main problems for an agent were these:
1. `get_dataset` can hide real label errors.
2. A GPU out-of-memory failure arrives as a 32 KB `internal` error, three times over.
3. Unknown tool arguments are silently ignored.
4. Several outputs are much larger than an agent's context should hold.

## What worked smoothly
- **The docs are accurate.** Every cookbook recipe worked as written on the first try. Field names in the
  examples matched the real output, apart from the gaps listed below.
- **Datasets.** `dataset synth` wrote 1000 images in 1.1 s. `dataset check` ran in 0.1–0.3 s and
  `dataset stats` in 0.3 s. The human tables with bar charts are pleasant to read.
- **Label errors (job 4).** `folo dataset check --fix-dry-run` caught each injected error, gave the file and
  line, and suggested a fix for each one:
  - a malformed line: `broken_label`, "not a finite number"
  - a line with only 4 values: `broken_label`
  - class 7 out of 3, and class -1: `class_overflow`
  - pixel coordinates and a box outside the image: `out_of_range_box`
  - a box slightly past the edge: `clipped_box`
  - a duplicated line: `duplicate_label`
  - a copied image: `duplicate_image`
  - a header-truncated or garbage JPEG: `corrupt_image`

  The messages say whether the loader skips the image. Edits were picked up on the next `get_dataset`, with no
  re-registration.
- **Errors (job 5).** Every wrong path, bad value and missing model produced the documented envelope with a
  stable code and the right exit code: 3 for not found, 2 for usage or config, 4 for not implemented. Most came
  with an actionable hint. On the MCP side, every failure was an `isError` result with `{code, message, hint}`.
  Unknown tools and resources were proper JSON-RPC errors, and no stray line ever appeared on stdout.
- **Resume is bit-exact.** A 5-epoch CLI run was killed after epoch 2 and then resumed with `--resume`. Its
  per-epoch losses and mAP were identical to an uninterrupted 5-epoch run, to every printed digit. On the MCP
  side, `stop_run` reached `stopped` 0.5 s after the call, `train {resume_run_id}` continued the same id, and a
  resume with a changed `epochs` was refused with a clear `invalid_config`.
- **Disconnect handling.** Closing stdin mid-run made `folo mcp` exit in 0.8 s. The run was `stopped` with a log
  line, and a new session listed it and resumed it.
- **The MCP surface.**
  - `initialize` takes about 30 ms and most tool calls about 1 ms.
  - The one-line `text` summaries are excellent for transcripts. For example: "run_86a2 (shapes-n-320):
    running, epoch 9/12, step 443, best mAP50-95 0.215 at epoch 8, ETA 40 s".
  - The FIFO queue ran two `train` calls back to back.
  - The `folo://runs/{id}` subscription produced about one update per second.
  - `compare_runs` returns a tidy `config_diff`.
  - `export` never overwrites (`-2`).
  - `predict_batch` with a `resource_link` keeps the result small.
  - `--read-only` and `--confine` both behave exactly as documented, including the dispatch-time rejection.
  - A 2026-07-28 `tools/call` without discovery just worked.
- **Inference is fast.** `val` on 200 images took 0.2–0.4 s (1.7 ms per image). `predict` took 10–15 ms per
  image at 320 px on the GPU, and `export` took 0.06 s.

## Timing
| what | time |
|---|---|
| `cargo build --release -p folo-cli` (cold, own target dir) | 4 min 49 s |
| `dataset synth` 1000 images at 320 | 1.1 s (CLI), 1.2 s (`create_synthetic_dataset`) |
| CLI `train` folo-n, 15 epochs, 320, batch 16 | 330 s (20–28 s per epoch, about 44 img/s; GPU contended) |
| CLI `train` folo-n, 10 epochs, 256, batch 32 | 396 s (27–63 s per epoch, about 26 img/s; heavily contended) |
| CLI `train` folo-n, 5 epochs, 320 (reference run) | 77 s |
| MCP `train` folo-n, 12 epochs, 320 | 160 s (about 13 s per epoch, 74 img/s) |
| MCP `train` folo-n, 8 epochs, 256, batch 32 | 88 s |
| CLI `train` folo-x, 1 epoch, 320 | 254 s |
| CLI `train` folo-x, 640 | GPU out of memory after 18 s (see F2) |
| `val` 200 images | 0.2–0.4 s |
| `predict` per image (GPU, 320) | 10–15 ms (the first image about 120 ms) |
| `export` folo-f16 | 0.03–0.06 s |
| MCP `initialize` / `run_status` / `get_dataset` (cached) | about 30 ms / 1 ms / 1 ms |
| MCP `folo_info` / first `train` / `register_dataset` | 0.85 s / 1.5 s / up to 2.3 s (GPU enumeration, indexing) |
| `stop_run` until `stopped` | 0.5 s |

The same `folo-n` at 320 px ran at 40–74 img/s depending on what else was on the GPU. That is slow for a 2.6M
parameter model on this card, but these runs cannot separate FOLO from contention. A benchmark on an idle GPU
should settle it (`folo bench` is still planned and exits 4).

## Token cost
Outputs an agent would pull into its context:

| output | size | note |
|---|---|---|
| `folo train --json`, 15 epochs at 50 batches | **143 KB** (771 lines; 750 are `step`) | No way to opt out of step records. On this slow, contended GPU, every batch got a record of about 190 bytes; the 10-per-second throttle never kicked in. |
| GPU out-of-memory failure (`train --json`) | **32.6 KB per record, three times** (log, `status.error`, error envelope) plus 1180 stderr lines | The wgpu validation error, repeated. |
| `folo://runs/{id}/metrics` (12 epochs) | **150 KB** | Every step is included, with no epoch-only view. |
| `tools/list` | **68.6 KB** (48.6 KB read-only) | `run_status` 11.2 KB, `list_runs` 8.4 KB and `stop_run` 6.9 KB each embed the full `Run` schema. |
| `folo describe --json` | **92 KB** | 72 KB of it is `mcp_tools`. A CLI-only agent pays for the MCP schemas too. |
| `diagnose_run` prompt | 7.5 KB | Pretty-printed JSON inside the message. |
| `dataset check --json` on a *clean* dataset | 6.6 KB | All 45 `empty_image` info records, with no severity filter. |
| `get_dataset` | 6.0–6.5 KB | 50 issues, mostly `empty_image` infos (see F1). |
| floats everywhere | about 40% of most records | For example `"lr0": 0.009999999776482582` and box coordinates such as `207.92527770996094`. |

Good: `run_status` is 1–2.5 KB, `predict_batch` 0.6 KB with a resource link, and every text summary is one line.

## Friction points
Severity: **blocker** stops the job; **annoying** costs an agent time, tokens or a wrong conclusion; **polish**
is cosmetic or inconsistent.

### F1. `get_dataset` hides real errors behind info items (annoying, high)
**Status:** fixed in #PR. `get_dataset` lists errors, then warnings (infos only counted in `issue_counts`), each issue with `severity`, `line` and `fix`; `list_dataset_issues {dataset_id, severity?, kind?}` pages through all of them over `GET /api/datasets/{id}/issues`, which gained `severity=` and `counts`.

- **Call:** `get_dataset {dataset_id}` on the broken copy, after adding `9 0.5 0.5 0.1 0.1` to
  `labels/val/000999.txt`.
- **Happened:** `issues_total: 57`, but only 50 issues were returned, in file order. 39 of them were
  `empty_image` infos, and the `class_overflow` in the val split was not among them. MCP issues also carry no
  `severity`, `line` or `fix`, unlike `folo dataset check`, so an agent cannot even tell which entry is the
  bad line. Recipe 3 tells agents to look here.
- **Expected:** errors first, then warnings, with infos summarised as counts.
- **Fix:**
  - Sort by severity and leave infos out of the list, or count them only (`counts.by_kind`, as the CLI does).
  - Add `severity`, `line` and `fix` to each issue.
  - Add a paged `list_dataset_issues {dataset_id, severity?, kind?}` tool.
- *This PR documents the workaround in the cookbook.*

### F2. GPU out of memory is a 32 KB `internal` error, three times (annoying, high)
**Status:** fixed in #PR. `gpu_out_of_memory` with a "lower batch / imgsz, smaller arch" hint on every front end; folo-gpu keeps the first wgpu error and counts the rest; engine messages are capped at 500 characters; the run log line only names the failure; a new CLI run that fails before its first checkpoint removes its folder. Deferred: free VRAM in the message (wgpu does not report it).

- **Command:** `folo train --data shapes --model folo-x --epochs 1 --json` (640 px on a GPU with 14 GB already in
  use by other processes).
- **Happened:** exit 1 with `{"code":"internal","message":"wgpu backend: wgpu error: Out of Memory\nValidation
  Error\n\nCaused by: ..."}`. The message was 32.6 KB, repeated in the `log` record and in `status.error`, and
  followed by 1180 lines on stderr. There was no hint. An empty `runs/shapes-x/` (with no weights) was left
  behind.
- **Expected:** `gpu_out_of_memory` with a hint like "lower --batch or --imgsz, or use a smaller --model",
  and the first wgpu error cut to a few hundred characters.
- **Fix:** map wgpu OOM to a dedicated code, deduplicate, cap error messages (MCP already cuts run errors to
  500 characters in lists, but not in the CLI), and report free VRAM where possible.

### F3. Unknown tool arguments are silently ignored (annoying)
**Status:** fixed in #PR. Every input schema (nested objects too) has `additionalProperties: false`, and unknown arguments are `invalid_arguments` with a suggestion (`lr` → `lr0`, `model` → `arch`).

- **Call:** `train {"dataset_id": "ds_66f9", "epochs": 1, "lr": 0.1}`
- **Happened:** a run started. The typo'd `lr` was dropped without a word.
- **Expected:** `invalid_arguments: unknown field lr, expected one of ...`. This is the classic agent mistake
  (`lr` instead of `lr0`, or `model` instead of `arch`).
- **Fix:** `#[serde(deny_unknown_fields)]` on tool inputs, or `additionalProperties: false` in the schemas plus
  validation.

### F4. `register_dataset` on the same path creates duplicates (annoying)
**Status:** fixed in #PR. Registering a folder again (by canonical path) returns the existing dataset, in the real and mock stores (HTTP and MCP); the MCP summary says "already registered".

- **Call:** `register_dataset {"name": "bad", "path": "D:/folo-dogfood/cli/bad"}`, three times across sessions.
- **Happened:** three datasets, `ds_132e`, `ds_79ad` and `ds_e7ef`, all named `bad`.
- **Expected:** the existing id back (idempotent), or a `dataset_exists` error that names it.
- **Fix:** key registration on the canonical path.

### F5. `folo train --resume` silently ignores other flags and finished runs (annoying)
**Status:** fixed in #PR. `--resume` refuses a changed flag (`invalid_config`, naming the flag and both values) and a completed run (`run_not_resumable`, with the `--model .../best.folo` command).

- **Command:** `folo train --resume runs/ref5/weights/last.folo --epochs 8 --json` on a completed 5-epoch run.
- **Happened:** exit 0, `"finished 5 epochs in 0.0 s"`, `status: completed`. `--epochs 8` was ignored.
- **Expected:** what MCP does: `invalid_config` for a changed field, and `run_not_resumable` (with the
  "start a new run from its weights" hint) for a completed run. The CLI never tells the agent that it should
  run `--model runs/ref5/weights/best.folo` to train further.
- **Fix:** reuse the MCP checks in the CLI.

### F6. An agent cannot stop CLI training gracefully on Windows (annoying)
**Status:** fixed in #PR. Ctrl+Break is handled like Ctrl+C on Windows, and `folo stop <run_dir>` writes a stop file the trainer polls (tested end to end; the Ctrl+Break path is not exercised by the tests).

- **Command:** a script starts `folo train --json` in its own process group and sends `CTRL_BREAK_EVENT` after
  epoch 2. A script cannot send Ctrl+C to a child on Windows.
- **Happened:** the process died at once (exit `0xC000013A`), with no `status: stopped` or `done` record. Resume
  still worked, because the epoch-2 checkpoint was already on disk.
- **Expected:** Ctrl+Break treated like Ctrl+C, meaning stop after the current batch and emit `stopped`.
- **Fix:** handle `ctrl_break` too, or add a stop file or `folo stop <run_dir>`. *Documented in CLI.md in this
  PR.*

### F7. A JPEG truncated mid-scan passes `dataset check` (annoying)
**Status:** fixed in #PR. `dataset check` (full decode) reports a JPEG without an end-of-image marker after its scan data as `corrupt_image` ("truncated JPEG").

- **Command:** `head -c 2000 000011.jpg` (the original is 17 342 bytes), then
  `folo dataset check bad --json`.
- **Happened:** no issue for `000011.jpg`, although the docs say the full decode catches truncated data. Cutting
  the file to 300 bytes, inside the headers, is caught as `corrupt_image`. The image presumably trains as
  half-grey pixels with labels for shapes it no longer shows.
- **Fix:** treat a missing EOI or a premature end of scan as `corrupt_image`, or add a warning
  `truncated_image`.

### F8. `train --json` step records: no opt-out (annoying, tokens)
**Status:** fixed in #PR. `folo train --json --json-events epoch` leaves the step records out; floats are written without f32 noise (F13).

- **Command:** `folo train ... --json` (see Token cost).
- **Happened:** 750 `step` records out of 771, one per batch, because each batch took longer than the 100 ms
  throttle.
- **Fix:** `--json-events epoch` (or `--no-steps`), and round floats in the records.
- *The docs said "about 10 per second", which reads as a rate rather than a cap. Fixed in this PR: throttled to
  about 10 per second, which on slow runs means one per batch.*

### F9. `tools/list` and `describe --json` are heavy (annoying, tokens)
**Status:** fixed in #PR. Schemas are compacted (no `$schema` / numeric `format` / `minimum: 0`, no output defaults or `$defs` descriptions, string enums): `tools/list` went from 68.6 KB to 44.9 KB with one more tool (`run_status` 11.2 → 5.3 KB, `list_runs` 8.4 → 3.7 KB, `stop_run` 6.9 → 3.3 KB), `describe --json` from 92 KB to 71 KB. `folo describe <command>` (about 7 KB for `train`) and `--no-mcp-tools` (27 KB). JSON Schema has no cross-tool `$defs`, so each tool still carries its shared types once.

- **Calls:** `tools/list`, `folo describe --json`.
- **Happened:** 68.6 KB and 92 KB respectively (see Token cost).
- **Fix:**
  - Hoist shared types (`Run`, `TrainConfig`, `Epoch`) into `$defs` once and reference them.
  - Shorten field descriptions in output schemas.
  - Add `folo describe <command>` and `--no-mcp-tools`.

### F10. The metrics resource has no compact view (annoying, tokens)
**Status:** fixed in #PR. `folo://runs/{id}/epochs` (subscribable): the epoch history without steps.

- **Read:** `folo://runs/{id}/metrics`, 150 KB for 12 epochs.
- **Happened:** every step is included, with no compact view. An agent wanting the mAP curve has to pull the
  whole history.
- **Fix:** add `folo://runs/{id}/epochs`, or a `?steps=none` option, or downsample steps.

### F11. Misleading `device gpu:7` error (annoying)
**Status:** fixed in #PR. `device_not_found` (HTTP 404, exit 3) on both front ends; the message names the devices FOLO can use and the hint lists the adapters ("gpu:0 (NVIDIA ..., vulkan)").

- **Command / call:** `folo train ... --device gpu:7`; `train {"device": "gpu:7"}`.
- **Happened:** "device gpu:7 has no registered backend in this build (GPU support is folo-gpu's `init()`); use
  --device cpu". The build does have GPU support; there is simply no adapter 7. The code is `not_supported`
  in the CLI and `not_implemented` in MCP, and the MCP message still says `--device`.
- **Fix:** `device_not_found` with "this machine has gpu:0 (NVIDIA GeForce RTX 4080 SUPER)" as the hint, and the
  same code on both front ends.

### F12. Paths in JSON mix separators on Windows (polish)
**Status:** fixed in #PR. CLI records, export paths, `duplicate_image` messages (now relative to the root) and `--confine` errors use `/` without `\\?\`; MCP registers dataset paths with `/`.

Examples:
- `"run_dir":"runs\\shapes-n"` and `"best":"runs\\shapes-n\\weights/best.folo"`
- `"yaml":"D:/folo-dogfood/cli/shapes\\data.yaml"`
- `"path":"D:/folo-dogfood/mcp\\exports\\shapes-n-320-best-f16.folo"`
- `"path":"runs/shapes-n/weights\\best-f16.folo"`
- `duplicate_image` messages: `same bytes as bad\\images/train\\000010.jpg`, which is not relative to `root`
  like `file` is.
- `--confine` errors print `\\?\D:\...` verbatim paths.

Fix: normalise every path in JSON output to `/` and strip the `\\?\` prefix.

### F13. Floats are serialised with f32→f64 noise (polish, tokens)
**Status:** fixed in #PR. CLI records, MCP results, resources and schemas print f32 values as f32 (`folo_api::json::clean_floats`). Box coordinates keep f32 precision (not rounded to 0.1 px).

- `"lr0": 0.009999999776482582`, `"hsv": [0.014999999664723873, ...]`, box coordinates with 15 digits, and
  metrics with 16.
- **Fix:** serialise f32 as f32, round box coordinates to 0.1 px, and round metrics to 4–5 significant digits.

### F14. Error codes differ between the CLI and MCP for the same failure (polish)
**Status:** fixed in #PR. One vocabulary in `folo_api::codes` (code, HTTP status, exit code; a test scans the sources). Bad arch → `invalid_config`, empty split → `invalid_dataset`, missing GPU → `device_not_found` everywhere; MCP reports the HTTP layer's `invalid_body` / `invalid_query` as `invalid_arguments`.

| failure | CLI code | MCP code |
|---|---|---|
| bad arch | `invalid_config` | `invalid_arguments` |
| empty split | `invalid_dataset` | `invalid_body` (an HTTP-layer name) |
| missing GPU | `not_supported` | `not_implemented` |

Agents that branch on codes need one vocabulary.

### F15. Run config shows a learning rate that is not used (polish)
**Status:** partly fixed in #PR: CLI.md explains `--optimizer auto` and where the used optimizer, `lr0` and accumulation are (`config.json`: `optimizer`, `accumulate`). Deferred: an `effective` object in `run_status` (a `Run` type change across the API, Studio and Desktop).

- **Call:** `run_status`, and `runs/*/config.json`.
- **Happened:** `config.lr0` is `0.01`, but the run log says `auto → adamw lr0 0.001429 ... batch 16 × accumulate 4`.
- **Expected:** the effective optimizer, `lr0`, accumulate and warmup as fields. An agent tuning `lr0` has to
  parse a log line to learn that `optimizer: auto` overrode its value.
- **Fix:** an `effective` object in `run_status` and `config.json`, and a sentence on `--optimizer auto` in
  CLI.md.

### F16. Smaller items (polish)
**Status:** mostly fixed in #PR, see the list.

- `folo train --lr0 -1` is a clap usage error ("unexpected argument '-1'"); only `--lr0=-1` reaches the
  range check. Fix: `allow_negative_numbers`. *Fixed in #PR (`allow_negative_numbers`).*
- `folo export --model shapes/data.yaml` says "invalid safetensors file: header length 8295717702782558243
  exceeds the 100000000-byte limit". Fix: "not a .folo weights file". *Fixed in #PR: "... is not a .folo weights file".*
- CLI `export` silently overwrites an existing `best-f16.folo`, while MCP `export` never overwrites (`-2`).
  Pick one behaviour and document it. *Fixed in #PR: the CLI never overwrites either (`-2` names, `output_exists`).*
- Not-found errors (`model_not_found`, `dataset_not_found`, `run_not_found`) have `hint: null`. Suggest
  `list_models` / `list_datasets` / `list_runs`, or `folo ... --help`. *Fixed in #PR: the hint names the list tool / endpoint.*
- `corrupt_image` messages read "cannot decode: image error at : Format error ..." (an empty path) and end in
  `\n`. *Fixed in #PR.*
- `dataset stats` reports `skipped_images: 2` (the corrupt images) while 6 more images with broken labels are
  also skipped (train went from 801 to 793). Count both, or name the field `unreadable_images`. *Deferred: renaming or splitting the field is an output change for its own PR.*
- `dataset check --json` has no severity filter: a clean dataset prints all its `empty_image` infos. Add
  `--min-severity warning`. *Fixed in #PR (`--min-severity`).*
- `folo_info.devices.gpus[].memory_bytes` is 0 and `doctor` shows no VRAM, so an agent cannot size `batch`. *Deferred: wgpu does not report VRAM.*
- `run_status.device` is `"NVIDIA GeForce RTX 4080 SUPER (vulkan)"` while `predict.device` is `"gpu:0"`. *Deferred: changing `Run.device` touches Studio and Desktop.*
- Annotated `--save` images and sample grids carry class *ids* (digit-only font) or no tags at all, and have no
  legend, so an agent looking at them must already know the palette. A legend strip would fix it. *Deferred (drawing work).*
- MCP `create_synthetic_dataset` has no `val_frac` or `vary_aspect`, which the CLI has. *Deferred (API and both stores).*
- There is no CLI `compare`. The cookbook's `jq -s` recipe needs `jq`, which a Windows machine usually lacks. *Deferred.*

## Doc gaps
- CLI.md listed `train` as "works (CPU)" while it trains on the GPU. *Fixed.*
- CLI.md and the cookbook said step records arrive "about 10 per second"; that is a cap, and slow runs get
  one per batch. *Fixed, with a size warning.*
- `folo val --json` had no documented shape, and the `val` and `predict` rows omitted `--device`. *Fixed;
  `val` now also reports `device`.*
- `get_dataset`'s 50-issue cap is in file order and can hide errors (F1). *Workaround documented in the
  cookbook.*
- Stopping CLI training from a script on Windows (F6). *Documented in CLI.md.*
- Annotated images are tagged with class ids, not names. *Clarified in CLI.md.*
- Not fixed: how `optimizer: auto` picks `lr0` and accumulation (F15). Also, which `epoch` value `run_status`
  shows during an epoch: it is the epoch in progress, so `epoch: 2` means one epoch has finished. That is
  inferable from "1-based" but worth one sentence in MCP.md.
- Not fixed: the cookbook's CLI examples print `runs/cats-s/...` paths, but on Windows they come back with `\`
  separators (F12).

## Top 10 improvements, ranked
1. **Make `get_dataset` issues trustworthy** (F1): severity first, infos as counts, `severity`/`line`/`fix` on *Fixed in #PR.*
   each issue, and a paged `list_dataset_issues` tool.
2. **Add a GPU OOM error code** (F2): `gpu_out_of_memory` with a hint, the message capped at about 500 *Fixed in #PR.*
   characters, no triple repetition, and no orphaned run folder.
3. **Reject unknown tool arguments** (F3), so typos do not silently train with defaults. *Fixed in #PR.*
4. **Cut the per-run token flood** (F8, F10): an epoch-only JSON mode for `folo train`, and an epoch-only *Fixed in #PR.*
   metrics resource.
5. **Shrink `tools/list` and `describe`** (F9): shared `$defs`, and a `describe <command>`. *Fixed in #PR.*
6. **Give the CLI a graceful stop for agents on Windows** (F6): handle Ctrl+Break, or add `folo stop`. *Fixed in #PR.*
7. **Detect JPEGs truncated mid-scan in `dataset check`** (F7). *Fixed in #PR.*
8. **Make `register_dataset` idempotent** per canonical path (F4). *Fixed in #PR.*
9. **Bring CLI `--resume` to MCP parity** (F5): refuse changed flags and completed runs, with the same hints. *Fixed in #PR.*
10. **Normalise the JSON surface** (F11–F14): `/` paths, rounded floats, one error-code vocabulary for the CLI
    and MCP, and `device_not_found` naming the adapters that exist. *Fixed in #PR.*

## Fixed in this PR
- `folo val --json` now includes `device` (as `predict` records already do), and the human line names the
  device. The CLI engine test checks it.
- docs/CLI.md:
  - `train` status changed from "works (CPU)" to "works".
  - Step records are described as throttled to about 10 per second (one per batch on slow runs).
  - Documented the `val` JSON shape.
  - Added `--device` to the `val` and `predict` rows.
  - Clarified the `--save` tag.
  - Explained Windows Ctrl+Break.
- docs/AGENT_COOKBOOK.md:
  - Step rate and size warning.
  - The `get_dataset` 50-issue caveat with the `folo dataset check` workaround.
  - The resume comment.
