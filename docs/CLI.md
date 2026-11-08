# The `folo` CLI

`folo` is the single binary for training, evaluation, inference, export, the Studio server and the MCP server. It is
built for people and for agents: every command follows the conventions below, and `folo describe --json` prints the
whole command surface (generated from the clap command tree, so it never drifts from the code).

```sh
cargo run -p folo-cli -- serve --mock     # mock Studio API on http://127.0.0.1:7878
cargo folo --help                         # alias for: cargo run --release -p folo-cli --
folo describe --json                      # every command, argument, type, default and help string
folo describe train --json                # one command (about 7 KB instead of the whole surface)
folo doctor                               # version, OS, CPU, memory, GPU adapters, web app build
folo --version                            # folo 0.1.0 (1a2b3c4d5e6f 2026-11-08): version, commit, build date
```

`folo --version` prints the workspace version (`[workspace.package] version` in the root `Cargo.toml`, the single
source of truth), the 12-character commit and the UTC build date, baked in by `crates/folo-cli/build.rs`. Outside a
git checkout the commit is `unknown` unless `FOLO_GIT_SHA` is set; `SOURCE_DATE_EPOCH` pins the date for
reproducible builds.

### The embedded Studio (`embed-web`)
Release binaries are built with `cargo build --release -p folo-cli --features embed-web`, which compiles the built
Studio into the binary (about 1 MB), so `folo serve` works anywhere. The feature needs a built `web/dist`
(`npm ci && npm run build` in `web/`), or `FOLO_WEB_DIST` pointing at one (relative paths are resolved against the
repository root); the build fails with a hint otherwise. Because of that, `cargo test --workspace --all-features`
also needs a built web app. Without the feature, `folo serve` serves `web/dist` from disk as before.

## Conventions (every command)

### `--json`
A global flag. With it, **stdout carries only JSON, one object per line** (JSON Lines), and all human text (progress,
tables, hints, logs) goes to **stderr**. Without it, human text goes to stdout and errors to stderr. Logs from the
server (`tracing`) always go to stderr; tune them with `RUST_LOG` (default `info`).

Paths in records use `/` separators (also on Windows, which accepts them), and floats are written as they were
stored: `0.01`, not the `0.009999999776482582` an f32 widened to f64 would print.

Long-running commands emit one record per event, each with an `"event"` field (e.g. `folo serve --json` prints
`{"event":"listening","url":"http://127.0.0.1:7878",...}` once bound and `{"event":"stopped"}` on shutdown).
One-shot commands print one object.

### Errors
A failing command prints its error as the last stdout line in `--json` mode, and always in human form on stderr:

```json
{"error":{"code":"not_implemented","message":"`folo train` is not implemented yet","hint":"it waits on ..."}}
```

`code` is a stable snake_case identifier, `message` is for humans (engine errors are cut to 500 characters), `hint`
says what to do next and is `null` when there is nothing to add. The codes are one vocabulary with the HTTP API
and `folo mcp`: `folo_api::codes` lists each with its HTTP status and exit code, and the same failure has the same
code everywhere (a bad `--model` arch is `invalid_config`, an empty split `invalid_dataset`, a `--device gpu:7`
this machine lacks `device_not_found` with the devices it has in the hint, a GPU out of memory
`gpu_out_of_memory` with "lower --batch or --imgsz, or use a smaller --model").

### Exit codes
| code | name | meaning |
|---|---|---|
| 0 | `ok` | success (also `--help` / `--version`) |
| 1 | `runtime_error` | the command ran and failed (IO, bind failure, bad data, ...) |
| 2 | `usage_error` | invalid arguments or flags (error code `usage`) |
| 3 | `not_found` | a named file, dataset, model, run or device does not exist |
| 4 | `not_implemented` | the command exists but is not implemented yet |

## Commands

| command | status | what it does |
|---|---|---|
| `serve [--mock] [--host 127.0.0.1] [--port 7878] [--web-dir DIR] [--data-dir DIR] [--device auto] [--precision f32] [--cors-any]` | works | Runs the HTTP API (`docs/API.md`) and serves the built Studio with an SPA fallback: from `--web-dir` when given, else the Studio embedded in the binary (`embed-web` builds), else `web/dist`. The `listening` record says which (`web_source`: `dir` or `embedded`, `web_dir`, `web_ready`). `--mock` serves deterministic mock data. `--port 0` picks a free port. `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`) holds the dataset registry, index and thumbnail caches, generated synthetic datasets, training runs, models and exports (see `docs/API.md`, Real mode). `--device` (env `FOLO_DEVICE`: `auto`, `cpu`, `gpu:<n>`) is where `/api/predict` and `/eval` run; each training run has its own `device`. `--precision` (env `FOLO_PRECISION`: `f32`, `f16`) is the precision of `/api/predict` and `/api/val` (docs/API.md); `f16` falls back to f32, with a warning in the log, where the device has no f16 compute. Ctrl+C shuts down gracefully: open SSE streams are closed, then an active training run stops after its current batch (keeping its last epoch's checkpoint, status `stopped`). |
| `describe [COMMAND...] [--no-mcp-tools]` | works | `folo describe train` (or `dataset check`) prints just that command (`path`, `about`, `args`, `subcommands`); `--no-mcp-tools` leaves out the MCP tool schemas, which are most of the full output. Prints the CLI surface as JSON: `global_args`, `conventions` (exit codes, error shape) and `commands` (recursive, each with `path`, `about`, `args` and `subcommands`). Each argument has `name`, `long`, `short`, `positional`, `type` (`string`, `path`, `integer`, `number`, `boolean`, `enum`, `count`), `value_name`, `required`, `multiple`, `default`, `possible_values`, `env` and `help`. Pretty-printed without `--json`, one line with it. |
| `doctor [--web-dir web/dist]` | works | Version (`version`, `git_sha`, `build_date`), OS, CPU name and threads, memory, every wgpu adapter (`gpus`: name, backend, type, driver, `shader-f16`, buffer and workgroup limits), the adapter FOLO picks (`gpu_selected`, honouring `FOLO_GPU_ADAPTER` / `FOLO_GPU_BACKEND`), whether the web app is built (`web_dist`: `path`, `exists`, `index_html`, `folo_wasm`), whether one is embedded (`web_embedded`: `present`, `files`, `folo_wasm`), which one `folo serve` would use (`web_source`) and whether that one includes folo-wasm (`web_folo_wasm`: its Playground can run models in the browser, docs/WASM.md; release builds always do). |
| `train --data D [--model folo-n] [--epochs 100] [--batch 16] [--imgsz 640] [--device auto] [--workers N] [--optimizer auto] [--lr0 0.01] [--lrf 0.01] [--cos-lr] [--warmup-epochs 3] [--patience 100] [augmentation flags] [--seed 0] [--project runs] [--name NAME] [--resume CKPT] [--json-events all\|epoch]` | works | Trains a detector (docs/MODEL.md §8). Arguments mirror `TrainConfig` in `docs/API.md`; `--arch` is an alias of `--model`, which also takes a `.folo` file to fine-tune from. Runs go to `<project>/<name>/` (default name `<dataset>-<arch suffix>`, numbered when it exists): `config.json`, `metrics.jsonl` (every event, every step), `weights/last.folo` and `weights/best.folo` (full checkpoints: EMA, optimizer, schedule), `samples/epoch_NNN.json`. Live view: epoch bar, losses, img/s, ETA and one table row per epoch (P, R, mAP50, mAP50-95). `--json` prints every event as a record: `{"event":"status"|"step"|"epoch"|"log", ...}` with the API.md payload fields (`step`s throttled to about 10 per second, the last batch of each epoch always sent, so a slow run gets one per batch; `metrics.jsonl` has every step), then `{"event":"done", "status", "run_dir", "best", "best_metrics", "weights"}`. `--json-events epoch` leaves the `step` records out (status, epoch, log and done only: about 1 KB per epoch instead of one record per batch; a 15-epoch run of 50 batches drops from about 140 KB to 15 KB). `--resume runs/<name>/weights/last.folo` continues bit-exactly with the stored settings: only `--workers`, `--device`, `--data`, `--project` and `--json-events` may differ, any other flag given must repeat the run's value (a changed one is `invalid_config`, exit 2, naming the flag and both values), and a run that completed (all epochs, or stopped early) is `run_not_resumable` (exit 1) with the command that trains further (`--model <run>/weights/best.folo`), as MCP and `POST /api/runs` answer. Stopping: Ctrl+C, on Windows also Ctrl+Break (what a script can send to a child started in its own process group), or `folo stop <run_dir>` from another process stops after the current batch, keeps the last epoch's checkpoint and ends with `status: stopped` and the `done` record; a second interrupt exits at once. A new run that fails before its first checkpoint (a bad device, a GPU out of memory) removes its run folder. `--optimizer auto` (the default) picks SGD for long runs and AdamW with its own `lr0` for short ones, scaling for the batch with gradient accumulation, so `--lr0` only matters with `--optimizer sgd` or `adamw`; the run's `config.json` has what is used (`optimizer.type`, `optimizer.lr0`, `accumulate`; `train` holds the settings as given) and its first log line says the same. `--device auto` (the default; also for `val` and `predict`) uses the GPU when folo-gpu finds an adapter (the model, its optimizer state, EMA, gradient buffers and the detection loss all run there; batches are staged on a thread and uploaded as `u8`), else the CPU; `--device gpu:<n>` without a usable adapter is `device_not_found` (exit 3; the hint lists the devices this machine has). |
| `stop RUN_DIR` | works | Asks the `folo train` writing `RUN_DIR` (its `run_dir`) to stop after its current batch: writes `RUN_DIR/stop`, which the trainer polls every 200 ms and removes. Returns at once with `{"event":"stop_requested","run_dir","stop_file"}`; the training process then ends with `status: stopped`. A folder that is not a run is `run_not_found` (exit 3). The portable way for a script to stop training, on Windows too. |
| `val --model W --data D [--split val] [--batch 32] [--imgsz <model's>] [--conf 0.001] [--iou 0.7] [--device auto] [--half] [--workers N]` | works | mAP@0.5, mAP@0.5:0.95, precision and recall (docs/METRICS.md) of the weights' EMA (fused) on a split, overall and per class. One JSON object with `--json`: `{"model","data","split","imgsz","images","boxes","map50","map50_95","precision","recall","per_class":[{"class","name",...}],"device","dtype","seconds"}` (`dtype`: `f32`, or `f16` with `--half` where the GPU computes in it). `--half` (also for `predict`): half-precision inference, f16 weights and activations with f32 accumulation, on a GPU with `shader-f16` (an f16 export is uploaded as it is; other files are fused in f32 and converted); elsewhere it runs in f32 and prints a `note:` on stderr. On trained models f16 mAP is within 0.002 of f32 (docs/BENCHMARKS.md). |
| `predict --model W SOURCES... [--conf 0.25] [--iou 0.7] [--max-det 300] [--imgsz <model's>] [--device auto] [--half] [--save DIR]` | works | Inference on images or folders (recursive), letterboxed `rect` (docs/MODEL.md §9). One `{"event":"prediction", "file", "image", "boxes": [ScoredBox], "timing", "device", "dtype", "saved"}` record per image (`dtype` as for `val`) (an unreadable image gives an `error` record and is skipped), then `{"event":"done", ...}` with mean timings. `--save` writes annotated copies (boxes coloured by class, tagged with the class id and score, e.g. `0 0.84`). |
| `export --model W [--format folo-f16\|folo\|onnx] [--output FILE] [--imgsz N] [--batch N] [--dynamic] [--half] [--decode]` | works | Fused EMA weights (FORMAT.md export). `folo-f16` (default) and `folo` write a `weights` file in f16 / f32; `onnx` writes an ONNX graph (opset 17) for onnxruntime, TensorRT, OpenVINO or the CoreML converters, see [ONNX export](#onnx-export) (`--imgsz`, `--batch`, `--dynamic`, `--half` and `--decode` are ONNX-only, a usage error otherwise). Default output next to the input: `<name>-f16.folo` / `<name>-fused.folo` / `<name>.onnx` / `<name>-f16.onnx` (`--half`), `-2`, `-3`, ... when taken: like MCP `export`, it never overwrites (an existing `--output` is `output_exists`, exit 1). A file that is not `.folo` weights is `invalid_weights` saying so. |
| `bench [--model folo-n] [--mode infer\|train] [--imgsz 640] [--batch 1] [--iters 100] [--warmup 10]` | planned (exit 4) | Speed benchmarks. |
| `dataset synth OUT_DIR [--images 1000] [--classes circle,square,triangle] [--imgsz 640] [--seed 0] [--val-frac 0.2] [--image-format jpeg\|png] [--vary-aspect]` | works | Writes a synthetic shapes dataset (YOLO layout). See [Datasets](#datasets). |
| `dataset stats PATH [--format auto\|yolo\|coco]` | works | Class counts, box sizes, boxes per image: a table with bar charts, or the `stats` object of `docs/API.md` with `--json`. |
| `dataset check PATH [--format ...] [--fix-dry-run] [--headers-only] [--min-severity error\|warning\|info]` | works | Every problem in a dataset, one issue per bad line / image. Exit 0 when clean, 1 with errors or warnings. |
| `dataset convert IN OUT --to yolo\|coco [--from auto\|yolo\|coco]` | works | Converts between YOLO and COCO; images are copied unchanged. |
| `mcp [--data-dir DIR] [--mock] [--read-only] [--confine] [--server URL] [--device auto]` | works | Model Context Protocol server over stdio (MCP 2026-07-28 and 2025-11-25), see `docs/MCP.md`. stdout carries protocol messages only (also with `--json`); logs go to stderr. By default it embeds the real store on `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`): datasets, training, models, `val`, `predict` and `export` on the FOLO engine, the same data `folo serve` uses. `--mock` serves mock data, `--server` forwards to a running `folo serve`, `--read-only` hides and rejects mutating tools, `--confine` only reads inside `--data-dir`, `--device` (env `FOLO_DEVICE`) is where inference and `val` run. Exits 0 when the client closes stdin; the runs it started are stopped first (an active run after its current batch, keeping its last checkpoint). Recipes: `docs/AGENT_COOKBOOK.md`. |

`folo describe --json` is the authoritative list of arguments and defaults. Its `mcp_tools` array lists every
`folo mcp` tool (`name`, `description`, `mutating`, `inputSchema`, `outputSchema`) with the schemas `tools/list`
publishes (about 43 KB of the 71 KB; `--no-mcp-tools` leaves them out, `folo describe <command>` prints one
command). Negative numbers are values (`--lr0 -1` reaches the range check and is `invalid_config`).

## ONNX export

`folo export --model W --format onnx` loads the weights for inference (the EMA when present), fuses Conv+BN in f32
(docs/MODEL.md §9.1) and writes an ONNX model, opset 17 (IR version 8), default domain only. The protobuf is FOLO's
own encoder; no ONNX library is involved. The graph is FOLO's forward pass op for op: `Conv`, SiLU as `Sigmoid` ×
`Mul` (opset 17 has no SiLU; runtimes fuse the pair), `MaxPool`, `Resize` (nearest, asymmetric, floor) for the
upsamples, `Concat`, `Split`, `Add`, and for the attention block `Reshape`, `Transpose`, `MatMul`, `Mul`, `Softmax`
and `Shape`. A model part the exporter has no mapping for fails with `not_implemented` (exit 4) naming it, and
never writes a partial file. No FOLO v1 model has such a part.

| flag | default | effect |
|---|---|---|
| `--imgsz N` | the model's training imgsz (`folo.imgsz`) | square input size, a multiple of 32 in 32..=8192 |
| `--batch N` | 1 | fixed batch size |
| `--dynamic` | off | symbolic `batch`, `height`, `width` (any batch, any H and W that are multiples of 32); excludes `--batch` |
| `--half` | off | f16 weights, input, activations and outputs (default name `<name>-f16.onnx`) |
| `--decode` | off | decode in the graph (below) instead of the raw head outputs |

**Input** `images: [B, 3, H, W]`, RGB in `[0, 1]` (`u8 / 255`, no mean/std), letterboxed as docs/MODEL.md §9.2
says (gray 114 padding). **Outputs**, raw (default): `box_raw: [B, 64, A]` (DFL logits, channel `side · 16 + bin`,
sides left, top, right, bottom) and `cls_raw: [B, nc, A]` (class logits), docs/MODEL.md §6.3. With `--decode`, one
output `detections: [B, A, 4 + nc]`: per anchor `x1, y1, x2, y2` in input pixels, then the `nc` class scores after
the sigmoid (docs/MODEL.md §6.4: the DFL softmax expectation and the anchor decode run in the graph). NMS is not in
the graph: apply docs/MODEL.md §9.3 (class-aware, `iou 0.7`) and map the boxes back through the letterbox. `A` is
the anchor count, `(H/8)(W/8) + (H/16)(W/16) + (H/32)(W/32)` (8400 at 640), level-major (stride 8 first), row-major
within a level; it is the symbolic `anchors` with `--dynamic`, where the decode builds its anchor grid from the
feature-map shapes. Weights are initializers under their `.folo` names (`backbone.stem.conv.weight`, ...).

`metadata_props` hold the model's `folo.*` metadata (`folo.arch`, `folo.config`, `folo.classes`, `folo.imgsz`,
`folo.dtype`, `folo.fused = "true"`, and `folo.metrics` / `folo.created` and unknown `folo.*` keys of the source),
`names` (`{"0": "<class 0>", ...}`) and `folo.onnx.*`: `outputs` (`raw` or `decoded`), `output_names`, `imgsz`
(`[H, W]`), `batch` (a number or `dynamic`), `dynamic`, `opset`, `strides`, `reg_max`.

With `--json`, one record: `{"path", "bytes", "format": "onnx", "dtype": "F32"|"F16", "fused", "source": "ema"|"model",
"arch", "classes", "opset", "imgsz", "batch", "dynamic", "decode", "inputs": [{"name", "dtype", "shape"}],
"outputs": [...], "seconds"}` (symbolic dimensions are strings in `shape`).

**Parity.** `crates/folo-cli/tests/onnx.rs` exports folo-n and folo-s (f32 and f16, raw and decoded, plus dynamic
graphs at other batches and rectangular sizes), runs them in onnxruntime with `bench/parity/onnx_check.py` (which
also runs `onnx.checker.check_model` with `full_check`) and compares with FOLO's forward pass element-wise,
`|onnx − folo| ≤ tol · max(|onnx|, |folo|, 1)`: f32 within `1e-4` (max) and `1e-5` (mean), f16 within `1e-1` (max)
and `5e-3` (mean), since an f16 graph rounds every activation to half precision while FOLO computes in f32. The test
needs a Python with `onnx`, `onnxruntime` and `numpy` (`FOLO_ONNX_PYTHON`, or the venv at `FOLO_ONNX_VENV`, default
`D:/folo-venvs/onnx`) and skips without one, unless `FOLO_ONNX_REQUIRED=1`, which CI's `onnx` job sets (docs/CI.md).

```sh
folo export --model best.folo --format onnx                            # best.onnx: f32, 1x3x640x640, box_raw + cls_raw
folo export --model best.folo --format onnx --decode --dynamic         # detections [batch, anchors, 4 + nc]
folo export --model best.folo --format onnx --half --imgsz 320         # best-f16.onnx
```

## Datasets

`PATH` is anything `folo_data::open` accepts (docs/DATA.md): a dataset folder, a `data.yaml` or a COCO instances
`.json`. Shared error codes: `path_not_found` (exit 3), `unknown_format` (nothing there looks like a dataset, exit 1),
`invalid_dataset` (a malformed `data.yaml` or COCO file, exit 1), `output_exists` (`synth` / `convert` into a folder
that exists and is not empty, exit 1), `usage` (bad flag values such as nine classes, exit 2).

### `folo dataset synth OUT_DIR`
Generates `--images` images with folo-data's shapes generator (deterministic in the flags) into `OUT_DIR/images/{train,val}`,
`OUT_DIR/labels/{train,val}` and `OUT_DIR/data.yaml`. `--classes` takes up to 8 distinct names: `circle`, `square`,
`triangle`, `star`, `ring`, `cross`, `hexagon` and `heart` get that shape, any other name the next shape no name has
claimed (`apple,square,pear` draws circles, squares and triangles); `data.yaml` keeps the names as given. Images
are square (`--imgsz` side) unless `--vary-aspect`. `--val-frac` (default 0.2) of the images go to `val`.

With `--json`: `{"event":"progress","done":N,"total":T}` records (at most 5 per second, always one with
`done == total`), then `{"event":"done","path","yaml","images","splits":{"train","val","test"},"boxes","classes",
"shapes","imgsz","seed","seconds"}`. Without it, progress is a `\r` line on an interactive stderr and a summary
goes to stdout.

### `folo dataset stats PATH`
`--json` prints one object: `{"name","path","root","format","classes","splits":{"train","val","test"},
"skipped_images","stats"}` where `stats` is exactly the `stats` object of `docs/API.md` (`boxes`, `empty_images`,
`broken_labels`, `class_counts`, `box_size_hist`, `boxes_per_image_hist`). The human form is a class table with
bars (top 40 classes by count when there are more) and both histograms as a sparkline plus one bar per bin.

### `folo dataset check PATH`
Reads every label line and every image (fully decoded; `--headers-only` reads just the headers, faster but blind
to truncated image data) and hashes file contents. Each issue has a kind and a severity:

| kind | severity | what |
|---|---|---|
| `broken_label` | error | a line that is not `class cx cy w h` or a polygon of finite numbers; an unreadable or non-UTF-8 label file |
| `class_overflow` | error | a class id that is negative, fractional, `>= nc` (when names are declared) or `>= 65536` |
| `out_of_range_box` | error | coordinates clearly outside `[0, 1]` (pixels, not normalised) or a negative size |
| `corrupt_image` | error | an image that cannot be read or decoded, or a JPEG cut off mid-scan (no end-of-image marker after the scan data; decoders fill the missing rows with grey, so only the full check sees it) |
| `missing_image` | error | an image named by an image list or COCO JSON that does not exist |
| `clipped_box` | warning | a box up to 0.01 outside the image (the loader clips it) |
| `zero_size_box` | warning | a box with no area inside the image (the loader drops it) |
| `duplicate_label` | warning | the same box twice in one file |
| `duplicate_image` | warning | an image with the same bytes as an earlier one (128-bit content hash) |
| `other` | warning | a split path that does not exist, a COCO image whose declared width / height differ from the file, ... |
| `empty_image` | info | an image without boxes (a background image; never makes the check fail) |

Errors make the loader skip the image (except a truncated JPEG, which decodes with grey rows: fix or remove it);
warnings are fixed up on load. `--min-severity warning` (or `error`) lists only the more serious issues in
`--json` (`counts` still covers all; a clean dataset with background images then prints no issues). `--json`
prints one report:
`{"path","root","format","classes","images","splits","clean","counts":{"error","warning","info","by_kind":{kind: n}},
"issues":[{"kind","severity","file","line","message","fix"?}],"seconds"}`. `file` is relative to `root` with `/`
separators; `line` is 1-based for label-line issues and `null` otherwise. Without `fix`, each issue is exactly the
object `GET /api/datasets/{id}/issues` returns. `--fix-dry-run` adds `fix` to each issue (what a
cleanup would do, e.g. `delete line 3`, `clip the box on line 2 to the image`, `remove this copy (and its label
file); keep images/train/a.jpg`) and prints it under each issue; nothing is ever modified.

Exit code 0 when `clean` (no errors or warnings). Otherwise the report is followed by the error envelope with code
`dataset_issues` and the exit code is 1.

### `folo dataset convert IN OUT --to yolo|coco`
Reads `IN` with folo-data and writes `OUT` (which must not exist or be empty): images are copied byte for byte to
`OUT/images/<split>/` (file stems made unique within a split, `a.jpg` + `a.png` → `a.jpg`, `a_1.png`; a name
without a readable extension, such as a COCO `file_name` of `a.bin`, gets one from the image content), plus
`labels/<split>/*.txt` and `data.yaml` for YOLO or `annotations/instances_<split>.json` (1-based category ids, pixel
`bbox`) for COCO. Images the loader skips (corrupt, missing, broken labels) are left out and counted in
`skipped_images`. An input with no readable images, or YOLO output from a test-only dataset (the YOLO reader needs
a train or val split), fails with `invalid_dataset` (exit 1) before anything is written.
`--json`: `{"input","output","from","to","classes","splits","boxes","skipped_images","seconds"}`.

## Adding a command
1. Add the clap `Args` struct in `crates/folo-cli/src/cli.rs` with doc comments on every field (they become the help
   text and `describe` output) and accurate defaults.
2. Return `Result<(), CliError>`; pick the exit code from the table above. Emit machine records with `Out::record`
   and human text with `Out::human`, never `println!` directly.
3. Add a test in `crates/folo-cli/tests/cli.rs` that runs it with `--json` and checks stdout parses line by line.
