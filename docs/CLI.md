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
{"error":{"code":"not_implemented","message":"`folo bench` is not implemented yet","hint":"..."}}
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
| 4 | `not_implemented`, `not_supported` | the command exists but is not implemented yet; `not_supported`: valid, but this machine cannot run it (`train --amp` without GPU tensor cores) |

## Commands

| command | status | what it does |
|---|---|---|
| `serve [--mock] [--host 127.0.0.1] [--port 7878] [--web-dir DIR] [--data-dir DIR] [--device auto] [--precision f32] [--tensor-cores] [--cors-any] [--token SECRET]` | works | Runs the HTTP API (`docs/API.md`) and serves the built Studio with an SPA fallback: from `--web-dir` when given, else the Studio embedded in the binary (`embed-web` builds), else `web/dist`. The `listening` record says which (`web_source`: `dir` or `embedded`, `web_dir`, `web_ready`). `--mock` serves deterministic mock data. `--port 0` picks a free port. `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`) holds the dataset registry, index and thumbnail caches, generated synthetic datasets, training runs, models and exports (see `docs/API.md`, Real mode). `--device` (env `FOLO_DEVICE`: `auto`, `cpu`, `gpu:<n>`) is where `/api/predict` and `/eval` run; each training run has its own `device`. `--precision` (env `FOLO_PRECISION`: `f32`, `f16`) is the precision of `/api/predict` and `/api/val` (docs/API.md); `f16` falls back to f32, with a warning in the log, where the device has no f16 compute. `--tensor-cores` (env `FOLO_GPU_TENSOR_CORES=1`) runs that f16 inference on the GPU's tensor cores where it has them (docs/GPU.md, Tensor cores), the regular kernels elsewhere; training runs with `"amp": true` need it (`POST /api/runs` answers `not_supported` without). The published weights pulled with `folo weights pull` list as models whose id is the published name (see [Weights](#weights)). A finished run gives its GPU memory back to the driver, and so does inference after 30 s idle (`/api/system`'s `gpu_pool`), so the server does not hold gigabytes between jobs. Requests that change something or fail are logged at `info`; successful reads (Studio polls `/api/runs` every 2 s) at `debug` (`RUST_LOG=folo_server=debug` shows them). `--token` (env `FOLO_TOKEN`; empty is a usage error) requires `Authorization: Bearer <token>` on every `/api` request but `GET /api/health` (`401 unauthorized` without; docs/CLOUD.md, Securing a server): use it whenever `--host` is not loopback, which the server otherwise warns about once in its log. The token is never logged or printed; the `listening` record's `token` says whether one is set. Ctrl+C shuts down gracefully: open SSE streams are closed, then an active training run stops after its current batch (keeping its last epoch's checkpoint, status `stopped`), and upload sessions in progress are dropped. |
| `describe [COMMAND...] [--no-mcp-tools]` | works | `folo describe train` (or `dataset check`) prints just that command (`path`, `about`, `args`, `subcommands`); `--no-mcp-tools` leaves out the MCP tool schemas, which are most of the full output. Prints the CLI surface as JSON: `global_args`, `conventions` (exit codes, error shape) and `commands` (recursive, each with `path`, `about`, `args` and `subcommands`). Each argument has `name`, `long`, `short`, `positional`, `type` (`string`, `path`, `integer`, `number`, `boolean`, `enum`, `count`), `value_name`, `required`, `multiple`, `default`, `possible_values`, `env` and `help`. Pretty-printed without `--json`, one line with it. |
| `doctor [--web-dir web/dist] [--tensor-cores]` | works | Version (`version`, `git_sha`, `build_date`), OS, CPU name and threads, memory, every wgpu adapter (`gpus`: name, backend, type, driver, `shader-f16`, `tensor_cores` (16×16×16 f16 → f32 cooperative matrices), buffer and workgroup limits), the adapter FOLO picks (`gpu_selected`, honouring `FOLO_GPU_ADAPTER` / `FOLO_GPU_BACKEND`; its `tensor_cores` is `on` with `FOLO_GPU_TENSOR_CORES=1` or `--tensor-cores`, else why they are off), what runs with `device: auto` can use (`capabilities`: `device`, `shader_f16`, `tensor_cores`, `amp` with the reason when unavailable; the same object `folo serve` reports in `/api/system`, docs/API.md, given the same `--tensor-cores`; the human `amp` line says when the GPU has tensor cores that only were not asked for: `folo train --amp` asks for them itself, `folo serve` and `folo mcp` need `--tensor-cores`), whether the web app is built (`web_dist`: `path`, `exists`, `index_html`, `folo_wasm`), whether one is embedded (`web_embedded`: `present`, `files`, `folo_wasm`), which one `folo serve` would use (`web_source`) and whether that one includes folo-wasm (`web_folo_wasm`: its Playground can run models in the browser, docs/WASM.md; release builds always do). |
| `train --data D [--format auto\|yolo\|coco] [--model folo-n \| --weights W] [--epochs 100] [--batch 16] [--imgsz 640] [--device auto] [--workers N] [--optimizer auto\|sgd\|adamw\|musgd] [--lr0 0.01] [--lrf 0.01] [--cos-lr] [--warmup-epochs 3] [--patience 100] [--amp] [--head o2m\|o2o] [augmentation flags] [--seed 0] [--set KEY=VALUE]... [--project runs] [--name NAME] [--resume CKPT] [--json-events all\|epoch]` | works | Trains a detector (docs/MODEL.md §8). Arguments mirror `TrainConfig` in `docs/API.md` (the server builds this command from a run's config: `GET /api/runs/{id}/cli`); `--format` reads `--data` as YOLO or COCO when a folder has both (default `auto`: detect); `--workers` (default: the logical CPU count, at most 16; also for `val`) sets the loader threads, which never change the batches; `--hsv` takes `h,s,v` or three values; `--arch` is an alias of `--model`, which also takes a `.folo` file (or a published name) to fine-tune from. `--weights W` (a `.folo` file or a published name such as `folo-n-coco`, see [Weights](#weights)) starts from those weights and takes their architecture (with an explicit `--model` it is a usage error); tensors whose shape or class list differ (the class outputs, for another dataset) keep their fresh initialisation, and the run logs `transferred N of M tensors`. Runs go to `<project>/<name>/` (default name `<dataset>-<arch suffix>`, numbered when it exists): `config.json`, `metrics.jsonl` (every event, every step), `weights/last.folo` and `weights/best.folo` (full checkpoints: EMA, optimizer, schedule), `samples/epoch_NNN.json`. Live view: epoch bar, losses, img/s, ETA and one table row per epoch (P, R, mAP50, mAP50-95). `--json` prints every event as a record: `{"event":"status"|"step"|"epoch"|"log", ...}` with the API.md payload fields (`step`s throttled to about 10 per second, the last batch of each epoch always sent, so a slow run gets one per batch; `metrics.jsonl` has every step), then `{"event":"done", "status", "run_dir", "best", "best_metrics", "weights", "resolved"}` (`best_metrics` is API.md's `Epoch.val`, with `best_f1_conf`; `resolved` is API.md's `Run.resolved`: the optimizer, effective `lr0`, momentum / betas, scaled weight decay, accumulation and warmup the run trained with). `--json-events epoch` leaves the `step` records out (status, epoch, log and done only: about 1 KB per epoch instead of one record per batch; a 15-epoch run of 50 batches drops from about 140 KB to 15 KB). `--resume runs/<name>/weights/last.folo` continues bit-exactly with the stored settings: only `--workers`, `--device`, `--data`, `--format`, `--project` and `--json-events` may differ, any other flag given must repeat the run's value (a changed one is `invalid_config`, exit 2, naming the flag and both values), and a run that completed (all epochs, or stopped early) is `run_not_resumable` (exit 1) with the command that trains further (`--model <run>/weights/best.folo`), as MCP and `POST /api/runs` answer. Stopping: Ctrl+C, on Windows also Ctrl+Break (what a script can send to a child started in its own process group), or `folo stop <run_dir>` from another process stops after the current batch, keeps the last epoch's checkpoint and ends with `status: stopped` and the `done` record; a second interrupt exits at once. A new run that fails before its first checkpoint (a bad device, a GPU out of memory) removes its run folder. `--optimizer auto` (the default) picks MuSGD for long runs (over 10,000 optimizer steps; SGD before `auto` rule 3) and AdamW with its own `lr0` for short ones (docs/MODEL.md §8.2: at least 0.001 since rule 2; a run resumes with the rule it started with), scaling for the batch with gradient accumulation, so `--lr0` only matters with `--optimizer sgd`, `adamw` or `musgd` (MuSGD: SGD plus Ultralytics 8.4's Muon update on every conv weight, the head's class branches at 3x the LR; docs/MODEL.md §8.2); the run's `config.json` has what is used (`optimizer.type`, `optimizer.lr0`, `accumulate`, and all of it as `resolved`, API.md's `Run.resolved`; `train` holds the settings as given) and its first log line says the same. A run too short to learn (from scratch with fewer than 200 optimizer steps, or a warmup over half of the run; docs/MODEL.md §8.4) gets a `warning:` line right after the first log line (a `log` record with `level: warn` under `--json`; `warnings` in `config.json`). `--device auto` (the default; also for `val` and `predict`) uses the GPU when folo-gpu finds an adapter (the model, its optimizer state, EMA, gradient buffers and the detection loss all run there; batches are staged on a thread and uploaded as `u8`), else the CPU; `--device gpu:<n>` without a usable adapter is `device_not_found` (exit 3; the hint lists the devices this machine has). `--amp` trains in mixed precision (docs/MODEL.md §8.8: f16 activations, convolutions and matrix products on the GPU's tensor cores, f32 master weights, optimizer state, loss, BatchNorm statistics and EMA, dynamic loss scaling; about 1.25x (folo-n) to 1.6x (folo-s) the f32 training step and 40% less memory, docs/BENCHMARKS.md). It asks for tensor cores itself (as `--tensor-cores` does for `predict`), also when resuming a mixed-precision run; without them (the CPU, another GPU) the run is `not_supported` (exit 4) before anything is written. Each epoch logs the loss scale and the optimizer steps skipped for gradient overflow (the first few steps of a run usually are). `--head o2o` (alias `end2end`; `TrainConfig.head`; default: the `--weights` file's head, else `o2m`) adds the NMS-free one-to-one head and trains with dual assignment (docs/MODEL.md §6.6: the one-to-many loss plus the one-to-one loss, weighted from 0.8 / 0.2 to 0.1 / 0.9 over the run; the reported losses are the weighted sums); validation, `best.folo`, `val` and `predict` then use the one-to-one head without NMS. Fine-tuning such a model from `o2m` weights (`--weights`) starts the one-to-one branch as a copy of the one-to-many head. `--set KEY=VALUE` (repeatable) sets any other key of the run's `config.json` `train` object (docs/MODEL.md §8.1), nested keys dotted, after the other flags: e.g. `--set momentum=0.9`, `--set loss.cls_gain=1.0`, `--set loss.tal.topk=13`, `--set augment.degrees=5`, `--set auto_rule=1`; the value is read as JSON when it parses, else as a string. An unknown key or a value of the wrong type is `invalid_config` (exit 2, listing the keys there are), and so is `--set` with `--resume` (a resumed run keeps its settings). |
| `stop RUN_DIR` | works | Asks the `folo train` writing `RUN_DIR` (its `run_dir`) to stop after its current batch: writes `RUN_DIR/stop`, which the trainer polls every 200 ms and removes. Returns at once with `{"event":"stop_requested","run_dir","stop_file"}`; the training process then ends with `status: stopped`. A folder that is not a run is `run_not_found` (exit 3). The portable way for a script to stop training, on Windows too. |
| `val --model W --data D [--split val] [--batch 32] [--imgsz <model's>] [--conf 0.001] [--iou 0.7] [--head o2m\|o2o] [--device auto] [--half] [--tensor-cores] [--workers N] [--coco] [--save-json FILE]` | works | mAP@0.5, mAP@0.5:0.95, precision and recall (docs/METRICS.md) of the weights' EMA (fused) on a split, overall and per class. Only that split is opened and checked (`folo_data::open_splits`: COCO's val2017 without probing the 118 287 train2017 images, 26 s instead of 60–150 s for `--data coco`); its classes are the full dataset's. `W` is a `.folo` file or a published name (see [Weights](#weights)). One JSON object with `--json`: `{"model","data","split","imgsz","images","boxes","map50","map50_95","precision","recall","best_f1_conf","per_class":[{"class","name",...}],"device","dtype","head","seconds","coco"}` (`dtype`: `f32`, or `f16` with `--half` where the GPU computes in it; `head`: the branch evaluated, the model's unless `--head` says otherwise: `o2o` (top-k, no NMS, docs/MODEL.md §6.6) for a model trained with `--head o2o`, where `--head o2m` evaluates its one-to-many branch with NMS; `--head o2o` on an `o2m` model is `invalid_config`). On a COCO dataset, `--coco` adds the pycocotools-exact evaluation (docs/METRICS.md, Full COCOeval: original-image boxes, crowd regions, annotation areas) as `coco: {"ap","ap50","ap75","ap_small","ap_medium","ap_large","ar1","ar10","ar100","ar_small","ar_medium","ar_large","images","truths","crowd","detections","missing_images","annotations","per_class":[{"class","name","truths","ap","ap50"}]}` (`null` without `--coco`) and prints `COCOeval.summarize()`'s table; GT images the loader skipped count with no detections (`missing_images`). `--save-json FILE` writes the detections as a COCO results file (`[{"image_id","category_id","bbox":[x,y,w,h],"score"}]`, original-image pixels, full f32 precision) for pycocotools. Both need exactly one COCO annotation file for the split (usage error otherwise). `--half` (also for `predict`): half-precision inference, f16 weights and activations with f32 accumulation, on a GPU with `shader-f16` (an f16 export is uploaded as it is; other files are fused in f32 and converted); elsewhere it runs in f32 and prints a `note:` on stderr. On trained models f16 mAP is within 0.002 of f32 (docs/BENCHMARKS.md). `--tensor-cores` (also for `predict`; the same as `FOLO_GPU_TENSOR_CORES=1`) runs the `--half` model's convolutions and matrix products on the GPU's tensor cores (wgpu's experimental cooperative matrices, docs/GPU.md); where they are not available, or without `--half`, it prints a `note:` and runs the regular kernels. Same mAP bound. |
| `predict --model W SOURCES... [--conf 0.25] [--iou 0.7] [--max-det 300] [--head o2m\|o2o] [--imgsz <model's>] [--device auto] [--half] [--tensor-cores] [--save DIR]` | works | Inference on images or folders (recursive), letterboxed `rect` (docs/MODEL.md §9). One `{"event":"prediction", "file", "image", "boxes": [ScoredBox], "timing", "device", "dtype", "saved"}` record per image (`dtype` as for `val`) (an unreadable image gives an `error` record and is skipped), then `{"event":"done", ...}` with `mean_timing` over the images after the first and `first_timing` (the first image, which includes pipeline compilation and the first allocations; with one image the mean is that image's) (`timing` as docs/API.md's `/api/predict`: on a GPU each stage waits for its own device work, so `inference_ms` is the whole forward pass). `--save` writes annotated copies (boxes coloured by class, tagged with the class id and score, e.g. `0 0.84`). A model trained with `--head o2o` predicts end to end (its one-to-one head, the `--max-det` best anchors above `--conf`, no NMS, so `--iou` is unused); `--head o2m` uses its one-to-many branch with NMS instead. |
| `export --model W [--format folo-f16\|folo\|weights\|onnx] [--output FILE] [--imgsz N] [--batch N] [--dynamic] [--half] [--decode] [--max-det 300] [--head o2m\|o2o]` | works | Fused EMA weights (FORMAT.md export). `W` is a `.folo` file or a published name (see [Weights](#weights)). `folo-f16` (default) and `folo` write a `weights` file in f16 / f32; `weights` writes the EMA weights **unfused** in f32 with no optimizer state (`fused: false` in the record): a small file to fine-tune from (`train --weights`) or to publish (a fused input is a usage error); `onnx` writes an ONNX graph (opset 17) for onnxruntime, TensorRT, OpenVINO or the CoreML converters, see [ONNX export](#onnx-export) (`--imgsz`, `--batch`, `--dynamic`, `--half`, `--decode`, `--max-det` and `--head` are ONNX-only; all but `--max-det` are a usage error otherwise). Default output next to the input: `<name>-f16.folo` / `<name>-fused.folo` / `<name>-weights.folo` / `<name>.onnx` / `<name>-f16.onnx` (`--half`) / `<name>-decoded.onnx` (`--decode`) / `<name>-decoded-f16.onnx` (both), `-2`, `-3`, ... when taken: like MCP `export`, it never overwrites (an existing `--output` is `output_exists`, exit 1). A file that is not `.folo` weights is `invalid_weights` saying so. |
| `bench [--model folo-n] [--mode infer\|train] [--imgsz 640] [--batch 1] [--iters 100] [--warmup 10] [--device auto]` | planned (exit 4) | Speed benchmarks. |
| `dataset synth OUT_DIR [--images 1000] [--classes circle,square,triangle] [--imgsz 640] [--seed 0] [--val-frac 0.2] [--image-format jpeg\|png] [--vary-aspect]` | works | Writes a synthetic shapes dataset (YOLO layout). See [Datasets](#datasets). |
| `dataset stats PATH [--format auto\|yolo\|coco]` | works | Class counts, box sizes, boxes per image: a table with bar charts, or the `stats` object of `docs/API.md` with `--json`. |
| `dataset check PATH [--format ...] [--fix-dry-run] [--headers-only] [--min-severity error\|warning\|info]` | works | Every problem in a dataset, one issue per bad line / image. Exit 0 when clean, 1 with errors or warnings. |
| `dataset convert IN OUT --to yolo\|coco [--from auto\|yolo\|coco]` | works | Converts between YOLO and COCO; images are copied unchanged. |
| `weights list` | works | Every published weights name (see [Weights](#weights)), whether it is downloaded, and its cache path. One record per name with `--json`: `{"name","release","asset","path","cached","about"}`. |
| `weights pull NAME [--force] [--repo himent12/folo]` | works | Downloads a published weights file from its GitHub release with the GitHub CLI (`gh release download`; the repository is private, so `gh auth login` first), checks its SHA-256 against the release's `SHA256SUMS` and that it opens as a `.folo` file, then moves it into the cache. Already cached: nothing is downloaded unless `--force`. `--json`: `{"event":"done","name","path","downloaded","sha256","bytes"}`. Unknown name: `weights_not_found` (exit 3); `gh` missing or failing: `gh_failed`; bad hash: `checksum_mismatch` (exit 1, nothing installed). |
| `cloud providers\|connect\|connections\|check\|disconnect\|offers\|instances\|terminate [--data-dir DIR]` | works | Cloud compute on the data dir, with or without a server running: see [Cloud](#cloud). |
| `mcp [--data-dir DIR] [--mock] [--read-only] [--confine] [--server URL [--token SECRET]] [--device auto] [--tensor-cores]` | works | Model Context Protocol server over stdio (MCP 2026-07-28 and 2025-11-25), see `docs/MCP.md`. stdout carries protocol messages only (also with `--json`); logs go to stderr. By default it embeds the real store on `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`): datasets, training, models, `val`, `predict` and `export` on the FOLO engine, the same data `folo serve` uses. `--mock` serves mock data, `--server` forwards to a running `folo serve` (`http://` or `https://`; `--token`, env `FOLO_TOKEN`, is the token that server was started with), `--read-only` hides and rejects mutating tools, `--confine` only reads inside `--data-dir`, `--device` (env `FOLO_DEVICE`) is where inference and `val` run, `--tensor-cores` (env `FOLO_GPU_TENSOR_CORES=1`) asks for the GPU's tensor cores as `folo serve --tensor-cores` does (`train` with `"amp": true` needs them; without, it answers `not_supported` before starting anything). Published weights pulled with `folo weights pull` are models (`train { "weights": "folo-n-coco" }`). Exits 0 when the client closes stdin; the runs it started are stopped first (an active run after its current batch, keeping its last checkpoint). Recipes: `docs/AGENT_COOKBOOK.md`. |

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
| `--decode` | off | decode in the graph (below) instead of the raw head outputs (default name `<name>-decoded.onnx`) |
| `--max-det N` | 300 | `--decode` of an `o2o` model: detections per image, `K = min(N, A)` |
| `--head o2m\|o2o` | the model's | the head branch to export: an `o2o` model (trained with `--head o2o`) exports its one-to-one branch, or with `--head o2m` its one-to-many branch; the other branch's tensors are left out |

**Input** `images: [B, 3, H, W]`, RGB in `[0, 1]` (`u8 / 255`, no mean/std), letterboxed as docs/MODEL.md §9.2
says (gray 114 padding, and FOLO's resize: a box prefilter on 2× or larger reductions, then bilinear). Parity with
`folo predict` needs the same preprocessing: another resize changes the input pixels, so scores, and with them boxes
and which detections pass `conf`, can differ (on R6 frames PIL's resize kept the boxes but moved scores by up to 0.15). **Outputs**, raw (default): `box_raw: [B, 64, A]` (DFL logits, channel `side · 16 + bin`,
sides left, top, right, bottom) and `cls_raw: [B, nc, A]` (class logits), docs/MODEL.md §6.3. With `--decode`, one
output `detections: [B, A, 4 + nc]`: per anchor `x1, y1, x2, y2` in input pixels, then the `nc` class scores after
the sigmoid (docs/MODEL.md §6.4: the DFL softmax expectation and the anchor decode run in the graph). NMS is not in
the graph: apply docs/MODEL.md §9.3 (class-aware, `iou 0.7`) and map the boxes back through the letterbox. `A` is
the anchor count, `(H/8)(W/8) + (H/16)(W/16) + (H/32)(W/32)` (8400 at 640), level-major (stride 8 first), row-major
within a level; it is the symbolic `anchors` with `--dynamic`, where the decode builds its anchor grid from the
feature-map shapes. Weights are initializers under their `.folo` names (`backbone.stem.conv.weight`, ...).

**End to end (`o2o` head).** The raw outputs of an `o2o` export are its one-to-one branch's, laid out the same way.
With `--decode` the graph also does the NMS-free selection of docs/MODEL.md §6.6, so its one output
`detections: [B, K, 6]` (`K = min(--max-det, A)`, symbolic `max_det` with `--dynamic`) is final: rows
`x1, y1, x2, y2, score, class` (class as a float) sorted by descending score, the `K` anchors with the highest
max-class score, then the `K` best (anchor, class) pairs among them (`ReduceMax`, `TopK`, `GatherND`). Apply your
`conf` (keep rows with `score > conf`) and map the boxes back through the letterbox; there is no NMS to run. An
`--half` graph does the selection in f32 and its `detections` stay f32 (an f16 class id is exact only up to 2048).

`metadata_props` hold the model's `folo.*` metadata (`folo.arch`, `folo.config`, `folo.classes`, `folo.imgsz`,
`folo.dtype`, `folo.fused = "true"`, and `folo.metrics` / `folo.created` and unknown `folo.*` keys of the source),
`names` (`{"0": "<class 0>", ...}`) and `folo.onnx.*`: `outputs` (`raw`, `decoded`, or `end2end` for a decoded `o2o`
head), `head` (`o2m` or `o2o`, the exported branch), `max_det` (end to end only), `output_names`, `imgsz`
(`[H, W]`), `batch` (a number or `dynamic`), `dynamic`, `opset`, `strides`, `reg_max`.

With `--json`, one record: `{"path", "bytes", "format": "onnx", "dtype": "f32"|"f16", "fused", "source": "ema"|"model",
"arch", "classes", "opset", "imgsz", "batch", "dynamic", "decode", "head", "inputs": [{"name", "dtype", "shape"}],
"outputs": [...], "seconds"}` (symbolic dimensions are strings in `shape`).

**Parity.** `crates/folo-cli/tests/onnx.rs` exports folo-n and folo-s (f32 and f16, raw and decoded, plus dynamic
graphs at other batches and rectangular sizes) and an `o2o` folo-n (raw, end to end in f32 and f16, dynamic, a small
`--max-det`, and its `--head o2m` branch), runs them in onnxruntime with `bench/parity/onnx_check.py` (which
also runs `onnx.checker.check_model` with `full_check`) and compares with FOLO's forward pass element-wise,
`|onnx − folo| ≤ tol · max(|onnx|, |folo|, 1)`: f32 within `1e-4` (max) and `1e-5` (mean), f16 within `1e-1` (max)
and `5e-3` (mean), since an f16 graph rounds every activation to half precision while FOLO computes in f32.
End-to-end outputs are compared row by row, so that near-equal scores may come out in either order (the score
columns must agree, every ONNX row must be one of FOLO's rows, the same class, score and box, and every FOLO row must be in the ONNX output unless it ties at the cutoff; `bench/parity/test_onnx_check.py` tests the comparison). The test
needs a Python with `onnx`, `onnxruntime` and `numpy` (`FOLO_ONNX_PYTHON`, or the venv at `FOLO_ONNX_VENV`, default
`D:/folo-venvs/onnx`) and skips without one, unless `FOLO_ONNX_REQUIRED=1`, which CI's `onnx` job sets (docs/CI.md).

```sh
folo export --model best.folo --format onnx                            # best.onnx: f32, 1x3x640x640, box_raw + cls_raw
folo export --model best.folo --format onnx --decode --dynamic         # detections [batch, anchors, 4 + nc]
folo export --model best.folo --format onnx --half --imgsz 320         # best-f16.onnx
folo export --model o2o.folo --format onnx --decode                    # o2o model: detections [1, 300, 6], no NMS
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

## Weights

Published pretrained weights are assets of GitHub releases of this repository, next to a `SHA256SUMS` asset; weights
are never committed to git. `folo weights pull NAME` downloads one into the cache, `$FOLO_WEIGHTS_DIR` or else
`<FOLO_HOME or ~/.folo>/weights/`. Wherever a command takes weights (`train --weights` / `--model`, `val --model`,
`predict --model`, `export --model`), a published name resolves to its cached file (an existing file of that name
wins); a name that is not downloaded fails with `weights_not_found` (exit 3) and the hint `folo weights pull NAME`.
`folo serve` and `folo mcp` (embedded) list the downloaded ones as models too, with the published name as the model
id (`source: published`, docs/API.md Models): Studio's "Start from", `POST /api/runs` with `"weights": "folo-n-coco"`
and MCP `train { weights }`, `predict`, `val` and `export` take the name as they take any model id. A file pulled
while the server runs lists at once; the server reads the same cache (`$FOLO_WEIGHTS_DIR` or `<FOLO_HOME or
~/.folo>/weights`) whatever its `--data-dir`.

| name | release | what |
|---|---|---|
| `folo-n-coco` | `weights-v1` | folo-n trained from scratch on COCO train2017 at 640 px (EMA, f32, unfused): fine-tune or predict. Budget and val2017 results: examples/coco/README.md |
| `folo-n-coco-f16` | `weights-v1` | the same, fused and cast to f16 (`folo export`): inference only |
| `folo-n-coco-v2` | `weights-v2` | interim: folo-n from scratch on COCO, epoch 62 of a paused 300-epoch run (EMA, f32, unfused), val2017 AP 0.317 against v1's 0.314. examples/coco/README.md, v2 |
| `folo-n-coco-v2-f16` | `weights-v2` | the same, fused and cast to f16 (`folo export`): inference only |

`weights-v1` and `weights-v2` are draft releases for now. `gh release download` finds draft releases by tag for
accounts with write access to the repository, so `folo weights pull` already works for them. Everyone else needs the
release published first. Without `folo`, `gh release download weights-v1 --repo himent12/folo --pattern 'folo-n-coco*' --pattern SHA256SUMS`
(or `weights-v2` with `'folo-n-coco-v2*'`) then `sha256sum -c --ignore-missing SHA256SUMS` does the same by hand.

## Cloud

`folo cloud` looks after the cloud compute of a data dir (docs/CLOUD.md): the connections in
`cloud/connections.json` and the instance ledger in `cloud/instances/`. It works on the files through
`folo_server::cloud`, so it needs no server, and a running `folo serve` on the same data dir sees its changes at
once (both take the data dir's cloud lock around every change). Every command takes `--data-dir` (as `folo serve`:
`$FOLO_DATA_DIR`, else `$FOLO_HOME` or `~/.folo`) and `--json`, which prints one record per item: the API's objects
(docs/API.md, *Cloud compute*). Without `--json` each prints a small aligned table. Errors are the API's codes with
their exit codes (`connection_not_found` and `instance_not_found` exit 3, `cloud_auth_failed`, `cloud_unavailable`
and `connection_in_use` exit 1, `invalid_body` exit 2).

| command | what it does |
|---|---|
| `cloud providers` | The providers and the fields a connection to each takes (`[key]` optional, `key (secret)` a credential). |
| `cloud connect PROVIDER [--name NAME] [--set KEY=VALUE]... [--secret KEY=env:VAR \| KEY=-]...` | Checks the values with the provider, then stores the connection (`POST /api/cloud/connections`). Values that are not secret go in `--set`; secrets come from an environment variable (`--secret api_key=env:RUNPOD_API_KEY`) or from one line of stdin (`--secret token=-`, once per command), never from the command line itself, which other processes and the shell history can read: a literal, or a secret field in `--set`, is a `usage` error (exit 2) that says so. The record shows each secret as `••••` and its last 4 characters (just `••••` below 8 characters); a `note` (an unencrypted `http://` URL) is printed too. |
| `cloud connections` | Every connection with what its last check found. |
| `cloud check ID` | Checks a connection's stored credentials again; a failed check is the record's `status: error` (exit 0). |
| `cloud disconnect ID` | Forgets a connection (`{"event":"disconnected","id"}`); `connection_in_use` while it has live instances or active runs. |
| `cloud offers CONNECTION [--refresh]` | What the connection can run on now, cheapest first; answers of the last minute are reused unless `--refresh`. |
| `cloud instances` | The instances FOLO launched: alive first, then those that ended in the last 7 days, newest first, each with its cost so far. |
| `cloud terminate ID` | Terminates an instance now, through the same path as `DELETE /api/cloud/instances/{id}`: `terminated` when the provider confirms, else `terminating` with the error (a server's reaper retries it). An instance that is gone answers its record. A run still using it fails with `cloud_instance_lost`. |

```sh
RUNPOD_API_KEY=rpa_... folo cloud connect runpod --secret api_key=env:RUNPOD_API_KEY
folo cloud connect remote --set url=https://gpu-box.example.com:7878 --secret token=- < token.txt
folo cloud offers cc_4d2a --json
folo cloud instances
```

## Adding a command
1. Add the clap `Args` struct in `crates/folo-cli/src/cli.rs` with doc comments on every field (they become the help
   text and `describe` output) and accurate defaults.
2. Return `Result<(), CliError>`; pick the exit code from the table above. Emit machine records with `Out::record`
   and human text with `Out::human`, never `println!` directly.
3. Add a test in `crates/folo-cli/tests/cli.rs` that runs it with `--json` and checks stdout parses line by line.
