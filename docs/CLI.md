# The `folo` CLI

`folo` is the single binary for training, evaluation, inference, export, the Studio server and the MCP server. It is
built for people and for agents: every command follows the conventions below, and `folo describe --json` prints the
whole command surface (generated from the clap command tree, so it never drifts from the code).

```sh
cargo run -p folo-cli -- serve --mock     # mock Studio API on http://127.0.0.1:7878
cargo folo --help                         # alias for: cargo run --release -p folo-cli --
folo describe --json                      # every command, argument, type, default and help string
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

Long-running commands emit one record per event, each with an `"event"` field (e.g. `folo serve --json` prints
`{"event":"listening","url":"http://127.0.0.1:7878",...}` once bound and `{"event":"stopped"}` on shutdown).
One-shot commands print one object.

### Errors
A failing command prints its error as the last stdout line in `--json` mode, and always in human form on stderr:

```json
{"error":{"code":"not_implemented","message":"`folo train` is not implemented yet","hint":"it waits on ..."}}
```

`code` is a stable snake_case identifier (the same vocabulary as the HTTP API's error codes where they overlap),
`message` is for humans, `hint` says what to do next and is `null` when there is nothing to add.

### Exit codes
| code | name | meaning |
|---|---|---|
| 0 | `ok` | success (also `--help` / `--version`) |
| 1 | `runtime_error` | the command ran and failed (IO, bind failure, bad data, ...) |
| 2 | `usage_error` | invalid arguments or flags (error code `usage`) |
| 3 | `not_found` | a named file, dataset, model or run does not exist |
| 4 | `not_implemented` | the command exists but is not implemented yet |

## Commands

| command | status | what it does |
|---|---|---|
| `serve [--mock] [--host 127.0.0.1] [--port 7878] [--web-dir DIR] [--data-dir DIR] [--device auto] [--cors-any]` | works | Runs the HTTP API (`docs/API.md`) and serves the built Studio with an SPA fallback: from `--web-dir` when given, else the Studio embedded in the binary (`embed-web` builds), else `web/dist`. The `listening` record says which (`web_source`: `dir` or `embedded`, `web_dir`, `web_ready`). `--mock` serves deterministic mock data. `--port 0` picks a free port. `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`) holds the dataset registry, index and thumbnail caches, generated synthetic datasets, training runs, models and exports (see `docs/API.md`, Real mode). `--device` (env `FOLO_DEVICE`: `auto`, `cpu`, `gpu:<n>`) is where `/api/predict` and `/eval` run; each training run has its own `device`. Ctrl+C shuts down gracefully: open SSE streams are closed, then an active training run stops after its current batch (keeping its last epoch's checkpoint, status `stopped`). |
| `describe` | works | Prints the CLI surface as JSON: `global_args`, `conventions` (exit codes, error shape) and `commands` (recursive, each with `path`, `about`, `args` and `subcommands`). Each argument has `name`, `long`, `short`, `positional`, `type` (`string`, `path`, `integer`, `number`, `boolean`, `enum`, `count`), `value_name`, `required`, `multiple`, `default`, `possible_values`, `env` and `help`. Pretty-printed without `--json`, one line with it. |
| `doctor [--web-dir web/dist]` | works | Version (`version`, `git_sha`, `build_date`), OS, CPU name and threads, memory, every wgpu adapter (`gpus`: name, backend, type, driver, `shader-f16`, buffer and workgroup limits), the adapter FOLO picks (`gpu_selected`, honouring `FOLO_GPU_ADAPTER` / `FOLO_GPU_BACKEND`), whether the web app is built (`web_dist`), whether one is embedded (`web_embedded`: `present`, `files`) and which one `folo serve` would use (`web_source`). |
| `train --data D [--model folo-n] [--epochs 100] [--batch 16] [--imgsz 640] [--device auto] [--workers N] [--optimizer auto] [--lr0 0.01] [--lrf 0.01] [--cos-lr] [--warmup-epochs 3] [--patience 100] [augmentation flags] [--seed 0] [--project runs] [--name NAME] [--resume CKPT]` | works (CPU) | Trains a detector (docs/MODEL.md §8). Arguments mirror `TrainConfig` in `docs/API.md`; `--arch` is an alias of `--model`, which also takes a `.folo` file to fine-tune from. Runs go to `<project>/<name>/` (default name `<dataset>-<arch suffix>`, numbered when it exists): `config.json`, `metrics.jsonl` (every event, every step), `weights/last.folo` and `weights/best.folo` (full checkpoints: EMA, optimizer, schedule), `samples/epoch_NNN.json`. Live view: epoch bar, losses, img/s, ETA and one table row per epoch (P, R, mAP50, mAP50-95). `--json` prints every event as a record: `{"event":"status"|"step"|"epoch"|"log", ...}` with the API.md payload fields (steps about 10 per second), then `{"event":"done", "status", "run_dir", "best", "best_metrics", "weights"}`. `--resume runs/<name>/weights/last.folo` continues bit-exactly with the stored settings (only `--workers` / `--device` may change; the dataset is reopened from the checkpoint unless `--data` is given). Ctrl+C stops after the current batch and keeps the last epoch's checkpoint. `--device auto` (the default; also for `val` and `predict`) uses the GPU when folo-gpu finds an adapter (the model, its optimizer state and EMA move there with `Module::to_device`), else the CPU; `--device gpu:<n>` without a usable adapter exits 4. |
| `val --model W --data D [--split val] [--batch 32] [--imgsz <model's>] [--conf 0.001] [--iou 0.7] [--workers N]` | works | mAP@0.5, mAP@0.5:0.95, precision and recall (docs/METRICS.md) of the weights' EMA (fused) on a split, overall and per class. One JSON object with `--json`. |
| `predict --model W SOURCES... [--conf 0.25] [--iou 0.7] [--max-det 300] [--imgsz <model's>] [--save DIR]` | works | Inference on images or folders (recursive), letterboxed `rect` (docs/MODEL.md §9). One `{"event":"prediction", "file", "image", "boxes": [ScoredBox], "timing", "device", "saved"}` record per image (an unreadable image gives an `error` record and is skipped), then `{"event":"done", ...}` with mean timings. `--save` writes annotated copies (boxes coloured by class, tagged `class score`). |
| `export --model W [--format folo-f16\|folo\|onnx] [--output FILE]` | works (`.folo`) | Fused EMA weights as a `weights` file (FORMAT.md export): `folo-f16` (default) casts to f16, `folo` keeps f32. Default output `<name>-f16.folo` / `<name>-fused.folo` next to the input. ONNX exits 4. |
| `bench [--model folo-n] [--mode infer\|train] [--imgsz 640] [--batch 1] [--iters 100] [--warmup 10]` | planned (exit 4) | Speed benchmarks. |
| `dataset synth OUT_DIR [--images 1000] [--classes circle,square,triangle] [--imgsz 640] [--seed 0] [--val-frac 0.2] [--image-format jpeg\|png] [--vary-aspect]` | works | Writes a synthetic shapes dataset (YOLO layout). See [Datasets](#datasets). |
| `dataset stats PATH [--format auto\|yolo\|coco]` | works | Class counts, box sizes, boxes per image: a table with bar charts, or the `stats` object of `docs/API.md` with `--json`. |
| `dataset check PATH [--format ...] [--fix-dry-run] [--headers-only]` | works | Every problem in a dataset, one issue per bad line / image. Exit 0 when clean, 1 with errors or warnings. |
| `dataset convert IN OUT --to yolo\|coco [--from auto\|yolo\|coco]` | works | Converts between YOLO and COCO; images are copied unchanged. |
| `mcp [--data-dir DIR] [--mock] [--read-only] [--confine] [--server URL] [--device auto]` | works | Model Context Protocol server over stdio (MCP 2026-07-28 and 2025-11-25), see `docs/MCP.md`. stdout carries protocol messages only (also with `--json`); logs go to stderr. By default it embeds the real store on `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`): datasets, training, models, `val`, `predict` and `export` on the FOLO engine, the same data `folo serve` uses. `--mock` serves mock data, `--server` forwards to a running `folo serve`, `--read-only` hides and rejects mutating tools, `--confine` only reads inside `--data-dir`, `--device` (env `FOLO_DEVICE`) is where inference and `val` run. Exits 0 when the client closes stdin; the runs it started are stopped first (an active run after its current batch, keeping its last checkpoint). Recipes: `docs/AGENT_COOKBOOK.md`. |

`folo describe --json` is the authoritative list of arguments and defaults. Its `mcp_tools` array lists every
`folo mcp` tool (`name`, `description`, `mutating`, `inputSchema`, `outputSchema`) with the schemas `tools/list`
publishes.

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
| `corrupt_image` | error | an image that cannot be read or decoded |
| `missing_image` | error | an image named by an image list or COCO JSON that does not exist |
| `clipped_box` | warning | a box up to 0.01 outside the image (the loader clips it) |
| `zero_size_box` | warning | a box with no area inside the image (the loader drops it) |
| `duplicate_label` | warning | the same box twice in one file |
| `duplicate_image` | warning | an image with the same bytes as an earlier one (128-bit content hash) |
| `other` | warning | a split path that does not exist, a COCO image whose declared width / height differ from the file, ... |
| `empty_image` | info | an image without boxes (a background image; never makes the check fail) |

Errors make the loader skip the image; warnings are fixed up on load. `--json` prints one report:
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
