# The `folo` CLI

`folo` is the single binary for training, evaluation, inference, export, the Studio server and the MCP server. It is
built for people and for agents: every command follows the conventions below, and `folo describe --json` prints the
whole command surface (generated from the clap command tree, so it never drifts from the code).

```sh
cargo run -p folo-cli -- serve --mock     # mock Studio API on http://127.0.0.1:7878
cargo folo --help                         # alias for: cargo run --release -p folo-cli --
folo describe --json                      # every command, argument, type, default and help string
folo describe train --json                # one command (about 7 KB instead of the whole surface)
folo doctor                               # version, OS, CPU/SIMD, memory, paths, GPU adapters, web app build
folo --version                            # folo 0.1.0 (1a2b3c4d5e6f 2026-11-08): version, commit, build date
```

`folo --version` prints the workspace version (`[workspace.package] version` in the root `Cargo.toml`, the single
source of truth), the 12-character commit and the UTC build date, baked in by `crates/folo-cli/build.rs`. Outside a
git checkout the commit is `unknown` unless `FOLO_GIT_SHA` is set; `SOURCE_DATE_EPOCH` pins the date for
reproducible builds.

`--release` (thin LTO, one codegen unit) is what releases and benchmark numbers use, and it takes minutes to build,
also after a small edit. For iterating on the CLI or the server, `cargo build --profile fast -p folo-cli` (the
`fast` profile in the root `Cargo.toml`: no cross-crate LTO, 16 codegen units, incremental) builds `target/fast/folo` in about
half the time from clean and in seconds after an edit, with CPU inference speed within noise of `--release`.

### The embedded Studio (`embed-web`)
Release binaries are built with `cargo build --release -p folo-cli --features embed-web`, which compiles the built
Studio into the binary (about 1 MB), so `folo serve` works anywhere. The feature needs a built `web/dist`
(`npm ci && npm run build` in `web/`), or `FOLO_WEB_DIST` pointing at one (relative paths are resolved against the
repository root); the build fails with a hint otherwise. Because of that, `cargo test --workspace --all-features`
also needs a built web app. Without the feature, `folo serve` serves `web/dist` from disk as before.

## Conventions (every command)

### Caches
Every command keeps two caches under `$FOLO_CACHE_DIR`, else `<FOLO_HOME or ~/.folo>/cache`: compiled GPU pipelines
and the lists of kernels each kind of run used, which the next such run compiles on worker threads before its first
step (`gpu/`, docs/GPU.md Pipeline cache) and dataset scans (`scan/`, docs/DATA.md Scan cache). They only make a second
start faster (COCO `folo train` reaches its first step in seconds instead of a minute) and do not change a result
as long as every change to a dataset file changes its length or modification time (the dataset is the one found
when the open starts); a missing, corrupt or stale entry is rebuilt silently. Both are bounded: the GPU cache is one file per adapter, at most 256 MB (a larger one is neither written
nor loaded), plus at most 64 kernel lists of at most 1 MB; the scan cache keeps one file per large folder or COCO annotation set (each at most 1 GiB, COCO
train + val take 30 MB) and at most 2 GiB in all, deleting the least recently written past that and anything not
written for 60 days. `FOLO_NO_CACHE=1` turns both off; deleting the folder is always safe.

### `--json`
A global flag. With it, **stdout carries only JSON, one object per line** (JSON Lines), and all human text (progress,
tables, hints, logs) goes to **stderr** (except `--help` and `--version`, which still print plain text to stdout and
exit 0). Without it, human text goes to stdout and errors to stderr. Logs from the
server (`tracing`) always go to stderr; tune them with `RUST_LOG` (default `info`).

Paths in records use `/` separators (also on Windows, which accepts them; `serve`, `doctor` and `predict` keep a
`\\?\` path as it is where it would name something else without that prefix, such as `\\?\Volume{GUID}\`, or where
the rest of it is 248 UTF-16 units or longer; off Windows a name that starts with `\\?\` keeps its spelling), and
floats are written as they were stored: `0.01`, not the `0.009999999776482582` an f32 widened to f64 would print.

Long-running commands emit one record per event, each with an `"event"` field (e.g. `folo serve --json` prints
`{"event":"listening","url":"http://127.0.0.1:7878","bind":"127.0.0.1:7878","api":"http://127.0.0.1:7878/api",...}` once bound and
`{"event":"stopped"}` on shutdown). One-shot commands print one object.

### Errors
A failing command prints its error as the last stdout line in `--json` mode, and always in human form on stderr
(`folo mcp` excepted: its stdout is the protocol's, so its errors, bad flags included, are on stderr only):

```json
{"error":{"code":"bench_not_found","message":"benchmark 'bn_3f9a' does not exist","hint":"`folo bench list` lists the benchmarks"}}
```

`code` is a stable snake_case identifier, `message` is for humans (engine errors are cut to 500 characters), `hint`
says what to do next and is `null` when there is nothing to add. For argument or flag parse failures (`code: "usage"`,
exit 2), `--json` preserves clap's suggestions (`did you mean ...?` for mistyped flags, subcommands or enum values) in `hint` and scopes the help pointer to the matched subcommand (e.g. `` run `folo train --help` or `folo describe train --json` ``).
The codes are one vocabulary with the HTTP API and `folo mcp`: `folo_api::codes` lists each with its HTTP status and
exit code, and the same failure has the same code everywhere (a bad `--model` arch is `invalid_config`, an empty split
`invalid_dataset`, a `--device gpu:7` this machine lacks `device_not_found` with the devices it has in the hint, a GPU
out of memory `gpu_out_of_memory` with "lower --batch or --imgsz, or use a smaller --model").

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
| `serve [--mock] [--host 127.0.0.1] [--port 7878] [--web-dir DIR] [--data-dir DIR] [--device auto] [--precision f32] [--tensor-cores] [--cors-any] [--token SECRET] [--allow-host NAME]...` | works | Runs the HTTP API (`docs/API.md`) and serves the built Studio with an SPA fallback: from `--web-dir` when given, else the Studio embedded in the binary (`embed-web` builds), else `web/dist`. Once bound, it prints the listening URL (for wildcard binds `0.0.0.0` or `::`, a clickable loopback URL alongside the bound address: e.g. `http://127.0.0.1:7878 (bound to 0.0.0.0:7878)` or `http://[::1]:7878 (bound to [::]:7878)`), API health URL, Studio source, data dir (outside `--mock`) and a token notice when `--token` is configured; in `--json` mode the `listening` record carries `url` (loopback for wildcard binds, bound host otherwise), `bind` (the actual bound socket address), `api`, `mock`, `web_source` (`dir` or `embedded`), `web_dir`, `web_ready`, `data_dir` and `token`. Bind errors exit 1 with `bind_failed` and a cause-specific hint: unresolved `--host` names, non-local `--host` addresses (`AddrNotAvailable`), occupied ports (`AddrInUse`, suggesting another `--port` or `0`), and `PermissionDenied` (distinguishing privileged ports `1–1023` from reserved or exclusively bound ports). `--mock` serves deterministic mock data. `--port 0` picks a free port. `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`) holds the dataset registry, index and thumbnail caches, generated synthetic datasets, training runs, models and exports (see `docs/API.md`, Real mode); it is created when missing, and a path that is not a folder is a `usage` error (exit 2), as for `mcp` and `cloud`. One process owns a data dir at a time (the mock uses none): `folo serve` and an embedded `folo mcp` hold the OS lock on `serve.lock` there, and a server writes its local address to `serve.json` as soon as it is bound (wildcard binds `0.0.0.0` and `::` are written as loopback `127.0.0.1` and `[::1]`, and legacy wildcard records are normalized on read; specific binds, including scoped IPv6 addresses, stay as bound). A second `folo serve` prints a `data_dir_in_use` record with that `url` and `data_dir` and fails with `data_dir_in_use` (exit 1), so it never stops the first one's active runs; a data dir that cannot be locked at all is `data_dir_lock_failed`. `--device` (env `FOLO_DEVICE`: `auto`, `cpu`, `gpu:<n>`) is where `/api/predict` and `/eval` run; each training run has its own `device`. `--precision` (env `FOLO_PRECISION`: `f32`, `f16`) is the precision of `/api/predict` and `/api/val` (docs/API.md); `f16` falls back to f32, with a warning in the log, where the device has no f16 compute. `--tensor-cores` (env `FOLO_GPU_TENSOR_CORES=1`) runs that f16 inference on the GPU's tensor cores where it has them (docs/GPU.md, Tensor cores), the regular kernels elsewhere; training runs with `"amp": true` need it (`POST /api/runs` answers `not_supported` without). The published weights pulled with `folo weights pull` list as models whose id is the published name (see [Weights](#weights)). A finished run gives its GPU memory back to the driver, and so does inference after 30 s idle (`/api/system`'s `gpu_pool`), so the server does not hold gigabytes between jobs. Requests that change something or fail are logged at `info`; successful reads (Studio polls `/api/runs` every 2 s) at `debug` (`RUST_LOG=folo_server=debug` shows them). `--token` (env `FOLO_TOKEN`; empty is a usage error) requires `Authorization: Bearer <token>` on every `/api` request but `GET /api/health` (`401 unauthorized` without; docs/CLOUD.md, Securing a server): use it whenever `--host` is not loopback, which the server otherwise warns about once in its log. The token is never logged or printed; the `listening` record's `token` says whether one is set. A server on loopback (the default `--host`, also with `--mock`) answers only requests whose `Host` is `localhost`, `127.x.y.z` or `[::1]` (any port), against DNS rebinding: any other `Host` is `403 forbidden_host` (docs/API.md, Conventions). `--allow-host NAME` (repeatable; a port in it is ignored) adds a name for a tunnel or reverse proxy that keeps its own `Host`; a server on a non-loopback `--host` checks no `Host`. Ctrl+C shuts down gracefully: open SSE streams are closed, then an active training run stops after its current batch (keeping its last epoch's checkpoint, status `stopped`), and upload sessions in progress are dropped. |
| `describe [COMMAND...] [--no-mcp-tools]` | works | `folo describe train` (or `dataset check`; resolves visible command aliases too) prints just that command (`name`, `aliases`, `path`, `about`, `groups`, `args`, `subcommands`); `--no-mcp-tools` leaves out the MCP tool schemas, which are most of the full output. Prints the CLI surface as JSON: `global_args`, `conventions` (exit codes, error shape) and `commands` (recursive, each with `name`, `aliases`, `path`, `about`, `groups`, `args` and `subcommands`). Each `groups` entry describes a constraint group (`name`, `required`, `exclusive`, `args`). Each argument has `name`, `long`, `short`, `aliases`, `positional`, `type` (`string`, `path`, `integer`, `number`, `boolean`, `enum`, `count`), `value_name`, `required`, `multiple`, `default`, `possible_values` (each `{value, aliases, help}`), `env`, `conflicts_with` (symmetric conflicts from `conflicts_with` and exclusive groups) and `help`. Pretty-printed without `--json`, one line with it. |
| `doctor [--web-dir web/dist] [--tensor-cores]` | works | Version (`version`, `git_sha`, `build_date`), OS, CPU name, threads and SIMD readiness (`cpu.simd`: `avx2`, `avx_vnni`, `neon`, `int8_fast`; the human `simd` line lists the detected ISAs and whether int8 uses the fast AVX-VNNI path or the portable path, noting `FOLO_Q8_SCALAR` when set), memory, resolved absolute directories (`paths`: `data_dir`, `weights_dir`, `cache_dir`, `cache_enabled`, reflecting `FOLO_DATA_DIR`, `FOLO_WEIGHTS_DIR`, `FOLO_CACHE_DIR`, `FOLO_HOME` and `FOLO_NO_CACHE`; printed as human `data`, `weights` and `cache` lines), every wgpu adapter (`gpus`: `index`, `name`, `backend`, `device_type`, `vendor`, `device`, `driver`, `driver_info`, `memory_bytes` (as in `/api/system`: 0 where the backend cannot say; the human line names the size when known), `shader_f16`, `tensor_cores` (16×16×16 f16 → f32 cooperative matrices), buffer and workgroup limits, `subgroup_size` as `[min, max]`), the adapter FOLO picks (`gpu_selected`, honouring `FOLO_GPU_ADAPTER` / `FOLO_GPU_BACKEND`; its `tensor_cores` is `on` with `FOLO_GPU_TENSOR_CORES=1` or `--tensor-cores`, else why they are off), what runs with `device: auto` can use (`capabilities`: `device`, `shader_f16`, `tensor_cores`, `amp` with the reason when unavailable; the same object `folo serve` reports in `/api/system`, docs/API.md, given the same `--tensor-cores`; the human `amp` line says when the GPU has tensor cores that only were not asked for: `folo train --amp` asks for them itself, `folo serve` and `folo mcp` need `--tensor-cores`), whether the web app is built (`web_dist`: `path`, `exists`, `index_html`, `folo_wasm`), whether one is embedded (`web_embedded`: `present`, `files`, `folo_wasm`), which one `folo serve` would use (`web_source`) and whether that one includes folo-wasm (`web_folo_wasm`: its Playground can run models in the browser, docs/WASM.md; release builds always do). |
| `train --data D [--format auto\|yolo\|coco] [--model folo-n \| --weights W] [--teacher T] [--epochs 100] [--batch 16] [--imgsz 640] [--device auto] [--workers N] [--cache none\|ram] [--optimizer auto\|sgd\|adamw\|musgd] [--lr0 0.01] [--lrf 0.01] [--cos-lr] [--warmup-epochs 3] [--patience 100] [--amp] [--head o2m\|o2o] [augmentation flags] [--seed 0] [--set KEY=VALUE]... [--project runs] [--name NAME] [--resume CKPT] [--json-events all\|epoch]` | works | Trains a detector (docs/MODEL.md §8). Arguments mirror `TrainConfig` in `docs/API.md` (the server builds this command from a run's config: `GET /api/runs/{id}/cli`); `--format` reads `--data` as YOLO or COCO when a folder has both (default `auto`: detect); `--workers` (default: the logical CPU count, at most 16; also for `val`) sets the loader threads, which never change the batches; `--cache ram` keeps the training images decoded and resized in memory between epochs (the same batches; refused before training starts, with the sizes, when it needs more than 80% of the available memory; a resume may change it); `--hsv` takes `h,s,v` or three values; `--arch` is an alias of `--model`, which also takes a `.folo` file (or a published name) to fine-tune from. `--weights W` (a `.folo` file or a published name such as `folo-n-coco`, see [Weights](#weights)) starts from those weights and takes their architecture (with an explicit `--model` it is a usage error); tensors whose shape or class list differ (the class outputs, for another dataset) keep their fresh initialisation, and the run logs `transferred N of M tensors`. Runs go to `<project>/<name>/` (default name `<dataset>-<arch suffix>`, numbered when it exists): `folo-train.json` and `train.lock` (what `folo stop` checks, see `stop`), `config.json`, `metrics.jsonl` (every event, every step), `weights/last.folo` and `weights/best.folo` (full checkpoints: EMA, optimizer, schedule), `samples/epoch_NNN.json`. Live view: epoch bar, losses, img/s, ETA and one table row per epoch (P, R, mAP50, mAP50-95). `--json` prints every event as a record: `{"event":"status"\|"step"\|"epoch"\|"log", ...}` with the API.md payload fields (`step`s throttled to about 10 per second, the last batch of each epoch always sent, so a slow run gets one per batch; `metrics.jsonl` has every step), then `{"event":"done", "status", "run_dir", "epochs_completed", "epochs", "early_stopped", "best", "best_metrics", "weights", "resolved"}` (`best_metrics` is API.md's `Epoch.val`, with `best_f1_conf`; `resolved` is API.md's `Run.resolved`: the optimizer, effective `lr0`, momentum / betas, MuSGD's step weights, scaled weight decay, accumulation and warmup the run trained with). `--json-events epoch` leaves the `step` records out (status, epoch, log and done only: about 1 KB per epoch instead of one record per batch; a 15-epoch run of 50 batches drops from about 140 KB to 15 KB). `--resume runs/<name>/weights/last.folo` continues with the stored settings (bit-exactly when the data, the device, the distillation settings and the teacher's weights are the same as before): only `--workers`, `--device`, `--data`, `--format`, `--cache`, `--project`, `--json-events`, `--teacher` and `--set distill.*` (below) may differ (left out, `--workers`, `--device`, `--data`, `--format` and stored distillation settings keep the run's), any other flag given must repeat the run's value (a changed one is `invalid_config`, exit 2, naming the flag and both values; `--weights` is refused: the run continues from its checkpoint), and a run that completed (all epochs, or stopped early) is `run_not_resumable` (exit 1) with the command that trains further (`--model <run>/weights/best.folo`; as in every command a hint prints, a path with anything but ASCII letters, digits and `-_./:`, such as a space, is in double quotes, so the command pastes into PowerShell, cmd or a POSIX shell when the path has none of `"`, `$`, `` ` ``, `%`, `!`, `\`, a typographic double quote or a control character; a path with one of those is not put in the command: it shows `<model>` there and gives the path after it, for the person to quote for their shell), as MCP and `POST /api/runs` answer. Stopping: Ctrl+C, on Windows also Ctrl+Break (what a script can send to a child started in its own process group), or `folo stop <run_dir>` from another process stops after the current batch, keeps the last epoch's checkpoint and ends with `status: stopped` and the `done` record (`completed` when the request comes after the last batch); a second interrupt exits at once. A new run that fails before its first checkpoint (a bad device, a GPU out of memory) removes its run folder. `--optimizer auto` (the default) picks MuSGD at `lr0` 0.01, momentum 0.937 and a bias warmup LR of 0.1 for every run, short or long, fine-tune or from scratch, with Muon / SGD step weights 0.5 / 0.5 from scratch and 0.2 / 1.0 with `--weights` (`auto` rule 5, docs/MODEL.md §8.2; rule 4 used 0.2 / 1.0 for every run, rule 3 picked MuSGD only over 10,000 optimizer steps and AdamW below, rules 1 and 2 SGD for long runs; a run resumes with the rule it started with), scaling for the batch with gradient accumulation, so `--lr0` only matters with `--optimizer sgd`, `adamw` or `musgd` (MuSGD: SGD plus Ultralytics 8.4's Muon update on every conv weight, the head's class branches at 3x the LR; docs/MODEL.md §8.2); the run's `config.json` has what is used (`optimizer.type`, `optimizer.lr0`, `accumulate`, and all of it as `resolved`, API.md's `Run.resolved`; `train` holds the settings as given) and its first log line says the same. A run too short to learn (from scratch with fewer than 200 optimizer steps, or a warmup over half of the run; docs/MODEL.md §8.4) gets a `warning:` line right after the first log line (a `log` record with `level: warn` under `--json`; `warnings` in `config.json`). `--device auto` (the default; also for `val` and `predict`) uses the GPU when folo-gpu finds an adapter (the model, its optimizer state, EMA, gradient buffers and the detection loss all run there; batches are staged on a thread and uploaded as `u8`), else the CPU; `--device gpu:<n>` without a usable adapter is `device_not_found` (exit 3; the hint lists the devices this machine has). `--amp` trains in mixed precision (docs/MODEL.md §8.8: f16 activations, convolutions and matrix products on the GPU's tensor cores, f32 master weights, optimizer state, loss, BatchNorm statistics and EMA, dynamic loss scaling; about 1.6x (folo-n) to 1.85x (folo-s) the f32 training step and 40% less memory, docs/BENCHMARKS.md). It asks for tensor cores itself (as `--tensor-cores` does for `predict`), also when resuming a mixed-precision run; without them (the CPU, another GPU) the run is `not_supported` (exit 4) before anything is written. Each epoch logs the loss scale and the optimizer steps skipped for gradient overflow (the first few steps of a run usually are). `--head o2o` (alias `end2end`; `TrainConfig.head`; default: the `--weights` file's head, else `o2m`) adds the NMS-free one-to-one head and trains with dual assignment (docs/MODEL.md §6.6: the one-to-many loss plus the one-to-one loss, weighted from 0.8 / 0.2 to 0.1 / 0.9 over the run; the reported losses are the weighted sums); validation, `best.folo`, `val` and `predict` then use the one-to-one head without NMS. Fine-tuning such a model from `o2m` weights (`--weights`) starts the one-to-one branch as a copy of the one-to-many head. `--teacher T` (a `.folo` file or a published name such as `folo-s-coco`, see [Weights](#weights)) distils from that frozen teacher (docs/MODEL.md §8.9: the student also learns the teacher's class probabilities and DFL distributions at the shared anchors; training only, the weights and their format do not change): it sets the `train` key `distill` to `{"teacher": T, "cls_gain": 0.5, "dfl_gain": 1.5, "temperature": 4.0}` (a file is stored as an absolute path, a published name as given), which `--set distill.<key>=<value>` then changes (e.g. `--set distill.temperature=2`; the gains must lie in 0 to 1e6 and the temperature in 0.05 to 100, else `invalid_config`; without `--teacher` a `distill.*` key is `invalid_config`). The teacher needs the student's number of classes, `reg_max` and strides (any scale: folo-s for folo-n), else the run fails before training (`invalid_config`); a published name that is not downloaded is `weights_not_found` with the pull hint. It runs on the training device (f16 with `--amp`) and costs one teacher forward per batch; each epoch logs `distillation cls … dfl …` (the two weighted terms, also for the first and last quarter of the epoch). With `--head o2o` the term is part of the one-to-many loss, so the dual loss weights it by that branch's 0.8 to 0.1 (the logged terms are not weighted). The teacher is stored with the run: `--resume` loads it again (a run whose teacher is gone fails, naming `distill.teacher`), and `--resume CKPT --teacher T [--set distill.<key>=<value>]...` is a resume override: distillation is on (or its teacher or gains changed) from the resumed epoch on, and the run's `config.json` and later checkpoints record it (with `--teacher`, a run that distilled keeps its stored gains unless `--set distill.*` changes them, a run that did not starts at the values above; `--set distill.*` alone changes a distilling run's gains and needs `--teacher` otherwise). The teacher is not part of the optimizer state, so nothing else of the resume changes; without these flags a resumed run keeps what its config stored. `--set KEY=VALUE` (repeatable) sets any other key of the run's `config.json` `train` object (docs/MODEL.md §8.1), nested keys dotted, after the other flags: e.g. `--set momentum=0.9`, `--set loss.cls_gain=1.0`, `--set loss.tal.topk=13`, `--set augment.degrees=5`, `--set auto_rule=4`, `--set warmup_floor=100` (the 100-iteration warmup floor of runs made before the key; default 0), `--set ema_run_fraction=0` (a fine-tune's EMA averages over the default 10,000-step window instead of a quarter of its optimizer steps, docs/MODEL.md §8.5; the configuration log line names the window), `--set musgd.muon=0.5` (MuSGD's step weights, docs/MODEL.md §8.2), `--set o2o_attached_epochs=10` (`--head o2o` only: the one-to-one loss also trains the backbone for the first 10 epochs, §6.6), `--set loss.tal.beta=4` (with `--head o2o`, `loss.tal.alpha`, `beta`, `eps` and `min_box_size` apply to both branches unless `o2o_keeps_tal` is `false`, `topk` and `topk2` to the one-to-many branch only, §6.6), and the opt-in Wave-7 and math-suite recipe keys (`loss.mal_gamma`, `loss.nwd_weight`, `loss.cls_loss=vfl|qfl`, `loss.vfl_alpha`, `loss.vfl_gamma`, `loss.qfl_beta`, `loss.box_loss=rpiou|saeiou|inner_ciou|wiou_v3`, `loss.rpiou_beta`, `loss.rpiou_kappa`, `loss.rpiou_delta`, `loss.dfl_ot_weight`, `loss.dfl_crps_weight`, `loss.tal.nwd_weight`, `loss.tal.eps_free`, `musgd.ortho=polar4|polar5|polar3_cs9|ns5`, `musgd.geo=geo`, `ema_mode=power`, `decoupled_wd`, `cautious_wd`, `wd_eff`, `gauge_norm`, `bn_freeze_close_mosaic`, `cls_lr_close_scale`, `backbone_lr_scale`, `freeze_backbone_epochs`, and with `--teacher`: `distill.bckd_gamma`, `distill.gt_verify`, docs/MODEL.md §7.2 / §7.4 / §8.1 / §8.2 / §8.5 / §8.9); the value is read as JSON when it parses, else as a string; an object value is merged into the key's object (`--set 'loss={"cls_gain":1.0}'` leaves the other loss keys as they were). An unknown key (at any depth, also inside an object value) or a value of the wrong type is `invalid_config` (exit 2, listing the keys there are), and so is `--set` with `--resume` (a resumed run keeps its settings; only `distill.*`, the resume override above, may change) and a key whose flag does more than set it (`name`, `dataset`, `data_format`, `arch`, `device`, `amp`: the message names `--name`, `--data`, `--format`, `--model`, `--device` or `--amp`). |
| `stop RUN_DIR` | works | Asks the `folo train` that owns `RUN_DIR` (its `run_dir`) to stop after its current batch: writes `RUN_DIR/stop`, which the trainer polls every 200 ms and removes when the run ends. Returns at once with `{"event":"stop_requested","run_dir","stop_file"}`; the training process then ends with `status: stopped`, or `completed` when the request comes after its last batch (during the final validation or the last checkpoint: completion wins, as for `POST /api/runs/{id}/stop`). Asking again while it runs is the same answer, and a stop file already there is left as it is. `folo stop` reads no training output: a run folder is one with `folo-train.json`, which `folo train` writes when it reserves (or resumes) the folder, and the run is live while its `folo train` process holds the OS lock on `train.lock` there (from that moment through setup, training and the final validation until the process ends; the OS releases it when the process is killed). A folder without `folo-train.json` (any other folder, and run folders of FOLO versions before it) is `run_not_found` (exit 3): a run of an older FOLO gets the file when it is resumed with this version (`folo train --resume <run_dir>/weights/last.folo`), and a finished one needs no stop. A run folder whose lock nobody holds (the run ended, or its process died) is `run_not_active` (exit 1), and a stop file left there is removed (also the one `folo stop` wrote if the run ended as it asked). A FOLO server's run folder (its `run.json`) is the CLI's to stop while a `folo train` process holds it (a server run resumed in place with `folo train --resume`), else the server's: `run_owned_by_server` (exit 1), with the hint to stop it in Studio or with `POST /api/runs/{id}/stop`. None of these writes anything. `folo train --resume` never overwrites a `folo-train.json` that FOLO did not write: it is `usage` (exit 2), naming the file to move or rename, and changes nothing. A second `folo train --resume` of a run that is training is `run_active` (exit 1). The portable way for a script to stop training, on Windows too. |
| `val --model W --data D [--split val] [--batch 32] [--imgsz <model's>] [--conf 0.001] [--iou 0.7] [--max-det 300] [--fusion none\|vote\|dfl\|sebf] [--fusion-iou 0.75] [--score none\|consensus\|elc\|dfl-consensus\|dfl-repulse\|dfl-cgr] [--head o2m\|o2o] [--device auto] [--half \| --precision f32\|f16\|int8] [--tensor-cores] [--workers N] [--coco] [--save-json FILE]` | works | mAP@0.5, mAP@0.5:0.95, precision and recall (docs/METRICS.md) of the weights' EMA (fused) on a split, overall and per class. Thresholds and sizes are validated before the model or dataset is opened: `--fusion-iou` outside `0..=1` is `usage` (exit 2), and `--conf` / `--iou` outside `0..=1`, `--max-det 0`, `--batch 0` or `--imgsz` not a positive multiple of 32 is `invalid_config` (exit 2). Only that split is opened and checked (`folo_data::open_splits`: COCO's val2017 without probing the 118 287 train2017 images; with the other `val` changes of the same release, `folo val` of folo-n-coco on COCO val2017 takes 8.9 s instead of 29.6 s on a quiet machine, and 10.8 s instead of a median 92 s with 20–24 cores busy elsewhere: docs/BENCHMARKS.md, docs/notes/hill/trainer-data.md); its classes are the full dataset's. `W` is a `.folo` file or a published name (see [Weights](#weights)). One JSON object with `--json`: `{"model","data","split","imgsz","images","boxes","map50","map50_95","precision","recall","best_f1_conf","per_class":[{"class","name",...}],"device","dtype","head","fusion","score","seconds","coco"}` (`dtype`: `f32`, `f16` with `--half` or `--precision f16` where the GPU computes in it, or `int8` with `--precision int8`; `head`: the branch evaluated, the model's unless `--head` says otherwise: `o2o` (top-k, no NMS, docs/MODEL.md §6.6) for a model trained with `--head o2o`, where `--head o2m` evaluates its one-to-many branch with NMS; `--head o2o` on an `o2m` model is `invalid_config`). On a COCO dataset, `--coco` adds the pycocotools-exact evaluation (docs/METRICS.md, Full COCOeval: original-image boxes, crowd regions, annotation areas) as `coco: {"ap","ap50","ap75","ap_small","ap_medium","ap_large","ar1","ar10","ar100","ar_small","ar_medium","ar_large","images","truths","crowd","detections","missing_images","annotations","per_class":[{"class","name","truths","ap","ap50"}]}` (`null` without `--coco`) and prints `COCOeval.summarize()`'s table; GT images the loader skipped count with no detections (`missing_images`). `--save-json FILE` writes the detections as a COCO results file (`[{"image_id","category_id","bbox":[x,y,w,h],"score"}]`, original-image pixels, full f32 precision) for pycocotools; its folder is created once the split is open and before the evaluation runs, and a folder that cannot be created or a folder in the file's place is refused then (whether the file itself can be written shows only after the evaluation). Both need exactly one COCO annotation file for the split (usage error otherwise). `--half` (also for `predict`): half-precision inference, f16 weights and activations with f32 accumulation, on a GPU with `shader-f16` (an f16 export is uploaded as it is; other files are fused in f32 and converted); elsewhere it runs in f32 and prints a `note:` on stderr. On trained models f16 mAP is within 0.002 of f32 (docs/BENCHMARKS.md). `--precision` (also for `predict`) is `f32` (default), `f16` (the same as `--half`, which it excludes) or `int8`: the backbone and neck in int8 on the CPU (docs/MODEL.md §9.6) with the calibration `folo quantize` stored in the file (a file without one is `invalid_weights` with a hint; `--device auto` picks the CPU and a GPU device is a usage error); `dtype` reads `int8`. int8 is fast only on a CPU with AVX-VNNI: elsewhere, AVX-512 VNNI CPUs included, its kernels take a portable path several times slower than `f32`, and a `note:` on stderr says so. `--tensor-cores` (also for `predict`; the same as `FOLO_GPU_TENSOR_CORES=1`) runs the `--half` model's convolutions and matrix products on the GPU's tensor cores (wgpu's experimental cooperative matrices, docs/GPU.md); where they are not available, or without `--half`, it prints a `note:` and runs the regular kernels. Same mAP bound. `--max-det N` keeps the `N` best detections per image (COCO's AP reads the best 100 of each class's; more lets low-scoring boxes of busy images count). `--fusion` (also for `predict`, off by default; recorded as `fusion`) refines each box NMS keeps from the candidates it suppressed (docs/MODEL.md §9.4): `vote` moves each edge to the score-weighted mean over the kept box's cluster (its class's candidates with IoU `>= --fusion-iou` with it; outside `0..=1` is a usage error), `dfl` weighs each member's edge by its score times the certainty of its DFL distribution on that side, and `sebf` (Student-t Exclusive Box Fusion) restricts fusion to exclusively owned cluster members weighted by `score · IoU^4 · precision` with 1-step Student-t (`ν = 3`) IRLS outlier rejection; scores, classes and which boxes are kept do not change. On COCO val2017, `--fusion dfl --iou 0.65 --max-det 1000` adds about 0.002–0.004 AP (folo-s, folo-n; intervals in docs/notes/hill/r3-fusion.md). `--score consensus` (also for `predict`, off by default; recorded as `score` by `val`) scales each kept box's score by the score mass of the candidates of its class at IoU at least 0.75 with it (a fixed cutoff, not `--fusion-iou`) and re-sorts (docs/MODEL.md §9.4); boxes, classes and which boxes are kept do not change, with or without `--fusion`. **The score scale changes**: a box that fired alone is divided by up to 3.16, so a threshold on the returned score, and `val`'s precision, recall and best-F1 confidence, mean something else with it on. `--conf` and `--max-det` still act on the class score, before it, so `--max-det` picks the boxes and the rule only re-ranks them. On COCO val2017 it adds 0.001–0.002 AP (folo-s, folo-n v5; intervals in docs/notes/hill/r6-consensus.md). `--score elc` (exclusive logit-space consensus), `--score dfl-consensus` (adding scale-free fused DFL precision), `--score dfl-repulse` (power-weighted exclusive consensus with DFL Shannon entropy, cross-stride agreement, and greedy log-survival soft repulsion on output boxes; best with `--fusion dfl --iou 0.75 --max-det 1000`), and `--score dfl-cgr` (confidence-gated greedy repulsion weighting each suppressor by `p_m^{0.25}` with `γ = 3.0`; best with `--fusion sebf --iou 0.78 --max-det 1000`, adding `+0.0076` AP on `folo-n-coco-v5` and `+0.0065` AP on `folo-s-coco` over default NMS) restrict each cluster to candidates exclusively suppressed by that kept box and preserve the `[0, 1]` score scale (docs/MODEL.md §9.4). `--head o2o` ignores `--fusion` and `--score`. |
| `predict --model W SOURCES... [--conf 0.25] [--iou 0.7] [--max-det 300] [--fusion none\|vote\|dfl\|sebf] [--fusion-iou 0.75] [--score none\|consensus\|elc\|dfl-consensus\|dfl-repulse\|dfl-cgr] [--head o2m\|o2o] [--imgsz <model's>] [--device auto] [--half \| --precision f32\|f16\|int8] [--tensor-cores] [--save DIR]` | works | Inference on images or folders (recursive; directory links are followed; a link whose resolved path is a folder already being walked is skipped, so a link cycle ends, while two links to one folder are both walked; a folder whose real path cannot be resolved, as on some network shares, is walked without that check), letterboxed `rect` (docs/MODEL.md §9). Thresholds and sizes are validated before the model is loaded: `--fusion-iou` outside `0..=1` is `usage` (exit 2), and `--conf` / `--iou` outside `0..=1`, `--max-det 0`, or `--imgsz` not a multiple of 32 in `32..=8192` is `invalid_config` (exit 2). One `{"event":"prediction", "file", "image", "boxes": [ScoredBox], "timing", "device", "dtype", "saved"}` record per image (`dtype` as for `val`) (an unreadable image gives an `error` record and is skipped), then `{"event":"done", ...}` with `mean_timing` over the images after the first and `first_timing` (the first image, which includes pipeline compilation and the first allocations; with one image the mean is that image's) (`timing` as docs/API.md's `/api/predict`: on a GPU each stage waits for its own device work, so `inference_ms` is the whole forward pass). `--save` writes annotated copies (boxes coloured by class, tagged with the class id and score, e.g. `0 0.84`) in `DIR`, which must be a non-empty path (not an existing file) lying outside the input folders: an empty `--save`, an existing file at `--save`, or a `--save` folder that is a source folder or lies inside one (at any depth, also before it exists), or that holds a source, is a `usage` error before the model is loaded, on the first run as on every later one (the copies would be read as images on the next run; paths are compared after resolving links, `.`, `..` and, on Windows, letter case; `.` and `..` are folded before links are followed). A copy that would replace a source image (e.g. through a link) is a `usage` error too, found after the model is loaded and before anything is written. Each annotated image is written through a short temporary file (`.folo-<pid>-<n>.tmp`, skipping any name colliding with the destination), synced (`sync_all`), and atomically renamed over the destination so existing files (including hard links) are replaced cleanly without overwriting in place and failed writes leave no temporary files behind. With the `o2m` head and `--score elc`, `dfl-consensus`, `dfl-repulse` or `dfl-cgr`, when `--conf > 0.10`, supporting anchors are gathered down to `0.10` before final calibrated scores are filtered by `--conf` (for `--conf <= 0.10`, calibrated scores below `--conf` are kept in full). A model trained with `--head o2o` predicts end to end (its one-to-one head, the `--max-det` best anchors above `--conf`, no NMS, so `--iou`, `--fusion` and `--score` are unused); `--head o2m` uses its one-to-many branch with NMS instead. |
| `quantize --model W --data D [--split train] [--images 512] [--seed 0] [--method percentile\|minmax\|mse] [--tail 0.0001] [--imgsz <model's>] [--batch 8] [--workers N] [--output FILE]` | works | Calibrates int8 inference (docs/MODEL.md §9.6) on the CPU: `--images` images of the split drawn at random by `--seed`, letterboxed to `--imgsz`, two passes of the `f32` model; each boundary's range is the `--tail`, `1 − --tail` quantiles (`percentile`), the minimum and maximum (`minmax`) or the range of least expected squared error (`mse`). Writes the fused `f32` weights (EMA when present) with the calibration (`folo.quant`, docs/FORMAT.md) and the calibration size as the file's `folo.imgsz` to `--output` [default `<name>-int8.folo` next to `W`, `-2`, `-3`, ... when taken; never overwrites, as `export`], checks that the file loads with it, and records `{"path","bytes","images","imgsz","method","tail","boundaries","seconds"}`. Use the file with `val` / `predict --precision int8`. About 2 minutes for folo-n on 512 COCO images. |
| `export --model W [--format folo-f16\|folo\|weights\|onnx] [--output FILE] [--imgsz N] [--batch N] [--dynamic] [--half] [--decode] [--max-det 300] [--head o2m\|o2o]` | works | Fused EMA weights (FORMAT.md export). `W` is a `.folo` file or a published name (see [Weights](#weights)). Flag combinations and bounds (`--batch 0`, `--max-det 0`, `--batch` with `--dynamic`, ONNX-only flags on non-ONNX formats, and `--imgsz` outside multiples of 32 in `32..=8192` for ONNX) are rejected upfront with `usage` (exit 2) before resolving or opening `W`. `folo-f16` (default) and `folo` write a `weights` file in f16 / f32; `weights` writes the EMA weights **unfused** in f32 with no optimizer state (`fused: false` in the record), preserving `folo.metrics`, `folo.created` and unknown `folo.*` keys while dropping checkpoint-only keys and stamping `folo.version` with this writer's version (matching `folo` and `folo-f16`): a small file to fine-tune from (`train --weights`) or to publish (a fused input is a usage error); `onnx` writes an ONNX graph (opset 17) for onnxruntime, TensorRT, OpenVINO or the CoreML converters, see [ONNX export](#onnx-export) (`--imgsz`, `--batch`, `--dynamic`, `--half`, `--decode`, `--max-det` and `--head` are ONNX-only; all but `--max-det` are a usage error otherwise). Default output next to the input: `<name>-f16.folo` / `<name>-fused.folo` / `<name>-weights.folo` / `<name>.onnx` / `<name>-f16.onnx` (`--half`) / `<name>-decoded.onnx` (`--decode`) / `<name>-decoded-f16.onnx` (both), `-2`, `-3`, ... when taken: like MCP `export`, it never overwrites (an existing `--output` is `output_exists`, exit 1). The file is written in the output's folder under a temporary name, `.folo-<pid>-<n>.tmp` (while it is being written, that name followed by `.<pid>.<n>.tmp`), and moved to its name in one step once complete, a move that fails when the name exists by then (`output_exists` for `--output`, the next free name otherwise). An export that fails or is stopped at any moment therefore leaves its name either absent (a retry works) or holding the complete file, never an empty or partial one. A failed export removes its temporary files (best effort); a killed one can leave in that folder `.folo-<pid>-<n>.tmp` (empty, partial or complete) and the writer's own `.folo-<pid>-<n>.tmp.<pid>.<m>.tmp`, which nothing reads and which are safe to delete. On Linux file systems without `RENAME_NOREPLACE` and on macOS the move is a hard link and then the temporary name's removal; when that removal fails, `.folo-<pid>-<n>.tmp` stays as a second name of the finished file (also safe to delete). A file system without hard links is `io_error` saying so, never an overwrite. `quantize` writes its file the same way. A file that is not `.folo` weights is `invalid_weights` saying so. With `--json`, `.folo` exports print `{"path","bytes","format","dtype","fused","source","arch","classes","seconds"}` (ONNX adds graph metadata; see [ONNX export](#onnx-export)). |
| `bench catalog\|list\|create\|show\|delete\|run\|status\|stop [--data-dir DIR] [--server URL [--token SECRET]]` | works | Vision benchmarks (docs/BENCH.md): frozen labeled images with a plan of stress conditions, run by FOLO models or by predictions made elsewhere and scored on five axes; through the data dir's `folo serve`, or in this process. See [Vision benchmarks](#vision-benchmarks). The speed numbers of docs/BENCHMARKS.md come from `cargo bench`. |
| `dataset synth OUT_DIR [--images 1000] [--classes circle,square,triangle] [--imgsz 640] [--seed 0] [--val-frac 0.2] [--image-format jpeg\|png] [--vary-aspect]` | works | Writes a synthetic shapes dataset (YOLO layout). See [Datasets](#datasets). |
| `dataset stats PATH [--format auto\|yolo\|coco]` | works | Class counts, box sizes, boxes per image: a table with bar charts, or the `stats` object of `docs/API.md` with `--json`. |
| `dataset check PATH [--format ...] [--fix-dry-run] [--headers-only] [--min-severity error\|warning\|info]` | works | Every problem in a dataset, one issue per bad line / image (`warn` is accepted as an alias for `warning`; `--min-severity` filters both human output and `--json`). Exit 0 when clean, 1 with errors or warnings. |
| `dataset convert IN OUT --to yolo\|coco [--from auto\|yolo\|coco]` | works | Converts between YOLO and COCO; images are copied unchanged. |
| `weights list` | works | Every published weights name (see [Weights](#weights)), whether it is downloaded, and its cache path. One record per name with `--json`: `{"name","release","asset","path","cached","about"}`. |
| `weights pull NAME [--force] [--repo himent12/folo]` | works | Downloads a published weights file and its release's `SHA256SUMS` over plain HTTPS, without credentials (`https://github.com/<repo>/releases/download/<tag>/<file>`, redirects followed; `$FOLO_WEIGHTS_BASE_URL` downloads `<base>/<tag>/<file>` instead). Transient HTTP (`408`/`429`/`5xx`), transport/IO, or truncated-body failures retry up to 3 times (4 total attempts) with exponential backoff (base `500 ms`, capped at `10 s`, honoring numeric and HTTP-date `Retry-After` headers; `FOLO_WEIGHTS_RETRY_DELAY_MS` adjusts the base delay, `0` disables sleeping). When HTTPS is refused (`404`, as for a private repository's release, a draft or a missing release, or `401`/`403`), it does not retry and falls back to the GitHub CLI (`gh release download`, which needs `gh auth login` and access to the repository). It checks the SHA-256 against `SHA256SUMS`, verifies that the file opens as a `.folo` file, syncs it to disk (`sync_all`), and atomically renames it into the cache (`rename_over`, preserving any existing cached file until the replacement is ready); on any failure nothing is installed. Already cached: nothing is downloaded unless `--force`. `--json`: `{"event":"done","name","path","downloaded","sha256","bytes","via"}` (`via`: `https`, `gh`, or `null` when nothing was downloaded). Unknown name: `weights_not_found` (exit 3); refused and then `gh` missing or failing: `gh_failed` (the message says both); no network, a timeout, a server error or a cut download: `download_failed` (no fallback to `gh`); bad hash: `checksum_mismatch` (exit 1). |
| `cloud providers\|connect\|connections\|check\|disconnect\|offers\|instances\|terminate [--data-dir DIR]` | works | Cloud compute on the data dir, with or without a server running: see [Cloud](#cloud). |
| `label agents\|jobs\|create\|add\|start\|stop\|status\|images\|view\|set\|review\|export [--data-dir DIR] [--server URL [--token SECRET]]` | works | Agent labeling (docs/LABELING.md): a job of images labeled by Claude Code, Codex or Antigravity, reviewed and corrected, exported as a dataset; through the data dir's `folo serve`, or in this process. See [Labeling](#labeling). |
| `mcp [--data-dir DIR] [--mock] [--read-only] [--confine] [--server URL [--token SECRET]] [--device auto] [--tensor-cores]` | works | Model Context Protocol server over stdio (MCP 2026-07-28 and 2025-11-25), see `docs/MCP.md`. stdout carries protocol messages only (also with `--json`, also for a usage error); logs and errors go to stderr. By default it embeds the real store on `--data-dir` (env `FOLO_DATA_DIR`, default `$FOLO_HOME` or `~/.folo`): datasets, training, models, `val`, `predict` and `export` on the FOLO engine, the same data `folo serve` uses. Embedded, it owns the data dir as `folo serve` does; when a `folo serve` already owns it (FOLO Desktop's, say), it goes through that server as with `--server` (after checking that server's `/api/system` `data_dir` is this one), except with `--confine`, which then fails with `data_dir_in_use`. `--mock` serves mock data, `--server` (env `FOLO_SERVER`; empty is a usage error) forwards to a running `folo serve` (`http://` or `https://`; `--token`, env `FOLO_TOKEN`, is the token that server was started with), `--read-only` hides and rejects mutating tools, `--confine` only reads inside `--data-dir`, `--device` (env `FOLO_DEVICE`) is where inference and `val` run, `--tensor-cores` (env `FOLO_GPU_TENSOR_CORES=1`) asks for the GPU's tensor cores as `folo serve --tensor-cores` does (`train` with `"amp": true` needs them; without, it answers `not_supported` before starting anything). Published weights pulled with `folo weights pull` are models (`train { "weights": "folo-n-coco" }`). Exits 0 when the client closes stdin; the runs it started are stopped first (an active run after its current batch, keeping its last checkpoint). Recipes: `docs/AGENT_COOKBOOK.md`. |

`folo describe --json` is the authoritative list of arguments and defaults. Its `mcp_tools` array lists every
`folo mcp` tool (`name`, `description`, `mutating`, `inputSchema`, `outputSchema`) with the schemas `tools/list`
publishes (about 77 KB of the 156 KB; `--no-mcp-tools` leaves them out, `folo describe <command>` prints one
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
and which detections pass `conf`, can differ (on 1080p frames PIL's resize kept the boxes but moved scores by up to 0.15). **Outputs**, raw (default): `box_raw: [B, 64, A]` (DFL logits, channel `side · 16 + bin`,
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
needs a Python with `onnx`, `onnxruntime` and `numpy` (set `FOLO_ONNX_PYTHON` to its interpreter, e.g. `python`, or
`FOLO_ONNX_VENV` to a venv folder) and skips without one, unless `FOLO_ONNX_REQUIRED=1`, which CI's `rust (linux)` job
sets (docs/CI.md).

```sh
folo export --model best.folo --format onnx                            # best.onnx: f32, 1x3x640x640, box_raw + cls_raw
folo export --model best.folo --format onnx --decode --dynamic         # detections [batch, anchors, 4 + nc]
folo export --model best.folo --format onnx --half --imgsz 320         # best-f16.onnx
folo export --model o2o.folo --format onnx --decode                    # o2o model: detections [1, 300, 6], no NMS
```

## Datasets

`PATH` is anything `folo_data::open` accepts (docs/DATA.md): a dataset folder, a `data.yaml` or a COCO instances
`.json`. Shared error codes: `path_not_found` (exit 3), `unknown_format` (nothing there looks like a dataset, exit 1),
`invalid_dataset` (a malformed `data.yaml` or COCO file, exit 1), `dataset_unfinished` (below, exit 1),
`output_exists` (`synth` / `convert` into a folder that exists and is not empty, exit 1), `usage` (bad flag values
such as nine classes, exit 2).

`synth` and `convert` keep a `.folo-unfinished` file in the output folder until the dataset is complete. A command
stopped part way leaves it: `stats`, `check`, `convert`, `folo train` and registering in `folo serve` refuse the
folder with `dataset_unfinished`, and `synth` / `convert` into it stop with `output_exists`; both hint "delete the
folder and run the command again", which is the way to recover. FOLO never deletes it for you (docs/DATA.md,
Unfinished datasets).

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
`broken_labels`, `class_counts`, `box_size_hist`, `boxes_per_image_hist`). The human form is a class table with numeric `id`, name (truncated to 24 characters with a trailing `…` when longer; `--json` always lists every class name in full), box count, share and bar (top 40 classes by count when there are more), plus both histograms as a sparkline and one bar per bin.

### `folo dataset check PATH`
Reads every label line and every image (fully decoded; `--headers-only` reads just the headers, faster but blind
to truncated image data) and hashes file contents. Each issue has a kind and a severity:

| kind | severity | what |
|---|---|---|
| `broken_label` | error | a line that is not `class cx cy w h` or a polygon of finite numbers; an unreadable or non-UTF-8 label file |
| `class_overflow` | error | a class id that is negative, fractional, `>= nc` (when names are declared) or `>= 65536` |
| `out_of_range_box` | error | coordinates clearly outside `[0, 1]` (pixels, not normalised) or a negative size |
| `corrupt_image` | error | an image that cannot be read or decoded, or a JPEG cut off mid-scan (no end-of-image marker after the scan data; decoders fill the missing rows with grey, so only the full check sees it) |
| `missing_image` | error | an image named by an image list or COCO JSON that does not exist; a YOLO label file whose image is gone (`labels/train/a.txt` with no `images/train/a.*`: training ignores it), in a label folder paired with an image folder that still holds an image the dataset reads |
| `clipped_box` | warning | a box up to 0.01 outside the image (the loader clips it) |
| `zero_size_box` | warning | a box with no area inside the image (the loader drops it) |
| `duplicate_label` | warning | the same box twice in one file |
| `duplicate_image` | warning | an image with the same bytes as an earlier one (128-bit content hash) |
| `other` | warning | a split path that does not exist, a COCO image whose declared width / height differ from the file, ... |
| `empty_image` | info | an image without boxes (a background image; never makes the check fail) |

Errors make the loader skip the image (except a truncated JPEG, which decodes with grey rows: fix or remove it);
warnings are fixed up on load. `--min-severity warning` (alias `warn`), `error` or `info` filters listed issues in
both human output (default: `warning`, hiding `empty_image` infos unless `info` is passed, and noting hidden warnings
under `error`) and `--json` (default: `info`, listing every issue). The human summary always reports per-severity
counts (`error(s), warning(s), info(s)`). `--json` prints one report:
`{"path","root","format","classes","images","splits","clean","min_severity","counts":{"error","warning","info","total","by_kind":{kind: n}},"filtered_out","issues":[{"kind","severity","file","line","message","fix"?}],"seconds"}`
(`counts` always covers every issue regardless of `--min-severity`, and `filtered_out` counts the issues omitted from
`issues`). `file` is relative to `root` with `/`
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
| `folo-s-coco` | `weights-v3` | folo-s trained from scratch on COCO train2017 at 640 px, 56 epochs (EMA, f32, unfused): fine-tune or predict. val2017 AP 0.416. examples/coco/README.md, folo-s |
| `folo-s-coco-f16` | `weights-v3` | the same, fused and cast to f16 (`folo export`): inference only |
| `folo-n-coco-v3` | `weights-v4` | folo-n from scratch on COCO: `folo-n-coco-v2` annealed for 16 more epochs, 78 in all (EMA, f32, unfused): fine-tune or predict. val2017 AP 0.340. examples/coco/README.md, v3 |
| `folo-n-coco-v3-f16` | `weights-v4` | the same, fused and cast to f16 (`folo export`): inference only |
| `folo-n-coco-v4` | `weights-v5` | folo-n from scratch on COCO train2017, 64 epochs, the second half distilled from the frozen `folo-s-coco` (EMA, f32, unfused): fine-tune or predict. val2017 AP 0.356. examples/coco/README.md, v4 |
| `folo-n-coco-v4-f16` | `weights-v5` | the same, fused and cast to f16 (`folo export`): inference only |
| `folo-n-coco-v5` | `weights-v6` | `folo-n-coco-v4` continued for 64 epochs on COCO train2017 with the frozen `folo-s-coco` teacher, 128 in all (EMA, f32, unfused): fine-tune or predict. val2017 AP 0.370, the best folo-n weights. examples/coco/README.md, v5 |
| `folo-n-coco-v5-f16` | `weights-v6` | the same, fused and cast to f16 (`folo export`): inference only |

`weights-v1` to `weights-v6` are published releases. `folo weights pull` downloads their files over plain
HTTPS: no `gh`, no account, nothing but the `folo` binary. Transient download failures (HTTP `408`/`429`/`5xx`,
retryable transport/IO errors, or a cut body) retry up to 3 times (4 attempts total) with exponential backoff (starting
at `500 ms`, capped at `10 s`, honoring numeric and RFC 9110 HTTP-date `Retry-After` headers). Only when HTTPS is
refused (`404` from a private fork or mirror or a draft, or `401`/`403`) does the pull fall back to `gh release
download`, which needs a logged-in [`gh`](https://cli.github.com) (`gh auth login`) with access to that repository. For
files of 1 MiB or more, an interactive terminal shows a live progress line with throughput (`KB/s` or `MB/s`, after
100 ms) and ETA, fitting narrow terminals cleanly. Each step has its own time limit, none the whole download: 15 s to
resolve the host name and 15 s to connect, 60 s to send the request and to get the answer's head, and 60 s for each
read of the file, so a stalled connection fails and a slow one does not.

- `FOLO_WEIGHTS_BASE_URL=<base>` downloads `<base>/<tag>/<file>` (and `<base>/<tag>/SHA256SUMS`) instead of the
  GitHub release: a mirror is a folder per release tag with the release's files, served over HTTP(S). A file the
  mirror answers `404` for falls back to `gh` as above.
- `FOLO_WEIGHTS_RETRY_DELAY_MS=<ms>` sets the initial retry backoff delay in milliseconds (default `500`, capped at
  `10000`; `0` disables sleeping between retries).
- `FOLO_GH=<program>` is the GitHub CLI the fallback runs (default `gh`, looked up on `PATH`).

A command that takes a published name that is not downloaded (`train --weights`, `--model` of `val` / `predict` /
`export`) does not download it: it stops at once with `weights_not_found` (exit 3) and the hint
``run `folo weights pull NAME`, then this command again``, so a run never starts on a download it did not ask for. A
first fine-tune is two commands ([QUICKSTART](QUICKSTART.md#fine-tune-from-pretrained-weights)):
`folo weights pull folo-n-coco-v5`, then `folo train --data <dataset> --weights folo-n-coco-v5`. Without `folo`,
`gh release download weights-v6 --repo himent12/folo --pattern 'folo-n-coco-v5*' --pattern SHA256SUMS` (or the
release URLs) then `sha256sum -c --ignore-missing SHA256SUMS` does the same by hand.

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
| `cloud offers CONNECTION [--refresh]` | What the connection can run on now, cheapest first. Each CLI call queries the provider live (the one-minute offers cache lives in memory in a `folo serve` or `folo mcp` process, so `--refresh` is a no-op here). |
| `cloud instances` | The instances FOLO launched: alive first, then those that ended in the last 7 days, newest first, each with its cost so far. |
| `cloud terminate ID` | Terminates an instance now, through the same path as `DELETE /api/cloud/instances/{id}`: `terminated` when the provider confirms, else `terminating` with the error (a server's reaper retries it). An instance that is gone answers its record. A run still using it fails with `cloud_instance_lost`. |

```sh
RUNPOD_API_KEY=rpa_... folo cloud connect runpod --secret api_key=env:RUNPOD_API_KEY
folo cloud connect remote --set url=https://gpu-box.example.com:7878 --secret token=- < token.txt
folo cloud offers cc_4d2a --json
folo cloud instances
```

## Labeling

`folo label` labels a folder of images with a coding agent's model (Claude Code, Codex or Antigravity,
docs/LABELING.md), lets
a person or a script check and correct the boxes, and exports the result as a YOLO dataset. It is the command-line
form of the MCP labeling tools (docs/MCP.md, *Labeling*) and runs the same code, so `--json` prints the same
records as those tools' `structuredContent`. Every command takes `--data-dir` (as `folo serve`: `$FOLO_DATA_DIR`,
else `$FOLO_HOME` or `~/.folo`), `--server URL` (env `FOLO_SERVER`, empty is a usage error; a `folo serve`, e.g. on
another machine; `--token` or `FOLO_TOKEN` for one started with `--token`) and `--json`.

Where the jobs live: when a `folo serve` owns the data dir (FOLO Desktop starts one), the command goes through it,
so Studio shows the job and its run at once. Otherwise the command opens the data dir's labeling jobs itself (only
those; `export` also opens its dataset registry) and holds its lock while it runs, as `folo mcp` does. A run
belongs to the process that started it, so `start` without a server **waits** for the run to end (Ctrl+C stops it,
also while it starts; labeled images stay), while through a server it returns at once unless `--wait` is given.
Paths are read in place, on the server's machine: an absolute path (of any OS, the server's may be another) is
passed as given, a relative one is made absolute against the current folder (on Windows `\photos` is on the
current drive).

Boxes are `[class, x1, y1, x2, y2]` with the corners in **pixels of a view of the image**: the image scaled so its
long side is 1024 (`view --size`; never upscaled), the size `view` writes and states (a 1920×1080 photo's view is
1024×576). On the command line a box is `CLASS,X1,Y1,X2,Y2`, where `CLASS` is a class index or name. Each image's
boxes have a revision (`view` prints it); `set` needs it (`--expect`), and `set` and `review --expect` are refused with
`label_boxes_changed` (exit 1) when the boxes changed since, e.g. a run labeled the image again.

| command | what it does |
|---|---|
| `label agents` | The agent CLIs (`claude`, `codex`, `agy`): installed or not, version, suggested models, default model and effort, and how to install a missing one. |
| `label jobs` | The jobs, newest first: `{"jobs":[{id, name, status, classes, counts, dataset_id}]}`. |
| `label create NAME --classes A,B [--instructions TEXT] [--path P]...` | Makes a job and adds the images of each `--path` (folders searched recursively, or image files): `{job_id, images, added, skipped, skipped_first?}`. |
| `label add JOB PATH...` | Adds images to a job (same record). |
| `label start JOB --agent claude\|codex\|antigravity [--model M] [--effort low\|medium\|high] [--fast] [--images ID,...\|--all] [--note TEXT] [--batch N] [--parallel N] [--wait]` | Starts an agent run on the unlabeled images (pending and failed), on `--images` (e.g. again with a `--note` about what was wrong), or on `--all`; `--fast` asks for Codex's Fast tier (more of the plan's usage), only with a model that offers it; a server that does not list it for that model (one from before the Fast tier, which would ignore it) is refused before the start (`usage`). Waiting, it prints `{"event":"started","job_id","total","model"}`, a `{"event":"progress","done","failed","total"}` record when the counts change, and `{"event":"finished", ...}` with the `status` record of the run as it was seen to end (the review count and the log are read after it: each is left out when its read fails, e.g. the job was deleted, and both when the job's run is another one after those reads); a run that ends `failed` exits 1 with `label_run_failed` (the agent CLI's message and hint). The run is followed by its id (`run.id`), and the finished record and the exit code come from the same read. When the job's run is another one by then (the run ended and someone started the next), it prints `{"event":"superseded","job_id"}` and exits 0. Not waiting: `{job_id, total, model}`. |
| `label stop JOB` | Stops the job's run: `{job_id, status}`. |
| `label status JOB [--log 5]` | Counts (labeled, failed, reviewed), the last run with its outcome, the tokens of every run (`calls`, `input_tokens`, ...) and the last log lines. |
| `label images JOB [--status pending\|labeling\|labeled\|failed] [--reviewed true\|false] [--boxes] [--offset 0] [--limit 50]` | A page of the images (`{items, next_offset, total}`; `next_offset` is `null` on the last page): id, file, status, box count, `source` (`agent` / `manual`), reviewed, and with `--boxes` the view `size`, the `revision` and the boxes. `--status labeled --reviewed false` is what is left to review. |
| `label view JOB [IMAGE] [-o FILE] [--size 1024] [--from 1]` | Writes the view of the image (default: the next to review) as a JPEG with its boxes outlined and numbered and pixel ticks on the edges (default file `<job>-<image>.jpg`; never over one of the job's images, however its path is spelled: that is `usage`, and through a `folo serve` too old to say whether a file is one of them, an existing file is not replaced either; the server compares paths as it sees them, so when it reaches the photos under other paths than yours (a container, another machine) it cannot recognise them: do not point `-o` at a job photo there), and prints its size, its revision and its boxes (300 at most, from number `--from`); the record adds `path`. |
| `label set JOB IMAGE (--box B... \| --clear \| [--relabel N=CLASS]... [--remove N]... [--add B]...) --expect REV [--size WxH]` | Changes the boxes: all of them (`--box`, repeated; `--clear` for none; a box given as `view` lists it keeps its exact pixels), or by the numbers `view` shows (untouched boxes keep their exact pixels); `--expect` is the revision `view` printed. `--size` is the view the coordinates are in, as `view` printed it, required with `--box` and `--add`. When boxes that differ look the same in the `--size` view, `--box` must list that look once for each of them or not at all; fewer is a `usage` error (which ones to keep is unknown): remove them by number (`--remove`), or give the boxes of a larger view. Saved as a hand edit and marked reviewed, like Studio's Save: the record is the image as `view` shows it, with its new `revision`. |
| `label review JOB IMAGE... [--undo] [--expect IMAGE=REV]...` | Marks labeled images reviewed (their boxes are right), like Studio's "Looks right", each with `--expect` only if its boxes are still that revision: `{updated, failed?, failed_total?, to_review}`. When nothing was marked and something failed, it exits with the first failure's code. |
| `label export JOB [--name N] [--val-fraction 0.2] [--seed 0]` | Writes the labeled images as a YOLO dataset under the data dir and registers it: `{dataset_id, name, path, train, val}`; `folo train --data PATH` trains on it. |

Errors are the API's codes with their exit codes: `label_job_not_found` and `image_not_found` exit 3, an agent
that is not installed `not_supported` exit 4, a run already active `label_job_active` exit 1, boxes changed since
their revision `label_boxes_changed` exit 1, bad boxes, flags or requests `usage` exit 2 (the API's `invalid_body`
and the tools' `invalid_arguments` are `usage` here), and `data_dir_in_use` (exit 1) when another process owns the data dir without
serving it.

```sh
folo label create street --classes car,person,bicycle --path ~/photos/street
folo label start lb_3f9a --agent claude --images 1,2,3,4,5      # a sample first
folo label view lb_3f9a -o next.jpg                              # the next image to review, its revision
folo label set lb_3f9a 4 --expect 2-1c9e04aa --size 1024x576 --relabel 2=person --add car,120,80,340,400
folo label review lb_3f9a 1 2 3 5
folo label start lb_3f9a --agent claude                          # the rest
folo label export lb_3f9a --json
```

## Vision benchmarks

`folo bench` makes vision benchmarks ([BENCH.md](BENCH.md)), runs contestants on them and reads their scores without
Studio. It shares its operations with the MCP benchmark tools (`folo_mcp::bench`), not its records: the table below
gives what `--json` prints, docs/MCP.md (Benchmarks) what the tools return. Every command takes `--data-dir` (as
`folo serve`: `$FOLO_DATA_DIR`, else `$FOLO_HOME` or `~/.folo`), `--server URL` (env `FOLO_SERVER`, empty is a usage
error; a `folo serve`; `--token` or `FOLO_TOKEN` for one started with `--token`) and `--json`.

Where the benchmarks live: when a `folo serve` owns the data dir (FOLO Desktop starts one), the command goes through
it, so Studio shows the benchmark and its entries at once. Otherwise the command opens the data dir's store itself
and holds its lock while it runs, as `folo mcp` does. A model's entry runs in the process that started it, so
`run --model` without a server **waits** for the entry to end (Ctrl+C stops it; nothing of it is scored), while
through a server it returns at once unless `--wait` is given, and `status --wait` follows it later by its id (the
entry is the server's: it goes on when the command exits). A wait reads the entry every 100 ms in process and every
500 ms through a server.

| command | what it does |
|---|---|
| `bench catalog` | The conditions a plan can name (`id`, `group`, `description`), the highest severity, the default plan and the axes of the score with their weights. |
| `bench list` | The benchmarks, newest first: `{"benchmarks":[{id, name, source_name, images, num_classes, conditions, entries, best, active}]}`. |
| `bench create NAME (--dataset ID \| --data PATH \| --label-job ID) [--split val] [--description T] [--max-images 500] [--seed 0] [--plan JSON\|FILE]` | Makes a benchmark: frozen copies of a dataset split's images and truth (`--data` registers the folder first, or finds the dataset it already is; a relative path is made absolute against the current folder) or of a labeling job's labeled images. `--plan` is a plan as JSON (`{"conditions":[{"condition":"haze","severities":[1,3],"weight":1}]}`) or a file of it, checked as the API checks it (an unknown condition, a severity outside 1 to 5, ... are `usage`); default: the catalog's default plan. The record is the benchmark as `show` prints it. |
| `bench show BENCH` | The benchmark (`id`, `name`, `description`, `source`, `source_name`, `classes`, `images`, `boxes`, `plan`, `seed`, `fingerprint`, `active`) and its `leaderboard`: one row per entry (`id`, `name`, `kind`, `status`, `score`, `partial`, `map50_95`, `ci95`, `stale`: graded on another fingerprint), ranked as `GET …/entries` ranks them. |
| `bench delete BENCH` | Deletes the benchmark with its entries (`bench_active`, exit 1, while one runs). |
| `bench run BENCH --model M [--name N] [--max-images N] [--no-conditions] [--wait] [--device D]` | Adds a model's entry. `M` is a model of the data dir by id or by name (a run's weights, a `.folo` file in `models/`, published weights pulled with `folo weights pull`, named as published); an unknown one is `model_not_found` (exit 3) listing the models there are. Waiting, it prints `{"event":"started", ...}` (the entry's state), a `{"event":"progress","done","total","condition","severity"}` record when it moves, and `{"event":"finished", ...}` with the state of the read that showed it over, its score included; an entry that ends `failed` exits 1 with `bench_entry_failed` (the entry's error and hint). Not waiting: the state. `--device` (default `$FOLO_DEVICE`, else `auto`) is for an entry run in this process: through a server it is `usage`, as the server runs entries on its own `--device`. |
| `bench run BENCH --predictions FILE --name N [--unscored]` | Adds an entry of predictions made elsewhere, read from `FILE` in this process: JSON Lines of the records `examples/bench_external.py` sends (`{"image", "condition"?, "severity"?, "boxes": [{"class", "score", "x1", "y1", "x2", "y2"}]}`, `class` the index of a benchmark class (a number), in pixels of the image as the server serves it; [BENCH.md](BENCH.md#who-can-run) has the rules: every unchanged image answered, a condition answered for every image at every planned severity or left out), one record per line of at most 1 MiB and 1,000 boxes (not arrays or `{"predictions": [...]}` documents, which are refused without being read whole). The file is read as it is sent, in parts of at most 20,000 records, 200,000 boxes and 32 MiB of record JSON (the separators and the request's wrapper come on top), to an open entry that is then finished and scored; the record is its state with the score, and an entry the server failed to score is `bench_entry_failed` (exit 1). A file that does not parse is `usage` naming the line. Ctrl+C (`interrupted`, exit 1) is checked before each record is read and before each request; while a read from a pipe blocks, it waits for that read. After a failed part, or a finish the server refuses with a 4xx other than `409`, the command reads the entry and, if the server reports it still open, tries to stop and remove it. After a finish with no answer, a 5xx or a `409` it stops and removes nothing (the entry may be being scored): it reads the entry once, and a scored entry is the result. In every other case the error is reported and an entry may be left on the server: `folo bench status` shows it, `folo bench stop` ends it, and an open one that gets no further part expires (after 30 minutes by default). `--unscored`: the model gives no confidences. |
| `bench status BENCH ENTRY [--wait]` | The entry's state: `benchmark_id`, `entry_id`, `name`, `status`, `done` / `total` images over all passes, the pass running (`condition`, `severity`), `images`, `conditions`, `missing_classes`, `error`, `hint`, and once `done` its `result`: `score` (0 to 100), `partial`, the five `axes` (`axis`, `value` 0 to 1 or `null` when not measured, `weight`), `accuracy` (`map50_95` with its `ci95`, `map50`, `map75`, `precision`, `recall`, `best_conf`), `conditions` (per condition of the plan it ran: `condition`, `weight`, `retention`, `points` as `[severity, map50_95, retention]`), `ece`, `mean_iou`, `self_labeling` (`threshold`, `precision`, `recall`, `readiness`) and `ms_per_image` (a model's); [BENCH.md](BENCH.md#definitions) defines each. `--wait` follows the entry as `run --wait` does. The reliability bins, label curve, IoU histogram and per-image detections are the HTTP API's (`…/entries/{id}/result`, `…/images`). |
| `bench stop BENCH ENTRY` | Stops a queued or running entry and prints its state (`stopping`, then `stopped`); one that has ended is `bench_entry_not_active` (exit 1). |

Errors are the API's codes with their exit codes: `bench_not_found`, `bench_entry_not_found` and `model_not_found`
exit 3; `bench_active`, `bench_entry_not_active` and `bench_class_mismatch` exit 1; bad flags, plans and predictions
are `usage`, exit 2 (the API's `invalid_body` and the tools' `invalid_arguments` are `usage` here);
`not_implemented` (exit 4) comes from a `folo serve --mock` or a server from before benchmarks; `data_dir_in_use`
(exit 1) when another process owns the data dir without serving it. Agent contestants, plans designed by an agent
and self-improvement loops spend an agent's tokens: they are left to Studio, FOLO Desktop and the HTTP API.

```sh
folo bench create pond --data ~/data/pond --description "a camera in a murky pond" --max-images 300
folo bench run bn_3f9a0c1e5b7d --model folo-n-coco                  # waits (no folo serve)
folo bench run bn_3f9a0c1e5b7d --predictions yolo.jsonl --name yolo26n
folo bench show bn_3f9a0c1e5b7d --json
```

## Adding a command
1. Add the clap `Args` struct in `crates/folo-cli/src/cli.rs` with doc comments on every field (they become the help
   text and `describe` output) and accurate defaults, and its variant in the `Command` enum there.
2. Put the implementation in a module of its own, declare it in `crates/folo-cli/src/main.rs` and add its arm to
   `run` there. List the command in the [Commands](#commands) table above.
3. Return `Result<(), CliError>`; pick the exit code from the table above. Emit machine records with `Out::record`
   and human text with `Out::human`, never `println!` directly.
4. Add a test in `crates/folo-cli/tests/cli.rs` that runs it with `--json` and checks stdout parses line by line.
