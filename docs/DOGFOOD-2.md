# Dogfood report 2: FOLO as a new user and a new agent see it

A fresh agent that had never used FOLO did three jobs. It started from README.md, docs/QUICKSTART.md, llms.txt
and docs/AGENT_COOKBOOK.md, and opened the source only where the docs ran out; each time it did, that is
recorded as a finding.
1. **As a human:** follow the README's 60-second quickstart. Train on the GPU, predict, export to f16 and ONNX,
   and serve Studio from a release `folo serve` (embedded Studio, with in-browser inference). Drive Studio with
   Playwright in the installed Chrome.
2. **As an agent:** run `folo mcp` on a 200-image copy of the R6 dataset. Register and check it, train, evaluate,
   compare, `predict_batch`, export, and try the error paths.
3. **Desktop:** build FOLO Desktop in release, point it at the same server, take a tour and screenshots.

- Build: `folo 0.1.0 (a45e82579e94 2026-11-08)`, main at #87.
- Machine: Windows 11, i7-14700KF, 32 GB, RTX 4080 SUPER (Vulkan), Chrome 156.
- **The machine was shared.** A long COCO training run kept the GPU at about 72% utilization and used 10 GB of
  VRAM, and other agents were building. Throughput numbers are upper bounds on wall time, not benchmarks.
- The MCP client was a 120-line Python stdio client: the 2025-11-25 handshake, plus one 2026-07-28 direct call.
  Token counts below are bytes / 4.
- Screenshots are in `D:/folo-dogfood2/shots` on the dev machine (not in git).

## Verdict
**FOLO is in good shape for both a person and an agent.** All three jobs were completed. The README quickstart
works as written: training took 138.7 s to mAP50 0.955, and the README says 137 s and 0.948. Studio did
everything asked of it: register a dataset, start and watch a run live, queue a second run, compare runs, open
the PR curves and the confusion matrix, export, and run the Playground on the server, on WebGPU and on the wasm
CPU, with identical boxes on all three. The MCP surface is the best part:
- every error path tried (31) returned a stable code, a clear message and usually a useful hint;
- most DOGFOOD-1 fixes held: F2 was not re-tested, F15 is still deferred, and F16 has one partial regression;
- `tools/list` stayed at 45 KB.

Nothing blocked a job outright. The worst problems were these:
1. **The Desktop release build crashed rustc 3 times in 3 attempts on this machine**, with the shipped
   thin-LTO + `codegen-units = 1` profile. Turning off LTO and using 16 codegen units built it (D1). That is a
   workaround, not an isolated cause.
2. **Every front end shows a wrong timing split.** GPU time is counted as postprocess or "NMS", for example
   "model 0.6 ms, NMS 17.5 ms" (X1).
3. **Short runs on small datasets silently spend almost all their steps in warmup**, and with no mosaic. An
   agent's first 12-epoch run ends at mAP 0.000 with no explanation (X2).
4. **Studio's Compare hides the setting that matters most:** that one run was fine-tuned and the other trained
   from scratch (S1).

## DOGFOOD-1 fixes: held or regressed
| # | fix | status now | evidence |
|---|---|---|---|
| F1 | `get_dataset` lists errors first, with `list_dataset_issues` | **held** | clean r6-200: `0 errors, 0 warnings, 15 infos`, 766 B; `list_dataset_issues {severity: error}` 292 B |
| F2 | `gpu_out_of_memory` with a short message | not re-tested | the GPU was shared, so no OOM was provoked |
| F3 | unknown arguments rejected | **held** | `lr` gives "did you mean `lr0`?"; `model` gives "did you mean `arch`?" |
| F4 | `register_dataset` idempotent | **held** | the same folder written as `D:\folo-dogfood2\r6-200\` (backslashes, trailing slash) returned `ds_4cf3`, "already registered" |
| F5 | CLI `--resume` refuses changed flags and completed runs | **held**, with a nit | `run_not_resumable` with the command to run; but the hint's `--data` path has `\\` separators (P3) |
| F6 | `folo stop` / Ctrl+Break | **held** | `folo stop <run_dir>` gave `status: stopped` and the `done` record 0.64 s later |
| F7 | truncated JPEG caught | **held** | `head -c 2000` gives `corrupt_image` "truncated JPEG" (its message had a run of spaces: fixed here) |
| F8 | `--json-events epoch` | **held** | 3 epochs: 7 records |
| F9 | smaller `tools/list` and `describe` | **held** | `tools/list` 45.0 KB (17 tools; 32 KB read-only); `describe --json` 73 KB, `--no-mcp-tools` 29.6 KB, `describe train` 6.7 KB |
| F10 | `folo://runs/{id}/epochs` | **held** | 4.9 KB for 12 epochs, against 31 to 58 KB for `/metrics` |
| F11 | `device_not_found` names the adapters | **held** | CLI and MCP, with the same hint |
| F12 | `/` paths in JSON | **held for JSON records**; still mixed elsewhere | human CLI output, hints, MCP error messages, run logs and `folo_info.data_dir` (P3) |
| F13 | f32 floats without noise | **held** | `0.9545887`, `lr0: 0.01` |
| F14 | one error-code vocabulary | **held** | bad arch is `invalid_config` everywhere (its message still starts with "invalid arguments:") |
| F15 | effective LR visible | still deferred; it now shows in Studio | Studio's run Config and Compare show "LR 0.01" while the run used 0.001429 (S5) |
| F16 | small items | mostly held | `--lr0 -1` reaches the range check; `export` of a yaml is "not a .folo weights file"; CLI export never overwrites (`output_exists`); `--min-severity` works. **Partly regressed:** `predict` on a non-image still says "image error at : The image format could not be determined" (M4). Still deferred: annotated images tagged with class ids, `Run.device` vs `predict.device` |

## Timing
| what | time |
|---|---|
| `cargo folo doctor`, cold (own target dir, `-j 6`) | 5 min 18 s |
| the next `cargo folo` call: rebuilt folo-cli for no visible reason | 2 min 46 s (C2) |
| `dataset synth` 1000 images at 320 | 1.1 s |
| `train` folo-n, 30 epochs, 320 (README quickstart) | **138.7 s** (2 min 20 s wall), mAP50 0.955 / mAP50-95 0.823, 200 to 320 img/s |
| `predict` 200 val images, GPU | 6.6 ms per image (pre 0.6, inference 0.8, **post 5.2**; X1) |
| `export` folo-f16 / ONNX | 0.06 s / 0.07 s; ONNX `--decode` matches the native top boxes to 0.1 px in onnxruntime 1.31 |
| `npm ci` + `npm run build` | 23 s |
| `scripts/build-wasm.sh` (release) | 1 min 47 s |
| `cargo build --release -p folo-cli --features embed-web` | 3 min 37 s |
| Studio: register a folder (1000 images) | 0.44 s |
| Studio: folo-n 8 epochs at 320 / fine-tune 5 epochs | 34 s / 20 s |
| Studio Playground, folo-n 320, 320×320 image | server 8.8 ms (post 7.3); WebGPU **15.0 ms** (first load 4.8 s: runtime, weights, compile); wasm CPU 93 ms |
| MCP `initialize` / `tools/list` / `folo_info` | 39 ms / 38 ms / 753 ms |
| MCP `register_dataset` (200 images, 1920×1080) | 53 ms |
| MCP `train` folo-n, 12 epochs, 640, 160 images | 153 s in training (165 s until `completed`); about 11 s per epoch, 15 to 19 img/s (X3) |
| MCP `train` folo-n, 12 epochs, 480, batch 8 | about 90 s |
| MCP `val` 20 test images, 1080p | 0.37 to 0.41 s (16 to 19 ms per image) |
| MCP `predict_batch` 20 images / `predict` one 1080p frame | 385 ms / 16.5 ms (pre 9.6, inference 0.8, post 6.0) |
| MCP `export` ONNX / folo-f16 | 46 ms / 28 ms |
| Desktop `cargo build --release` (own target dir) | **failed 3 times** (5 min 5 s, 36 s, 20 s); 3 min 51 s with LTO off and 16 codegen units (D1) |
| Desktop Local GPU, 320×320 image | 19.3 ms ("pre 1.2, model 0.6, **NMS 17.5**") |

## Token cost (MCP)
| output | size | note |
|---|---|---|
| `tools/list` | 45.0 KB (about 11.3k tokens) | `run_status` 5.4 KB, `train` 3.7 KB, `list_runs` 3.7 KB; 32 KB with `--read-only` |
| `run_status` while training | 0.8 to 1.9 KB | the one-line text summary is excellent |
| `get_dataset` / `register_dataset` | 0.8 / 0.6 KB | |
| `val`, `compare_runs`, `list_runs`, `predict_batch` | 0.5 to 1.5 KB each | `predict_batch` keeps per-image boxes in a resource (5.7 KB for 20 images) |
| `predict` with `conf: 0.001` | 18.6 KB | 182 boxes; fine, since the agent asked for them |
| `folo://runs/{id}/epochs` | 4.9 KB for 12 epochs | the one to read |
| `folo://runs/{id}/metrics` | 31 KB (120 steps), 58 KB (240 steps) | as documented |
| `diagnose_run` prompt | 6.6 KB | |
| `folo://docs/MCP` | 26 KB | |
| `folo://runs/{id}/samples/12` | **250 KB** | a base64 JPEG grid of 16 frames at 1080p |
| `folo://datasets/{id}/images/{image_id}` | **350 KB** | the full 1920×1080 original, with no thumbnail option (M2) |
| subscription traffic | 110 `resources/updated` notifications in 165 s | under the 1 per second cap |

## Friction points
Severity: **blocker** stops the job; **annoying** costs time, tokens or a wrong conclusion; **polish** is cosmetic
or inconsistent. The IDs group findings by surface:
- **X**: across surfaces
- **C**: CLI
- **M**: MCP
- **S**: Studio
- **D**: Desktop
- **P**: paths and docs

### Across surfaces

#### X1. The timing split counts GPU compute as postprocess or NMS (annoying)
- **Repro:** run any prediction on the GPU and read the timing:
  - `folo predict --model runs/shapes-n/weights/best.folo datasets/shapes/images/val`: "pre 0.6, inference
    0.8, post 5.2"
  - Studio Playground on the server: "Inference 0.87 ms 10%, Postprocess 7.34 ms 83%"
  - MCP `predict`: `inference_ms: 0.84, postprocess_ms: 6.03`, with 0 boxes
  - Desktop Local GPU: "model 0.6 ms, NMS 17.5 ms" for 4 boxes
- **Contrast:** the same model on WebGPU in the browser reports "Inference 12.7 ms, Postprocess 0.90 ms".
  BENCHMARKS.md puts the folo-n forward pass at about 3 ms, so a 0.6 to 0.9 ms "inference" means the timer stops
  when the GPU work is submitted, and the wait for the GPU lands in postprocess at readback.
- **Expected:** inference includes the GPU work, as on WebGPU, where decode and readback finish before the
  inference timer stops and postprocess is only CPU NMS and rescaling.
- **Fix:** stop the inference timer after the output buffer is mapped, or add a `gpu_wait_ms` field. Then check
  whether CPU postprocess really costs 5 ms. Until then, anyone tuning NMS is chasing the wrong stage.

#### X2. Short runs on small datasets are almost all warmup, with no mosaic, and nothing says so (annoying, high)
- **Repro:** MCP `train {dataset_id: r6-200, arch: folo-n, epochs: 12, imgsz: 640}` (160 train images). The
  first log line says "batch 16 × accumulate 4, … 10 batches = 3 optimizer steps per epoch, warmup 33 steps".
  The run has 36 optimizer steps, so **33 of them are warmup**. "closing mosaic for the last 10 epochs" means
  only 2 epochs use mosaic.
- **Result:** best mAP50-95 0.000 at epoch 7, and `run_status` says "completed … best mAP50-95 0.000". The same
  happens in Studio. For an 8-epoch run, the default Close mosaic of 10 means no mosaic at all; Studio does label
  that as "no mosaic", which helps.
- **Expected:** a warning in the first log line, in `run_status` and in Studio's summary when warmup is more
  than half of the optimizer steps or `close_mosaic >= epochs`, with a suggestion such as "fewer epochs of
  warmup or more epochs".
- **Cause:** this follows the spec. MODEL.md §8.4 sets `W = min(max(round(3 · spe), 100), (epochs − 1) · spe)`.
  The 100-step floor (as in Ultralytics) means any run with fewer than about 100 optimizer steps stays in warmup
  until its last epoch. With `auto`'s nominal batch of 64, 160 images make only 3 steps per epoch.
- **Fix:** add those warnings. Also consider replacing the 100-step floor with a fraction of the run (for
  example `min(100, 10% of T)`) when the run is short.

#### X3. 1080p training on a small dataset runs at 15 to 19 img/s (annoying; needs an idle-GPU check)
- **Repro:** folo-n at 640 on 160 frames of 1920×1080 plus 40 val frames.
  - MCP: about 11 s per epoch (the step records say `imgs_per_s` 18.8).
  - CLI with `--workers 16`: 34.2 s for 3 epochs. With the default 8 workers: 29.3 s.
- **For comparison:** examples/r6 reports 12 to 21 s per epoch for **2030** frames (about 100 to 170 img/s).
  Doubling the workers did not help, so this run was not limited by decode threads. The GPU was shared, so this
  report cannot pin it on FOLO.
- **Fix:** time it on an idle GPU. Then report per-epoch time split into load, train, val and checkpoint in the
  epoch record, so an agent can see where the time goes. The MCP `train` tool has no `workers` field at all.

### CLI

#### C1. Two user-facing messages had long runs of spaces (polish; fixed in this PR)
- `dataset check`: "it decodes with grey                  where the data is missing".
- `folo doctor`: "build it with              scripts/build-wasm".
- Both string literals had lost a word or a line break. Fixed in `crates/folo-data/src/check.rs` (now "grey
  rows", as CLI.md says) and in `crates/folo-cli/src/main.rs`.

#### C2. A second `cargo folo` call rebuilt folo-cli for 2 min 46 s with no source change (polish)
- **Repro:** `git switch -c x`, then `cargo folo doctor` (cold build), then `cargo folo dataset synth …`. The
  second call printed "Compiling folo-cli" and took 2 min 46 s. Every later call took about 2 s.
- **Cause unconfirmed.** One guess: `build.rs` re-runs when HEAD, the branch ref or `packed-refs` changes, and
  the ref `git switch -c` had just created was written during the first build.
- **Fix:** diagnose it first with `CARGO_LOG=cargo::core::compiler::fingerprint=info cargo folo --version` right
  after a fresh `git switch -c`. Keep the HEAD, ref and `packed-refs` watches (they keep the baked sha right). If
  it is the mtime race, the cost is one rebuild per new branch.

#### C3. Small items (polish)
- **Mixed separators in human output:**
  - "Run folder: runs\shapes-n"
  - "Weights: runs\shapes-n\weights/best.folo"
  - "datasets/shapes/images/val\000800.jpg: 4 boxes"
  - "yaml datasets/shapes\data.yaml"
- **The `--resume` hint mixes separators:** `--model C:/…/best.folo --data C:\\Users\\…\\shapes`.
- **Null hints:** `invalid_config` for `epochs 0` and for `imgsz 333`, and `run_not_active` for `stop_run` on a
  completed run, have `hint: null`. The `--lr0` range error does have a hint (`folo describe train --json`).
  Apply that to every range error.

### MCP

#### M1. `folo_info` in proxy mode reports the wrong data dir (annoying)
- **Repro:** `folo serve --data-dir D:/folo-dogfood2/studio-data`, then
  `folo mcp --server http://127.0.0.1:7878`, then `folo_info`.
- **Happened:** `"mode": "proxy", "data_dir": "C:\\Users\\himent\\.folo"`. That is the MCP process's own default,
  with backslashes, and it is not where the runs are.
- **Expected:** the server's data dir (add it to `/api/system` or `/api/health`), or `null` in proxy mode.
- **Why it matters:** an agent that builds paths from it (`exports/…`) looks in the wrong place.

#### M2. Image resources have no small form (annoying, tokens)
- **Read:** `folo://datasets/ds_4cf3/images/200`.
- **Happened:** a 350 KB blob, the full 1920×1080 JPEG. `folo://runs/{id}/samples/12` is 250 KB.
- **Expected:** a thumbnail, as the HTTP API has (`?thumb=256`). The MCP resource has no such option, so an
  agent that wants to glance at a frame pays for the full original.
- **Fix:** `folo://datasets/{id}/images/{image_id}?thumb=512` (or a `/thumb/{px}` template), and a smaller
  default size for sample grids, with the full size as an option.

#### M3. Per-class metrics carry class ids only (polish)
- **Where:** `val.per_class`, `last_epoch.val.per_class` and the epoch records.
- **Happened:** they hold `{"class": 0, …}` with no name, while `list_models` and `get_dataset` do carry the
  names.
- **Fix:** add `name` to each per-class entry. It is a few bytes, and it saves the agent a join.

#### M4. Some error messages are still raw (polish)
- `predict` on a non-image file: "not an image: image error at : The image format could not be determined".
  The path after "at" is empty, the same defect F16 fixed for `dataset check`.
- `register_dataset` errors print paths with `\\`, for example
  `'D:\\folo-dogfood2\\does-not-exist' does not exist`, and `unknown_format`. `predict`'s `path_not_found` echoes
  the path as given.
- `invalid_config` for `arch: "folo-q"` reads "invalid arguments: unknown variant …". The code and the message
  prefix disagree.

### Studio

#### S1. Compare hides the starting weights (annoying)
- **Repro:** run A is folo-n from scratch for 8 epochs. Run B fine-tunes the imported shapes-n for 5 epochs.
  Runs → select both → Compare.
- **Happened:** "Side by side · **2 settings differ**": Epochs (5 vs 8) and Duration (20 s vs 34 s). "Show 8
  identical settings" lists Dataset, Model, Batch, Image size, Optimizer, LR, Mosaic and Device.
  - Compare has no row for the start weights (`weights: m_67a4c8c1` vs none), and they explain 0.833 vs 0.168.
  - Close mosaic (5 vs 8 after clamping) has no row either.
  - MCP `compare_runs` on the same pair, in proxy mode, says "4 config fields differ".
- **Model page:** it shows "trained by studio-n-ft5" but not that it was fine-tuned from shapes-n.
- **Fix:** add "Start from" and "Close mosaic" rows to Compare's settings. Add the parent model to the model page
  and to the runs table.

#### S2. Model page shows a benchmark latency as the model's latency (polish)
- **Where:** Models → studio-n-ft5-best → Details.
- **Happened:** "Latency 5.18 ms at 640 px · End to end, one image · RTX 4080 SUPER · fp32", on a 320 px model.
  The number is the README's folo-n benchmark, not a measurement of this model.
- **Fix:** label it "folo-n benchmark (640 px)", or show the model's own Playground or `val` timing.

#### S3. Export stays on the server (polish)
- **Happened:** Export → ONNX writes `D:/folo-dogfood2/studio-data/exports/studio-n-ft5-best.onnx` and shows the
  path ("The file is on the machine running folo serve"). There is no download link, and there are no ONNX
  options (`--decode`, `--dynamic`, `--half`).
- **Why it matters:** fine on localhost, but a remote Studio user cannot get the file. MCP `export` also has no
  ONNX options (documented in MCP.md).
- **Fix:** a download button (`GET /api/models/{id}/weights?format=…` exists for f16; add `onnx`), and the
  three ONNX switches.

#### S4. The in-browser loader's error for a file that is not a .folo is raw (polish)
- **Repro:** the Playground's hidden "use a local .folo" input (the first `input[type=file]`, `accept=".folo"`)
  given a JPEG. A Playwright script did this by mistake; a person could do it by picking "All files".
- **Happened:** WebGPU and CPU fail with "invalid safetensors file: header length 5064878326892452095 exceeds
  the 100000000-byte limit". The JPEG then stays as that model's weights until the page reloads.
- **Fix:** reuse the CLI's message ("… is not a .folo weights file", F16), and offer to clear the local file.

#### S5. Smaller items (polish)
- **Effective LR:** the run page's Config and Compare show "Initial LR 0.01" while the run used
  `auto → adamw lr0 0.001429` (F15, still deferred).
- **No folo-x:** the New run model picker offers n, s, m and l. The CLI, MCP and `folo_info` have folo-x.
- **Image size:** New run defaults to 640 px on a dataset whose images are all 320×320. It could default to the
  dataset's size.
- **Truncated learning rate:** the stat on the run page reads "3.42 × 10…" at 1440 px wide.
- **Log noise:** `folo serve` logs every request at INFO, including Studio's 2-second `/api/runs` poll, so
  stderr is mostly noise. Log polling GETs at DEBUG.

### Desktop

#### D1. The documented release build crashes rustc on a 32 GB machine (annoying, high; it blocked job 3 as written)
- **Repro:** in `apps/desktop`, with `CARGO_TARGET_DIR=D:/folo-build/dogfood2-desktop`, run
  `cargo build --release -j 6` (what `cargo run --release` runs).
- **Happened:** the final `folo-desktop` crate failed three times in a row:
  1. after 5 min: `rustc-LLVM ERROR: out of memory` / `STATUS_STACK_BUFFER_OVERRUN`;
  2. after 36 s: `STATUS_STACK_BUFFER_OVERRUN`;
  3. after 20 s, at 1.2 GB working set: `STATUS_ILLEGAL_INSTRUCTION`.

  The machine had 13 GB of RAM and 23 GB of commit free, and other agents were building.
  `CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release` then built it in
  3 min 51 s, and the app ran fine. Another worktree built Desktop in release earlier the same day, so this is
  intermittent, probably memory pressure.
- **Expected:** the build the README documents works on a loaded dev machine.
- **Fix:** keep `lto = "thin"` + `codegen-units = 1` in a `dist` profile for releases, and make `release` use the
  defaults (or `codegen-units = 16`). Also document the override in apps/desktop/README.md.

#### D2. Desktop and Studio disagree on units and time zones (polish)
- File size: Studio shows 39.8 MiB, Desktop 41.7 MB.
- Log times: Studio uses local time (13:50:56), Desktop UTC (21:52:04).
- The Desktop run log line "transferred 417 of 417 tensors from D:/folo-dogfood2/studio-data\models\shapes-n.folo"
  mixes separators.
- The Desktop learning-rate chart reads "lr 0.14" with "× 10⁻⁴" in the corner, which takes a moment to decode.

Everything else in the Desktop tour worked on real data:
- Overview, the runs list, both run dashboards, the model page (PR curves, a confusion matrix that matches
  Studio's), the datasets screens and the Playground;
- Playground **Local GPU** (19.3 ms) on a validation image;
- `tools/shoot.ps1 -Scenario none -ExtraArgs "--server … --open run:<id>"`, which captures live screens.

### Docs

#### P1. Tool count and the ONNX snippet (polish; fixed in this PR)
- **Tool count:** README, QUICKSTART §9 and llms.txt said "16 tools". `tools/list` has 17, and llms.txt's list
  left out `list_dataset_issues`. MCP.md already said 17.
- **ONNX snippet:** QUICKSTART §6 fed `np.zeros((1, 3, 640, 640))` to the shapes model's static 320 px graph,
  which fails with "index: 3 Got: 640 Expected: 320". It now uses 320 and prints `(1, 2100, 7)`.

#### P2. Missing prerequisites (polish; partly fixed in this PR)
- **wasm-bindgen-cli:** `scripts/build-wasm.sh` needs it at Cargo.lock's exact version. QUICKSTART §7 now says so
  (the script already prints the install command).
- **Studio without the wasm runtime:** the README quickstart builds Studio without it, so a reader who follows
  it gets a Playground without WebGPU/CPU. That is fine for a quickstart; the Studio section says so.
- **onnxruntime:** the snippet needs `pip install onnxruntime numpy`. Not added.
- **Real-data screenshots:** apps/desktop/README.md did not say that `shoot.ps1` can capture real data
  (`-Scenario none -ExtraArgs`), or that it defaults to the debug build. Added.

#### P3. Path separators outside JSON records (polish)
F12 fixed the JSON records, but `\` still shows up in the CLI's human output, the `--resume` hint, MCP error
messages, `folo_info.data_dir` (proxy), and the trainer's log line about loaded weights (C3, M1, M4, D2). One
`display_path()` helper used for every user-facing path would finish the job.

## What worked smoothly
- **Studio.**
  - Registering a folder (Windows path, backslashes) took 0.44 s, with a clear "does not exist" message and hint
    for a bad path.
  - The live run page updated over SSE with mAP, samples, per-class metrics, logs and ETA. A second run queued
    behind the first.
  - Compare overlays the curves.
  - The PR curves, per-class table and confusion matrix are clear ("Most confused: square → circle 7.1%").
  - The Playground runs the same model on the server, WebGPU and wasm CPU with identical boxes, and the
    thresholds re-filter instantly.
- **MCP.**
  - Every error path tried returned the documented code, mostly with an actionable hint. That covered 31 tool
    calls plus an unknown tool, bad resource URIs, a garbage stdin line, an unsupported protocol version,
    `--read-only` and a dead `--server`.
  - The one-line text summaries make a transcript easy to read.
  - The FIFO queue, subscriptions, `predict_batch` with a resource link, collision-free exports (`-2`), proxy
    mode and the 2026-07-28 direct call all worked as documented.
- **CLI.** The quickstart matched its documented time and accuracy, ONNX export matched the native predictor,
  and `folo stop` and the dataset checks did what DOGFOOD-1 asked for.

## Top 10, ranked
1. **D1:** make the Desktop release build survive a loaded 32 GB machine: a `dist` profile for thin LTO with one
   codegen unit, and plain `release` for development.
2. **X1:** fix the timing split so the GPU wait counts as inference, not postprocess or NMS, on all front ends.
3. **X2:** warn when warmup takes most of a short run or `close_mosaic >= epochs`, and consider capping warmup.
4. **S1:** show the start weights and Close mosaic in Studio's Compare, and the model's lineage on its page.
5. **X3:** measure small-dataset 1080p throughput on an idle GPU, and break epoch time into load, train, val and
   checkpoint.
6. **M2:** thumbnails for image resources and smaller sample grids (350 KB and 250 KB today).
7. **M1:** `folo_info` in proxy mode should report the server's data dir.
8. **P3 / C3 / M4:** route every user-facing path through one `/`-normalising helper, and give every range error
   a hint.
9. **S3:** download exports from Studio, and expose the ONNX options in Studio and MCP.
10. **S2, M3, S5:** smaller polish: an honest latency label, class names in per-class metrics, folo-x in the
    New run picker, the effective LR in run configs.

## Fixed in this PR
- README, QUICKSTART §9 and llms.txt: 17 MCP tools; llms.txt lists `list_dataset_issues`.
- QUICKSTART §6: the onnxruntime check uses the 320 px input of the shapes model.
- QUICKSTART §7: `build-wasm.sh` needs wasm-bindgen-cli.
- apps/desktop/README.md: `shoot.ps1` on real data, and its default binary.
- `crates/folo-data/src/check.rs` and `crates/folo-cli/src/main.rs`: the two messages with runs of spaces (C1).
- llms.txt: links this report.
