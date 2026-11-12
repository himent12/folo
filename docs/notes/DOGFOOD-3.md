# Dogfood report 3: the docs site, pretrained weights, AMP and the NMS-free head on real data

A fresh agent covered what landed between report 2 (#93) and #131, as three people:
1. **A newcomer** following README.md → docs/QUICKSTART.md → the docs site (`site/`). They built and served the
   site, ran the README quickstart, and checked the pages against each other and against the code. A sub-agent
   cross-checked every command and number in README, QUICKSTART and AGENT_COOKBOOK against the clap definitions and
   docs/BENCHMARKS.md.
2. **A practitioner fine-tuning their own data.** They pulled `folo-n-coco` from the draft `weights-v1` release and
   fine-tuned it on a 1080p dataset (`fasteners`, 2,030 train / 203 val / 93 test frames at 1920×1080).
   They trained four ways: f32, `--head o2o`, `--amp`, and `--amp --head o2o`. Then they checked `--half` and
   `--tensor-cores`, the ONNX export in onnxruntime, `folo val --coco`, and the runs in Studio (Playwright, headless
   Chrome) and in FOLO Desktop (release build, `tools/shoot.ps1` on real data).
3. **An agent** driving it all over MCP (embedded and proxying `folo serve`) and `--json`. That included the queue
   control (`reorder_run`), `Run.resolved` and the run estimates, which merged during the session (#129, #130). The
   agent rebased and tested them.

- **Builds.** `folo 0.1.0 (7963ae414945)`, main at #126, then rebased onto `2cbd07b` (#130) for the queue,
  estimate and precision checks.
- **Machine.** Windows 11, i7-14700KF, 32 GB, RTX 4080 SUPER (Vulkan), Chrome 156, onnxruntime 1.31.1. The CPU was
  shared with other agents' builds and CI, and the FOLO Desktop build overlapped the first epochs of one run (noted
  where it matters). Treat timings as wall time on a busy machine, not benchmarks.
- **Clients.** The report-2 Python stdio MCP client (2025-11-25 handshake); token counts are bytes / 4.
  Screenshots are in `D:/folo-dogfood3/shots` on the dev machine (not in git).

## Verdict
**The new features work end to end on real data, and the docs site is a real improvement.** The site builds in
under a second, its link checker passes, and search finds `--amp`, `o2o`, `weights pull` and `queue` in the right
sections. On the practitioner's side:
- `folo weights pull folo-n-coco` worked against the draft release in 3.7 s, checksum verified.
- 20 epochs fine-tuned from it reached mAP50-95 0.432 on val in 6 min 14 s.
- `--head o2o` trained and then predicted without NMS.
- `--amp` trained 1.3× faster per mosaic-free epoch.
- The f32 ONNX export matched `folo predict` box for box with score difference 0.0000 once the inputs agreed.
- `val --coco` agreed with `val` (AP 0.433 against mAP50-95 0.432).
- Studio and Desktop showed the same numbers for the same model, down to the confusion matrix.

Over MCP, the queue control, `resolved`, the per-run warnings and the measured run estimates did what API.md says.
The estimate predicted 64.6 s for a run that took 65 s.

Nothing blocked a job. These were the worst problems:
1. **`folo serve` keeps about 7 GB of VRAM after its runs finish** (X1). Every other GPU job then slowed 3–7×: a CLI
   fine-tune's epochs went from 22 s to 59–99 s until the server was restarted.
2. **The pretrained weights exist only for the CLI** (X2). Studio's "Start from" and MCP `train { weights }` take
   model ids, so `folo-n-coco` is `model_not_found` until someone copies the file into the data dir.
3. **AMP over MCP fails late, and the advice is for the CLI** (M1). Embedded `folo mcp` has no way to ask for tensor
   cores, and `train { amp: true }` is accepted, then the run fails. The message suggests `folo train --amp` and
   `folo serve --tensor-cores`, neither of which an MCP client can use.
4. **The ETA was wrong by 2–5× on loader-bound runs** (X3, fixed here).
5. **Studio's "CLI command" for a fine-tune does not run** (S1: `--model` together with `--weights`, and a server
   model id as the weights), and its time estimate was 3.5× optimistic (S3).

## DOGFOOD-2 fixes: held or regressed
| # | fix | status now | evidence |
|---|---|---|---|
| X1 | timing split | **held** | `predict` on 1080p frames: inference 3.0 ms, postprocess 0.24–1.8 ms; o2o model: postprocess 0.41 ms |
| X2 | short-run warnings, capped warmup and close mosaic | **held** in the engine; **the UIs ignore the cap** | a 4-epoch fine-tune: `run_status` "warning: warmup takes 96 of the run's 128 optimizer steps …"; "close_mosaic 10 is more than half of the 15 epochs: mosaic closes for the last 7 instead". Studio and Desktop still draw 10 closing epochs (S2) |
| M1 | `folo_info.data_dir` in proxy mode | **held** | `"data_dir": "D:/folo-dogfood3/studio"` |
| M2 | image resources at 512 px, `?size=` | **held** | `folo://datasets/ds_ba61/images/2233?size=256`: 14.8 KB |
| M3 | class names in per-class metrics | still deferred | MCP `val` and the CLI `done` record still carry `{"class": 0, …}`; `folo val --json` has `name` |
| C3 / P3 | `/` in user-facing paths | **regressed in one new command** | `folo weights list/pull --json`: `"path":"D:/folo-dogfood3/home\\weights\\folo-n-coco.folo"` (C2, fixed here) |
| S1 | Studio's Compare and lineage | **mostly held** | the Runs table has a "Started from" column (`folo-n-coco`); the run header says "Fine-tuned from folo-n-coco"; Compare not re-checked |
| S3 | ONNX switches and download | **held** for the API (#129: `GET /api/models/{id}/weights?format=onnx`); MCP `export` with `decode` worked | |
| S5 | log noise | **held** | 11 `INFO` lines from `folo serve` in a session of Studio polling and 3 runs |
| D1 | Desktop release build | **held** | `cargo build --release` in apps/desktop: 6 min 15 s on a busy machine, no crash |
| F15 | effective LR | **held in the API** (`Run.resolved.lr0 0.001667`, #129); Studio still shows "Initial LR 0.01" (S4) | |

## Timing
| what | time |
|---|---|
| `cargo build --release -p folo-cli --features embed-web`, cold, own target dir | 4 min 0 s (3 min 23 s to 3 min 36 s incremental after engine edits) |
| docs site: `npm run check` / `npm run build` | 2.8 s / 0.4 s (25 pages) |
| `folo weights pull folo-n-coco` (10.6 MB, draft release) | 3.7 s, sha256 verified |
| README quickstart `train` (folo-n, 30 epochs, 320) | **82.5 s**, mAP50 0.955 / mAP50-95 0.823 (README said 137 s, 0.948 / 0.814; report 2 got the same 0.955 / 0.823) |
| Studio: register 1080p dataset (2,326 frames) | 0.75 s |
| `folo dataset convert fasteners fasteners-coco --to coco` | 0.8 s |
| fine-tune from `folo-n-coco`, f32, 20 epochs, 640 (CLI) | 373.7 s; best mAP50-95 **0.432** (val), 0.426 (test); mosaic epochs 23–33 s, mosaic-free 9.6–10.5 s |
| the same with `--head o2o` | 394.1 s; best 0.414 (val), 0.413 test (o2o branch, no NMS), 0.415 with `--head o2m` |
| MCP (proxy), `amp: true`, 15 epochs | 322 s (the Desktop build overlapped epochs 1–8); best 0.427; mosaic-free epochs **7.1–8.2 s** vs 9.6–10.5 s in f32 (1.3×); mosaic epochs about 24 s in both |
| MCP (proxy), `amp: true, head: "o2o"`, 15 epochs | 219 s; best 0.394 |
| `val` on the 93 test frames, f32 / `--half` / `--half --tensor-cores` | mAP50-95 0.4261 / 0.4258 / 0.4260 |
| `predict` one 1080p frame, f32 / `--half` / tensor cores (median of 19 after the first) | inference 3.01 / 3.01 / 2.95 ms; total 13.3 / 12.6 / 15.2 ms (JPEG decode and letterbox are 9–10 ms); the first image 52 / 60 / 124 ms (pipeline compile) |
| `export --format onnx` (raw, `--decode`, `--decode --half`, `--dynamic`) | 0.03–0.05 s each |
| onnxruntime CPU, decoded folo-n, 640 | 30–35 ms per image (dynamic rect input), 98 ms (static 640×640) |
| FOLO Desktop `cargo build --release` (own target dir) | 6 min 15 s |
| MCP `tools/list` | 48.9 KB (17 tools) at #126; **54.7 KB** (18 tools) after #129 / #130 |

## Token cost (MCP)
| output | size | note |
|---|---|---|
| `tools/list` | 54.7 KB (about 13.7k tokens) | `run_status` 6.5 KB, `train` 4.9 KB, `list_runs` 4.2 KB, `stop_run` 3.9 KB; up from 45 KB in report 2 |
| `folo://system/throughput` | **36 KB** | every reference row for 5 archs × 2 sizes × precisions × devices; an agent wants one row (M3) |
| `run_status` (running, `log_lines: 5`) | 1.6–2.0 KB | the warnings come first in the summary line |
| `run_status` (queued) | 0.9 KB | `queue_position` after #129 |
| `list_models` (7 models) | 2.9 KB | `head`, `trained_with`, `precision`, `fused` present |
| `compare_runs` (2 runs) | 1.0 KB | `config_diff` lists `head` and `name` (`amp` matched) |
| `val` / `predict` / `export` / `reorder_run` | 0.6 / 0.6 / 0.26 / 0.2 KB | |
| `folo://runs/{id}/epochs`, 15 epochs | 7.2 KB | |

## Friction points
Severity: **blocker** stops the job; **annoying** costs time, tokens or a wrong conclusion; **polish** is cosmetic
or inconsistent. IDs group findings by surface:
- **X**: across surfaces
- **C**: CLI
- **M**: MCP
- **S**: Studio
- **D**: Desktop
- **P**: docs and the docs site

### Across surfaces

#### X1. `folo serve` keeps its training runs' GPU memory after they finish (annoying, high; fixed by `fix/dogfood-3`)
- **Repro:**
  1. `folo serve --data-dir D:/folo-dogfood3/studio --port 7979 --tensor-cores`.
  2. Train two fine-tunes through it (MCP proxy, `amp: true`, 15 epochs each) and let both complete.
  3. With nothing training, `nvidia-smi` reports 8089 MiB in use; the idle baseline is about 1.1 GB.
  4. Run `folo train --weights folo-n-coco --data fasteners --epochs 6 --close-mosaic 2` from the CLI.
- **Happened:** that CLI run's epochs took 24 / 24 / 25 / **73 / 59 / 99 s**, 321 s in total. After stopping the
  server (8089 → 1056 MiB), the same command took **114 s** (epochs 23–24 s, then 9.7 s).
- **Expected:** a finished run gives its buffer pool back, or trims it to a small cap, so the shared GPU is usable.
  PR #128 saw the same effect from two runs side by side ("filled its memory and slowed both about 5×"). This is the
  idle case: the server holds the memory with nothing running.
- **Fix:** drop or trim the GPU buffer pool when the worker goes idle, or after each run. Also report the server's
  current pool size in `/api/system`, so Studio can show it. Engine / server.

#### X2. Published weights are CLI-only (annoying; fixed by `fix/dogfood-3`)
- **Repro:**
  - MCP: `train { "dataset_id": "ds_ba61", "weights": "folo-n-coco" }` returns `model_not_found` "model
    'folo-n-coco' does not exist", with the hint "list_models … lists the models that exist".
  - Studio: New run → "Start from" offers only "From scratch" and the data dir's models.
- **Workaround that worked:** `cp ~/.folo/weights/folo-n-coco.folo <data dir>/models/`. It then lists as
  `m_d849e303` and fine-tunes from Studio and MCP (`metrics: null`, `trained_with: null`).
- **Expected:** the names `folo weights list` knows work everywhere a model is taken, as they do for
  `train --weights`, `val --model` and `export --model`.
- **Fix:** have the server list the downloaded published weights (`FOLO_WEIGHTS_DIR`) as models (source
  `published`, with the `about` text and the COCO metrics), or accept published names in `TrainConfig.weights`.
  Until then, the `model_not_found` hint could name the copy step when the id is a published name. Engine / server /
  MCP, then Studio.

#### X3. The training ETA was 2–5× off on loader-bound runs (annoying; fixed in this PR)
- **Repro:** fine-tune on 1080p JPEGs with mosaic. With MCP `run_status` every 30 s on the `amp + o2o` run:
  - at epoch 1 it said "ETA 103 s" when about 200 s were left;
  - at epoch 3, "89 s" for about 170 s.
- **Cause 1.** The per-batch clock started *after* the loader handed over the batch, so the wait for the loader
  (half of a mosaic epoch here) was never counted. #130 fixed this on main in parallel.
- **Cause 2** (what this PR adds). One rate covered the whole run, but a mosaic epoch took 24 s and a mosaic-free
  one 10 s. With the wait counted, the ETA overshot by 1.3–1.9× instead.
- **Fixed:** `Clock` keeps per-phase wall and step times. The remaining mosaic epochs are charged at the mosaic wall
  time per batch. The closed epochs are charged at their own wall time once they have run, and before that at the
  mosaic batches' *step* time, since without mosaic the loader keeps up (`eta_s`, unit-tested).
- **A/B** with the same 6-epoch fine-tune, back to back, ETA ÷ actual remaining time, at 15 / 30 / 50 / 75% of
  the run:

  | binary | 15% | 30% | 50% | 75% |
  |---|---|---|---|---|
  | before | 0.80 | 0.77 | 0.78 | 0.98 |
  | after | **1.08** | **0.96** | **0.96** | **0.95** |

  Under X1's memory pressure the old ETA read 0.19–0.47.
- **Also:** `eta_s` is rounded to 0.1 s (it was `119.9689651910345`). On a 4-epoch run the first epoch's pipeline
  compile still inflates the early ETA (2.3× at epoch 1). The throughput table already separates `startup_s`, and
  the ETA could too.

#### X4. Mosaic epochs on 1080p frames are loader-bound, so `--amp` only helps without mosaic (annoying)
- **Measured** (folo-n, 640, batch 16, 1080p JPEGs):

  | | mosaic epochs | mosaic-free epochs |
  |---|---|---|
  | f32 | 23–27 s | 9.6–10.5 s |
  | `--amp` | about 24 s | 7.1–8.2 s |

  The 2,030 frames are 1920×1080 JPEGs, so a mosaic sample decodes four of them.
- **Expected:** a pre-resized cache of the training images at the training size, or a decode at reduced size (JPEG
  DCT scaling) when an image is at least 2× the target. Report 2's X3 asked for the per-epoch split into load,
  train and validation time, and that is still open.
- **Fix:** engine (folo-data).

#### X5. Timings printed with f64 noise (polish; fixed in this PR)
- **Where:** `predict --json` per-image `timing` (`"inference_ms": 81.48469999999999`), its `done.mean_timing`
  (`49.05455500000001`), and MCP `predict` / `/api/predict` (`"preprocess_ms": 10.516300000000001`).
- **Fixed:** `folo_api::Timing` and `ValTiming.per_image_ms` serialize milliseconds rounded to 0.1 µs
  (`folo_api::json::round_ms`), and the CLI means use the same rounding. Tests: `folo-api` json tests and the CLI
  engine test, which checks every timing has at most 4 decimals.
- **Still open:** `mean_timing` includes the first image's pipeline compile: 124 ms for that image with
  `--tensor-cores`, against 3 ms after it. Over 20 images the "mean inference" read 105 ms. Leave the first image out
  of the means, or report a median.

### CLI

#### C1. `val --coco` on a YOLO dataset said no dataset was found (polish; fixed in this PR)
- **Repro:** `folo val --model best.folo --data fasteners --coco`.
- **Happened:** "--coco: could not detect a dataset in fasteners: expected a data.yaml, COCO instances JSON, or
  images/ + labels/ folders". `fasteners` has a data.yaml.
- **Now:** "--coco: D:/…/fasteners is a YOLO dataset, not a COCO one". The hint names `folo dataset convert fasteners <out> --to
  coco`, which worked: the converted copy evaluated at AP 0.433, with the full COCOeval table.
- **Test:** folo-data `coco_ground_truth_of_a_yolo_dataset_says_it_is_yolo`.

#### C2. `folo weights` JSON paths had `\` (polish; fixed in this PR)
- **Repro:** `folo weights list --json` and `folo weights pull NAME --json`.
- **Happened:** `"path":"D:/folo-dogfood3/home\\weights\\folo-n-coco.folo"`, a mixed separator (F12 / P3
  regressed in the new command). A pull of an already-cached name returned no `sha256` or `bytes`.
- **Now:** paths go through `folo_api::json::path`, and the cached `done` record has both fields.
- **Test:** `tests/engine.rs` (`weights_names_export_and_coco_val`) asserts the paths and the cached record.

#### C3. Export naming and casing (polish; fixed by `fix/dogfood-3`, except `-e2e`)
- **Decode name:** `export --format onnx --decode` next to an existing raw `best.onnx` writes `best-2.onnx`.
  `--half` gets `-f16`, but `--decode` gets no suffix, so the raw and decoded graphs are told apart only by a
  counter. Suggest `<name>-decoded.onnx`, and `-e2e` for an o2o model.
- **dtype casing:** the export record says `"dtype": "F32"`, while `val` and `predict` say `"f32"`.

#### C4. `folo doctor` says AMP is "not available" on a machine where `folo train --amp` works (polish; fixed by `fix/dogfood-3`)
- `amp  not available: … gpu:0 has tensor cores off (not requested …)`.
- The line is right about `device: auto` servers. A newcomer reads it as "this GPU can't". Say "available with
  `--amp` (it asks for tensor cores); `folo serve` needs `--tensor-cores`".

#### C5. Smaller items (polish)
- **`--resume` help (fixed):** the help text and the `invalid_config` hint said only `--workers`, `--device` and
  `--data` may change. `--project` and `--json-events` may too, as CLI.md says.
- **`--half` and tensor cores at batch 1:** no speed-up for folo-n on 1080p frames (inference 3.01 / 3.01 / 2.95
  ms). The 9–10 ms of JPEG decode and letterbox dominate. Worth a sentence in CLI.md, so nobody expects `--half` to
  halve the latency.
- **Transfer log wording:** "transferred 417 of 519 tensors (51 skipped, 102 missing)" for an o2o fine-tune. The
  copied o2o branch is in "transferred", which is right, but "missing" reads like a failure. "51 skipped (class
  outputs: 80 → 2 classes), 102 kept at init" would say what happened.

### MCP

#### M1. AMP over MCP: no way to ask for tensor cores, accepted then failed, CLI-only advice (annoying; fixed by `fix/dogfood-3`)
- **Embedded:** `folo mcp --data-dir D` reports `amp.available: false` ("tensor cores off (not requested …)").
  `train { amp: true }` returns "started run_868f", then the run fails. Its message advises "`folo train --amp`
  asks for them; start `folo serve --tensor-cores`". `folo mcp` has no `--tensor-cores` flag; only
  `FOLO_GPU_TENSOR_CORES=1` in the MCP client's config works, and MCP.md says so, but the error does not.
- **Proxy, CPU:** `train { amp: true, device: "cpu" }` was also accepted ("started"), then failed with
  `not_supported`. The CLI refuses before anything is written.
- **Expected:** `folo mcp --tensor-cores`, or embedded runs that ask for tensor cores as `folo train --amp` does.
  `train` should check `amp` against the capabilities synchronously and return `not_supported` with an MCP-shaped
  hint (`FOLO_GPU_TENSOR_CORES=1` in the server's `env`, or `amp: false`).
- **Fix:** engine / MCP.

#### M2. `train` accepts `weights` together with `arch` (polish; fixed by `fix/dogfood-3`)
- `train { arch: "folo-n", weights: "m_d849e303" }` is accepted. The CLI makes `--model` with `--weights` a usage
  error, because the weights decide the architecture. Reject it, or say in the summary that `arch` was ignored.

#### M3. Size and precision of new outputs (polish)
- **`folo://system/throughput` is 36 KB:** every reference row for every arch, size, precision and device. An agent
  deciding on one run wants the matching row or the `folo://estimate/...` resource. Offer filters (`?arch=&imgsz=`)
  or list the reference rows in a separate resource.
- **`tools/list` grew to 54.7 KB** (18 tools): `ResolvedTrain` is inlined in three `Run`-returning tools (#129 said
  so). A `$defs` reference would share it.
- **Per-class metrics still carry class ids only** (report 2's M3): `val.per_class[0]` is `{"class": 0, …}`.

### Studio

#### S1. The New run "CLI command" for a fine-tune does not run (annoying)
- **Repro:** New run → fasteners → folo-n → Start from `folo-n-coco` → Mixed precision on → "CLI command":
  `folo train --data D:/folo-dogfood3/fasteners --model folo-n --epochs 15 --batch 16 --imgsz 640 --amp --weights
  m_d849e303 --head o2m --name fasteners-n-2`.
- **Happened:** exit 2, because `--model` and `--weights` conflict. Without `--model` it is exit 3: "m_d849e303 does
  not exist", since the CLI takes a path or a published name, not a server model id.
- **Fix:** leave `--model` out when fine-tuning, and give the model's file path. The API does not expose it today:
  add `Model.path`, or a `cli` string from the server. Studio + API.

#### S2. Studio and Desktop draw the close-mosaic phase uncapped (polish)
- **Repro:** a 15-epoch run with the default `close_mosaic` 10. The engine logs "mosaic closes for the last 7
  instead" (the X2 cap).
  - Studio's progress bar hatches the last 10 epochs ("The hatched end of the bar marks the last 10 epochs").
  - Desktop's "close mosaic" marker sits at epoch 5.
  - Both are 4 epochs early; the measured epoch times show the switch after epoch 8.
- **Fix:** use the capped value: `close_mosaic_epochs` in the run's `config.json`, or
  `min(close_mosaic, floor(epochs / 2))`. Or have the API return the effective value in `Run.resolved`.

#### S3. The New run time estimate was 3.5× optimistic (annoying)
- **Repro:** fasteners, folo-n, 15 epochs, from `folo-n-coco`, Mixed precision on.
- **Happened:**
  - Studio said "Est. time ~2 min (~640 img/s)"; the run took 5 min 22 s at 80–93 img/s.
  - For folo-s, 100 epochs from scratch, it said "~19 min (~320 img/s)". The step-only benchmark rate leaves out
    the loader and validation.
  - Before `/api/system` loaded, the panel briefly read "Training on the CPU is roughly 14× slower" with
    "~2 min (~20 img/s)", and "0 steps / epoch . 1 batches of 16".
- **Now possible:** #130's `POST /api/runs/estimate` gave **60.6 s and 64.6 s** for 4-epoch runs that took **66 s
  and 65 s**, once it had measured rows. Use it.
- **Also:** "Est. GPU memory ~1.7 GiB / GPU memory unknown". The server reports `memory_bytes: 0` for the Vulkan
  adapter.

#### S4. The run page does not show precision, head or the effective optimizer (polish)
- **Repro:** the `mcp-amp` run page.
  - Nothing in the header or Config says Mixed precision.
  - The o2o run says nothing about its head.
  - Config shows "Optimizer auto, Initial LR 0.01", while `Run.resolved` (#129) says AdamW at 0.001667.
  - "Weights m_d849e303" is an id, while the header says "Fine-tuned from folo-n-coco".
- **Fix:** show `config.amp`, `config.head` and `resolved` on the run page and in Compare, and the model's name
  instead of its id.

#### S5. Smaller items (polish)
- **Path URLs:** `http://127.0.0.1:7979/datasets` (a path, not `#/datasets`) shows Overview, and the address
  becomes `/datasets#/datasets`. Studio uses hash routes; redirect a known path to its hash route.
- **Browser backends without the runtime:** the Playground offers WebGPU and CPU in a build without the wasm
  runtime. Picking one fails with "Prediction failed: this build of Studio does not include the in-browser
  runtime" and a console MIME error. The server knows (`folo doctor`'s `web_folo_wasm`), so disable the two with that
  tip.
- **Queue controls:** none in the UI yet. #129's API and `queue` SSE event are there.

### Desktop

#### D1. The run screen: close-mosaic marker and missing precision (polish)
- **Close mosaic:** the marker sits at epoch 5 of 15 (S2).
- **Header:** "folo-n · fasteners · RTX 4080 SUPER · Vulkan · started 4 min ago". It does not show Mixed precision or the
  head.
- **What worked:**
  - `--server … --open run:<id>` and `tools/shoot.ps1 -Exe <release>`;
  - the live AMP run (throughput sparkline, loss on a log axis, the overflow log lines);
  - the model page, which matches Studio's to the digit: 0.432 / 0.725 / 0.811 / 0.674 and confusion 81 / 19,
    59 / 41.

### Docs and the docs site

#### P1. README and QUICKSTART were stale against BENCHMARKS and the code (annoying; fixed in this PR)
- **README headline table** (also the site's landing page, "every number from docs/BENCHMARKS.md"):
  - CPU end to end read "26.9 ms … FOLO behind"; BENCHMARKS row 8 now has 21.7 ms against 24.8.
  - Training read "AMP still ahead"; FOLO AMP is 307 img/s against 252.
  - The paragraph under it said f16 had not been benchmarked.
- **README "Not yet"** still listed "f16 compute on the GPU", but `--half`, `--tensor-cores` and `--amp` all ship.
- **QUICKSTART:**
  - It said `--device gpu:<n>` "exits with code 4"; it is `device_not_found`, exit 3.
  - It said "FOLO ships no pretrained weights yet".
  - It never mentioned `--amp`, `--head o2o`, `--half`, `val --coco` or `--weights`.
  - §8's wasm check skipped `build-wasm.sh`.
- **Quickstart numbers:** README and QUICKSTART said 137 s and 0.948 / 0.814; two runs a day apart gave 82.5 s and
  0.955 / 0.823.
- **Tool count:** QUICKSTART and llms.txt said "17 tools" after #129 made it 18, and llms.txt's list lacked
  `reorder_run`. That is the third report in a row to find a stale count, so a test now checks it:
  `folo-mcp docs_name_every_tool` (README, QUICKSTART, MCP.md and llms.txt against `TOOLS`).

#### P2. The resize ONNX users must match was undocumented (polish; fixed in this PR)
- **Repro:** letterbox a 1080p frame with PIL's `BILINEAR` and run the decoded ONNX export (`--dynamic`, rect input as
  `predict` uses).
- **Happened:** all 43 boxes matched `folo predict`, but scores differed by up to **0.147**. Feeding both the same
  pre-resized 640×360 PNGs gave a score difference of **0.0000**.
- **Cause:** FOLO's resize averages `floor(src/dst)` blocks first (a box prefilter) on 2× or larger reductions.
  MODEL.md §9.2 said only "bilinear".
- **Fixed:** §9.2 now specifies the resize (pixel-centre alignment, 11-bit weights, the prefilter; a 1080p frame at
  640 is an exact 3×3 block average) and what PIL and OpenCV do differently. CLI.md's ONNX input section points to
  it.
- **Still open:** a static export is square (640×640), while `predict` letterboxes `rect` by default. With a static
  graph, 41 of 43 boxes matched. A `predict --square` switch, or a note in CLI.md, would help anyone checking parity.

#### P3. Docs-site and cross-page items (polish)
- **Fixed:**
  - AGENT_COOKBOOK and MCP.md pointed to a README section "Use FOLO from Claude Code / Codex" that is now "Use FOLO
    from an agent"; they are links now, so the site's checker covers them.
  - CLI.md's error example used `folo train` as the not-implemented command; it is `folo bench` now.
  - CLI.md's `bench` row lacked `--device`.
- **Open:**
  - The CLI reference is one table whose cells run to 2,137 characters (`train`). The site's table of contents
    cannot link to a command, and search lands on "Commands" as a whole. Per-command H3 sections would fix both.
  - BENCHMARKS.md:154 asks to list params/GFLOPs, which are listed; :269 calls tensor cores "the next step", and they
    are the next section.
  - WASM.md:136 still compares against 8.69 ms native end to end (now 5.18 ms).
  - ROADMAP's shapes gate quotes epoch 23 (0.975 / 0.878), not the final 0.980 / 0.898.
  - MCP.md and CLI.md disagree on the tool-schema size (about 45 KB against 43 KB; now 53–55 KB).
  - `site/` and `web/` both run Vite dev servers on :5173.

## What worked smoothly
- **The docs site.**
  - `npm run check` (2.8 s) checks every link and anchor, and `npm run build` takes 0.4 s.
  - Search finds sections (`--amp` → MODEL.md §8.8 and BENCHMARKS' AMP section; `weights pull` → CLI.md Weights).
  - The landing page renders the README; `npm run shoot` captured light, dark and phone screenshots, and the two
    looked at (landing, CLI) were clean.
- **Pretrained weights.**
  - `weights list`, `pull` (checksum verified, cached) and `train --weights folo-n-coco` ("transferred 366 of 417
    tensors"): 20 epochs to 0.432, against #128's 100-epoch 0.441 test result.
- **The NMS-free head.**
  - It trains, validates and predicts without NMS.
  - `--head o2m` evaluates the other branch of the same file.
  - Exports with `--decode`.
  - o2m vs o2o on the same budget: 0.432 vs 0.414 (f32), 0.427 vs 0.394 (AMP).
- **AMP.**
  - The loss-scale log lines are clear (4 overflow skips in epoch 1, then none).
  - `trained_with.amp` shows on the model in the API, MCP, Studio ("Trained in Mixed precision (f16)") and Desktop.
- **Export and evaluation.**
  - ONNX f32, f16, decoded and dynamic all load in onnxruntime 1.31.
  - `val --coco` / `--save-json` work on a converted COCO copy.
  - `--half` is within 0.0003 mAP of f32.
- **MCP queue control and estimates (#129, #130).**
  - `reorder_run { action: "top" }` moved the third run ahead of the second.
  - A running run gets `run_not_queued`, and `position: 0` gets a clear 1-based hint.
  - `queue_position` and `resolved` appear in `run_status`.
  - The estimate became `high` confidence and within 1–9% of the actual time after one measured run.
- **Studio and Desktop on real data.**
  - Register, dataset browser, live run with samples and logs, model page with PR curves and the confusion matrix,
    and the server Playground on a 1080p frame.

## Top 10, ranked
1. **X1** (engine / server): release or trim the GPU buffer pool when the server goes idle. Today a finished run
   holds about 7 GB and slows every other GPU job 3–7×.
2. **X2** (server / MCP, then Studio): make published weights usable from Studio and MCP, by listing the cached ones
   as models or accepting their names.
3. **M1** (MCP / engine): `folo mcp --tensor-cores` or self-requested tensor cores for `amp` runs, a synchronous
   `not_supported` from `train`, and an MCP-shaped hint.
4. **X3** (fixed): an ETA that counts loader waits and charges mosaic and mosaic-free epochs at their own rates.
5. **S1 + S3** (Studio): a CLI command that runs, and `POST /api/runs/estimate` instead of the static-rate estimate.
6. **X4** (engine): the loader-bound mosaic epochs on large JPEGs (a resized cache or DCT-scaled decode), and report
   2's per-epoch time split.
7. **S2 / D1 / S4** (Studio, Desktop): the capped close-mosaic phase, and precision, head and the resolved optimizer
   on the run screens.
8. **P1 / P2** (fixed): README, QUICKSTART and llms.txt against BENCHMARKS and the code, with a tool-count test, and
   the resize spec for ONNX users.
9. **C1 / C2 / X5** (fixed): the `val --coco` message on a YOLO folder, `/` paths in `folo weights`, timings
   without f64 noise.
10. **C3, C4, M2, M3, S5, P3** (open, polish): export names and casing, doctor's AMP line, `weights` + `arch` over
    MCP, the size of the throughput resource and of `tools/list`, Studio's path URLs and browser backends, and the
    docs-site leftovers.

## Fixed in this PR
- **ETA** (`crates/folo-vision/src/train/epoch.rs`): per mosaic phase, loader waits included, rounded to 0.1 s; unit
  test `eta_by_mosaic_phase`; docs/API.md's ETA paragraph (X3).
- **Timings** (`crates/folo-api`): milliseconds serialized to 0.1 µs (`json::round_ms` / `serialize_ms`);
  `predict`'s `mean_timing` uses the same rounding; docs/API.md says so (X5).
- **`val --coco`** (`crates/folo-data/src/dataset/coco.rs`, `crates/folo-cli/src/coco.rs`): "is a YOLO dataset,
  not a COCO one", with a convert hint (C1).
- **`folo weights`** (`crates/folo-cli/src/weights.rs`): `/` paths in `list` and `pull`; a cached `pull` reports
  `sha256` and `bytes` (C2).
- **`--resume`**: the help text and the hint name all five flags that may change (C5).
- **Docs:**
  - README: headline table rows, the f16 status, the quickstart numbers.
  - QUICKSTART: exit 3, pretrained weights, `--amp` / `--head o2o` / `--half` / `--coco`, the wasm check, 18
    tools, the quickstart numbers.
  - llms.txt: 18 tools with `reorder_run`, and a link to this report.
  - MODEL.md §9.2: the resize.
  - CLI.md: the ONNX input resize, `bench --device`, the error example.
  - AGENT_COOKBOOK.md and MCP.md: the README links.
  - site NAV: this report (P1, P2, P3).
- **Test** `folo-mcp docs_name_every_tool`: the tool count in four docs and llms.txt's list follow `TOOLS`.

## For the orchestrator
**Engine / CLI / MCP / server** (status after the `fix/dogfood-3` PR)
- X1 **fixed**: a finished run gives the pool back to the driver before it is reported finished
  (`GpuContext::release_memory`: the cached buffers, the cap back to the live bytes, the uniform arena, then a wait
  so wgpu frees them now), inference does after 30 s idle and drops its cached predictors after 5 min; `/api/system`
  reports `gpu_pool` (live, cached, peak bytes). Measured with
  `nvidia-smi` around a `folo serve` that trained folo-n at 640 px, batch 16, 4 epochs on 320 synthetic images:
  idle server after the run **8260 MiB** in use on main (1608 MiB before the run: 6.65 GB held), **2101 MiB** with
  the fix (1398 before: 0.70 GB, the device, its pipelines and the driver allocator's retained blocks;
  `gpu_pool.cached_bytes` 0); the run's peak live bytes were 3.5 GB.
- X2 **fixed**: `folo serve` and `folo mcp` list the published weights in the `folo weights` cache as models (id =
  the published name, `source: published`, `about`), so `train { weights: "folo-n-coco" }` works over HTTP, MCP and
  Studio's "Start from"; a name that is not downloaded is `model_not_found` with `folo weights pull <name>` as the
  hint. The COCO metrics are not in the file (`metrics: null`); the `about` line points to examples/coco.
- M1 **fixed**: `folo mcp --tensor-cores` (tensor cores must be asked for before the GPU is first used, so a start-up
  flag rather than a runtime toggle); `POST /api/runs`, MCP `train` and resuming refuse an `amp` run synchronously
  with `501 not_supported` (the same check as `/api/system`'s `capabilities.amp`), and MCP's hint names `amp: false`
  or the `folo mcp` / `folo serve` restart with `--tensor-cores`.
- X4 **deferred** (folo-data: resized cache or DCT-scaled decode, the per-epoch time split).
- X3 follow-up **deferred** (first epoch's startup in the ETA).
- X5 follow-up **fixed**: `predict --json`'s `mean_timing` leaves the first image out (with two or more) and
  `first_timing` reports it.
- C3 **fixed**: `--decode` exports are `<name>-decoded.onnx` (`-decoded-f16` with `--half`), in the CLI and the
  server's export; the export record's `dtype` is lowercase (`f32` / `f16`). `-e2e` for o2o models **deferred**.
- C4 **fixed**: doctor's `amp` line says "available … `folo train --amp` asks for tensor cores itself; `folo serve`
  and `folo mcp` need `--tensor-cores`" when the adapter has them and only the request is missing.
- C5 transfer wording **fixed**: "transferred N of M tensors from F; the other K keep their fresh initialisation (S of
  the file's tensors were not used: another shape, such as the class outputs of another class list, or not part of
  this model)". The `--half` latency note **deferred**.
- M2 **fixed**: a fine-tune takes the weights' architecture on the server too (it built `arch` and transferred
  almost nothing into it); MCP `train` with an explicit `arch` that differs from the weights' is `invalid_config`.
- M3 **deferred** (filtered throughput resource, `$defs` for `ResolvedTrain`, class names in per-class metrics).
- P2 follow-up (`predict --square` or a note) **deferred**.
- `memory_bytes: 0` for Vulkan adapters **deferred** (wgpu does not report it).

**Studio**
- S1: the CLI command for fine-tunes (needs `Model.path` or a server-built command). **Server side done** by
  `fix/dogfood-3`: `RunEstimate.cli`, `GET /api/runs/{id}/cli` (`{ cli, argv }`, feature `run_cli`) and
  `Model.path`; the "Start from" list gets published models (`source: published`). Studio to switch over.
- S3: use `POST /api/runs/estimate`; no CPU warning or zero schedule before `/api/system` arrives.
- S2: the capped close-mosaic phase on the progress bar. **Server side done** by `fix/dogfood-3`:
  `Run.resolved.close_mosaic` and `RunEstimate.close_mosaic` carry the capped value.
- S4: precision, head, `resolved` optimizer and LR, and the start model's name on the run page and in Compare.
- S5: path URLs to hash routes; WebGPU / CPU disabled when the build lacks the runtime (**server side done** by
  `fix/dogfood-3`: `/api/system` lists the `web_wasm` feature when the served build has it); queue controls (#129).

**Desktop**
- D1 / S2: the close-mosaic marker at the capped epoch; precision and head in the run header.
