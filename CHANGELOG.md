# Changelog

All notable changes to FOLO. Versions follow [Semantic Versioning](https://semver.org); the release process is in
[docs/RELEASE.md](docs/RELEASE.md).

## Unreleased

### Usability pass
Testers used Studio, FOLO Desktop and the server as people would and wrote down what got in the way; these are the
fixes.
- **Lists that stay current**: the models list catches up on every screen (Models, the Playground, a model page,
  Overview's Best models) when a run ends or is deleted, and when the tab is shown again; the Datasets list reads
  again after a change; Overview reads the sidebar's run list, so the two never disagree.
- **Runs**: Remove from queue deletes a run that never started (`DELETE /api/runs/{id}`, Studio and Desktop) instead
  of leaving a "Stopped" run with Resume; a run stopped before its first epoch says so and offers Start again; a run
  the server's shutdown stopped says "Interrupted" (`error_code: "interrupted"`); a failed run says what went wrong in
  plain words; "Epoch 1 of 5" everywhere, with no time left shown until the first epoch is done; run names are tidied
  as typed and capped at 64 characters (server, Studio and MCP); the Runs list sorts, keeps its state on Back, shows
  durations, row menus and its compare checkboxes; the log has one line per epoch.
- **A removed dataset is still named**: runs record their dataset's name and classes (`Run.dataset`), so their pages
  and every list say "shapes (removed)" instead of an id or "deleted dataset".
- **Datasets**: a folder that was moved is marked "Folder missing" with Remove from FOLO, never as deleted; pasted
  paths in quotes are taken and relative ones refused (datasets and labeling); default names are tidy and unique;
  removing a generated sample set can delete its images too (`?files=true`, refused while another dataset or a run
  reads them); a dataset registered again while runs use it gets its id back, and an id never moves to another
  dataset (tombstones in `datasets.json`); a folder of unlabeled images points to
  Label images; health guidance and counts that are true ("1 class").
- **Not found and errors**: one not-found state for every deleted run, model, dataset, benchmark and labeling job;
  error states name what could not be loaded; a real offline state that recovers by itself, its notice in the top bar;
  Studio no longer falls back to demo data by itself: a server that is slow to answer or not there shows "Server
  offline" and is asked again until it answers (demo data only with `?mock=1` or Settings › Use demo data); server
  hints written for agents stay out of people's way; messages that start with an identifier keep its case.
- **Dialogs and small things**: a double-click that opens a dialog no longer closes it; closing a dialog after its
  notice changed tone no longer throws; toasts expire and never stack; Back and Forward restore the scroll position;
  New run keeps its draft (for the server it was made on), follows a changed link and never starts on Enter alone; a
  dataset whose folder is missing cannot be picked to train; the Playground keeps its image, follows an edited URL,
  opens on the best model, marks a model's first (cold) prediction and lists one set of image formats; labeling has a
  reviewed state (Looks right, Enter; with changes, Enter or Ctrl+S saves).
- **FOLO Desktop**: delete runs and datasets with a confirmation; says when `folo serve` is lost; empty Runs and
  Models say what to do first; picks the best model by default; Studio's wording for learning rate, loss, plurals and
  stopped runs; a removed dataset by name.
- **GPU memory**: the server reports a Vulkan GPU's memory size (`/api/system` `memory_bytes`), and `folo doctor`
  prints it.

### Studio on fw
- FOLO Studio no longer runs on Svelte: it runs on **fw**, our own UI framework (`web/framework/`): JSX compiled at
  build time into cloned DOM templates with one fine-grained binding per dynamic part, on a push-pull signal graph,
  with Svelte 5's semantics where Studio relied on them (scoped CSS with Svelte's exact selector rewriting, event
  delegation, `bind:`, transitions, effect and creation order). Every screen and state looks exactly as before: all
  1193 screenshot states of the Svelte app were matched pixel for pixel, and the DOM too (ids included) except where
  the Svelte app had a bug (the command palette kept a page's commands after leaving it) or was not deterministic.
- Faster and lighter, measured on the production build against the Svelte app (`scripts/perf.mjs`, in-browser mock):
  first paint 36–52% sooner on every main screen, 18–67% less script on a cold load, 21–30% less JS heap, 33–43% fewer
  DOM nodes, startup JS 58 KB instead of 430 KB gzipped (each screen is a chunk of its own, the first one requested
  before anything else runs), and on a live run 55% less script per second (83% less on Overview).
- Each screen's chunk loads when it is first needed (or ahead: a link hovered, the palette opened, idle time).
- Tooling: `npm run check` is `tsc` (strict) instead of svelte-check, plus the token, contrast, icon and spacing
  checks; `scripts/parity.mjs` compares a shoot of the app with a stored one, and `scripts/perf.mjs --ref <git ref>`
  measures the app against another version's build. The lucide icons Studio draws are vendored
  (`web/framework/icons/lucide.json`); Svelte, svelte-check and @lucide/svelte are no longer dependencies.

### Pretrained weights
- The weights releases `weights-v1` (`folo-n-coco`) and `weights-v2` (`folo-n-coco-v2`) are published.
- **folo-s COCO weights**: `folo-s-coco` and `folo-s-coco-f16` (release `weights-v3`), folo-s trained from scratch
  on COCO train2017 for 56 epochs on one rented GPU, val2017 **AP 0.416 / AP50 0.580** (pycocotools-exact,
  cross-checked with pycocotools; f16 the same), 0.05–0.06 AP behind the published YOLO11s / YOLO26s
  ([examples/coco](examples/coco/README.md#folo-s-56-epochs-on-a-rented-gpu)). `folo weights pull folo-s-coco`.
- **MuSGD validated at scale**: that run is the first in which `auto` resolved to MuSGD (rule 3: lr0 0.01,
  momentum 0.9, warmup_bias_lr 0) at full scale; it trained without a non-finite batch. One run with no SGD twin:
  it shows the recipe works, not how much it gains over SGD on full COCO.
- The 300-epoch folo-n COCO run stays paused at epoch 62 (`folo-n-coco-v2`, AP 0.317); it can be resumed from its
  checkpoint.
- **Fine-tuning benchmark** (`bench/finetune/`, docs/BENCHMARKS.md, Fine-tuning on small datasets): FOLO's COCO
  weights against Ultralytics' YOLO26 of the same size on four small public datasets, three seeds each, both trained
  on the same rented GPU type and scored by one pycocotools scorer. folo-n is level with YOLO26n on three of them
  and 0.020 AP behind on BCCD; folo-s is 0.020 behind YOLO26s on BCCD, level on pills, and ahead on signatures
  (0.012) and brain-tumor (0.080, where YOLO26s does worse than YOLO26n).
- `folo weights pull` downloads over plain HTTPS first, without credentials (streamed, redirects followed, a stalled
  connection fails after 60 s, a progress line on a terminal), so once the repository is public it needs nothing but
  the `folo` binary. A refused download (`404`: a private repository, a draft, a missing release) falls back to
  `gh release download` as before; the repository is private for now, so today a pull still needs access to it
  through a logged-in `gh`. When `gh` is missing or fails then, one `gh_failed` error says both. No network, a
  timeout, a server error or a cut download is the new `download_failed`, without a fallback. The checksum and
  `.folo` checks are unchanged: nothing is installed unless both pass. `--json` adds `via` (`https`, `gh`, `null`).
- `FOLO_WEIGHTS_BASE_URL` downloads from a mirror (`<base>/<tag>/<file>`), `FOLO_GH` names the GitHub CLI program.
- The QUICKSTART has the two-command path to a first fine-tune (`folo weights pull folo-n-coco-v2`, then
  `folo train --weights folo-n-coco-v2`); `train` still never downloads by itself, and its hint says what to run.

### Vision benchmarks
- Make a benchmark from your own labeled images and run **any vision model** on it: a FOLO model, Claude Code or
  Codex looking at the images, or predictions from a model that runs anywhere else
  ([docs/BENCH.md](docs/BENCH.md); HTTP API under `/api/bench`, feature `vision_bench`; **Benchmarks** in
  Studio). A benchmark is frozen (images and truth copied in, a fingerprint over all of it) and carries a plan of
  stress conditions: twelve deterministic image changes (low light, overexposure, low contrast, defocus, motion
  blur, sensor noise, haze, colour cast, JPEG compression, low resolution, distance, occlusion) at severities 1
  to 5, served as the same bytes to every contestant. A coding agent can design the plan from the benchmark's
  description and a few of its images (one request).
- Every entry gets a score from 0 to 100 made of five axes with exact definitions: accuracy (mAP@0.5:0.95 with a
  bootstrap interval), robustness (the accuracy kept under the plan's conditions), self-labeling (the share of
  objects it labels by itself while 9 in 10 labels are right), calibration (1 − ECE) and localization. Studio
  shows the leaderboard with every entry's interval on one scale, and per entry where the score comes from, the
  accuracy kept per condition and severity, the reliability diagram, the label curve and the images it gets most
  wrong; two to four entries compare side by side.
- **Self-improvement loops** test whether a model can teach its successor: the model labels a pool of images by
  itself, a student (the same or a larger architecture) trains on those labels alone, the benchmark grades the
  student, and the student is the next generation's teacher. No true label enters training or the choice of the
  best checkpoint; the verdict is read against the first teacher's interval.
- New crate `folo-bench` (conditions, plans, scoring): pure, deterministic for every thread count, each number
  tested against a hand-worked case.

### Engine hill-climb
- Faster everywhere the GPU is used, measured against the previous build on the same machine
  ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#hill-climb-2026-11-09), notebooks in `docs/hill/`): folo-n
  batch-1 GPU forward 2.88 → 2.09 ms, end to end at batch 16 960 → 1898 img/s on tensor cores, the training step 242
  → 327 img/s in f32 and 345 → 537 with `--amp` (folo-s 121 → 151 / 194 → 273), a whole `folo train` epoch 216 → 347
  img/s (f32) and 302 → 500 (AMP), `folo val` on COCO val2017 29.6 → 8.9 s with the same mAP. On the CPU the training
  step is 25% faster and `predict_batch` 32–47%; batch-1 latency is unchanged.
- **`--optimizer musgd`** (Ultralytics 8.4's MuSGD: a Muon-orthogonalised update plus SGD) and two `auto` rule
  changes: rule 2 floors the AdamW LR of short runs at 0.001 (80-class runs from scratch learned almost nothing at
  the old 0.000119), rule 3 picks MuSGD for runs over 10,000 iterations (its +0.031 AP over SGD on the COCO proxy was measured with `--optimizer musgd` at momentum 0.937 and warmup_bias_lr 0.1; rule 3 ships momentum 0.9 and warmup_bias_lr 0, and no measured run has resolved `auto` to MuSGD (the proxy's 6,260 iterations and the shapes gate resolve AdamW), so the shipped rule-3 recipe itself had not been validated by a run when this merged; the folo-s COCO run under Pretrained weights, above, has since run it at full scale). Checkpoints resume with the rule they were
  started with (`TrainConfig.auto_rule`; a missing field is rule 1), so resumed runs keep their optimizer;
  `--set auto_rule=2` keeps SGD for a new long run. `--set KEY=VALUE` sets any `train` config key.
- The data loader defaults to up to 16 workers (was 8); the next epoch's loader starts once the previous epoch has taken
  its last batch (it works during validation and the checkpoint writes), and checkpoints download in one transfer
  per dtype and 128 MiB chunk.
- The GPU submits its command batch every 64 dispatches (was 1024), starting at 16 after an idle wait so the GPU
  starts early (`FOLO_GPU_FLUSH_EVERY` overrides it); `FOLO_GPU_PROFILE=1` turns on the per-kernel GPU timer
  (docs/GPU.md, Profiling).
- Gradient clipping runs in three multi-tensor kernels. A GPU run resumed from a checkpoint written by an earlier
  build continues normally, but not bit-identically to how that build would have continued while clipping is active
  (the sum of squares adds in another fixed order); resume within one build stays bit-exact.
- CPU `predict_batch` runs a batch as concurrent forwards of two images; each image's predictions are those it gets
  alone.

### Cloud compute
- Train a run on a GPU that is not in this machine: one rented from **RunPod** for the length of the run, or
  another FOLO server of yours (`remote`). The run stays a run of this server, with the same metrics, logs,
  samples and models ([docs/CLOUD.md](docs/CLOUD.md); `TrainConfig.compute`, HTTP API under `/api/cloud`,
  feature `cloud`; `folo cloud`; MCP `cloud_*` tools; Settings → Cloud and New run → Run on in Studio and
  Desktop).
- Nothing rented is left behind: every instance is written to a ledger before the provider is asked for it,
  released on every way out of a run (also a crash or a bug), terminated with retries until the provider confirms,
  and bounded by the run's time limit (`compute.max_hours`), which the instance also enforces on itself.
- A cloud run that trains does not need this server: close the laptop, and the next `folo serve` follows it again
  and fetches what it missed. A stopped or failed one resumes on a new instance from the checkpoint copied here.
- `folo serve --token` (bearer auth for a server that listens beyond loopback), `/api/uploads` (datasets and
  weights in retryable parts), cursors on a run's metrics and logs, `Range` on weights downloads and `checkpoint`
  on `POST /api/runs`: what makes any FOLO server usable as a worker, and useful on their own. `folo mcp --server`
  takes `--token` and `https://`.
- **RunPod checked against the real service** (2026-11-10, docs/CLOUD.md, *Checking a provider against the real
  service*): runs trained on rented A40 pods from launch to the weights in the local run folder, and what the real
  pods showed was fixed:
  - A pod found no GPU: the Vulkan manifest the NVIDIA runtime mounts names the driver's GLX library, which gives
    no device without an X server. The worker's setup now checks `folo doctor` and falls back to the driver's EGL
    library.
  - A worker that cannot be downloaded (a release that is not published, a private repository) is found before a
    pod is rented: the run fails with `cloud_boot_failed` at once, and the connection says so in Settings → Cloud
    and `folo cloud connections`. Before, the pod was rented and the run waited 15 minutes.
  - A pod whose setup fails terminates itself; the run now says so within seconds instead of at the boot timeout.
  - `weights/last.folo` could not be copied while epochs were shorter than its download (every attempt met a
    newer file), and each attempt held up the live metrics and a stop by about 10 seconds. A download now takes
    the rest of a file in one request when it fits (a file replaced in between: all of it in one), the copy runs
    on a thread of its own, and a failed one is tried again a minute later.
  - RunPod's proxy answered an upload with its own `404` a second after the worker had answered; such an answer
    is now tried again instead of failing the run with `cloud_sync_failed`.
  - A run whose pod ended while nothing followed it fails with `cloud_instance_lost` (it can be resumed), not
    `cloud_sync_failed`, and the cost of an instance that was gone when FOLO looked again is counted to when it
    should have terminated itself, not to the next start.
  - `Run.cloud.cost_usd` is there from the first phase (it was empty for the first minute of training), and a
    run's log says when its worker stops answering and when it answers again.
- New run's GPU menu lists the offers that can start first (a provider's cheapest are mostly out of stock), in
  Studio and Desktop.
- **FOLO Desktop**: the sidebar counts the runs that train as a chip (`Runs  2`) instead of a pulsing dot and
  "2 live", live marks breathe instead of throwing rings, and a connection's warning wraps under its row.
  **Studio**: Settings → Connection says `Connected` with a check, as its cloud connections do, instead of a
  ringed dot and "Live".

### Agent labeling
- Unlabeled images to a dataset: `folo serve` hands a set of images to **Claude Code** or **Codex** on the
  server's machine, Studio shows every box for review and correction, and the labeled images become a
  registered YOLO dataset ([docs/LABELING.md](docs/LABELING.md); HTTP API under `/api/label`, feature
  `agent_labeling`).
- Built to be cheap: one model request per batch of images with the agent's tools, config, MCP servers and
  session switched off (a Claude Code call's fixed cost is a few hundred tokens; Codex's drops from about
  20000 to about 8000), images scaled to at most 1024 px, and a run that ends on the first failure another call
  would only repeat. On 24 COCO photos the default (Claude Code, Haiku) found 79% of the objects with 92% of its
  boxes right for under a cent.

### Desktop
- The window works on Windows: Minimize, Maximize and Close in the title bar, dragging and double-clicking the
  title bar (and the sidebar's FOLO row) did nothing, so the window could not be maximized. They work now, and so
  does resizing from the top edge; also with the command palette or a dialog open, which stays open.
- Full screen: `F11`, or Toggle full screen in the command palette; the title bar's buttons work in full screen
  too (Restore leaves it).
- The window opens where it was closed: its size, place and maximized or full-screen state, and the theme, are
  kept in `~/.folo/desktop.json` (not for `--scenario`, `--width` or `--height`). It reopens on the display it
  was on, also with displays at different scales; a place on a display that is gone opens centred on the main one.
- FOLO Desktop runs on its own UI framework, **folo-ui** ([apps/desktop/ui/README.md](apps/desktop/ui/README.md)),
  instead of GPUI. Nothing you see changes: the same screens, layout, text, colours, keys and window behaviour.
  It is lighter: the UI renders on the CPU and redraws only what changed. Measured against the GPUI app on the
  same PC ([apps/desktop/ui/PERF.md](apps/desktop/ui/PERF.md)): the first frame in 70 ms instead of 442 ms, no
  GPU memory or GPU load (GPUI held 48–93 MB and kept 34 % of the GPU busy on a live run), a working set of
  58–69 MB instead of 90–100 MB and private memory of 42–53 MB instead of 153–166 MB, a live training run's page
  at 9 % of one core instead of 103 %, a smaller binary (the shipped build 30.4 MB instead of 33.9 MB) from 168
  crate versions instead of 465, and a clean release build in 140 s instead of 194 s.
- For developers: folo-ui keeps GPUI's API for what the app uses, so the app's code reads as before; the
  vendored GPUI and its patch are gone. `tools/shoot.ps1` and `shoot-all.ps1` render headless through
  `folo-desktop --shot` (no window, no cursor; all 98 shots in a few seconds, `-Scale` instead of the display's);
  `capture.ps1`, `capture-all.ps1` and `perf.ps1` check the real window off screen. CI now also builds and tests
  the app on Linux, renders a headless screenshot there, and type-checks folo-ui for macOS.

### Server
- DNS rebinding: a server on loopback (the default `--host 127.0.0.1`, also `--mock`) answers only requests whose
  `Host` is `localhost`, `127.x.y.z` or `[::1]` (any port, any case). A website whose name re-resolves to 127.0.0.1
  could read every dataset, image and weights file and start agent labeling from a visitor's browser; such requests
  are now `403 forbidden_host` before any handler runs, GETs, SSE and Studio's files included.
  `folo serve --allow-host NAME` (repeatable) adds a name for a tunnel or reverse proxy that keeps its own `Host`. A
  server bound beyond loopback (`--host 0.0.0.0`, cloud workers) is not checked: use `--token` there.

## 0.1.0 (2026-11)

The first release: a from-scratch Rust vision framework with a YOLO-style detector, its own tensors, autograd, CPU
and GPU kernels, training, evaluation and export, driven from one `folo` binary, an HTTP API, an MCP server, FOLO
Studio (web) and FOLO Desktop (GPUI).

Headline numbers, folo-n against Ultralytics yolo26n on an RTX 4080 SUPER and an i7-14700KF at 640×640
([docs/BENCHMARKS.md](docs/BENCHMARKS.md); FOLO in strict f32 unless noted):

| | folo-n | yolo26n (best config) |
|---|---:|---:|
| GPU forward, batch 1 (median) | **2.99 ms** | 4.95 ms |
| GPU forward, batch 16 | 1385 img/s | 1454 img/s (TF32) |
| GPU end to end, batch 1 / batch 16 | **5.18 ms** / **722 img/s** | 6.92 ms / 615 img/s (fp16) |
| Training step, batch 16, f32 / AMP | **219** / **307 img/s** | 202 (TF32) / 252 img/s |
| Training step peak memory, f32 / AMP | **3474** / **2022 MiB** | 4517 / 2646 MiB |
| CPU forward, batch 1 | **21.4 ms** (contended) | 22.8 ms |
| CPU end to end, batch 1 | **21.7 ms** (quietest normal-priority round) | 24.8 ms |
| COCO val2017 AP, 50 epochs from scratch | 0.314 | 0.409 (Objects365 + 245 COCO epochs) |

### Engine
- folo-core: tensors, views, dtypes, autograd with finite-difference gradient checks, CPU kernels (GEMM about
  1 TFLOPS, AVX2 epilogues, pack-free conv, deterministic reductions) and a backend registry (#11, #39, #55, #137).
- folo-gpu: wgpu compute (Vulkan, DX12, Metal, WebGPU) with its own WGSL kernels: tiled SGEMM (about 31 TFLOPS),
  implicit-GEMM conv, fused Conv+BN+SiLU, BatchNorm and optimizer kernels, a buffer pool, whole models on the GPU
  (#10, #31, #38). f16 inference (#80) and opt-in tensor-core f16 GEMM and conv on cooperative matrices (#89).
- folo-nn: layers, init, SGD and AdamW, LR schedules, EMA, `.folo` weight files (a safetensors-compatible format,
  docs/FORMAT.md) (#1, #21).
- folo-vision: the folo-n/s/m/l/x detector family (#4, #28), box ops, IoU family, NMS, decode, task-aligned assigner,
  mAP and confusion matrix with pycocotools parity (#5, #15, #20), the NMS-free one-to-one head, `--head o2o` (#109).
- CPU end to end under the yolo26n bar: fused letterbox, a direct stem conv, in-place parallel decode (#127); per-core
  conv efficiency (B packing, stride-2 phase split, register epilogue) (#137).

### Training
- Detection loss (TAL + BCE + CIoU + DFL), the trainer and predictor; the shapes gate passes on CPU and GPU
  (#27, #33). Resume (bit-exact), EMA, mosaic and the usual augmentations; the folo-data pipeline with YOLO and COCO
  readers, statistics, checks, conversion and synthetic datasets (#8, #29).
- GPU training end to end with the loss and assigner on the device (#46); mixed-precision training, `--amp`, on
  tensor cores with dynamic loss scaling (#101).
- COCO 2017 support with 6x less loader memory, a pycocotools-exact COCOeval and `folo weights` to pull pretrained
  weights (#72); `folo-n-coco`, val2017 AP 0.314 / AP50 0.455 after 50 epochs from scratch, in the release
  `weights-v1` (#128; a draft when 0.1.0 was cut, published on 2026-11-11).
- Short-run schedules, timing split and other dogfooding fixes (#69, #99).

### Inference and export
- `folo predict` and `folo val` on CPU or GPU, `--half` and `--tensor-cores` for f16 (#33, #80, #89).
- Export to `.folo` (f32 or f16) and to ONNX (opset 17, f32 or f16, raw or decoded, static or dynamic shapes),
  parity-tested against onnxruntime in CI (#87).
- folo-wasm: the detectors in the browser on WebGPU or the wasm CPU, about 14 ms for folo-n at 640 in Chrome (#51);
  shipped inside the release builds' Studio (#78) and, from this release, as the `folo-wasm.tar.gz` asset.

### Server, API and MCP
- `folo serve`: an axum HTTP + SSE API (docs/API.md, types in folo-api) with real datasets, runs, models, evaluation,
  export and predict, plus `--mock` demo data (#7, #42, #60).
- Capability reporting in `/api/system` (f16, tensor cores, AMP) (#108); `head`, `trained_with`, `precision` and
  `fused` on models (#117, #123); operating-point confidence, the resolved optimizer, queue control and ONNX download
  (#129); measured throughput, run estimates and a benchmark probe (#130); evaluation provenance (#138).
- `folo serve` releases a run's GPU memory when it ends and an idle model's after inference, lists published weights
  as models and refuses `--amp` up front where it cannot run (#140).
- `folo mcp`: an MCP server (rmcp, stdio) with 18 tools, `folo://` resources and prompts, embedded or proxying a
  running `folo serve`, able to drive real training end to end (#2, #36, #49).
- The CLI is agent-friendly: `--json` everywhere, stable exit and error codes, `folo describe --json`, `folo doctor`
  (#69).

### Studio
- FOLO Studio (Svelte 5, its own design system) embedded in the `folo` binary: Overview, Datasets, Runs (live
  metrics over SSE, compare, resume), Models (evaluation, export) and the Playground, which runs models on the server
  or in the browser (#3, #6, #13, #19, #57, #70).
- Design-review rounds 1 to 23; one responsive system from phone to desktop, with blank-section and overflow QA in CI
  (#126, #135, #141).

### Desktop
- FOLO Desktop, a native GPUI app with the same screens, a command palette, keyboard navigation and dark and light
  themes. It talks to `folo serve` over HTTP + SSE, starts `folo serve --mock` when no server answers, and runs local
  GPU or CPU inference in its Playground (#9, #32, #73).
- Design-review rounds 3 to 51 (#26 to #142).

### Docs
- README, QUICKSTART, CLI, API, MCP, MODEL, FORMAT, DATA, GPU, NN, WASM, METRICS and BENCHMARKS, an agent cookbook
  and `llms.txt` (#71); a static docs site generated from the Markdown with search and link checking (#122).
- Three dogfooding reports and their fixes (#59, #93, #134).
- Release builds for Windows and Linux with the embedded Studio, install scripts, checksums and a draft-release
  workflow on self-hosted runners (#22, #78, #104).

### Known issues
- No macOS build: the macOS job needs GitHub-hosted runners, which are unavailable for this repository, so
  `install.sh` stops on macOS. Build from source there.
- The repository is private: the install scripts need a token (`GH_TOKEN`, `GITHUB_TOKEN` or a logged-in `gh`).
- `weights-v1` (`folo-n-coco`) was a draft release, so only repository writers could `folo weights pull` it. It
  was published on 2026-11-11 (Unreleased, Pretrained weights).
