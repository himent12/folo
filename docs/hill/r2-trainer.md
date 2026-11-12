# Hill-climb 2: trainer start-up, validation and loader caches (lane `trainer`)

Goal: what a user waits for around the (already fast) training step: the start of `folo train`, the first epoch's
pipeline compilation, `folo val`'s readback and the decode of large images. Baseline `main` 647b981f, built in
`D:/folo-build/hill2-base`. Round 1 (method, tools, earlier wins): [trainer-data.md](trainer-data.md).

## Method

- **Machine.** i7-14700KF (28 threads), RTX 4080 SUPER (Vulkan), shared by six lanes: GPU runs under
  `lock.sh acquire gpu trainer`, numbers are A/B pairs alternated inside one hold. "others" = cores used by other
  processes during a run (`bench/trainer/ab.py`'s accounting).
- **Start-up.** `bench/trainer/startup.py` (new): a fresh `folo train --json` per run, read from a pipe, stopped by
  PID at the first `step` record with `step >= --until-step N` (a timer kills a run that passes `--timeout`).
  `step` counts batches and the optimizer updates once per `accumulate = nbs / batch` batches, so at batch 16
  `--until-step 4` is process start to the first optimizer step; the default, 1, is the end of the first batch.
  `--warm 1` runs every binary once first (a second start) and then keeps each run alive 3 s after the step, so the
  pipeline cache is on disk for the next one; `--clear PATH` deletes a cache before every run (a cold start).
- **Compilation.** `FOLO_GPU_TRACE_COMPILE=1` (docs/GPU.md, Pipeline cache): folo-gpu sums each `Kernel::compile`'s
  `create_shader_module` time and total time and prints `folo-gpu: compiled N pipelines, module X s, total Y s` to
  stderr every 25 pipelines and when `folo` ends. `bench/trainer/first_epoch.sh` reads that line from one row-10
  epoch (`folo train --data D:/folo-bench/data --model folo-n --epochs 1 --batch 16 --imgsz 640 --workers 8
  --close-mosaic 0 --device gpu:0 [--amp]`). The numbers below came from a temporary timer that did the same but
  printed only every 25 pipelines (`TRACE compile ...`), so they miss up to 24 pipelines.
- **Dataset open.** `crates/folo-data/examples/open_time.rs`: `cargo run -p folo-data --release --example
  open_time -- PATH [noverify]` times `folo_data::open_with` (here on `D:/folo-data/coco`, train + val), with the
  scan cache in `SCAN_CACHE=DIR` when set.

## PR 1: persistent pipeline cache + dataset scan cache

### Where the start-up time was (main, COCO, `folo train --model folo-n --batch 16 --device gpu:0`)

| phase | time | |
|---|---|---|
| process start to the first record (dataset open) | 75.6 s | one run, my release build compiling beside it |
| dataset open alone, cold / warm OS cache | 79.9 / 49.0 s | `open_with`, train + val, probing every header |
| of which: JSON parse + candidates (no probing) | 4.6-8.0 s | 448 MB `instances_train2017.json` |
| of which: one `is_file` stat per image | 1.8 s | 123 287 opens and closes on Windows |
| first record to the first step | 3.9 s | first batch + compiling the training pipelines |

Pipeline compilation in one row-10 epoch (f32, ~170 pipelines): 3.0-4.1 s, of which WGSL parsing and validation
(`create_shader_module`) 0.7-0.9 s and pipeline creation (naga's SPIR-V + the driver) the rest. A second run was
only 1 s faster, so the driver's own shader cache does not remove it.

### What changed

- **folo-gpu `pipeline_cache.rs`**: a `wgpu::PipelineCache` (Vulkan `VkPipelineCache`) every pipeline compiles
  through, loaded from and saved to `<FOLO_HOME>/cache/gpu/wgpu_pipeline_cache_vulkan_<vendor>_<device>.bin`. FOLO's
  header (magic, format, folo-gpu version, adapter name and driver, payload length, FNV-1a checksum) rejects a
  file from another driver / adapter / version, a torn or corrupt file, or one over 256 MB; then the cache starts
  empty (and wgpu + the driver check their own headers too). A saver thread writes atomically after a quiet second;
  `folo` writes what is left when a command ends. The device asks for `Features::PIPELINE_CACHE` only when a
  folder is set and the adapter has it (`context/device.rs`, a neighbour's file: three lines).
- **folo-data `dataset/cache/`**: the scan cache. Per folder: each image's header probe and each YOLO label's parse
  with the file's length and modification time, checked against one directory listing (no per-file stat); per
  COCO annotation set: everything derived from the JSON files, checked against their stamps. `open_jsons` is split
  into the JSON half (`read_annotations`, cacheable) and the file-system half (image folder, candidates). Results
  are exactly those of an uncached open (tests compare whole `LoadedDataset`s, with corrupt and stale caches).
- **folo-cli**: `init_caches()` turns both on under `$FOLO_CACHE_DIR` or `<FOLO_HOME>/cache`; `FOLO_NO_CACHE=1`
  leaves them off. Library defaults are off, so tests write nothing to the user's home.

### Results

COCO open (`open_with`, train + val, 123 287 images), same binary: **51.1 s cold → 0.38 / 0.29 s** with a warm scan
cache (24 MB COCO entry + 6 MB of folder entries).

**`folo train` on COCO, process start to the first optimizer step** (head 8c249e96, `bench/trainer/startup.py --bin
base=D:/folo-build/hill2-base/release/folo.exe --bin new=<this PR's folo.exe> --warm 1 --rounds 3 --until-step 4
--out D:/folo-data/runs-hill2/trainer/startup2 -- --data D:/folo-data/coco --model folo-n --epochs 1 --batch 16
--imgsz 640 --device gpu:0` with `FOLO_CACHE_DIR` a fresh folder, alternated in one GPU lock hold; the machine was
heavily loaded (main's runs swing 78-128 s; the script records no core counts):

| | main 647b981f | this PR |
|---|---|---|
| second start (warm caches): step 4 (first optimizer step), median (range) | **107.2 s** (78.1-127.5) | **3.5 s** (3.5-4.1) |
| per run, alternated | 107.2 / 127.5 / 78.1 | 4.1 / 3.5 / 3.5 |
| first start (the untimed warm-up run, no cache yet) | 310.4 s | 117.3 s |

An earlier measurement (head 89d3ce2d, lighter load), taken before `--until-step` existed, timed the first `step`
record, i.e. the end of the first batch, not the first optimizer step (`--until-step 1`;
`bench/trainer/startup.py --warm 1 --rounds 3 -- --data D:/folo-data/coco --model folo-n --epochs 1 --batch 16
--imgsz 640 --device gpu:0`, alternated in one GPU lock hold, `FOLO_CACHE_DIR` set to a fresh folder for the new binary, my workspace test build running beside it):

| | main 647b981f | this PR |
|---|---|---|
| second start (warm caches), median (range) | 68.4 s (67.9-70.5) | **3.7 s** (2.9-4.7) |
| first start (no cache yet; the untimed warm-up run) | 48.7 s | 41.8 s (a directory listing instead of a stat per image) |

**First epoch, row-10 set** (`bench/trainer/first_epoch.sh`: main, this PR with the pipeline cache deleted
before the run, this PR with the cache the previous run left; 3 rounds alternated; the compile total is the
temporary timer's last report, every 25 pipelines, so it misses up to 24 of them; see Method):

| | main | PR, cold cache | PR, warm cache |
|---|---|---|---|
| f32 epoch 1 (`duration_s`) | 6.84 / 6.28 / 5.15 s | 6.79 / 6.37 / 4.87 s | 5.79 / 6.13 / 5.89 s |
| f32 compile total (of which `create_shader_module`) | not timed | 2.29 / 1.85 / 1.99 s (1.25 / 1.01 / 1.10) | 1.24 / 1.29 / 1.42 s (1.03 / 1.07 / 1.14) |
| AMP epoch 1 | 2.96 / 3.00 / 4.33 s | 4.06 / 3.53 / 6.19 s | 3.30 / 3.35 / 5.28 s |
| AMP compile total (of which `create_shader_module`) | not timed | 2.21 / 2.09 / 3.08 s (1.12 / 1.03 / 1.52) | 1.51 / 1.59 / 1.77 s (1.24 / 1.32 / 1.40) |

What this says:
- The persistent cache removes most of the **pipeline-creation** part (cold 0.8-1.6 s → warm 0.2-0.4 s). On this
  machine the NVIDIA driver's own shader cache already holds most compiled pipelines after any earlier run: the
  very first AMP run of this binary compiled for 31.1 s (the tensor-core kernels), every later one ~2 s even with
  FOLO's cache deleted. The FOLO cache is what a run gets when the driver's cache is cold or evicted (new driver,
  cleared cache, another machine), and it is the only cache on drivers without one.
- What remains (~1.0-1.4 s) is WGSL parsing and validation in naga (`create_shader_module`), which no pipeline
  cache covers. The <= 1 s target is therefore **not met** by this PR alone; see leftovers (parallel precompile).
- Epoch-1 times are inside the noise between all three (the epoch is 5-7 s with validation; the compile is 1-2 s
  of it): no slower on a cold cache.

**Row 10 not worse** (`bench/trainer/ab.py train ... --epochs 2 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0
--device gpu:0`, f32, 3 rounds alternated, 15-25 other cores busy): epoch-2 sustained median main **312.9 img/s**
(162.1-337.7), PR **309.8** (143.3-321.1): a tie.

Review: 3 Sol reviewers, one Opus fixer, 3 re-review rounds (merged as #183, 4dfdaf1d). What mattered: lexical cache
keys aliasing symlinked folders, a prune that could delete other files in the cache folder, cache files read whole
before any check, test subprocesses writing into `~/.folo/cache`, a GPU test that could not see lost reuse, and the
start-up script timing the first batch instead of the first optimizer step (rerun above). Merged with `--admin`:
`tiny_amp_run_learns` fails identically on main d3032648 (another lane's merge), Linux CI refused for billing.

## PR 2: validation downloads only the multi-label candidate pairs

### Where the time was
Round 1 left `folo val`'s scoring thread as the limit: every batch of 32 downloaded the decoded head `[B, A, 4 + nc]`
(32 x 8400 x 84 f32 = 90 MB) for multi-label NMS at conf 0.001.

### What changed
- folo-gpu `kernels::detect::multi_label_candidates` (`shaders/det_candidates.wgsl`): one thread per (image, anchor)
  over the decoded `[B, 4 + nc, A]` (the dense storage behind `decoded_on_device()`'s transposed view), counts its
  classes with a score `> conf`, reserves them with one atomic add into the image's counter and writes
  `(anchor, class, score bits)` up to a capacity; the count keeps going, so the host sees an overflow. The scores are
  copied, not recomputed.
- folo-vision `val.rs`: a batch downloads the pairs and the boxes `[B, 4, A]`; `nms::non_max_suppression_pairs` sorts
  the pairs back to `(anchor, class)` order and runs the same `finish` as `non_max_suppression_rows`. The capacity
  follows the most pairs an image had (+25%, at least 16 Ki) but never past half the decode (then that batch uses the
  full decode); an image with more pairs than room downloads its whole decode. Only with a positive normal `conf`.
- folo-cli tests: the three stop / resume tests train 1000-epoch runs that `train_until_stopped` always stops after an
  epoch (with a 120 s deadline), so they no longer race the trainer's speed.

### Results
`bench/trainer/ab.py val --bin base=D:/folo-build/hill2-base/release/folo.exe --bin new=<PR folo.exe> --rounds 3 --
--model D:/folo-weights/folo-n-coco.folo --data D:/folo-data/coco --device gpu:0`, run twice (6 rounds), alternated in
one GPU-lock hold, 16-21 other cores busy:

| | main 647b981f | PR |
|---|---|---|
| process wall, median (range) | 8.8 s (8.1-10.4) | **6.7 s** (6.2-8.0; one first run 11.8) |
| evaluation (`seconds`) | 7.3 s (6.7-8.7) | **5.2 s** (4.8-6.0; first run 9.4) |
| mAP50 / mAP50-95 | 0.44944638 / 0.30999148 | **0.44944638 / 0.30999148**, every run |

Per run (wall / evaluation / other cores), base then new: 9.79 / 7.46 / 20.6 and 11.81 / 9.37 / 17.1; 8.13 / 6.74 /
16.3 and 6.57 / 5.09 / 17.4; 8.39 / 6.95 / 16.8 and 6.37 / 4.83 / 18.2; 8.68 / 7.21 / 16.3 and 6.24 / 4.86 / 16.2;
10.37 / 8.75 / 18.5 and 7.97 / 6.04 / 21.4; 8.96 / 7.45 / 17.7 and 6.81 / 5.40 / 19.3. The `<= 6.5 s` wall target is
missed by 0.2 s at the median on this loaded machine; the evaluation itself is 29% faster. What remains is the forward
passes, the loader and host NMS.

## PR 3: `--cache ram`

### Where the time was
Round 1: for 1920x1080 JPEGs, decode (12 ms) and resize (5 ms) per image, four per mosaic sample, are ~89% of a
sample; the loader made 83-140 samples/s, below the step.

### What changed
- folo-data `ram_cache.rs`: `CacheMode::Ram` keeps every training image as `resize_long_side` made it, plus its decoded
  size, filled the first time a worker loads it. The transform takes a `Fetched { index, sample, orig_size }`: the
  mosaic's resize of a cached image is the identity, and the letterbox path letterboxes the cached pixels (a resize to
  the same size copies rows) but maps the normalised boxes and reports `orig_size` / `ratio` / `pad` with the original
  size's geometry. Batches are identical with and without the cache (pixels, targets, meta).
- The first version mapped boxes through the resized image's geometry: same pixels, targets off by float rounding
  (`ram_cache_gives_the_uncached_batches` caught it).
- The size is checked up front against 80% of the memory available to the process: the host's available memory and,
  on Linux, the tightest limit among the process's cgroups and their ancestors (found through `/proc/self/mountinfo`,
  so co-mounted v1 controllers and mounted subtrees resolve), less what other live RAM caches reserved.
- `folo train --cache none|ram`; a resume may change it.

### Results
`cargo bench -p folo-data --bench pipeline -- --data D:/folo-data/runs-hill2/trainer/synth1080/data.yaml --workers 16
--rounds 2 --no-stages --no-val [--cache ram]`, the set made with `folo dataset synth
D:/folo-data/runs-hill2/trainer/synth1080 --images 600 --imgsz 1920 --vary-aspect --image-format jpeg --val-frac 0.1
--seed 7` (~400 KB JPEGs, long side 1920), full train transform, 16 workers, under the CPU lock, alternated:

| round | no cache | `--cache ram` |
|---|---|---|
| 1 | 134.9 samples/s (118.9-134.9) | **801.9** (799.0-801.9) |
| 2 | 116.1 (49.9-116.1) | **717.7** (639.3-717.7) |

About **6x** after the cache is filled (the filling epoch runs at the uncached speed). Off by default: training
without `--cache` is unchanged.

Review: 2 Sol reviewers, an Opus fixer, then two rounds fixed by me; what mattered: the memory budget ignoring
cgroup limits (container, then nested scopes, then co-mounted / subtree mounts), tests that could not see the cache
stop working or store the unresized image, two loaders admitted against the same memory.

## Leftovers

1. **Per-epoch checkpoint writes off the training thread** (item 4 of the brief, not done: three PRs merged).
   The epoch event promises `last.folo` / `best.folo` on disk; moving the write needs the event to wait for it or a
   new event, a server / Studio contract change.
2. **Parallel pipeline precompile**: write the `(kernel, spec)` list a run compiled next to the pipeline cache and
   compile it on all cores when the next context starts, so the ~1.1 s of naga parsing (and any driver compile)
   overlaps the dataset open instead of the first steps. Needs a name → `KernelDef` table.

# Second wave (lane `trainer2`)

Same machine and method as above. Every A/B below alternates the binaries run by run inside one GPU-lock hold:
`647b981f` (the baseline, `D:/folo-build/hill2-base`), `ed265317` (the main this wave branched from) and the PR's
build. Load: the bench scripts sample the machine's CPU load before every run (`warm_start.sh`, a 1 s snapshot, so
noisy) or the cores other processes used (`ab.py`).

## PR 4: parallel pipeline precompile

### Where the time was
After PR 1 a warm start still compiled every pipeline on the training thread as the first step met it: 0.8-1.3 s
(f32) and 1.3-2.2 s (AMP) in this wave's runs, most of it WGSL parsing (`create_shader_module`), which no cache
stores. With the NVIDIA driver's cache cold (a new driver, a new machine, or simply new shader code) the first AMP
epoch compiled for 30 s, one kernel after the other.

### What changed
- folo-gpu `kernel_list.rs` + `context/precompile.rs`: `GpuContext::precompile(kind, shape)` records the kernels a
  context first uses (name + spec, in order) beside the pipeline cache and, at the next call with the same kind and
  shape, compiles that list on one worker thread per core. A thread that needs a kernel takes it ready, waits for
  the worker compiling it, or compiles it itself if no worker started it. No list for the shape: the newest list
  of the same kind (most kernels are shared). Same header and checksum as the pipeline cache, but the identity
  leaves out the driver (a list names kernels, not compiled code). Bounds: 4096 kernels, 1 MB, 64 lists per folder.
  Written only when the run used a kernel the file lacks (a run stopped early keeps the longer list).
- `kernels::find_def`: every kernel module lists its definitions (`DEFS`), plus the per-layout optimizer kernels, so
  a recorded name finds its definition again. `gpu_train` / `gpu_amp` assert that every kernel a GPU run and its
  validation used is in that registry (`GpuContext::kernels_without_registry`).
- Callers: the trainer (`train <arch> <f32|amp> head <head> <optimizer> nc <n>`, shape `imgsz <n> batch <n>`: tile
  sizes and bounds checks follow the shapes, see a list's lines), `folo predict` / `folo val` (`<command> <arch>
  <precision> nc <n>` with their imgsz / batch / head options) and `folo serve`'s model loads.
- `FOLO_GPU_TRACE_COMPILE=1` prints a second line: the time the threads that needed kernels spent getting them.

### Results
Warm start, one row-10 epoch (`OUT=D:/folo-data/runs-hill2/trainer2/ws-f32 bench/trainer/warm_start.sh train
base=<647b981f> main=<ed265317> new=<PR>`, and `AMP=--amp OUT=.../ws-amp`; 3 rounds, every binary with its own warm
cache folder; 647b981f has no compile timer):

| | 647b981f | ed265317 | PR |
|---|---|---|---|
| f32: compile on the training thread, per round | not timed | 1.30 / 0.93 / 0.82 s | **0.240 / 0.001 / 0.001 s** |
| f32: epoch 1 `duration_s`, median (range) | 4.37 s (3.45-10.92) | 2.92 s (2.52-3.72) | **2.03 s** (1.94-2.82) |
| AMP: compile on the training thread | not timed | 1.87 / 1.28 / 2.19 s | **0.008 / 0.020 / 0.018 s** |
| AMP: epoch 1, median (range) | 2.78 s (1.97-3.58) | 3.24 s (2.65-3.90) | **2.32 s** (1.26-2.33) |

(The PR's first f32 round waited 0.24 s for 5 kernels still on a worker; the AMP rows' remaining ~10 ms are the 3
tensor-core self-test kernels the context compiles when it is created, before any run could start a precompile.)

First-ever start, AMP (`OUT=.../cold-amp bench/trainer/cold_start.sh --amp`, 3 rounds): every run gets an empty
NVIDIA shader cache through `__GL_SHADER_DISK_CACHE_PATH` (its Vulkan driver honours it; a copy of the exe at a new
path or with bytes appended does not make the driver cache cold, it is keyed by shader code):

| | compile on the training thread | epoch 1 | process wall |
|---|---|---|---|
| ed265317, first-ever | 30.8 / 30.1 / 29.9 s | 30.8 s (30.5-34.0) | 32.2 s (32.2-36.0) |
| PR, first-ever (no list yet) | 28.3 / 31.3 / 27.4 s | 28.9 s (27.8-32.0) | 30.3 s |
| PR, driver and pipeline caches cold, list kept | **3.25 / 3.20 / 2.81 s** | **3.89 s** (3.48-4.16) | **5.40 s** (5.05-5.67) |

The workers' compile time summed over threads was 60-69 s there (the driver compiles in parallel, ~20x). A truly
first-ever run (no list at all) is unchanged: nothing says which kernels it will need before it meets them (see
leftovers).

`folo predict` of one image, process start to exit, warm caches (`ROUNDS=5 OUT=.../ws-predict warm_start.sh predict
...`, `D:/folo-weights/folo-n-coco.folo`, 33 pipelines): **647b981f 0.84 s (0.69-0.85), ed265317 0.66 s
(0.59-0.82), PR 0.47 s (0.46-0.52)**; the PR's predict spends 0.000-0.011 s on its kernels.

Row 10 not worse (`ab.py train --bin main=<ed265317> --bin new=<PR> --rounds 3 -- --data D:/folo-bench/data --model
folo-n --epochs 2 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0 --device gpu:0`, `FOLO_CACHE_DIR` warm): epoch-2
sustained main **344.1 img/s** (340.2-349.5), PR **339.1** (324.2-358.1): a tie (others 6.7-7.2 vs 7.9-12.0 cores).
Epoch-1 training img/s: 197.5 → 297.1.

### Negative results
- Parsing each WGSL module once (shared naga IR) was not needed: on the workers the ~1.1-1.4 s of parsing is spread
  over the cores, and the training thread's remaining time is 0.001-0.02 s on a warm start. It would also need the
  specialisation constants as pipeline-overridable constants, which cannot size workgroup arrays on every backend
  (GPU.md).
