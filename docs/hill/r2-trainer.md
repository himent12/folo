# Hill-climb 2: trainer start-up, validation and loader caches (lane `trainer`)

Goal: what a user waits for around the (already fast) training step: the start of `folo train`, the first epoch's
pipeline compilation, `folo val`'s readback and the decode of large images. Baseline `main` 647b981f, built in
`D:/folo-build/hill2-base`. Round 1 (method, tools, earlier wins): [trainer-data.md](trainer-data.md).

## Method

- **Machine.** i7-14700KF (28 threads), RTX 4080 SUPER (Vulkan), shared by six lanes: GPU runs under
  `lock.sh acquire gpu trainer`, numbers are A/B pairs alternated inside one hold. "others" = cores used by other
  processes during a run (`bench/trainer/ab.py`'s accounting).
- **Start-up.** `bench/trainer/startup.py` (new): a fresh `folo train --json` per run, read from a pipe, stopped by
  PID at the first `step` record (emitted after the first optimizer step), so the number is process start to the
  first optimizer step. `--warm 1` runs every binary once first (a second start); `--clear PATH` deletes a cache
  before every run (a cold start).
- **Compilation.** A temporary build with a timer in `Kernel::compile` (not committed) sums the time spent compiling
  pipelines in one row-10 epoch (`folo train --data D:/folo-bench/data --model folo-n --epochs 1 --batch 16 --imgsz
  640 --workers 8 --close-mosaic 0 --device gpu:0 [--amp]`).
- **Dataset open.** A temporary example timing `folo_data::open_with` on `D:/folo-data/coco` (train + val).

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

(A/B tables below.)
