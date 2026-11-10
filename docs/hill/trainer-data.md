# Hill-climb: trainer data pipeline, host loop and validation (`hill/trainer-data`)

Goal: shorter `folo train` / `folo val` wall time on the dev machine (i7-14700KF, 28 threads, RTX 4080 SUPER) without
changing what a batch contains. Every change keeps batches bit-identical for a seed (any worker count) and resume
bit-exact; validation results are identical (same mAP to the last digit in every run below).

## Method

- **Tools.** `crates/folo-data/benches/pipeline.rs` (CPU only: single-thread stage table, loader img/s per worker
  count, `--data` for a real dataset) and `bench/trainer/ab.py` (alternates binaries, fresh process per run, reads
  `metrics.jsonl` / `folo val --json`, reports medians, best, spread and the cores other processes used).
- **Tracing.** Temporary `eprintln!` timers behind an env var (not committed) split an epoch into loader waits, step
  time, validation, checkpoint download and write, and validation into loader wait, forward + readback and host
  post-processing.
- **Locks.** GPU runs under `lock.sh acquire gpu`, CPU timing under `acquire cpu`. The machine was never quiet: other
  processes used **7-26 cores** during almost every run (column "others"), and in one session my GPU lock was taken
  over mid-run by another agent (those train numbers are discarded). Numbers below are therefore A/B pairs run
  alternately, not absolute bests.
- **Row 10 protocol** (docs/BENCHMARKS.md): the incumbent's own set `D:/folo-bench/data` (512 train / 64 val,
  640x640 JPEG, 3 classes), `folo train --model folo-n --epochs 2 --batch 16 --imgsz 640 --workers 8
  --close-mosaic 0 --device gpu:0 [--amp]`, sustained = the epoch-2 `imgs_per_s` of the last step (512 images over
  the epoch's wall time from its start, loader included), fresh process, 3 rounds.

## Where the time was

| what | measurement | finding |
|---|---|---|
| `folo val` on COCO | process wall 43-152 s for a 14-29 s evaluation | `open` probed all 123 287 image headers (train2017 too): ~6 ms for a file not opened since boot (antivirus / cold open; a re-open is 0.01 ms), 0.35 ms each at 28 threads |
| validation (COCO val2017, multi-label NMS at conf 0.001) | forward + full-decode readback 11-19 s, split 0.7-3.3 s, NMS 0.4-4 s, metrics 0.1 s | serial: the GPU waits for the host and vice versa; each batch of 32 downloads 32 x 8400 x 84 f32 (90 MB) |
| row 10, epoch 2 | f32: 2278 ms, of which loader waits 62 ms (first batch 52 ms), step 2216 ms | the loader keeps up (0 ms waits after batch 2); the epoch is the GPU step |
| checkpoints (folo-n) | 1347 tensors, 169-297 ms to download one by one; the write 30-110 ms, up to 10 s under disk contention | one GPU wait per tensor |
| R6 loader (1920x1080 JPEG, mosaic) | decode 12 ms + resize 5 ms per image, x4 per sample = ~89% of a sample; mosaic 2, affine 4, HSV 2.3 ms (loaded machine) | loader-bound: 83 samples/s on 8 workers, 140 on 16 |
| synthetic 640 px loader | 675 samples/s on 8 workers, 1003 on 16 | above the 500/s target already |

## Wins (A/B, alternated, medians)

| change | commit | before | after | how measured |
|---|---|---|---|---|
| `folo val` opens only the evaluated split (`folo_data::open_splits`) | 722a05f4 | process wall 61-152 s (median 92) | 25.4-26.8 s (median 25.7) | COCO val2017, folo-n-coco, 3+3 runs, others 20-24 cores |
| validation download + NMS + scoring on a second thread, overlapping the next forward passes | 37b0d2be | evaluation 21.0-29.3 s (median 26.6), wall median 28.7 s | 10.9-13.5 s (median 13.3), wall median 16.1 s | same, others 12-26 cores |
| NMS straight from the downloaded `[A, 4+nc]` rows (no split copy) | ce58b409 | evaluation 8.4-17.9 s (median 15.1) | 7.8-16.1 s (median 8.5) | same, others 7-24 cores: **inside the noise**, kept because it is the same computation without a 90 MB copy per batch |
| next epoch's loader starts when the previous epoch took its last batch | c444f759 | epoch-2 loader wait 56-62 ms (first batch 40-52 ms) | 2-5 ms (first batch 0 ms) | traced, row 10, f32 and amp, 6 runs |
| checkpoint tensors in one download per dtype | c444f759 | 169-297 ms per checkpoint | 21-42 ms | both paths timed back to back in the same process, 7 pairs |
| default loader workers min(cores, 16) instead of min(cores, 8) | 5b6b311a | R6 loader 83 samples/s | 140 samples/s | pipeline bench, R6 train split, loaded machine |

COCO `folo val` end to end: process wall median **92 s (main) → 10.8 s** (9.9-19.9) at ce58b409, evaluation 26.6 → 8.5 s
median, across the three sessions above.

Memory of the overlapped validation (review check, 2026-11-09, integration head 448d9da9 against `main` 93c25129):
with `sync_channel(1)` up to three batches of decoded outputs can be on the device at once, plus the contiguous copy
`to_vec` makes of the transposed `[B, A, 4+nc]` view. `folo val --model D:/folo-weights/folo-n-coco.folo --data
D:/folo-data/coco --device gpu:0`, 3 alternated rounds, GPU otherwise idle (under the GPU lock), sampled every 150 ms
(the whole GPU's `memory.used` above its level before the run, since WDDM does not report per-process memory; the
buffer pool keeps its buffers for the run, so this is its high-water mark plus the context):

| | `main` | head | head / `main` |
|---|---|---|---|
| GPU memory above the idle level | 2883 / 3035 / 2925 MiB | 3886 / 3712 / 3805 MiB | 1.29x |
| host peak working set | 1328 / 1305 / 1305 MB | 786 / 751 / 751 MB | 0.58x |
| process wall | 54.6 / 41.9 / 39.6 s | 9.6 / 10.2 / 10.0 s | |

mAP50 0.44944638 and mAP50-95 0.30999148 from both. The GPU peak is under 1.5x `main`'s, so the channel stays as it is
(a bound would cost overlap for about 0.9 GB on a 16 GB card); the host peak is lower than `main`'s.

## Row 10 (FOLO on the incumbent's protocol)

Epoch-2 sustained img/s, 3 rounds alternated with `main` (c50211ba), 12-23 other cores busy:

| precision | main best / median | this branch best / median | bar |
|---|---|---|---|
| f32 | 204.6 / 183.2 | **201.3** / 197.5 | 199 |
| amp | 243.9 / 241.0 | **336.2** / 300.2 | 224 |

A lighter-loaded single traced run of `main` gave 224.7 (f32) and 330.6 (amp). The f32 difference between the two
binaries is inside the noise (the epoch is the GPU step: 2216 of 2278 ms); amp's spread (163-336) is load, not code.
Sustained / step-only: f32 201 / 247 = 81% under load, 225 / 247 = 91% in the lighter run; amp exceeds the 307
step-only figure of docs/BENCHMARKS.md (that figure is older than the current step).

## Negative results and things not done

- **Loader on the row-10 set**: nothing to win. 675 samples/s on 8 workers against a 247 img/s step; after the second
  batch of an epoch the trainer never waits for data.
- **Per-epoch checkpoint writes in the background**: not done. The epoch event (and Studio / the server) expects
  `last.folo` and `best.folo` on disk when it arrives, and the event cannot move without reordering it against the
  next epoch's steps. Only the download got faster; the write itself (30 ms - 10 s under disk contention) remains.
- **More probe threads for `open`**: 64 threads probe COCO headers 30% faster than 28 (0.248 vs 0.354 ms/file) but
  `folo train` on COCO still pays ~30 s at start; not changed (an IO pool beside rayon, small gain per run).
- **Reduced-DCT JPEG decode**: not done. zune-jpeg has no scaled IDCT, and a 1/2-scale decode followed by the
  bilinear resize does not give the pixels of the full decode + 3x box filter, so it would need an accuracy study.

## Leftovers (largest first)

1. **Validation readback**: the scoring thread is now the limit (download 90 MB per batch + NMS at conf 0.001,
   multi-label). A device-side multi-label candidate kernel (folo-gpu; the single-label `detect_candidates` exists)
   would download only the pairs above `conf`.
2. **`--cache ram` for large images** (R6: decode + resize are ~89% of a mosaic sample). Caching the resized image
   keeps the pixels and targets identical, but `Transformed::orig_size` and the letterbox ratio of a sample would
   then describe the cached image: the transform (`augment/pipeline.rs`) needs to take the original size from the
   cache, which is agent learn's file, so it was left for a coordinated change.
3. **COCO `folo train` start-up**: ~30-50 s probing train2017 headers every process start; a probe cache keyed by
   path, size and mtime (as Ultralytics' `labels.cache`) would remove it.
4. **First epoch on the GPU** (4 s f32, 13 s amp in these runs): pipeline compilation in folo-gpu; a persistent
   wgpu pipeline cache would help every run after the first.
