# Wave 6, lane `loader`: faster training samples with the same bytes

Brief: `D:/folo-wave6/briefs/loader.md` (raise the default loader's throughput on COCO-like JPEG data with
bit-identical batches; target 1.25x images/s of the loader alone at 8 workers on COCO train2017 with the default
augmentation at 640 px, and a visible gain in a real `folo train`). Logs, binaries and scripts:
`D:/folo-wave6/lanes/loader/` (`logs/`, `bin/`, `scripts/`, `profile-*.txt`); paths below are relative to it.

Machine: 28 logical CPUs (8 P-cores with two threads each, 12 E-cores), one RTX GPU, Windows 11. All day the
machine also ran two WSL2 VMs (about 27 GB), other lanes' builds and an 8-11 GB training process of another lane;
free commit memory was often 2-4 GB. "Load" below is the whole machine's CPU busy percentage (`% Processor Time`,
2-3 one-second samples) just before and just after a run; the measured processes were not alone on the CPU.

## The bytes stay the same

`crates/folo-data/tests/golden.rs` (commit 9b078721, before any optimisation) writes 14 textured samples of 7 sizes
(31 to 321 px, odd sizes included) to memory and to JPEG and PNG files and pins an FNV digest of each of the first
3 batches of epoch 1, for each path of the transform: mosaic (u8 and f32 output), mosaic with rotation and shear,
mosaic with mixup and up-down flips, letterbox + affine (mosaic 0), letterbox with rotation, and validation. The
digests were blessed on `origin/main`'s code; every later commit passes unchanged. After the review a second set
was added, blessed the same way: from the files at imgsz 192 (where decoded images, resized tiles, letterboxes and
warped samples are all 64 KiB or more, so the loader's buffer pool serves them) for mosaic, rotation + shear,
letterbox + affine and validation, with no workers (no pool) and with 3 workers, both against the same digests. The existing determinism tests
(batches independent of the worker count) stay green, and each new kernel has a test against the kernel it
replaces (below).

## Profile (before)

`cargo run --release -p folo-data --example loader_profile -- D:/folo-data/coco 512` (new example): 512 mosaic
samples at 640 from COCO train2017 on one thread, each stage timed alone, allocations counted by a counting global
allocator. `profile-main.txt`, `origin/main`'s code:

| stage | ms / sample | share | allocs / sample | MB allocated / sample |
|---|---|---|---|---|
| file read (4 files) | 0.437 | 3.3% | 8.0 | 0.66 |
| JPEG decode (4 images) | 6.750 | 50.7% | 49.4 | 4.11 |
| resize long side | 0.784 | 5.9% | 4.9 | 0.67 |
| mosaic canvas (`2s x 2s`) | 1.092 | 8.2% | 5.0 | 4.92 |
| affine warp | 1.675 | 12.6% | 2.0 | 1.23 |
| HSV | 1.944 | 14.6% | 0.0 | 0.00 |
| flips | 0.125 | 0.9% | 0.0 | 0.00 |
| collate u8 (collator thread) | 0.494 | 3.7% | 1.0 | 1.23 |
| total | 13.306 | 100% | 75.3 | 12.82 |

Half the time is the JPEG decoder (zune-jpeg through `image`), which bit-identical work cannot change. The rest:
mosaic canvas + warp 21%, HSV 15%, resize 6%. A single thread does not show what 8-16 workers do: the loader's
scaling on `origin/main` (`scaling-main.txt`, pipeline bench, medians of 3 epochs) was 63 img/s at 1 worker, 237 at
8, 399 at 16, 445 at 28: 8 workers gave 3.8x one worker, not 8x. A micro-benchmark of a fresh buffer written whole
against a reused one (not committed) showed why:

| buffer | threads | fresh | reused (zeroed) |
|---|---|---|---|
| 825 KB (a decoded COCO image) | 1 | 0.021 ms | 0.022 ms |
| 825 KB | 8 | 0.466 ms | 0.049 ms |
| 825 KB | 16 | 0.641 ms | 0.065 ms |
| 1.2 MB (a 640 px sample) | 1 | 0.241 ms | 0.034 ms |
| 1.2 MB | 8 | 0.544 ms | 0.061 ms |
| 1.2 MB | 16 | 0.961 ms | 0.112 ms |
| 4.9 MB (the mosaic canvas) | 8 | 2.419 ms | 0.936 ms |
| 4.9 MB | 16 | 5.263 ms | 2.724 ms |

On Windows a buffer of a megabyte or more comes fresh from the system and its first writes fault every page in; with
many threads doing that at once the faults serialise in the kernel. A mosaic sample allocated about 13 MB.

## What changed (all bit-identical)

1. **Mosaic fused into the warp** (`warp.rs` `Plane`, `mosaic::mosaic_affine`). The warp reads the four resized
   tiles where they lie on the `2s x 2s` canvas (a `Plane`: images placed on one fill colour) instead of painting a
   4.9 MB canvas first. With rotation or shear the canvas is still painted and warped as before.
2. **Row path of the warp.** When the inverse matrix's vertical step rounds to 0 in 16.16 (no rotation or shear, as
   in the default augmentation) every output row reads the same two source rows: blend them once into `u32` sums,
   then interpolate along them. Same four products and the same single rounding `(sum + 2^21) >> 22`, so the same
   bytes (test `row_path_matches_the_general_path`, and `plane_warps_as_its_rendered_image` for the plane).
3. **Buffer reuse** (`buffers.rs`). A process-wide pool of image-sized buffers, used only on the loader's threads
   (worker and collator; elsewhere `vec!` as before): decoded images, resized tiles, warped samples. Buffers are
   zeroed where `vec![0; n]` was and otherwise overwritten whole, so no byte depends on reuse. At most 512 MB of
   free buffers; emptied when the loader stops.
4. **JPEG decode into a pooled buffer** (`image.rs` `decode_jpeg`): the same `JpegDecoder` with the limits
   `ImageReader::decode` sets, writing into a pooled buffer, then EXIF orientation through a scratch buffer (test
   `jpeg_decodes_as_the_generic_path`, on our encoder's files and on Pillow-written 4:2:0, 4:2:2, 4:4:4,
   progressive and CMYK files in `tests/fixtures/jpeg`). The `--cache ram` entries, kept for the whole run, are
   shrunk to their exact size, so a pooled buffer of a larger size class does not hold memory the cache's
   reservation does not count.
5. **HSV on AVX2** (`augment/hsv.rs`). The per-pixel kernel with branches stays as the fallback. On CPUs with AVX2,
   blocks of 64 pixels are copied into channel arrays and one loop without branches computes them (the hue sector
   and the max channel become masks and selects, table reads become gathers). Test `kernels_agree_on_all_colours`
   checks both against each other on all 2^24 colours. In-process timing on 64 COCO train2017 images
   (`hsv-kernels.txt`, 5 rounds, load 48-56%): per-pixel 5.65-6.51 ns/px, AVX2 blocks 2.77-4.15 ns/px, **1.38-2.33x,
   median 2.04x**.

## Profile (after)

`profile-simd-x.txt` (head of the branch, 256 samples; the profile before used 512). The two profiles are separate
runs at different, unmeasured loads: the whole table here is 2.1x slower, the decode included, whose code did not
change. The shares and the ratio of each stage to the decode are what the two runs observed, not a controlled
comparison (the A/B below is that):

| stage | before, ms (share) | after, ms (share) | stage / decode before → after |
|---|---|---|---|
| JPEG decode | 6.750 (50.7%) | 16.399 (58.4%) | 1 → 1 |
| mosaic + affine warp | 2.767 (20.8%) | 3.094 (11.0%) | 0.41 → 0.19 |
| HSV | 1.944 (14.6%) | 2.928 (10.4%) | 0.29 → 0.18 |
| resize long side | 0.784 (5.9%) | 1.669 (5.9%) | 0.12 → 0.10 |
| collate u8 | 0.494 (3.7%) | 1.086 (3.9%) | 0.07 → 0.07 |
| MB allocated / sample | 12.82 | 7.95 | |

(On one thread outside the loader the pool is not used; the 4.9 MB fewer is the canvas.)

## A/B: the loader alone

`scripts/ab2.ps1`: the pipeline bench (`benches/pipeline.rs`) built from `origin/main` (`bin/pipeline-base.exe`,
with this branch's bench file for `--same-epoch` and `SCAN_CACHE`) and from the branch (`bin/pipeline-new.exe`),
one process per arm, arms alternated, under the `cpu` lock. Each process opens COCO train2017 (118 287 images)
through the scan cache, runs epoch 1's first 125 batches of 16 (2 000 images) as a discarded warm-up pass (so both
arms read equally warm files), then 2 measured passes over the same 2 000 images. img/s below is the mean of the
two passes (their median): the bench's own "median" column printed the faster pass when the count was even, a bug
the review found and this branch fixes; the rates here are recomputed from the two passes each log records
(`scripts/recompute.py`). "CPU ms/img" is the process's total CPU time (all threads, warm-up included) over the
6 000 images it made: what a sample costs, which other processes' load changes much less than the wall rate.
Default augmentation, 640 px, u8 batches. `logs/w{4,8,16}-a/`.

| workers | round | base img/s | new img/s | new / base | base CPU ms/img | new CPU ms/img | load before/after (base; new) |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 323.2 | 419.7 | 1.30 | 24.98 | 18.10 | 93/78; 59/67 |
| 8 | 2 | 344.2 | 472.9 | 1.37 | 23.85 | 17.22 | 66/50; 49/78 |
| 8 | 3 | 304.5 | 375.3 | 1.23 | 24.99 | 19.74 | 78/95; 89/83 |
| 8 | 4 | 320.4 | 452.1 | 1.41 | 23.71 | 17.83 | 74/56; 49/54 |
| 16 | 1 | 539.2 | 583.4 | 1.08 | 27.28 | 19.95 | 64/82; 76/90 |
| 16 | 2 | 441.9 | 523.5 | 1.18 | 26.99 | 20.84 | 99/96; 100/100 |
| 16 | 3 | 439.2 | 675.0 | 1.54 | 26.49 | 20.22 | 100/76; 82/93 |
| 16 | 4 | 519.0 | 653.9 | 1.26 | 26.77 | 20.53 | 88/81; 86/93 |
| 4 | 1 | 144.1 | 200.2 | 1.39 | 25.08 | 18.64 | 66/67; 94/75 |
| 4 | 2 | 83.4 | 104.3 | 1.25 | 25.32 | 21.38 | 91/85; 86/82 |
| 4 | 3 | 111.0 | 96.4 | 0.87 | 26.21 | 19.21 | 97/92; 90/48 |
| 4 | 4 | 129.8 | 177.1 | 1.36 | 19.68 | 17.86 | 41/76; 37/47 |

Medians:

| workers | base img/s (range) | new img/s (range) | median of the pair ratios (lowest) | CPU ms/img base → new |
|---|---|---|---|---|
| 4 | 120.4 (83.4-144.1) | 140.7 (96.4-200.2) | 1.31x (0.87) | 25.2 → 18.9 (−25%) |
| **8** | **321.8 (304.5-344.2)** | **435.9 (375.3-472.9)** | **1.34x (1.23)** | **24.4 → 18.0 (−26%)** |
| 16 | 480.4 (439.2-539.2) | 618.6 (523.5-675.0) | 1.22x (1.08) | 26.9 → 20.4 (−24%) |

The 8-worker target (1.25x) is met on the median pair; the weakest 8-worker pair is 1.23x. The CPU cost per image
falls by about a quarter at every worker count, with less spread than the wall rates. The machine was busy (load
37-100%), so the wall rates move by up to 1.7x between rounds of the same arm, the 4-worker ones most.

## Real training

`folo train --model folo-n --amp --epochs 2 --batch 16 --imgsz 640 --workers 8 --close-mosaic 0` (mosaic on in both
epochs) on `coco5k/` (COCO val2017's 5 000 JPEGs and labels as the training split; the brief's example was an
8 000-image train2017 subset, but the val2017 files were already on disk as one folder), `bin/folo-base.exe`
(`origin/main`) against `bin/folo-new.exe` (this branch), alternated, under the `gpu` lock. The rate is the last
epoch-2 `imgs_per_s` of `metrics.jsonl` (the running mean of epoch 2). Two sets: A, `scripts/train_ab.sh train-w8 3
8` (3 rounds, `logs/train-w8/summary.txt`), and B, `scripts/train_ab2.ps1` (5 rounds, `logs/train-w8-b/summary.csv`,
which also records the process's CPU time over the 10 000 training images and the wall time of the whole run,
start-up and validation included). Load is the machine's CPU busy percentage just before and just after each run.

| set, round | base img/s | new img/s | new / base | base CPU ms/img | new CPU ms/img | load before/after (base; new) |
|---|---|---|---|---|---|---|
| A 1 | 447.4 | 545.0 | 1.22 | | | 33/65; 39/22 |
| A 2 | 358.2 | 378.9 | 1.06 | | | 22/68; 98/65 |
| A 3 | 350.4 | 299.7 | 0.86 | | | 58/75; 88/100 |
| B 1 | 284.5 | 364.6 | 1.28 | 29.64 | 23.95 | 100/47; 68/99 |
| B 2 | 293.1 | 203.8 | 0.70 | 29.06 | 26.17 | 99/90; 88/50 |
| B 3 | 169.1 | 258.3 | 1.53 | 30.10 | 22.89 | 50/63; 44/65 |
| B 4 | 226.5 | 253.8 | 1.12 | 28.09 | 22.93 | 66/38; 36/56 |
| B 5 | 365.6 | 447.3 | 1.22 | 27.27 | 22.36 | 48/65; 76/58 |

- **Rate:** median of the 8 pair ratios **1.17x** (0.70-1.53). The machine was 22-100% busy around the runs and
  single runs of one arm moved by 2x (set B's base arm: 169-366 img/s). Two pairs went the other way (A 3, B 2);
  the before/after load readings do not explain B 2 (new 88/50 against base 99/90), and two readings per run cannot
  say what ran during it. Both arms' rates stay below what the trainer sustains on synthetic data (folo-n AMP 613
  img/s, docs/BENCHMARKS.md row 10), which suggests the data side limited these runs; this check does not measure
  the trainer's wait for batches directly.
- **CPU per training image** (set B, the whole process: loader, trainer, driver threads): base 27.3-30.1 ms, median
  29.1; new 22.4-26.2, median 22.9: **−21%, lower in all 5 pairs**, about the 6.4 ms per image the loader alone saved.
- **Same training:** all 16 runs end epoch 2 with the same total train loss (11.849166) and the same validation
  metrics, as identical batches would give; `tests/golden.rs` is what pins the bytes.

So a gain shows in real training, about 1.2x on the median pair at 8 workers with a wide spread; the CPU time per
image is the steadier number.

## What lost

- **Scalar HSV without branches.** The first try turned the per-pixel kernel's branches into masks and selects on
  plain `u8`/`i32` pixels, one at a time. In-process A/B against the kernel with branches: 3.8-4.8 ms against
  2.5-3.6 ms per image (6 rounds): slower: every pixel then computes all the cases, and one pixel at a time
  over interleaved RGB gives the compiler nothing to vectorise. Reverted; the AVX2 block kernel (channel arrays, 64 pixels) is
  what won.

## Leftovers, by expected value

1. **The decoder** is now about 58% of a sample. Bit-identical options are few: zune-jpeg's own output is fixed;
   decoding at a reduced scale (DCT scaling) for images much larger than the target would change the bytes and is a
   judged change with the accuracy check. COCO images are mostly 640 px already, so little to gain there.
2. **Worker count.** `auto` is `min(16, cores)` = 16 here. On `origin/main` 20 and 28 workers gave 426 and 445 img/s
   against 399 at 16 (`scaling-main.txt`), within the spread this machine shows under load; not resolvable today.
   The batches do not depend on it, so it can change without an accuracy check.
3. **Resize long side** (~6%): vertical-first separable passes over contiguous rows could save about 3% of a
   sample; it must keep the rounding order.
4. **Collate into reused batch buffers.** The collator still allocates each batch tensor (79 MB of f32 or 20 MB
   of u8 per batch of 16 at 640); the trainer owns the batch after the hand-off, so recycling needs a return path.
5. **Letterbox path** (mosaic off: validation, `close_mosaic` epochs): the same plane fusion as the mosaic (letterbox
   canvas read in place by the warp) would save one image-sized copy per sample.
6. **One resampling pass** (decode → warp directly, skipping the resize) changes the bytes: a separate, judged PR with
   the 320 px proxy accuracy check, as the brief says.
7. Not measured: the channel hand-off (workers blocked on a full channel, the trainer on an empty one). The training
   check's rates, below the trainer's on synthetic data, suggest the trainer waits on the loader at 8 workers;
   timing that wait is the way to know.
