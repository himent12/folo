# Hill-climb 2: fold-in (merged `main` against 647b981f)

Nine lanes (gpu-dispatch, gpu-gemm, gpu-train, cpu, trainer, learn; then cpu2, gpu-pool, trainer2) merged #181–#205
into `main` overnight on 2026-11-11/12, each measured under the others' load. This notebook re-measures the merged
`main` against `main` before the climb (647b981f) on a quieter machine, names the log behind every cell, and lists
where a number differs from a lane's own. The results are folded into docs/BENCHMARKS.md ("FOLO targets",
"Hill-climb 2 (2026-11-12)", "Next optimisations").

## What was measured

- **New** (`new`): `main` at a37f7f23 (#203, the last climb PR when measuring started, 07:30). Two PRs merged while
  this ran: #206 (lane trainer2: per-epoch checkpoint writes on a writer thread) and #207 (its notebook), together
  e4fa2b43. Row 10, the one row #206 can move, was measured on e4fa2b43 too (`head`): it ties a37f7f23. The other
  trainer rows (start-up, first epoch, `folo val`) are a37f7f23's; #206 changes neither the first step nor `val`.
- **Baseline** (`base`): 647b981f, built from a private worktree (`D:/folo-wt/hill2-foldin-base`) with two changes to
  the `gpu` bench file only (`crates/folo-vision/benches/gpu.rs`), none to library code:
  - the merged bench's latency warm-up (#203: at least 25 calls *and* a second of back-to-back calls before a
    synchronised latency row). 647b981f's bench warmed up for 25 calls (about 40 ms) after the idle second before
    each section, so its batch-1 rows partly timed the GPU's clocks ramping back up (docs/hill/r2-gpu-dispatch.md,
    Pool PR 2). With the patch both builds' batch-1 rows are steady state;
  - a count of the pool buffers created per training step (`PoolStats::misses` over the 10 measured steps), which
    the merged bench prints.
- Machine: as docs/BENCHMARKS.md, Machine (RTX 4080 SUPER, driver 621.01, i7-14700KF). The PC had rebooted at
  06:34. Lane trainer2 was still working (CPU builds, one GPU hold), CI ran `cargo test --workspace` twice on the
  local runners, and the user had a game open for part of the morning: every run records the load it saw, and the
  sections that were loaded were run again later.

## Method

- **Builds.** `cargo build --release -p folo-cli`, `cargo bench --no-run -p folo-vision --bench gpu --bench cpu_e2e
  --bench train_step` (and `--bench gpu_latency` for `new`) and `cargo bench --no-run -p folo-data --bench pipeline`,
  `CARGO_TARGET_DIR=D:/folo-build/hill2-foldin` (`new`) and `D:/folo-build/hill2-foldin-base` (`base`); the
  binaries copied to `D:/folo-data/runs-hill2/foldin/bin/{base,new}-<bench>.exe` (and `head-folo.exe`,
  `pr186-folo.exe`, see below). Every path below is relative to `D:/folo-data/runs-hill2/foldin/`.
- **GPU sections**: `bin/ab.py <dir> <rounds> --env ... -- base=<exe>|--bench new=<exe>|--bench` (written for this
  notebook): alternates the binaries run by run, the order flipped every other round, at High priority. Before each
  run it waits (at most 5 min) until no other process uses more than 5% of a GPU engine (the Windows "GPU Engine"
  counters per process, as bench/baselines does); during the run it samples the other processes every ~3 s; after
  it, the cores other processes used (system busy CPU time minus the run's own, over the wall time). The first line
  of every run's log (`<dir>/<label>-r<round>.txt`) is that `# load:` record. The trainer rows use the lanes'
  scripts: `bench/trainer/ab.py` (records the other processes' cores per run), `startup.py`, `warm_start.sh`
  (records the CPU load before each run), `cold_start.sh` (records none) and `bin/cold_start_extra.sh`
  (`cold_start.sh` passing `$EXTRA` to `folo train`). All under `bash D:/folo-build/hill/lock.sh acquire gpu foldin`,
  never longer than 30 minutes.
- **CPU sections**: `tools/bench-ab.ps1` (High priority, 28 threads, `busy` = the other processes' cores), under
  the `cpu` lock.
- **Summaries**: `bin/summ.py <dir>[,<dir>...] base new` (every row of the `gpu` bench; several folders pool their
  rounds) and `bin/summ_cpu.py <log>`. A cell is the median of the runs' medians and the min–max over the runs;
  img/s at batch 16 end to end and for the training step is 16 / median call (the range from the slowest and the
  fastest run). Inside ±3% is a tie.

## GPU inference (rows 1–6)

`FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=forward,predict` (folo-n and folo-s; f32, f16, f16 on tensor cores),
8 rounds pooled from two sessions: `infer/` (5 rounds, 07:39–07:50; 0–2% other GPU in rounds 1, 3 and 5, bursts of
15–21% from short-lived processes in rounds 2 and 4; 1.4–13.9 other cores) and `infer-q/` (3 rounds, 09:02–09:08;
0–1% other GPU, 1.0–2.0 other cores). Summary: `infer-pooled.txt` (`bin/summ.py infer,infer-q base new`).

| row | folo-n 647b981f | folo-n `main` | folo-s 647b981f | folo-s `main` |
|---|---|---|---|---|
| f32 forward b1, median (p90) | 2.13 ms (2.07–3.04), p90 2.20 | **1.14 ms** (1.13–1.33), p90 1.17 | 3.96 ms (3.85–7.68), p90 4.24 | **2.24 ms** (2.16–2.28), p90 2.27 |
| f32 forward b16, pipelined | 1619 img/s (1598–1650) | **2081 img/s** (2048–2098) | 624 img/s (197–642) | **819 img/s** (743–826) |
| f32 e2e b1 | 2.92 ms (2.85–4.46), p90 3.00 | **1.75 ms** (1.69–2.05), p90 1.83 | 4.90 ms (4.53–6.04) | **3.04 ms** (2.82–5.65) |
| f32 e2e b16 | 1189 img/s (1094–1233) | **1469 img/s** (1440–1500) | 580 img/s (560–584) | **699 img/s** (681–713) |
| f16 forward b1 | 2.16 ms (2.05–3.47) | **1.15 ms** (1.11–1.33) | 3.80 ms (3.63–5.29) | **2.14 ms** (2.10–2.53) |
| f16 forward b16, pipelined | 1953 img/s (1852–2020) | **2339 img/s** (2297–2413) | 804 img/s (724–807) | **934 img/s** (908–935) |
| f16 e2e b1 | 2.95 ms (2.59–4.81) | **1.69 ms** (1.66–2.43) | 4.37 ms (4.27–9.35) | **2.84 ms** (2.72–5.55) |
| f16 e2e b16 | 1405 img/s (1226–1444) | **1600 img/s** (1530–1655) | 681 img/s (648–688) | **767 img/s** (653–779) |
| f16 tensor cores forward b1, median (p90) | 1.85 ms (1.59–2.37), p90 2.00 | **0.95 ms** (0.87–1.55), p90 1.07 | 2.18 ms (2.12–4.02), p90 2.25 | **1.36 ms** (1.35–1.44), p90 1.40 |
| f16 tensor cores forward b16, pipelined | 3081 img/s (2729–3112) | **4159 img/s** (3127–4187) | 1237 img/s (1147–1250) | **1799 img/s** (1730–1825) |
| f16 tensor cores e2e b1 | 2.52 ms (2.14–3.49), p90 2.74 | **1.56 ms** (1.41–2.48), p90 1.66 | 2.80 ms (2.72–3.91) | **2.12 ms** (1.91–2.60) |
| f16 tensor cores e2e b16 | 1839 img/s (1584–1916) | **2156 img/s** (1724–2312) | 979 img/s (941–990) | **1337 img/s** (1279–1348) |
| upload + letterbox b16 (f32) | 2.5–2.9 ms | 2.4–2.7 ms, tie | | |
| new bind groups per forward call | one per dispatch (no cache) | 0.0 | | |

- Every row is faster on `main`. The batch-1 rows of `main` hold their range (1.13–1.33 ms for folo-n f32 over 8
  rounds; 647b981f 2.07–3.04).
- The folo-n f16 tensor-core batch-16 forward of `main` read 4149–4187 img/s in all five rounds of `infer/` and
  3127–3886 in the three of `infer-q/`, with no other process on the GPU in either: the GPU's own clock state, not
  load; the pooled median is 4159. 647b981f moved less (2729–3112).
- folo-s f32 batch 16 is bimodal in both builds, as the 2026-11-09 fold-in saw for folo-s f16: one round of
  647b981f read 197 img/s pipelined and both builds have rounds whose p90 is 120–180 ms. The medians are unaffected.

**Batch-1 synchronised latency p50 / p90** (the steady-state rows above, folo-n): f32 forward 2.13 / 2.20 → 1.14 /
1.17 ms; f16 tensor cores 1.85 / 2.00 → 0.95 / 1.07 ms; end to end f32 2.92 / 3.00 → 1.75 / 1.83 ms, tensor cores
2.52 / 2.74 → 1.56 / 1.66 ms.

**The wait spin (#203) on a quiet machine** (`spin/`: `new-gpu.exe` with `FOLO_GPU_WAIT_SPIN_US=0` against the default
2 ms, folo-n, f32 and f16 tensor cores, 5 rounds alternated, 0–1% other GPU but one 17% burst in `spin0` round 4,
0.9–4.2 other cores):

| folo-n | spin 0 | spin 2 ms (default) |
|---|---|---|
| f32 forward b1, median (p90) | 1.19 ms (1.17–1.24), p90 1.23 | **1.14 ms** (1.14–1.15), p90 1.18 |
| f16 tensor cores forward b1 | 0.96 ms (0.93–2.07), p90 1.06 | **0.90 ms** (0.88–0.92), p90 0.98 |
| f32 e2e b1 | 1.79 ms (1.77–3.70) | **1.73 ms** (1.72–1.73) |
| f16 tensor cores e2e b1 | 1.63 ms (1.59–3.47) | **1.57 ms** (1.52–1.58) |
| f32 / f16 tensor cores forward b16 | 7.85 / 4.15 ms | 7.82 / 4.32 ms, ties |

The spin is 4–6% faster at batch 1 here and steadier (narrower ranges); the f16 tensor-core regression lane
gpu-dispatch left unresolved (1.25 → 1.50 ms) does not reproduce. Its larger gains in the lane's table (end to end
−15% / −31%) came from a busy CPU, where the blocked thread woke on another core.

**Accuracy** (`val/`, below): f32 AP moves in the 7th digit since #186.

## GPU training step (row 9) and the pool

`FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=train`, batch 16 at 640, 3 rounds each: SGD (`train/`, 07:51–07:54;
0–8% other GPU, 1.0–3.7 other cores), `FOLO_BENCH_CLIP=1` (`train-clip/`, 07:54–07:56; 0–7% other GPU but 16% from
a game in `base` round 3, 1.1–10.7 cores) and `FOLO_BENCH_OPTIM=musgd` (`train-musgd/`, 07:56–07:58; 0–6% but 11%
from the game in `base` round 2, 1.1–9.2 cores). img/s at the median step:

| row | folo-n 647b981f | folo-n `main` | folo-s 647b981f | folo-s `main` |
|---|---|---|---|---|
| f32, SGD | 333 img/s (330–334) | **380** (375–380), +14% | 151 (151–152) | **162** (162–162), +7% |
| AMP, SGD | 544 (540–546) | **594** (591–609), +9% | 276 (276–278) | **299** (299–300), +8% |
| f32, clip 10 | 345 (344–346) | **376** (371–376), +9% | 150 (147–150) | **161** (161–161), +7% |
| AMP, clip 10 | 526 (512–559) | **613** (588–616), +17% | 268 (143–275) | **296** (293–300), +10% |
| f32, MuSGD | 316 (292–317) | **366** (331–370), +16% | 136 (72–138) | **154** (154–154), +13% |
| AMP, MuSGD | 443 (237–454) | **592** (565–593), +34% | 229 (128–231) | **276** (251–277), +21% |
| MuSGD step time over SGD, f32 / AMP | +5.4% / +22.6% | **+3.9% / +0.2%** | +11.1% / +20.7% | **+5.4% / +8.3%** |
| split, f32: forward / backward / SGD (ms) | 13.1 / 34.9 / 1.2 | 12.4 / 29.4 / 1.2 | 31.6 / 73.3 / 1.5 | 29.5 / 68.6 / 1.4 |
| split, AMP: forward / backward / unscale + SGD | 7.7 / 18.7 / 2.1 | 7.3 / 17.3 / 1.9 | 16.8 / 38.9 / 2.2 | 14.9 / 36.9 / 2.1 |
| MuSGD optimizer phase, f32 / AMP (ms) | 6.8 / 8.2 | **2.1 / 2.7** | 14.6 / 14.4 | **5.8 / 6.8** |
| dispatches per step, f32 / AMP | 1109 / 1275 | 1117 / 1276 | 1109 / 1275 | 1120 / 1273 |
| new bind groups per step, f32 / AMP | 1109 / 1275 (every dispatch) | **0 / 0** | 1109 / 1275 | **0 / 0** |
| new pool buffers per step, f32 / AMP | 55 / 0 | **0 / 0** | 53 / 0 | **0 / 0** |
| new pool buffers per step, MuSGD f32 | 298 | **0** | 311 | **0** |
| peak live memory, f32 / AMP (MuSGD) | 3467 / 2022 MiB (3478 / 2033) | the same | 6520 / 3713 MiB (6558 / 3751) | the same |
| held by the pool / wgpu reserved, f32 | not printed | 4196 / 4608 MiB | not printed | 7701 / 9216 MiB |

- The MuSGD overhead compares the SGD and MuSGD holds above (same binaries, 2–4 minutes apart), not one interleaved
  run.
- The AMP step runs after the f32 step in the same process, so the pool already holds the f32 step's buffers: 0 new
  buffers per AMP step on both builds here, where an AMP step in a process of its own churned 46 per step on
  647b981f's pool (docs/hill/r2-gpu-dispatch.md, Pool PR 1).
- The clipped folo-s AMP and the MuSGD rows of 647b981f have slow rounds (143, 72, 128, 237 img/s) that coincide
  with the game's load in `train-clip/base-r3` and `train-musgd/base-r2`; the medians are of the other rounds' level.

## The trainer (row 10)

`bench/trainer/ab.py train ... -- --data D:/folo-bench/data --model folo-n --epochs 2 --batch 16 --imgsz 640 --workers
8 --close-mosaic 0 --device gpu:0 [--amp]`, `FOLO_CACHE_DIR=cache`, 5 rounds, three binaries alternated: 647b981f,
`main` a37f7f23 and e4fa2b43 (`head`, with #206) (`row10-f32-d/`, `row10-amp-d/`, 08:58–09:00; 1.1–3.9 other cores):

| epoch-2 sustained img/s | 647b981f | `main` a37f7f23 | e4fa2b43 (#206) | bar |
|---|---|---|---|---|
| f32 | 358 (357–358) | **388** (388–391), +8% | 387 (387–388) | 199 |
| AMP | 553 (550–555) | **613** (612–615), +11% | 608 (605–614) | 224 |

- `auto` resolves differently: 647b981f picks AdamW (rule 3: lr0 0.001429 for 3 classes, 64 iterations), `main`
  MuSGD (rule 5: lr0 0.01, momentum 0.937, weights 0.5 / 0.5). `main` is faster while doing the heavier optimizer.
- Earlier windows (`row10-f32/`, `row10-f32b/`, `row10-amp/`, `row10-f32-c/`, `row10-amp-c/`; 3–17 other cores)
  swung both binaries between 130 and 384 img/s (f32) and 293 and 598 (AMP) from round to round. Their best rounds
  (f32 348 vs 384, AMP 535 vs 598) agree with the quiet table; their medians are not used.

## Start-up, first epoch, `folo val`, loader (the rows this climb added)

**`folo train` on COCO, process start to the first optimizer step** (`bench/trainer/startup.py --until-step 4 -- --data
D:/folo-data/coco --model folo-n --epochs 1 --batch 16 --imgsz 640 --device gpu:0`; 3 rounds; the script records no
load; GPU lock held):

| | 647b981f | `main` | log |
|---|---|---|---|
| warm caches (`--warm 1`: second start, FOLO caches filled by an untimed run) | 36.5 s (24.1–39.4) | **1.28 s** (1.12–1.65) | `startup-warm/` |
| the untimed first run of that session (no FOLO cache yet) | 122.1 s | 34.7 s | `startup-warm/` |
| cold FOLO caches (`--clear`: the cache folder deleted before every run) | 32.7 s (31.8–40.2) | 33.0 s (29.7–45.5), tie | `startup-cold/` |

**First epoch: pipeline compilation** (the row-10 set, one epoch; `warm_start.sh train` with a warm FOLO cache per
binary, 3 rounds, CPU load 4–85% before the runs; `cold_start.sh` with an empty NVIDIA shader cache per run, 3
rounds in a quiet window, 08:42–08:51, load not recorded per run):

| | 647b981f | `main` | log |
|---|---|---|---|
| f32 warm: epoch 1 (`duration_s`) | 2.67 s (2.60–3.36) | **1.89 s** (1.84–1.99) | `ws-f32/` |
| f32 warm: compile on the training thread | not timed (no timer in 647b981f) | **0.001 s** (194 of 194 pipelines precompiled) | `ws-f32/` |
| AMP warm: epoch 1 | 3.88 s (3.07–7.11) | **2.27 s** (2.12–3.33) | `ws-amp/` |
| AMP warm: compile on the training thread | not timed | **0.008–0.011 s** (215 of 218 precompiled; 3 tensor-core self-test kernels compile with the context) | `ws-amp/` |
| f32, driver cache cold, first-ever run (no kernel list) | 13.80 s (13.66–14.02) | 14.02 s (13.85–14.28), tie (+1.6%) | `cold-f32b/` |
| f32, driver and pipeline caches cold, kernel list kept | | **2.86 s** (2.83–2.94); 1.23–1.32 s on the training thread | `cold-f32b/` |
| AMP, driver cache cold, first-ever run | 21.34 s (21.21–21.51) | 22.94 s (22.86–22.98), **+7.5%** | `cold-ampb/` |
| AMP, the same with `--optimizer adamw` on both | 21.66 s (21.48–22.19) | 23.11 s (22.97–23.95), **+6.7%** | `cold-amp-adamw/` |
| AMP, driver and pipeline caches cold, kernel list kept | | **3.48 s** (3.46–3.51); 2.93–3.00 s on the training thread | `cold-ampb/` |
| `folo predict` of one image, process wall (5 rounds) | 0.48 s (0.45–0.65) | **0.37 s** (0.36–0.98) | `ws-predict/` |

A first-ever AMP run (a new machine or driver, no kernel list) is about 1.6 s slower on `main` in all six rounds,
with either optimizer, so it is the compilation of more kernel variants, not MuSGD: `main` compiles 203 pipelines
for this AdamW AMP epoch (218 with MuSGD); 647b981f has no counter. Earlier, noisier runs of the same
(`cold-f32/`, `cold-amp/`) read 18.5 → 24.3 s (f32) and 24.7 → 25.5 s (AMP) with overlapping ranges.

**`folo val` on COCO val2017** (`bench/trainer/ab.py val -- --model D:/folo-weights/folo-n-coco.folo --data
D:/folo-data/coco --device gpu:0`, 3 rounds, `val/`, 4.2–9.3 other cores):

| | 647b981f | `main` |
|---|---|---|
| process wall | 8.0 s (6.9–9.7) | **4.5 s** (4.4–5.7) |
| evaluation (`seconds`) | 6.21 s (5.69–6.97) | **4.13 s** (4.02–4.63) |
| mAP50 / mAP50-95 (every run) | 0.44944638 / 0.30999148 | 0.4494464 / 0.30999136 |

Not identical: mAP50-95 −1.2·10⁻⁷, mAP50 +2·10⁻⁸. `folo` built at d3032648 (#186, split-K for forward convs) gives
`main`'s digits exactly (`val-186/`, binary `bin/pr186-folo.exe`): split-K sums a conv's `K` in chunks, a different
f32 rounding order. Identical to 5 digits, as lane gpu-gemm reported; lane trainer2 saw the same digits.

**`--cache ram` on 1080p JPEGs** (`pipeline --bench --data D:/folo-data/runs-hill2/trainer/synth1080/data.yaml
--workers 16 --rounds 2 --no-stages --no-val [--cache ram]`, 3 rounds, `cacheram/`, CPU lock, 3.2–12.6 other cores):
647b981f 174 samples/s (156–193), `main` 185 (160–187), `main --cache ram` **1206** (1185–1337): 6.5x once filled.

## CPU (rows 7, 8)

`cargo bench -p folo-vision --bench cpu_e2e` (`FOLO_BENCH_ITERS` 60 / 40 / 20, `FOLO_BENCH_SCALE`, batch 16 with
`FOLO_BENCH_BATCH=16 FOLO_BENCH_ITERS=8`) and `--bench train_step` (folo-n, 320 px, batch 16, AdamW), 28 threads, 3
rounds (train step 5), quiet window 09:00–09:02 (`cpu/n28c.txt`, `s28c`, `m28c`, `b16c`, `trainc`):

| row (28 threads) | folo-n 647b981f | folo-n `main` | folo-s 647b981f | folo-s `main` | folo-m 647b981f | folo-m `main` |
|---|---|---|---|---|---|---|
| other cores busy | 0.7–1.3 | | 1.1–4.5 | | 0.8–1.3 | |
| forward b1 | 15.23 ms (15.14–15.38) | **12.74** (12.63–12.75), −16% | 33.27 (32.30–34.70) | **29.24** (27.89–29.60), −12% | 77.74 (76.76–78.30) | **68.94** (68.73–70.34), −11% |
| e2e b1, 640×640 | 15.90 (15.53–16.18) | **13.12** (13.12–13.14), −17% | 33.44 (33.32–36.03) | **29.66** (27.97–30.05), −11% | 79.66 (79.34–80.13) | **68.75** (68.30–70.26), −14% |
| e2e b1, 1280×720 | 12.36 (12.12–12.82) | **10.30** (10.19–10.84) | 23.25 (23.24–25.93) | **21.08** (20.23–21.44) | 54.30 (53.87–55.19) | **47.69** (47.04–48.15) |
| `predict_batch` 16 (0.6–5.6 busy) | 152.1 ms (140.4–154.8), 105 img/s | 143.6 (140.0–147.6), 111 img/s: tie, ranges overlap | | | | |
| train step, 320 px b16 (5 rounds, 3.1–4.8 busy) | 246.9 ms (245.0–250.1) | 238.8 (235.5–238.9), −3.3% | | | | |
| of which forward + loss / backward | 79.4 / 166.9 ms | 70.4 / 167.0 ms | | | | |

Earlier runs under more load (`cpu/n28.txt` 8–13 busy, `n28b` 2–9, `s28`, `m28`, `b16`, `b16b`, `train`, `trainb`)
gave the same direction with slower absolute numbers: folo-n forward 16.9–17.4 → 14.0–14.2 ms, folo-s 35.8 → 29.4,
folo-m 87.2 → 77.0, `predict_batch` 162.6 → 148.5–151.2, train step 254–261 → 243–244. The 8-pinned-P-core rows of
the 2026-11-09 fold-in were not re-run.

## Learning (quoted, not re-run)

From docs/hill/r2-learn.md, Headline (three seeds per side unless noted): COCO proxy P1 (folo-n from scratch, 320 px,
20 epochs) AP 0.0757 (647b981f `auto`) → **0.1234** (`auto` rule 5), Ultralytics 8.4.198's single runs MuSGD 0.1133,
SGD 0.0853; shapes gate mAP50-95 0.892 → **0.934** (one seed; bar 0.75); BCCD fine-tune test AP 0.607 → **0.625**
(yolo26n 0.626); medical-pills / signature / brain-tumor 0.804 / 0.989 / 0.452 → 0.807 / 0.989 / 0.450 (yolo26n
0.803 / 0.982 / 0.436).

## Against the lanes' own claims

Where a number here differs from the lane's by more than noise:
- **cpu2, folo-n CPU forward b1**: 12.33 ms claimed (PR 3, 2–6 cores busy); 12.74 here with about one core busy
  (+3%, a tie), but 14.0–14.2 ms in the three runs with 2–13 cores busy. The gain against 647b981f is −16% here
  (15.23 → 12.74) against −22% in the notebook (15.81 → 12.33); folo-s −12% here against −20% (34.30 → 27.34; this
  machine's 647b981f read 33.27, its merged 29.24). folo-m agrees (68.94 vs 68.39).
- **gpu-dispatch, the wait spin**: the f16 tensor-core batch-1 regression (1.25 → 1.50 ms, "unresolved") does not
  reproduce on a quiet machine (0.96 → 0.90 ms with the spin); the end-to-end batch-1 gains (−15% / −31%) are −3% /
  −4% here. The spin pays when the CPU is busy.
- **gpu-train, MuSGD overhead**: f32 "none (MuSGD measured 4–6% faster than SGD)" and AMP +7%; here MuSGD is 3.9%
  slower in f32 and 0.2% in AMP (folo-n), 5.4% / 8.3% (folo-s). The faster-than-SGD f32 MuSGD does not reproduce.
- **trainer, first start without a cache**: 48.7 → 41.8 s claimed (a directory listing instead of a stat per image);
  a tie here with the cache folder deleted before every run (32.7 vs 33.0 s).
- **Identical AP** (trainer, gpu-gemm): identical to 5 digits; at 8, mAP50-95 moved by −1.2·10⁻⁷ with #186.
- Everything else matches within noise or is better here: gpu-gemm's final forward table (folo-n f32 b16 2107 vs
  2081 img/s here, f16 tensor cores 4168 vs 4159, folo-s 830 vs 819 and 1822 vs 1799); gpu-pool's 0 new bind groups
  and buffers per step; the trainer's warm start (3.5 s claimed under heavy load, 1.28 s here), `folo val` (6.7 s,
  4.5 s here) and `--cache ram` (6x, 6.5x here); trainer2's warm first-epoch compile (0.001–0.02 s) and cold
  epoch with a list (3.7–5.2 s, 3.5 s here).

## Rows worse than 647b981f

- **A first-ever AMP epoch** (cold NVIDIA shader cache, no kernel list: a new machine, a driver update): +1.6 s
  (21.3 → 22.9 s, +7.5%), also with AdamW on both builds (+6.7%). More pipeline variants to compile (or slower
  ones): the climb added kernel variants in #184 (stem weight-gradient tile), #186 (split-K and its finish kernel),
  #199 (the tap-mask gather) and #202 (64×64 / 64×128 tensor-core tiles), and #191's Muon kernels under MuSGD; not
  bisected. Every later run
  compiles from the kernel list on the workers (3.5 s) or finds the pipelines cached (2.3 s). The f32 first-ever epoch
  ties.
- **`folo val` AP in the 7th digit** (#186): not an accuracy loss, but no longer the same bits as 647b981f.

Nothing else measured here is slower than 647b981f.

## Logs

Every cell's log, under `D:/folo-data/runs-hill2/foldin/`:

| cells | logs |
|---|---|
| GPU inference rows (forward, e2e, f32 / f16 / tensor cores, upload, bind groups) | `infer/{base,new}-r1..5.txt`, `infer-q/{base,new}-r1..3.txt`; summary `infer-pooled.txt` |
| wait spin on / off | `spin/{spin0,spin2ms}-r1..5.txt` |
| training step, SGD; MuSGD overhead (SGD side); counters; memory | `train/{base,new}-r1..3.txt` |
| training step, clip 10 | `train-clip/{base,new}-r1..3.txt` |
| training step, MuSGD; MuSGD overhead; MuSGD counters | `train-musgd/{base,new}-r1..3.txt` |
| row 10 | `row10-f32-d/`, `row10-amp-d/` (`{base,new,head}-r0..4.log` and `.metrics.jsonl`), `row10-f32-d.json`, `row10-amp-d.json`; loaded windows `row10-f32/`, `row10-f32b/`, `row10-amp/`, `row10-f32-c/`, `row10-amp-c/` |
| COCO start-up, warm / cold | `startup-warm/`, `startup-cold/` |
| first epoch, warm, and `folo predict` | `ws-f32/log-*.txt`, `ws-amp/log-*.txt`, `ws-predict/log-*.txt` (the summary lines are in this notebook; each log has its run's output) |
| first epoch, cold driver cache | `cold-f32b/log-1..9.txt`, `cold-ampb/log-1..9.txt`, `cold-amp-adamw/log-1..9.txt`; noisier `cold-f32/`, `cold-amp/` |
| `folo val` | `val/` (`*.log`), `val.json`; #186 attribution `val-186/`, `val-186.json` |
| `--cache ram` | `cacheram/{base,new,ram}-r1..3.txt` |
| CPU rows | `cpu/n28c.txt`, `s28c.txt`, `m28c.txt`, `b16c.txt`, `trainc.txt`; loaded runs `cpu/n28.txt`, `n28b.txt`, `s28.txt`, `m28.txt`, `b16.txt`, `b16b.txt`, `train.txt`, `trainb.txt` |
| scripts | `bin/ab.py`, `bin/summ.py`, `bin/summ_cpu.py`, `bin/cold_start_extra.sh` |
