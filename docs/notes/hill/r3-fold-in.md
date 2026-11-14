# Hill-climb 3: fold-in (merged `main` against e5a6ccf0)

Hill-climb 3 (2026-11-13) shipped new algorithms rather than kernel tuning: lanes lazy (candidate-first prediction,
#254, #255), fusion (box fusion after NMS and the validation clip, #253, #256, #257), plan (recorded forward plans,
#258, and same-canvas batches, #259) and refine (a learned box refiner, a negative result, #260). Each lane measured
under the others' load and, in the evening, beside a game on the GPU. This notebook measures the merged `main`
against `main` before the round (e5a6ccf0) with the two binaries alternated run by run, names the log behind every
cell, and lists where a number differs from a lane's own. The results are folded into docs/BENCHMARKS.md ("FOLO
targets", "Hill-climb 3 (2026-11-13)", "Summary", "Accuracy", "Next optimisations").

## What was measured

- **New** (`new`): `main` at ffa0f8cb (#258, the last code PR of the round). #261 (lane weights: `folo-n-coco-v3`
  registered, release `weights-v4`) merged during the measurements as 8ea21bab; it changes the weights registry and
  docs only, so the speed binaries stay ffa0f8cb's. The accuracy of `folo-n-coco-v3` was measured with a `folo`
  built at 8ea21bab (`main-folo`), since ffa0f8cb's `folo weights pull` does not know the name.
- **Baseline** (`base`): e5a6ccf0, built from a private worktree (`D:/folo-wt/hill3-foldin-base`) with `main`'s
  bench files applied uncommitted, none of the library code:
  - `crates/folo-vision/benches/gpu.rs` (the 1280×720 and mixed `predict_batch` rows) without its one line
    `model.set_forward_plans(true)` (e5a6ccf0 has no plans);
  - `cpu_e2e.rs` (the trained-weight rows; it uses only the public `Predictor` API);
  - `gpu_trained.rs` (new in the round) with the plans toggle stubbed out, and its `[[bench]]` entry;
  - the two bench rows this PR adds to both trees (below).
- **Bench rows added by this PR** (both trees): `gpu_trained`'s **b1 640** (`predict` on each of the 16 images
  resized to 640×640: the targets table's end-to-end protocol, decoded 640×640 images in), **b16 640**
  (`predict_batch` of the 16 resized to 640×640: row 6's protocol) and **b16 hd** (`predict_batch` of the 16 images
  resized to 1280×720, one rect canvas 640×384), and `cpu_e2e`'s trained `predict_batch 16 x 1280x720`. The first
  `gpu_trained` pass (`trained/`) ran before **b1 640** existed, the first two before **b16 640** (`bin/v2/` keeps
  the second pass's binaries; the third pass's are `bin/{base,new}-gpu_trained.exe`, `bin/{base3,new3}-commit.txt`;
  `new3` is built from this branch on 8ea21bab, whose library code is ffa0f8cb's).
- Weights: the published `folo-n-coco` and `folo-s-coco` for every trained-weight speed row
  (`~/.folo/weights/folo-{n,s}-coco.folo`); the first 16 val2017 images by name. Accuracy: every published COCO
  checkpoint (`folo-n-coco`, `folo-n-coco-v2`, `folo-s-coco`, `folo-n-coco-v3`).
- Machine: as docs/BENCHMARKS.md, Machine. In use the whole evening: a browser, music, a game client
  (RobloxPlayerBeta) open for at least part of it, the T3 / Codex apps, lane weights' build and `folo val` passes,
  and the Windows CI runners (a `gpu_amp` test ran on the GPU during one training-step run).

## Method

- **Builds.** `bin/build.sh` per tree: `cargo build --release -p folo-cli` and `cargo bench --no-run -p
  folo-vision --bench gpu --bench cpu_e2e --bench gpu_trained --bench train_step`, `CARGO_TARGET_DIR`
  `D:/folo-build/hill3-foldin` (`new`) and `D:/folo-build/hill3-foldin-base` (`base`), the binaries copied to
  `D:/folo-data/runs-hill3/foldin/bin/{base,new}-<bench>.exe` (`{base,new}-commit.txt` hold the commit and the
  tree's uncommitted files). The `gpu_trained` binaries of the first pass are kept in `bin/v1/`. Every path below
  is relative to `D:/folo-data/runs-hill3/foldin/`.
- **Runs.** `bin/ab.py <dir> <rounds> --env ... -- label=exe|args[|env]` (hill-climb 2's fold-in runner, two
  additions): alternates the binaries run by run, the order reversed every other round, at High priority. Before
  each run it waits (at most 5 minutes) until no other process uses more than 5% of a GPU engine **and** the machine
  uses at most `--wait-cpu` cores over a 2 s sample (4 for the GPU sections; 2 for the CPU ones' first folo-n round,
  then 5, since the machine's idle floor that evening was 4–7 cores and every run waited out the 5 minutes); during the run it
  samples the other processes' GPU load every ~3 s; after it, the other processes' GPU load once more and the cores
  other processes used during the run (system busy time minus the run's own, over the wall time). The first line of
  every run's log (`logs/<section>/<label>-r<round>.txt`) is that `# load:` record: other GPU before / during /
  after, `cpu before` (the last 2 s sample) and `other cores`. GPU sections: `bin/gpu-runs.sh <section>` under
  `bash D:/folo-build/hill/lock.sh acquire gpu foldin`, one hold per section, each under 25 minutes; CPU sections:
  `bin/cpu-runs.sh <scale>` under the `cpu` lock. `nvidia-smi` at the start and end of each hold:
  `logs/<section>-{start,end}.txt`.
- **Quiet.** A run is quiet when the other processes' GPU load stayed at or under 10% while it ran (at or under 5%
  before and after) and the other processes used at most 3.5 cores (GPU sections) or 3 cores (CPU sections) during
  it. Cells are from quiet runs; a loaded run is dropped with its pair (the other binary's run of the same round)
  where the section had enough quiet pairs left, and kept, said so, where it had not. The waits were capped, so
  some runs started loaded; the `# load:` lines say which.
- **Summaries.** `bin/summ3.py <dir>[,<dir>] base new [nop] [--skip label-rN,...]` (every row of the `gpu`,
  `gpu_trained`, `cpu_e2e` and `train_step` output; the median of the runs' medians, min–max over runs, and new / base
  of the medians), written to `logs/<section>-summary.txt`. img/s at batch 16 end to end and for the training step
  is 16 / the median call. Inside ±3% with overlapping ranges is a tie.

## GPU inference, random weights (rows 1–6 as the 2026-11-12 table measured them)

`FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=forward,predict`, three labels alternated: `base`, `new` and `nop`
(the new binary with `FOLO_GPU_NO_PLAN=1`: forward plans off, the rest of the round on), 5 rounds, 22:01–22:23
(`logs/infer/`). Load: 0–8% other GPU during every run (a browser and the window manager), 0–3% after; other cores
1.3–2.4, except `nop` round 1 (8.0) and `new` round 2 (6.4), dropped (`logs/infer-summary.txt`: `--skip
nop-r1,new-r2`; `base` keeps all 5 rounds, the others 4).

| row | folo-n e5a6ccf0 | folo-n `main` | `main`, plans off | folo-s e5a6ccf0 | folo-s `main` | `main`, plans off |
|---|---|---|---|---|---|---|
| f32 forward b1 (p90) | 1.13 ms (1.13–1.32), p90 1.16 | **1.09** (1.08–1.13), p90 1.11 | 1.13 (1.12–1.18) | 2.21 (2.16–2.67), p90 2.27 | **2.16** (2.11–2.18), p90 2.21 | 2.29 (2.22–2.46) |
| f16 forward b1 | 1.14 (1.12–1.44) | **1.09** (1.08–1.09) | 1.15 (1.14–1.50) | 2.16 (2.11–2.72) | 2.13 (2.06–2.33) | 2.20 (2.15–2.26) |
| f16 tensor cores forward b1 (p90) | 1.21 (0.86–1.31), p90 1.43 | **0.85** (0.82–1.01), p90 0.87 | 0.94 (0.87–0.96) | 1.65 (1.32–1.95), p90 1.96 | **1.30** (1.28–1.31), p90 1.32 | 1.41 (1.38–1.57) |
| f32 forward b16, pipelined | 2064 img/s (1804–2083) | 1962 (1827–2089) | 1977 (1882–2074) | 767 (39–826) | 825 (97–828) | 822 (802–828) |
| f32 forward b16, synchronised median | 7.81 ms (7.77–9.47) | 7.84 (7.78–8.76) | 8.63 (7.99–8.95) | 20.05 (19.63–22.31) | 19.65 (19.58–19.87) | 19.76 (19.62–20.62) |
| f16 tensor cores forward b16, pipelined | 4148 img/s (3685–4164) | 4166 (4144–4184) | 4093 (4052–4185) | 1738 (1493–1823) | 1800 (1626–1823) | 1654 (1573–1818) |
| f32 e2e b1 | 2.64 ms (1.71–3.16) | 1.79 (1.44–4.17) | 2.09 (1.74–2.91) | 4.27 (2.92–6.54) | 2.68 (2.50–4.82) | 2.94 (2.69–5.49) |
| f16 tensor cores e2e b1 | 2.15 (1.43–3.09) | 1.37 (1.31–1.50) | 1.91 (1.70–2.01) | 2.46 (1.85–3.30) | 2.09 (1.68–2.32) | 2.15 (1.95–2.31) |
| f32 e2e b16 | 13.95 ms (10.85–15.44), 1147 img/s | 10.49 (9.30–11.46), 1525 img/s | 11.12 (9.64–12.60) | 23.21 (22.63–25.15), 689 img/s | 21.22 (20.80–21.67), 754 img/s | 21.12 (20.90–21.20) |
| f16 tensor cores e2e b16 | 7.38 ms (6.83–9.27), 2168 img/s | 7.21 (6.13–7.58), 2219 img/s | 7.20 (6.84–7.56) | 13.69 (11.81–14.76), 1169 img/s | 11.52 (10.97–11.63), 1389 img/s | 11.86 (11.33–12.41) |
| `predict_batch` 16 × 1280×720, f32 | 14.16 ms (13.65–16.15) | **10.20** (9.62–10.97), −28% | 10.14 (9.77–10.41) | 25.92 (25.39–31.87) | **17.90** (17.87–18.03), −31% | 17.86 (17.76–19.55) |
| `predict_batch` 16 × 1280×720, f16 tensor cores | 10.07 (9.49–11.52) | **7.98** (7.68–8.99), −21% | 8.22 (8.08–8.27) | 15.46 (14.50–18.02) | **11.64** (11.38–13.43), −25% | 11.29 (11.20–12.50) |
| `predict_batch` 16 mixed shapes, f32 | 14.53 (13.32–36.91) | 23.46 (10.49–38.61) | 12.28 (11.27–13.59) | 24.40 (23.66–27.52) | 21.82 (21.48–22.06) | 21.86 (21.82–22.48) |
| new bind groups per forward call | 0.0 | 0.0 | | | | |
| dispatches per forward, folo-n f32 b1 / b16 | 146 / 137 | 146 / 137 | | | | |

- **Random weights pass no candidate at `conf` 0.25**, so on `main` the e2e rows run no box branch at all, where
  e5a6ccf0 runs the dense head: these rows overstate candidate-first. The trained-weight rows below are the ones to
  read for rows 5 and 6.
- **The e2e b1 rows are bimodal on all three binaries tonight**: per-run medians of folo-n f32 1.71–3.16 ms
  (e5a6ccf0), 1.44–4.17 (`main`), 1.74–2.91 (plans off), with no load in the records that explains which runs were
  slow. The 2026-11-12 fold-in read 1.69–2.05 on its build. Both benches warm up for a second before each latency
  row; the `gpu_trained` bench on trained weights did not show it (below). Not explained.
- **e5a6ccf0's f16 tensor-core forward b1 is bimodal**: 0.86 and 0.91 ms in two runs, 1.21–1.31 in three, against
  0.82–0.85 (one run 1.01) for `main` and 0.87–0.96 for `main` with plans off. The plan's own share is therefore
  read on the same binary: 0.94 → 0.85 ms (−10%) for folo-n, 1.41 → 1.30 (−8%) for folo-s; in f32 1.13 → 1.09
  (−3.5%) and 2.29 → 2.16 (−6%). Regular f16 1.15 → 1.09 (−5%).
- **folo-n f32 `predict_batch` of mixed shapes has a slow mode on both builds**: 36.9 ms in e5a6ccf0's round 1,
  38.6 and 35.0 ms in `main`'s rounds 3 and 4, and 18.7 in the dropped `nop` round 1; the fast mode is 10.5–14.6 ms.
  Lane plan saw the same mode on its old binaries and could not reproduce it on `main`; it is there on e5a6ccf0 too,
  so it predates the round. Leftover.
- Batch 16 forward: ties (folo-n f32 synchronised 7.81 vs 7.84 ms, per-run pipelined 7.66–8.87 ms on both builds;
  tensor cores 4148 vs 4166 img/s). folo-s f32 / f16 batch 16 keep the bimodality the 2026-11-12 fold-in saw (one
  run of each binary at 39–97 img/s pipelined).

## GPU prediction with trained weights (rows 5 and 6, the record)

`FOLO_BENCH_WEIGHTS=~/.folo/weights/folo-{scale}-coco.folo FOLO_BENCH_IMAGES=D:/folo-data/coco/val2017
FOLO_BENCH_PRECISION=f32,f16tc FOLO_GPU_TENSOR_CORES=1`, the `gpu_trained` bench (the public `Predictor`, conf 0.25):
**b1** each of the 16 images on its own rect canvas (mostly 640×480), **b1 one** the first image only, **b1 640**
each image resized to 640×640, **b16** `predict_batch` of the 16 (padded square 640), **b16 hd** the 16 resized to
1280×720. Two passes, 5 rounds each: `trained/` (22:23–22:34, before **b1 640** existed) and `trained2/`
(22:36–22:50). Quiet pairs: `trained` rounds 4 and 5 (rounds 1–3 had a run at 4.1–6.4 other cores or a browser at
12% of the GPU) and `trained2` rounds 2–5 (round 1: 4.7 and 6.0 other cores), pooled as `logs/trained-pooled/`
(`SOURCES.txt` maps them; summary `logs/trained-pooled-summary.txt`): 6 pairs, 4 for **b1 640**. Load of those
runs: 0–5% other GPU, 1.3–3.3 other cores.

| row | folo-n e5a6ccf0 | folo-n `main` | folo-s e5a6ccf0 | folo-s `main` |
|---|---|---|---|---|
| f32 b1 640 (row 5), median (p90) | 1.77 ms (1.73–1.78), p90 1.88 | **1.60** (1.58–1.62), p90 1.70, −10% | 4.57 (3.16–4.58), p90 6.12; see below | **2.65** (2.63–2.69), p90 2.77 |
| f16 tensor cores b1 640 (row 5) | 1.51 (1.46–1.52), p90 1.65 | **1.33** (1.28–1.35), p90 1.40, −12% | 1.98 (1.96–2.01), p90 2.19 | **1.82** (1.80–1.84), p90 1.90, −8% |
| f32 b1, rect | 1.43 (1.42–1.44) | **1.38** (1.36–1.40), −4% | 2.42 (2.36–2.45) | **2.29** (2.27–2.40), −6% |
| f16 tensor cores b1, rect | 1.43 (1.36–1.57) | **1.24** (1.21–1.31), −13% | 1.69 (1.65–1.75) | 1.63 (1.60–1.67), −4% |
| f32 b1 one | 1.47 (1.45–1.48) | **1.38** (1.36–1.47), −6% | 2.41 (2.35–2.86) | 2.30 (2.21–2.55), −4% |
| f16 tensor cores b1 one | 1.47 (1.33–1.52) | **1.29** (1.26–1.40), −12% | 1.69 (1.65–2.13) | 1.62 (1.59–1.82), −4% |
| f32 b16, val2017 images (padded square) | 10.46 ms (10.05–10.89), 1530 img/s | **9.30** (9.04–10.25), 1721 img/s, −11% | 22.37 (22.02–22.77), 715 img/s | 21.53 (20.41–22.45), 743 img/s, −4% |
| f16 tensor cores b16, val2017 images | 6.24 (6.16–6.30), 2566 img/s | **5.81** (5.74–6.06), 2754 img/s, −7% | 11.22 (11.18–11.45), 1427 img/s | **10.70** (10.64–10.93), 1496 img/s, −5% |
| `predict_batch` 16 × 1280×720, f32 | 14.13 (13.43–25.49) | 13.95 (10.02–21.17), bimodal | 25.59 (25.37–25.83) | **17.44** (16.05–17.91), −32% |
| `predict_batch` 16 × 1280×720, f16 tensor cores | 9.67 (9.62–9.90) | **7.96** (7.90–8.10), −18% | 14.61 (14.55–14.85) | **11.01** (10.95–11.04), −25% |
| detections per image (b1 / b1 640 / b16) | 3.6 / 3.8 / 3.8 | the same | 4.6 / 5.3 / 4.8 | the same |

- **folo-s f32 b1 640 on e5a6ccf0 is bimodal**: 2.95 and 3.16 ms in two runs, 4.57–4.58 in three (round 1 of
  `trained2`, dropped, read 2.95); `main` read 2.63–2.84 in all five. Against e5a6ccf0's fast runs the gain is
  −10 to −16%, against its median −42%; the cell gives both. Not seen on folo-n or with tensor cores.
- **folo-n f32 16 × 1280×720 on `main` is bimodal**: 9.94–10.02 ms in 3 of the 10 runs of both passes, 11.0–21.2 in
  the others (e5a6ccf0 13.4–15.1, one 25.5). The random-weight bench reads the same row 10.20 ms (9.62–10.97) in
  every run, and the f16 tensor-core row is stable on trained weights (7.90–8.33). Not explained: a leftover.
- The square b16 row runs the 16 val2017 images of different shapes, padded square: a batch of mixed canvases,
  so it is candidate-first and not the rect canvas that gains.

### Third pass: row 6 (`trained3/`)

The same bench with **b16 640**, 5 rounds, 01:08–01:15 (`logs/trained3/`). Other GPU 0–1% throughout; other cores
1.6–2.7 except `main` rounds 1 and 2 (17.0, 10.5) and e5a6ccf0 round 5 (10.4). Those three loaded runs are left out
(`logs/trained3-quiet-summary.txt`, `summ3.py ... --skip base-r5,new-r1,new-r2`): on the tensor-core rows they read
7–9% slower (folo-n: `main` 6.57–6.60 ms quiet, 7.05–7.10 loaded; e5a6ccf0 7.12–7.41 quiet, 8.01 loaded), and on
folo-n's f32 b16 640 row one of `main`'s read faster (10.35 ms). All five runs: `logs/trained3-summary.txt` (folo-n
tensor cores 7.37 → 6.60 ms, f32 13.49 → 12.16, −10%). `main`'s round 4 **crashed** after its folo-n rows (exit
0xC0000005, an access violation, below), so its folo-s rows have 2 quiet runs; e5a6ccf0 has 4, folo-n's `main` 3.

| row | folo-n e5a6ccf0 | folo-n `main` | folo-s e5a6ccf0 | folo-s `main` |
|---|---|---|---|---|
| f32 b16 640 (row 6) | 12.79 ms (11.38–15.53), 1251 img/s, p90 23–68 ms on both | 12.49 (12.16–12.65), 1281 img/s, −2%: a tie | 22.64 (22.53–22.76), 707 img/s | **20.94** (20.89–20.98), 764 img/s, −8% |
| f16 tensor cores b16 640 (row 6) | 7.36 (7.12–7.41), 2174 img/s | **6.59** (6.57–6.60), 2428 img/s, −10% | 12.65 (12.26–12.72), 1265 img/s | **11.66** (11.48–11.84), 1372 img/s, −8% |
| f32 b1 640 | 1.73 (1.71–2.20) | 1.56 (1.56–1.59), −10% | 2.78 (2.75–2.81) | 2.67 (2.62–2.72), −4% |
| f16 tensor cores b1 640 | 1.48 (1.45–1.51) | 1.36 (1.35–1.40), −8% | 1.97 (1.96–2.03) | 1.86 (1.83–1.88), −6% |
| f32 b16 hd | 14.16 (14.00–14.26) | **10.23** (10.17–10.24), −28% | 24.95 (24.86–25.17) | **16.06** (16.03–16.08), −36% |
| f16 tensor cores b16 hd | 9.92 (9.87–10.02) | **8.02** (7.98–8.08), −19% | 15.05 (14.84–15.15) | **11.21** (11.15–11.27), −25% |

- folo-s f32 b1 640: e5a6ccf0 ran in its fast mode in all five runs here (2.75–2.81 ms quiet, 3.47 in the loaded
  fifth), and `main` is 4% faster than that. The evening's −42% is e5a6ccf0's slow mode; the targets cell keeps the
  evening's median and says so.
- folo-n f32 b16 hd is steady here on `main` (10.17–10.24 ms quiet, 10.38 at most with the loaded two), where the
  evening passes were bimodal.
- folo-n f32 b16 640 is a tie on the quiet runs (−2%) and −10% on all five: e5a6ccf0 is bimodal on this row
  (11.38–15.53 ms quiet, p90 25–62 ms), so it is not read as a gain.

**Batch-1 synchronised latency, folo-n, the row-5 protocol**: f32 1.77 / 1.88 → 1.60 / 1.70 ms (p50 / p90); tensor
cores 1.51 / 1.65 → 1.33 / 1.40.

## GPU training step (row 9)

`FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=train`, batch 16 at 640, SGD, 5 rounds, 22:52–23:14 (`logs/train/`).
Only round 3 was quiet (1.4–1.7 other cores); the others had 3.8–16.5 other cores and in round 2 a CI test
(`gpu_amp`) at 10–11% of the GPU beside e5a6ccf0's run. The step is GPU-bound and the medians held their range
through that load, so all 5 rounds are used (`logs/train-summary.txt`; round 3 alone: `logs/train-quiet-r3-summary.txt`).

| row | folo-n e5a6ccf0 | folo-n `main` | folo-s e5a6ccf0 | folo-s `main` |
|---|---|---|---|---|
| f32, median step | 41.73 ms (41.53–42.11), 383 img/s | 41.93 (41.67–42.01), 382 img/s: tie (+0.5%) | 99.06 (97.79–99.19), 162 img/s | 98.22 (97.59–98.64), 163 img/s: tie |
| AMP, median step | 27.44 (25.87–36.58), 583 img/s | 26.30 (26.03–27.27), 608 img/s: −4% inside overlapping ranges (one slow e5a6ccf0 round); the quiet round 3 a tie | 53.82 (53.03–56.45), 297 img/s | 53.28 (53.20–54.69), 300 img/s: tie |
| round 3 only, f32 / AMP | 41.53 / 25.87 ms | 41.67 / 26.03 ms | 97.79 / 53.15 ms | 97.59 / 53.28 ms |
| dispatches, new bind groups, new pool buffers per step (f32) | 1117, 0, 0 | 1117, 0, 0 | 1120, 0, 0 | 1120, 0, 0 |
| peak live memory f32 / AMP | 3467 / 2022 MiB | the same | 6520 / 3713 MiB | the same |

Lane plan's unresolved +5.4% on the folo-n f32 step (52.5 → 55.3 ms beside another GPU process) does not reproduce:
+0.5% here, inside the run-to-run range of both builds. A training step records no plan and runs the same kernels.

## CPU (rows 7, 8, `predict_batch`)

`cpu_e2e`, 28 rayon threads, `bin/cpu-runs.sh <scale>`: `FOLO_BENCH_ITERS` 60 / 40 / 20 (folo-n / s / m),
`FOLO_BENCH_BATCH=16`, and for folo-n and folo-s `FOLO_BENCH_WEIGHTS=~/.folo/weights/folo-{n,s}-coco.folo
FOLO_BENCH_IMAGES=D:/folo-data/coco/val2017` (the trained rows). Runs, all under the `cpu` lock:

| dir | what | rounds | time (log files written) | other cores (other GPU) |
|---|---|---|---|---|
| `cpu-n/` | whole bench, folo-n | 6 (round 1 at `--wait-cpu 2`, 2–6 at 5) | runs ending 23:20–23:32 | 3.7–7.4 (the game at 26–33% of the GPU from round 3) |
| `cpu-s/` | whole bench, folo-s | 5 | 23:34–23:58 | 4.6–6.0, one 3.4 (the game until about 23:50) |
| `cpu-m/` | whole bench, folo-m (untrained only) | 3 | 00:03–00:10 | 3.8–4.4 |
| `cpu-rect-s/` | trained rows alone (`FOLO_BENCH_BATCH=0 FOLO_BENCH_UNTRAINED=0`, 60 calls), folo-s | 4 | 00:11–00:19 | 2.8–8.7 (runs of 4 s: a coarse estimate) |
| `cpu-rect-n/` | the same, folo-n | 3 | 00:20–00:26 | 1.3–4.6 |
| `cpu-rect2-s/` | folo-s whole bench without the batches (`u1b0`) or without the untrained rows (`u0b16`), four labels | 3 | 00:28–00:35 | 1.9–4.4 |
| `cpu-s-late/` | whole bench, folo-s again | 3 | 00:37–00:42 | 1.8–4.2 |
| `cpu-bisect-s/` | whole bench, folo-s, with f04dce45 (#254 + #255, before #258 / #259) as a third binary | 3 | 00:45– | see `cpu-bisect-s-summary.txt` |

Most runs were above the 3-core quiet rule (the machine's floor that night was 4–7 cores until about 23:50 and 2–4
after), so nothing was dropped and every cell below gives its load. The runs that met it (the `# load:` header of
each log, `other cores` ≤ 3) agree with the cells: row 8 for folo-n with the rows alone, e5a6ccf0 13.41 and 13.58 ms
(`cpu-rect-n/base-r2`, `-r3`) against `main` 13.54 (`new-r3`); folo-s's trained 1280×720 frame in the whole bench
21.05–22.80 ms without #259 (`cpu-s-late/base-r3`, `cpu-bisect-s/base-r2`, `f04-r2`, `cpu-bisect2-s/f04-r1..r3`)
and 26.92–29.20 with it (`cpu-s-late/new-r3`, `cpu-bisect-s/new-r1`, `-r2`, `cpu-bisect2-s/new-r1`, `-r3`,
`r259-r1`, `-r3`). Summaries `logs/<dir>-summary.txt`.

| row (28 threads) | folo-n e5a6ccf0 | folo-n `main` | folo-s e5a6ccf0 | folo-s `main` | folo-m e5a6ccf0 | folo-m `main` |
|---|---|---|---|---|---|---|
| forward b1 (row 7), whole bench | 14.64 ms (13.61–15.09) | 14.46 (14.16–14.90), tie | 30.48 (29.76–30.70) | 31.41 (30.28–31.75), tie (+3.1%, ranges overlap) | 75.32 (74.71–75.76) | 75.47 (75.20–76.07), tie |
| e2e b1, trained, 640×640 (row 8), rows alone | 13.58 (13.41–14.87) | 13.57 (13.54–14.13), tie | 31.00 (30.30–31.48) | 29.40 (29.05–29.64), −5% | | |
| the same, whole bench | 15.65 (14.39–15.79) | 15.36 (14.73–16.13), tie | 31.37 (30.97–32.92) | 31.19 (30.16–31.70), tie | | |
| e2e b1, trained, 1280×720 (`rect` 640×384), rows alone | 11.73 (10.88–11.94) | 11.10 (10.82–11.40), −5% | 22.27 (21.08–22.97) | 21.25 (20.48–21.40), −5% | | |
| the same, whole bench | 12.52 (11.19–12.96) | 12.04 (10.43–12.64), −4% | 22.08 (21.46–23.15) | **27.29** (26.39–31.07), +24% | | |
| e2e b1, untrained, 640×640 | 15.45 (13.74–15.63) | 13.84 (12.81–14.05), −10% | 31.18 (30.48–31.61) | 30.11 (29.54–30.44), −3% | 76.46 (75.85–76.92) | 74.23 (73.09–74.54), −3% |
| e2e b1, untrained, 1280×720 | 12.46 (11.21–12.64) | 11.38 (10.33–11.58) | 21.41 (20.97–22.89) | 21.68 (20.22–22.38) | 52.78 (51.50–53.31) | 50.52 (49.98–50.56) |
| `predict_batch` 16, trained, val2017 images (padded square) | 151.7 (149.6–153.7) | 130.2 (129.0–134.0), −14% | 385.5 (379.5–392.7) | 354.0 (347.5–363.3), −8% | | |
| `predict_batch` 16 × 1280×720, trained | 146.7 (143.0–150.6) | 96.5 (93.9–100.2), −34% | 382.5 (380.3–398.2) | 275.2 (272.8–283.5), −28% | | |
| `predict_batch` 16 × 1280×720, untrained | 148.0 (140.2–150.1) | 95.7 (92.2–98.5), −35% | 382.5 (380.6–407.2) | 279.4 (275.2–283.7), −27% | 1045.8 (1041.2–1056.9) | 667.6 (666.4–669.6), −36% |
| `predict_batch` 16 mixed shapes, untrained | 149.6 (144.0–151.2) | 129.2 (124.9–130.6), −14% | 378.6 (377.4–397.9) | 352.7 (345.8–366.0), −7% | 1056.5 (1054.2–1062.4) | 998.9 (982.4–1001.8), −5% |
| detections per image, trained 640 / 1280×720 (whole bench, 40 or 60 calls) | 3.9 / 3.9 | the same | 4.6 / 3.9 | the same | | |

- Rows alone: `logs/cpu-rect-{n,s}-summary.txt`; whole bench: `logs/cpu-{n,s,m}-summary.txt`.
- **Row 8 does not move for folo-n on trained weights** (13.58 → 13.57 ms alone, −2% in the whole bench). On untrained
  weights `main` runs no box branch (−10%: the most candidate-first can give on this row).
- **folo-s trained 1280×720 in the whole bench, +24 to +29% on `main`**: 22.08 → 27.29 (`cpu-s/`, 5 rounds) and 21.88
  → 28.21 ms (`cpu-s-late/`, 3 rounds, 1.8–4.2 other cores), slower in every run; the timing split moves into
  inference (`inf` 21.0 → 25.0 ms in `cpu-s/` round 1), the detections are the same (3.9 per image). It is 5% faster
  on `main` with the trained rows alone (22.27 → 21.25), without the batches (`u1b0`: 22.01 → 20.91) and without the
  untrained rows (`u0b16`: 21.94 → 20.87; `cpu-rect2-s-summary.txt`). It needs both: the untrained model's rows
  (with its `predict_batch` rows) before the trained model's 1280×720 row in one process. folo-n's whole bench does
  not show it.
- **Bisect** (`cpu-bisect-s/`: e5a6ccf0, f04dce45, ffa0f8cb; `cpu-bisect2-s/`: f04dce45, 25666428, ffa0f8cb; 3 rounds
  each, 1.7–4.9 other cores; `bin/f04-cpu_e2e.exe` and `bin/r259-cpu_e2e.exe` built from `D:/folo-wt/hill3-foldin-f04`
  at each commit with `main`'s `cpu_e2e.rs`, `bin/{f04,r259}-commit.txt`): the trained 1280×720 row reads 22.45 /
  21.29 / 27.29 ms and 21.45 / **29.13** / 26.92 ms. It arrives with #259, whose only library change is
  `predict.rs`'s rect batch: in the whole bench the untrained model's `predict_batch` of 1280×720 frames now runs at
  640×384 (in chunks of two on the CPU), and the trained model's single 640×384 calls after it are slower. Not
  diagnosed; a process-wide cache or scratch keyed by shape is the first place to look. The same runs give #259 alone
  on the batch rows: 16 × 1280×720, trained, 347.8 → 276.1 ms (−21%), untrained 342.3 → 274.1 (−20%).

## Accuracy: the fusion flags on the published weights

`folo val --model <weights> --data D:/folo-data/coco --coco --device gpu:0 --json` with and without `--fusion dfl
--iou 0.65 --max-det 1000` (`bin/acc.sh` with ffa0f8cb's `folo`; `bin/acc-v3.sh` with 8ea21bab's for v3), one run each
(deterministic), `logs/acc/<weights>-{default,bstar}.json` and `logs/acc/summary.txt`:

| weights | default AP / AP50 / AP75 | fusion flags AP / AP50 / AP75 | gain | lane fusion's B0 → B* |
|---|---|---|---|---|
| `folo-n-coco` (v1) | 0.3138 / 0.4554 / 0.3392 | **0.3181** / 0.4612 / 0.3424 | +0.0043 | 0.3138 → 0.3181 |
| `folo-n-coco-v2` | 0.3168 / 0.4578 / 0.3428 | **0.3202** / 0.4634 / 0.3457 | +0.0034 | 0.3168 → 0.3202 |
| `folo-s-coco` | 0.4162 / 0.5798 / 0.4519 | **0.4186** / 0.5840 / 0.4537 | +0.0025 | 0.4162 → 0.4186 |
| `folo-n-coco-v3` (f32) | 0.3396 / 0.4874 / 0.3671 | **0.3434** / 0.4922 / 0.3692 | +0.0038 | 0.3396 → 0.3434 (landed) |
| `folo-n-coco-v3-f16` (`--half`) | 0.3395 / 0.4874 / 0.3669 | not run | | |

Every number matches lane fusion's notebook (r3-fusion.md, Headline and "B* on val2017") to its four digits, and
v3 matches #261's verification (0.339555 / 0.487367 f32, 0.343357 with the flags; here 0.3395551 / 0.4873670 and
0.3433569). The intervals of the gains are lane fusion's paired bootstrap (1000 resamples, all above zero), not
measured again. v3 is a pull of release `weights-v4` (`folo weights pull folo-n-coco-v3` into
`weights/`, checksums verified by the command, via `gh`: the release is not public); its sha256 (`f366a2fd...`)
equals the landing run's `best-weights.folo`. The `detections` field of the JSON (910,288 for v1's default) is lower
than lane fusion's count (970,945) because it counts after COCOeval's cap of 100 detections per image and class
(`cocoeval.rs`, `num_dets`); AP is computed on the same capped set by both.

Evaluation time (`seconds`, one run each, not an A/B): v1 5.3 → 6.3 s with the flags, v2 4.1 → 6.4, folo-s 8.8 →
14.1, v3 7.0 → 5.9. The flags cost an extra readback of the side moments and score up to 1000 detections per image;
lane fusion's A/B has the per-image cost (r3-fusion.md, Timing).

## Candidates per image (quoted)

From r3-lazy-head.md (all 5000 val2017 images at conf 0.25, published weights): folo-n median 21 of 8400 anchors
need a box, p99 119, max 227; folo-s median 26, p99 143, max 249. Not measured again.

## Against the lanes' own claims

Each lane's own number, then this notebook's (the cells above):

| claim (notebook) | the lane | here | reading |
|---|---|---|---|
| plan: folo-n f32 training step (r3-plan.md, Training step) | 52.5 → 55.3 ms, +5.4%, beside another GPU process | 41.73 → 41.93 ms, +0.5% | **does not reproduce** (a tie) |
| plan: AMP training step | 38.3 → 35.0 ms, −8.6% | 27.44 → 26.30 ms, −4% inside overlapping ranges (one slow e5a6ccf0 round); 25.87 → 26.03 in the quiet round | does not reproduce |
| plan: rows 1 / 2, two binaries | −23% / −35% at 5–88% CPU load | plans off → on on one binary: −3.5% (f32), −10% (tensor cores) | the lane's quieter `gpu_latency` rounds (−3.5% / −5.7%) and its in-process toggle (about −5% / −10%) agree; the loaded two-binary gains do not reproduce |
| plan: folo-s b1, plans | a tie | −6% (f32), −8% (tensor cores), plans off → on | larger here |
| plan: rows 3, 4, 6 at batch 16 | +2.4 to +5.3% slower | ties (forward); no plan at batch 16 | **does not reproduce** |
| plan: mixed-batch slow mode "no longer on `main`" | not reproduced on `main` | 35–39 ms in 3 of 9 runs, on both builds | **reproduces**, predates the round |
| plan: GPU 16 × 1280×720 (#259 alone, on ca37efc5) | −21 / −14 / −32 / −21% (n f32, n tc, s f32, s tc) | against e5a6ccf0 (with candidate-first): random weights −28 / −21 / −31 / −25%; trained n tc −18%, s −32 / −25%, n f32 bimodal | consistent (this includes candidate-first) |
| plan: CPU 16 × 1280×720 (#259 alone) | −16.6% (155.2 → 129.5 ms, 4–100% CPU load) | −20% untrained, −21% trained (bisect, f04dce45 → 25666428, folo-s) | agrees, a little larger |
| lazy: GPU batch 16, end to end | n f32 −10.5%, n tc −11.7%, s f32 −8.9%, s tc −4.1% | trained, val2017 images: −11%, −7%, −4%, −5% | agrees for folo-n f32 and folo-s tensor cores; folo-n tensor cores and folo-s f32 smaller here |
| lazy: GPU row 5 | "not established" | −10% f32, −12% tensor cores (folo-n, 640×640, trained); folo-s −8% (tc), f32 bimodal on e5a6ccf0 | established here |
| lazy: CPU row 8, folo-n 640² trained | 17.49 → 15.02 ms, −14%, at 32–55% load | 13.58 → 13.57 (rows alone), 15.65 → 15.36 (whole bench) | **does not reproduce** (a tie) |
| lazy: CPU folo-n 1280×720 trained / batch of 16 | −8% / −12% | −5% / −14% | agrees |
| lazy: CPU folo-s | unreadable under load | −5% at 640×640 and 1280×720 (rows alone); +24–29% on 1280×720 in the whole bench (#259, below) | |
| counsel: CPU row 8 | −9 to −12% (estimate) | a tie on trained weights | not reached |
| fusion: AP with the flags, every published checkpoint | r3-fusion.md Headline | the same to four digits | **reproduces** |
| #261: `folo-n-coco-v3` | 0.339555 / 0.487367; 0.343357 with the flags | 0.3395551 / 0.4873670; 0.3433569 | **reproduces** |


## Rows worse than e5a6ccf0

- **folo-s, CPU, a trained 1280×720 frame after an untrained model's rect `predict_batch` in the same process**:
  +24 to +29% (CPU section), from #259 (bisect). The only row slower than e5a6ccf0 at its median.
- **folo-n f32 `predict_batch` 16 × 1280×720, GPU, trained**: a tie at the median (14.13 → 13.95 ms), bimodal on
  `main` (3 of 10 runs at 10.0 ms, the others 11–21).
- **folo-n f32 forward b16, pipelined**: 2064 → 1962 img/s (−5%) with overlapping ranges; the synchronised median ties
  (7.81 / 7.84 ms) and the code path did not change (no plan at batch 16): noise.
- **folo-s CPU forward b1**: +3.1% in the whole bench (30.48 → 31.41 ms, ranges overlap), −4.7% and −1.3% in the later
  bisect runs (`cpu-bisect-s`: 29.52 → 28.14 against e5a6ccf0; `cpu-bisect2-s`: 28.86 → 28.49 against f04dce45):
  noise; the forward did not change.

Nothing else measured is slower than e5a6ccf0.

## What could not be measured

- The incumbent was not measured again (as the brief says).
- folo-m on trained weights: there are no published folo-m weights, so its CPU rows are the random-weight ones,
  where candidate-first runs no box branch.
- Rows 10 and 11 (the trainer and its memory): not re-measured, since row 9 did not move.
- A quiet training-step A/B: one quiet round of five (above).
- A quiet CPU A/B: most CPU runs were above the 3-core rule and only a few of each row were quiet (CPU section);
  every CPU cell gives its load.
- Candidate-first's bit-identity and the plan's zero mismatches on val2017 were not re-tested; the detections per
  image of every trained row are the same on both builds.
- The GPU clocks were not logged per run, so the bimodal rows above (e5a6ccf0's tensor-core forward b1, folo-s f32
  b1 640, the e2e b1 rows of the random-weight bench, folo-n f32 1280×720 batches on `main`) cannot be tied to them.

## Leftovers

- **An access violation in `main`'s `gpu_trained`** (`logs/trained3/new-r4.txt`: exit 3221225477 = 0xC0000005 after
  23 s, at the switch from folo-n to folo-s: folo-n's predictors dropped, folo-s's being opened; no Windows event-log
  entry). Once in 30 runs of `main`'s GPU bench binaries tonight (`gpu` 10 with plans on or off and 5 training-step
  runs, `gpu_trained` 15) and in none of e5a6ccf0's 25. Not reproduced. Forward plans pin buffers per predictor, so a plan outliving its
  model or context is the first thing to rule out. Reproduce: the `trained` section of `bin/gpu-runs.sh` in a loop.
- **folo-n f32 `predict_batch` of 16 × 1280×720 frames on trained weights reads 10.0 or 11–21 ms run by run on
  `main`** in the evening passes (above), steady at 10.2 ms in the third pass. Reproduce: `FOLO_BENCH_WEIGHTS=<weights dir>/folo-{scale}-coco.folo
  FOLO_BENCH_IMAGES=D:/folo-data/coco/val2017 FOLO_BENCH_SCALES=n FOLO_BENCH_PRECISION=f32 FOLO_GPU_TENSOR_CORES=1
  cargo bench -p folo-vision --bench gpu_trained`, several runs, the `b16 hd` row. The random-weight bench and f16
  on tensor cores do not show it.
- **folo-n f32 `predict_batch` of mixed shapes: a 35–39 ms mode on both builds** (random weights, the `gpu` bench's
  `predict b16 mixed` row, 3 of 15 runs), against 10.5–14.6 ms.
- **The random-weight bench's e2e b1 rows are bimodal tonight on every binary** while the trained bench's are not;
  both warm up for a second first. Not explained; GPU clocks per run would be the first thing to log.

## Leftovers (CPU)

- **folo-s on the CPU: single 1280×720 frames +24–29% after another model's rect `predict_batch` in the same
  process**, since #259. Reproduce: `FOLO_BENCH_SCALE=s FOLO_BENCH_ITERS=40 FOLO_BENCH_BATCH=16
  FOLO_BENCH_WEIGHTS=<weights dir>/folo-s-coco.folo FOLO_BENCH_IMAGES=D:/folo-data/coco/val2017 cargo bench -p
  folo-vision --bench cpu_e2e`, the `trained 1280x720 (rect 640x384)` row (about 27–29 ms on `main`, 21–22 ms on
  f04dce45); with `FOLO_BENCH_BATCH=0` or `FOLO_BENCH_UNTRAINED=0` it reads about 21 ms. A server holding several
  models (the LRU of predictors) can meet the same order. Not diagnosed.
- **A quiet CPU repeat**: most CPU runs were loaded, only three of row 8's quiet. Row 8's tie and the folo-s numbers
  should be read again on an idle machine, with the 8-pinned-P-core rows of the first fold-in.

## Logs

Every cell's log, under `D:/folo-data/runs-hill3/foldin/`:

| cells | logs |
|---|---|
| GPU inference, random weights (forward, e2e, batches, plans off) | `logs/infer/{base,new,nop}-r1..5.txt`; `logs/infer-driver.txt` (the `# load:` lines); summary `logs/infer-summary.txt` |
| GPU prediction, trained weights | `logs/trained/{base,new}-r1..5.txt`, `logs/trained2/{base,new}-r1..5.txt`, pooled quiet pairs `logs/trained-pooled/` (`SOURCES.txt`); summaries `logs/trained-summary.txt`, `logs/trained2-summary.txt`, `logs/trained-pooled-summary.txt`; row 6: `logs/trained3/`, `logs/trained3-quiet-summary.txt` (the cells), `logs/trained3-summary.txt` (all five runs), `logs/trained3-{driver,start,end}.txt` |
| GPU training step | `logs/train/{base,new}-r1..5.txt`; `logs/train-summary.txt`, `logs/train-quiet-r3-summary.txt` |
| CPU rows | `logs/cpu-{n,s,m}/`, `logs/cpu-s-late/`, `logs/cpu-rect-{n,s}/`, `logs/cpu-rect2-s/`, `logs/cpu-bisect-s/`, `logs/cpu-bisect2-s/` (`{base,new,f04,r259,...}-rN.txt`), each with `logs/<dir>-summary.txt` and `logs/<dir>-driver.txt`; bisect binaries `bin/{f04,r259}-cpu_e2e.exe`, `bin/{f04,r259}-commit.txt`, `logs/build-{f04,r259}.log` |
| accuracy | `logs/acc/*.json`, `logs/acc/*.err`, `logs/acc/summary.txt` |
| builds | `logs/build-{new,new2,base}.log`, `bin/{base,new,main}-commit.txt` |
| scripts | `bin/ab.py`, `bin/summ3.py`, `bin/build.sh`, `bin/gpu-runs.sh`, `bin/cpu-runs.sh`, `bin/cpu-rect.sh`, `bin/cpu-rect2.sh`, `bin/cpu-bisect.sh`, `bin/cpu-bisect2.sh`, `bin/acc.sh`, `bin/acc-v3.sh` |
