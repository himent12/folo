# Hill-climb 3, lane `cpufix`: the fold-in's two CPU findings and one crash

The fold-in (r3-fold-in.md) left three things unexplained: (A) candidate-first prediction gave folo-n nothing on
the CPU at 640×640 with trained weights (13.58 → 13.57 ms) although skipping the box branch entirely was worth 10%;
(B) folo-s single 1280×720 frames ran 24–29% slower after another model's rect `predict_batch` in the same process,
since #259; (C) one access violation in 30 runs of the GPU bench at a model switch. Brief:
`D:/folo-build/hill3/briefs/cpufix.md`. Logs, binaries and scripts: `D:/folo-data/runs-hill3/cpufix/` (`logs/`,
`bin/`, `scripts/`); every path below is relative to it.

Outcome: B was the CPU buffer cache, which never evicted, and is fixed (#264). A had no large fixed cost: at the
fold-in's 60 calls the tie was noise, and in the whole bench B's mechanism hid the gain. Candidate-first is worth
about 7% of the model step on idle cores and 9–10% under load; the one change I tried on top is a tie. C did not
reproduce in 2×(500 + 4000) predictor lifetimes, with and without forward plans, nor in 20 more runs of the
fold-in's GPU bench.

## Method
- Machine as docs/BENCHMARKS.md, at night (2026-11-14, 01:40–05:00), mostly quiet: 1.4–3.0 other cores in most
  timed runs (each log's `# load:` line), the GPU idle. CI of #264 ran once in between, and no run was timed through
  it (the runner waits for ≤ 3 other cores before each run).
- Runner: the fold-in's `bin/ab.py` (alternates binaries or settings run by run, order flipped every other round,
  High priority unless said otherwise, `# load:` header), driven by `bin/run.sh <dir> <rounds> <wait-cores> ...`
  under the `cpu` lock; summaries by the fold-in's `bin/summ3.py` (median of the run medians, min–max over runs).
- Baseline `bin/base-*.exe`: `origin/main` at the branch point (8ea21bab), from `D:/folo-wt/hill3-cpufix-base`
  with this branch's `cpu_e2e.rs` (the control row below) uncommitted; `bin/base-commit.txt`.
- Scratch probes, not in any PR (their sources in `scripts/`): `cpu_probe.rs` (call sequences across two predictors,
  printing `buffer_cache_bytes`), `lazy_probe.rs` (candidate-first against dense in one process, alternated call by
  call, with the stage timers that `scripts/instrument_lazy.py` adds to `model/lazy.rs`; `scripts/prof-tree.diff`
  is the whole profiling tree), `gpu_stress.rs` (part C).

## B. Single frames slower after a rect batch: the CPU buffer cache never evicted

folo-core's CPU buffer cache (`cpu/pool_alloc.rs`) keeps freed f32 buffers of at least 16 KiB by exact length, up to
1 GiB, and dropped any returned buffer that did not fit. Once full, it held whatever had filled it first for the life
of the process. `cpu_e2e`'s untrained rows end with three `predict_batch` rows of 16 images; on the CPU each runs 8
chunks of 2 images side by side, so the cache fills to 1 GiB with batch-2 lengths. Since #259 the 1280×720 batch runs
at 640×384 instead of 640×640, which adds a second set of batch lengths. The trained model's 1280×720 frames then
need lengths that are not there: its box branch at 640×384, which the untrained model never ran because it passes no
candidate. Those lengths can never get in, so every such buffer is a fresh allocation (page faults included) on every
call. #259 exposed it; the cache policy caused it.

Smallest sequence (`cpu_probe`, folo-s, 28 threads, `logs/probe2/`, `probe3/`, `probe3fix/`; S, B, M = the untrained
model's `predict_batch` of 16 square, 1280×720 and mixed images, 4 calls each; then a trained predictor's 16 square
frames and its 1280×720 frames):

| sequence | cache after | trained 1280×720 frames |
|---|---|---|
| trained frames only | 413 MiB | 20.4 ms |
| S, B, M, then trained | 1023 MiB (full) | **36.3 ms** (38.2 in a second run) |
| S, M (no rect batch), then trained | 1020 MiB | 20.8 ms |
| S, B, then trained | 1023 MiB | 38.8 ms |
| B, M or B alone, then trained | 1023 MiB | 21.0 / 20.9 ms (the trained square row, which runs first, pays instead: 34.3 / 31.6 against 27.3 ms) |
| S, B, M, `clear_buffer_cache()`, trained | 233 MiB | 20.3 ms |
| cache limit 8 GiB, S, B, M, trained | 2039 MiB | 20.4 ms |
| **eviction (#264)**, S, B, M, trained | 1009 MiB | **20.8 ms** |

The batches pay too: B after S ran 267–275 ms per call with the cache full, 183–189 ms with room.

**Fix (#264)**: when a returned buffer does not fit, the stale lengths are evicted, then the buffer is kept if it
now fits. A length is stale once it has gone unused (neither taken nor returned) for two limits (2 GiB) of traffic
through the cache, and for twice the longest it ever went unused before being asked for again. A scan is skipped
until the soonest kept length can be stale. The second bound came from the review (below): with the first alone,
a workload that cycles through more than twice the cache evicts each cached length just before it comes back and
ends with no hits, where the old policy kept a cache-sized part. To learn that, a length's record (last use,
longest idle stretch) outlives its buffers, and the records of lengths with no buffer are dropped once unused for 16
limits and stale (so for max(16 limits, twice the longest idle stretch)). A return longer than the limit is refused
before it counts as traffic. Changing the limit resets the scan deadline. Tests on the cache struct, each failing
when its part is taken out:
- three workloads in turn, each admitted after about two limits of traffic (also fails with the old policy, and
  with the empty-cache bound bug the first version had);
- cyclic workloads of 1.5, 3, 6 and 12 limits per cycle keep 4 of their lengths cached and evict nothing once
  settled;
- returns over the limit age nothing;
- a smaller limit ages by the new limit at once;
- lengths used once are forgotten;
- no rescan before anything can be stale.

`cpu_e2e` gains the control row `trained 1280x720, buffer cache cleared`.

`cpu_e2e`, whole bench, base against #264's first version, 4 rounds each, 2.4–4.0 other cores (`logs/prB-{s,n}/`,
`prB-{s,n}-summary.txt`):

| row | folo-s base | folo-s #264 | folo-n base | folo-n #264 |
|---|---|---|---|---|
| trained 1280×720 (rect 640×384) | 26.99 ms (25.81–28.04) | **21.40** (20.20–21.84), −21% | 11.34 (11.00–11.43) | 11.62 (11.07–11.68), tie |
| ... buffer cache cleared (control) | 20.87 | 21.31 | 11.11 | 11.54 |
| trained 640×640 (row 8, whole bench) | 30.09 (27.75–30.48) | **28.26** (27.03–28.48), −6% | 14.98 (14.78–15.21) | **14.31** (13.99–14.92), −4.5% |
| `predict_batch` 16 × 1280×720 | 272.0 (265.9–281.6) | **196.2** (190.9–196.6), −28% | 94.0 (93.8–97.1) | **72.3** (70.1–77.7), −23% |
| `predict_batch` 16 × 640×640 / mixed | 325.7 / 330.2 | 314.3 / 319.2 | 121.4 / 127.0 | 120.2 / 125.2 |

The +2.5 to +4.7% on folo-n's single 1280×720 rows sits beside +3.9% on the control row, where the two policies do
the same thing (the cache never fills there), and the new side's runs had the more load (up to 4.0 cores): noise.
The 16 × 1280×720 batch rows are #259's real CPU gain, which the full cache had been eating. For scale only, from
other runs: the fold-in measured that row at 382.5 ms at e5a6ccf0 and 275.2 at `main` (#262,
`docs/notes/hill/r3-fold-in.md`, its CPU table).

The final policy (`v2`) against the base and the first version (`prB`), the same protocol, 4 rounds each in turn,
1.4–2.0 other cores (`logs/v2-{s,n}/`, `v2-{s,n}-summary.txt`); folo-s had 4 more rounds of `prB` against `v2`
(`logs/v2b-s/`), and its `prB`/`v2` cells are over all 8:

| row | folo-s base | prB | v2 | folo-n base | prB | v2 |
|---|---|---|---|---|---|---|
| trained 1280×720 | 26.36 ms (25.21–27.84) | 20.94 | **21.26**, −19% | 10.61 | 10.59 | 11.10 (10.45–11.30), tie |
| ... buffer cache cleared (control) | 20.17 | 20.66 | 20.95 | 10.59 | 10.31 | 10.46 |
| trained 640×640 (row 8) | 29.48 (28.99–29.64) | 28.07 | **28.43** (27.79–29.65), −3.6% | 14.48 (14.35–14.58) | 13.61 | **13.61**, −6% |
| `predict_batch` 16 × 1280×720 | 264.2 | 194.4 | **194.6**, −26% | 89.4 | 69.8 | **68.8**, −23% |
| untrained predict 1280×720 | 20.29 | 20.23 | 20.28 | 9.87 | 10.48 | 9.81 |

`v2` against `prB` reads +1.3 to +1.5% on folo-s's trained rows with the control row at +1.4%, where the two do the
same thing: a tie. folo-n's trained rect row at +4.8% for `v2` against `prB` (and `prB` at +6% against the base on the
untrained rect row in the same runs) are within what folo-n's rows move from one set of runs to the next (the
control row: `prB` +3.9% against the base in the first set, −2.6% here): noise at this scale.

CPU training is the workload that cycles through more than the cache (`train_step`, folo-n b16, base against `v2`,
3 rounds each, 1.1–2.7 other cores, `logs/v2-ts320/`, `v2-ts640/`): 320² base 216.7–226.3 ms, `v2` 212.1–218.3;
640² base 960.8–984.3 ms, `v2` 953.5–963.0. Ties. The first version measured the same (`logs/ts320/`, `ts640/`),
but that was before its empty-cache bound fix, so those runs do not stand for its final form. A draft that measured
staleness against the refused length's own reuse distance (`scripts/pool_v2_chunk.rs`) was dropped unmeasured: it
used the last reuse distance, which within a training step is short, where `v2` keeps the longest.

## A. Candidate-first on the CPU at 640×640: no large fixed cost

### Against the number of candidates
`lazy_probe`, folo-n `folo-n-coco`, val2017 000000000139 letterboxed to 640×640, conf chosen per target so exactly
that many anchors pass, `predict_candidates` + `finish_candidates` and dense `forward` + `detections` alternated
call by call, 60 calls each (`logs/prof1/`, one run, 3.7 other cores, so the two medians are noisy; the stage times
are means over the calls):

| candidates | candidate-first | dense | box stage (wall) | `conv0` + `conv1` + `out` (summed over levels) |
|---|---|---|---|---|
| 0 | 12.92 ms | 14.64 | 0.02 ms | 0 |
| 1 | 13.68 | 14.67 | 0.33 | 0.19 + 0.11 + 0.01 |
| 5 | 13.87 | 14.89 | 0.36 | 0.18 + 0.14 + 0.01 |
| 21 | 13.41 | 14.43 | 0.35 | 0.41 + 0.17 + 0.03 |
| 119 | 13.11 | 13.99 | 0.38 | 0.64 + 0.23 + 0.07 |

The sparse box branch has a fixed cost of about 0.3 ms (0 → 1 candidate). Beyond that it barely grows, and it is
small beside the 12 ms of backbone and neck and the 1.0 ms of the three class branches (side by side; 2.1–2.5 ms
summed). The dense head costs about 2.6 ms (dense minus features), the candidate-first head about 1.35.

### The bench's protocol, in one process
The first 16 val2017 images letterboxed square at conf 0.25 (27.6 candidates per image, row 8's input), 400 calls
per side, alternated (`logs/profA-n/`, 3 rounds, 1.4–2.3 other cores): candidate-first 12.11 / 12.71 / 12.66 ms
against dense 13.01 / 13.59 / 13.54, a ratio of 0.931–0.937 (−7%). Stages: features 10.93 ms, class branches and
candidates 0.89, box stage 0.34 (conv0 0.35, conv1 0.14, out 0.03 summed over levels), decode 0.01. So 7% (the head
saved) plus 2.6% (the box stage kept) is the 10% that "no box branch at all" gives.

### Idle and loaded
The same at normal priority with my own busy-loop processes (`scripts/spin.py`) on 4 and 8 cores
(`logs/profA-load{0,4,8}/`, 3 rounds each):

| other cores | candidate-first / dense, as on `main` | with the per-level pipeline (below) |
|---|---|---|
| 1.5–2.0 | 0.931–0.937 | 0.915–0.925 |
| 5.6–6.3 | 0.916–0.921 | 0.901–0.906 |
| 9.5–10.2 | 0.904–0.906 | 0.891–0.900 |

The gain grows with load (the dense box branches beside the class branches compete for cores), which explains part
of lane lazy's −14% at 32–55% load. On idle cores the dense box branch is not free either: about 7% of the step.

### Why the fold-in read a tie
- **Row 8 alone, 60 calls**: e5a6ccf0 13.41 / 13.58 against `main` 13.54, three quiet runs. With 200 calls, 4
  rounds (`logs/row8-n/`, 1.4–4.0 other cores), the fold-in's own binaries read e5a6ccf0 13.88 ms (13.38–14.58),
  `main` 13.04 (12.36–13.45): **−6%**. Later and quieter, `main` (8ea21bab) alone read 12.28 (12.21–12.54) at
  1.5–2.5 cores (`logs/prA-row8-n/`). The tie was the spread of short runs.
- **Row 8 in the whole bench** (15.65 → 15.36): the untrained rows fill the buffer cache, and the trained model's
  box-branch `_at` buffers (full-map outputs of `conv0` and `conv1`, which the untrained model never needed) are then
  fresh allocations on every call: B's mechanism. With #264 that row reads 14.98 → 14.31 against the build before
  (−4.5%; folo-s −6%).

### Tried, a tie
- **Per-level pipeline** (`scripts/pipeline-tried.diff`): for an `o2m` head each level runs its class branch, its
  candidates and its box branch in one task, without the barrier between the class phase and the box phase (`o2o`
  keeps it: its top-k spans the levels). Bit for bit (`lazy_head` passed). In process it moved candidate-first's
  ratio by 1–1.5 points (table above; −3% of the candidate-first step at idle). Through the `Predictor`, rows alone,
  4 rounds, 1.0–2.5 other cores (`logs/prA-row8-{n,s}/`): folo-n row 8 12.28 → 12.42 ms, folo-s 27.82 → 27.82, the
  1280×720 rows −2%. A tie, so not proposed.
- **Small `_at` Winograd passes on one thread** (below `PAR_FLOPS`, or 4× that: no two-pass fork over 28 threads for
  a handful of tiles). Ties in process (`logs/profA-n/`, `ser1`, `ser4`: ratios 0.927–0.937).

What is left on this row is the backbone and neck (about 85% of the step) and the dense class branches. The sparse
box branch's 0.3 ms floor would be the next thing to shave, about 2%.

## C. The GPU access violation: not reproduced

`gpu_stress` (`scripts/gpu_stress.rs`; main's library, built in the profiling tree, which differs from main only in
CPU timers): per cycle, for folo-n then folo-s, f32 then f16 on tensor cores (`FOLO_GPU_TENSOR_CORES=1`; the
fold-in's crash came at the switch from folo-n's f16 tensor-core predictor to folo-s's f32 one): open a `Predictor`
from the published weights, `predict` one image 3 times (rect: a plan records at the 2nd call and replays from the
3rd), `predict_batch` the 16 images 3 times (square) and the 16 resized to 1280×720 3 times (rect canvas), then
queue a `forward` without reading it back and drop the predictor (its plans) while that forward may still run. Each
run under the GPU lock (`bin/stress.sh`, `logs/stress/`, with `nvidia-smi` before and after):

| run | cycles (predictor lifetimes) | plans | result |
|---|---|---|---|
| `s1-on` | 125 (500) | on | exit 0, 57 s |
| `s1-off` | 125 (500) | off (`FOLO_GPU_NO_PLAN=1`) | exit 0, 55 s |
| `s2-on` | 1000 (4000) | on | exit 0, 437 s |
| `s2-off` | 1000 (4000) | off | exit 0, 430 s |
| the fold-in's own `gpu_trained` (`foldin/bin/new-gpu_trained.exe`, its env), 20 runs back to back | 20 runs × 4 (folo-n, folo-s × f32, f16 tensor cores) | on | 20 × exit 0, about 33 s each |

The last row is the run that crashed once in the fold-in's 30 (`logs/stress/trained-loop2/`; with the fold-in's
runs, 1 in 50). No fix, since there is nothing to fix that I can show. The fold-in's crash came once on a PC whose
CPU is suspected of degradation (rare impossible-state crashes of rustc and of `loss_cases` during this lane, too).
Reproduce: `bin/stress.sh <cycles> <tag>` (builds `gpu_stress` from `scripts/`).

## Leftovers
- The sparse box branch's fixed cost, about 0.3 ms (2% of row 8 on folo-n): conv0's two-pass Winograd over a few
  tiles and the full-map outputs it allocates. Measured, not attacked beyond the serial try above.
- The cache still churns a workload whose cycle is longer than 16 limits (16 GiB of returns at the default limit),
  since it forgets a length's record after that long. Nothing in the repo runs one; a larger `FORGET_LIMITS` costs
  only records.
- A per-length cache does not help shapes that vary per call (the `_at` passes' gathered columns, Winograd's `V` per
  block count). They are small and bypass or barely touch the cache today.
- The fold-in's entries that this resolves (docs/BENCHMARKS.md: row 8's bullet in "Hill-climb 3", "Rows worse than
  e5a6ccf0", "Next optimisations"; CHANGELOG.md; docs/ROADMAP.md) were updated by the lead in #264 after #262 merged,
  with this notebook's numbers. Row 8's cell keeps the fold-in's 60-call number.
