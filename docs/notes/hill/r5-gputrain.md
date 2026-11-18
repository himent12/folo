# Hill-climb 5: GPU training glue, backward GEMM shapes, BatchNorm in the conv (lab notebook)

Lane `gputrain` of hill-climb 5 (2026-11-16). Brief: `D:/folo-build/hill5/briefs/gputrain.md`. Target: the folo-n
b16 training step of `docs/BENCHMARKS.md` row 9 (AMP 599 → ≥ 700 img/s, f32 369 → ≥ 420) without touching
inference numerics. Three proposals, run in the order 1, 3, 2, each behind a ceiling gate measured first:

- proposal 1, training glue: gradient fan-in, `Tensor::cat`'s zero fill, the training concat;
- proposal 3, backward GEMM shapes: the stride-2 input gradient as one sub-pixel conv, 1×1 weight gradients as a
  plain GEMM;
- proposal 2, BatchNorm inside the conv GEMMs: statistics in the conv epilogue, BN backward `dx` in the
  weight-gradient GEMM.

**Status on `main`: nothing of this lane is merged.** Its one PR, #294 (branch `gpu/train-glue`, proposal 1 lite
below), was **closed, not merged**: its A/B is a tie under the round's ±3 % rule (f32 +2.4 %, AMP +1.9 %), for 576
added lines (about 190 of them code in folo-core's autograd, backend and tensor and in folo-gpu's backend and
kernels, the rest tests and docs) whose reviews found two real races (both fixed, the last round clean). Proposal 2's branch `gpu/bn-epilogue` got
no PR. Both branches stay on origin for reference; the gate branch `gate/ceil` was never pushed. The code, tests
and docs this notebook describes as "new" exist only on those branches.

## Method

- Machine: RTX 4080 SUPER (Vulkan), shared with the other lanes; every measured run under
  `D:/folo-build/hill/lock.sh` (`gpu gputrain`), none while `gpu-training.txt` existed.
- Bench: `cargo bench -p folo-vision --bench gpu` with `FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=train
  FOLO_BENCH_SCALES=n FOLO_BENCH_TRAIN=f32,amp` (folo-n, batch 16, SGD; the bench's median over its steps).
- A/B: the bench binary of each side copied to `D:/folo-build/hill5-gputrain-bin/`, runs alternated per round
  at High priority by `ab.sh` (summary by `ab.py`: per side the median of the round medians, the range, and the
  fastest single step). GPU and CPU load are logged before each run (`loads.log`). Inside ±3 % is a tie.
- Ceiling gates: a throwaway branch (`gate/ceil`, not pushed) with toggles read from `FOLO_GATE`, so one binary
  measures the step with a piece of work skipped (`nofanin`: drop the second gradient instead of accumulating;
  `noclear`: no zero fill in `uninit`; `nostats`: BatchNorm statistics from image 0 only; `nodx`: skip `bn_bwd_dx`;
  `nostep`: skip the optimizer step so the skipped work cannot blow up the weights). Numbers from such runs are
  upper bounds on what removing that work gives, not results.
- Logs: `D:/folo-data/runs-hill5/gputrain/logs/` (`ab-<tag>/` per A/B: `<side>-<round>.txt`, `loads.log`,
  `table.txt`).

## Ceiling gates

Every row compares sides of the same gate binary (toggles off against on); the load is the range of the CPU
samples in `loads.log` (the GPU also read 49–89 % busy before some runs: other lanes).

| gate | toggles | f32 img/s | AMP img/s | rounds | load | log |
|---|---|---|---|---|---|---|
| 1: glue (fan-in + zero fill) | none → `nofanin,noclear` | 379 (376–383) → 397 (391–397), +4.7 % | 605 (582–612) → 605 (559–631), 0 % | 3 | cpu 5–86 % | `ab-gate1` |
| 1 again | `nostep` → `nostep,nofanin,noclear` | 376 (370–378) → 393 (383–398), +4.5 % | 573 (421–581) → 577 (465–619), +0.7 % | 5 | cpu 53–87 % | `ab-gate2` |
| 1, AMP only | `nostep` → `nostep,nofanin,noclear` | — | 590.5 (545–613) → 591.5 (568–626), tie | 8 | cpu 56–100 % | `ab-gate3` |
| 2: BN in the conv | `nostep` → `nostep,nostats,nodx` | 376 (370–378) → 407 (402–410), +8.2 % | 573 (421–581) → 620 (434–626), +8.2 % | 5 | cpu 53–87 % | `ab-gate2` |

(`ab-gate1` also holds main's binary as `base`, 376 / 588, and a `g2` side, `nostats,nodx` without `nostep`, whose
runs diverged and gave no number: hence `nostep` from `ab-gate2` on.)

- **Gate 1 fails for AMP.** The brief asks for ≥ 3 % on both precisions. Removing all fan-in accumulation and every
  zero fill is worth +4.5 to +4.7 % in f32 but nothing in AMP, in three A/Bs (the fastest step moves 2–5 %, the
  median does not: AMP's step is bound elsewhere). So the fan-in-in-the-producer's-epilogue machinery (a hint from
  autograd to the producing kernel to accumulate instead of write) was **not built**; only its two cheap,
  bit-identical parts were proposed (below; #294, closed). Where fan-in happens (AMP, `fanin-log-amp.txt`: 24 identical steps of 40
  accumulations): per step the second gradient comes from a conv 23 times, a `narrow` (C2f's chunk) 11, `max_pool`
  3, `upsample` 2, `matmul` 1; 32 of the 40 accumulators are exclusive, 8 are not (5 `narrow`, 2 `max_pool`, 1
  `add`), so an epilogue accumulate would also need strided, shared targets.
- **Gate 2 passes** (+8 % on both): BatchNorm statistics and the BN backward `dx` pass are worth building.

## Proposal 1 lite: cat without the clear, in-place GPU accumulation (PR #294, `gpu/train-glue`, closed)

Two changes, both bit-identical to main (not merged: see the status above):

- `Tensor::uninit_with` (whose one caller is `Tensor::cat`) asked the backend for a zero-filled buffer (`full(..,
  0.0)`), so every GPU `cat` paid a `clear` it then overwrote. New `Backend::empty` (default: `full(.., 0.0)`, so the
  CPU keeps its zeros); the GPU backend returns a pooled buffer as is (except where the length ends inside a 32-bit
  word, still cleared so the spare bytes stay zero). Zeros-then-scatter (`zeros_with`) still clears, as it must.
- Gradient fan-in (`autograd::accumulate`) on the GPU was, for an exclusive F32 accumulator, `acc + g` out of place plus
  a copy back, and for F16 one out-of-place add. New `Backend::add_into(dst, layout, src)`, a strided elementwise
  `dst[i] += src[i]` (`ew::add_into`, KIND 5 of the elementwise generator), used by `accumulate` for any float dtype off
  the CPU, on its exclusive accumulator only (and not one that itself requires grad) (`Tensor::is_exclusive`: no other
  handle, view or graph shares the storage, so no other thread can hold its lock or check its version). The GPU declines
  (`NotSupported`) other dtypes and destinations that overlap themselves or share the source's buffer, and `accumulate`
  then adds out of place. Locks as every in-place op takes them: the source's read, then the destination's write, the
  version bumped once the backend has written (a decline leaves it to the fallback). Review history: a first version
  also routed the GPU `add_assign` through it; on shared tensors that raised lock-order (against `copy_from`) and
  version-timing questions over three rounds, and training never calls GPU `add_assign` (SGD and gradient accumulation
  use the fused optimizer kernels), so `add_assign` is main's again.

Tests (on the branch, `crates/folo-gpu/src/tests/backend/accumulate.rs`): the kernel bit for bit against the CPU (f32 and f16, the
whole destination buffer) over dense views (vec4 and scalar, also at an offset), channel slices, a transposed and a
broadcast source; fan-in gradients against the CPU in F32 and F16 (vec4 and odd shapes), and bit for bit against the
GPU's own out-of-place add, each asserting that the fan-in ran the in-place kernel (a per-thread count of the backend's
successful `add_into` calls, test builds only); `cat` into a NaN-filled buffer (whole-word and half-word f16 runs). F16
cases are skipped where the device does not compute in f16. Mutation checks: KIND 5 as a copy (`Y = X0`) and a linear
destination that ignores its offset fail the kernel test, one extra rounding (`· 1.0000001`) fails the bit-for-bit
fan-in test, a GPU `add_into` that always declines fails both fan-in tests, a GPU `cat` that skips its one-channel part
fails the `cat` test. The `cat` test runs with `poison_uninit` on (every result buffer NaN-filled), so it does not
depend on which recycled buffer the pool hands out; it does not detect a restored clear (a leftover).

A/B (`ab-glue1`, 5 rounds, base = main cf65f7a5, PR side = the WIP commit 8aad6628, which differs from the PR in
the precompile bounds and the review fixes; it also sent GPU `add_assign` through `add_into`, which a training step
never calls, so none of that touches a steady-state step. CPU 69–100 %: base ran at 96–100 % in rounds 3–5 and
had its two slowest AMP rounds there (545, 537); the PR side ran at 100 % in rounds 3–4 and at 69 % in round 5, and
its rounds 4–5 were its fastest (592, 593), so the load fell more on base's side):

| | base | PR | change |
|---|---|---|---|
| f32 img/s | 371 (365–378) | 380 (375–386) | +2.4 % (tie band; faster in 4 of 5 rounds) |
| AMP img/s | 567 (537–571) | 578 (543–593) | +1.9 % (tie) |
| f32 fastest step | 41.85 ms | 40.91 ms | −2.2 % |
| AMP fastest step | 26.59 ms | 25.78 ms | −3.0 % |

## Proposal 3 gates (measured before proposal 2's build)

`gate3-bwd.txt`, `gate3b-bwd.txt` (throwaway `bwd_gate` bench, folo-n training shapes, batch 16, µs per call):

- **Stride-2 input gradient as one sub-pixel conv** (a k2 s1 conv `C_out → 4·C_in` over `dy`, then a pixel
  shuffle, against today's four phase convs). The named gate (64→64 @160 on tensor cores ≤ 450 µs) **fails**: 594
  µs against 681 µs for the phases. On tensor cores the sub-pixel form wins every shape (1.14–2.21×; the six shapes
  sum to 2048 against 2766 µs), in f32 it wins most on 16→32 @320 (953 → 385 µs).
- **1×1 weight gradient as a plain `A·Bᵀ` GEMM** over `K = N·H·W`: 2.6–62× slower without a split-K (one workgroup
  walks 400k columns). A batched GEMM per image and K chunk (partials not reduced in the timing) beats the current
  kernel on tensor cores (256→64 @80: 292 → 81 µs; 48→64 @160: 358 → 270; 192→128 @40: 71 → 40) but barely in f32
  (except 384→128 @40: 278 → 123). Possible tensor-core saving about 0.4 ms per step, before the reduction it needs.

Decision: neither is built. The brief's rule is to skip an item whose gate fails, and both named gates fail: the
sub-pixel gradient misses its 450 µs bar on the gate shape (594), and the plain `A·Bᵀ` GEMM with a K-chunk floor
is the slow form above. What they would be worth instead is a leftover (below): the sub-pixel form about 0.7 ms of
the six stride-2 input gradients on tensor cores, the batched-chunk weight gradient about 0.4 ms before its
reduction, together about 4 % of an AMP step if nothing else moved.

## Proposal 2: BatchNorm statistics in the tensor-core conv epilogue (built, not shipped)

Branch `gpu/bn-epilogue` (pushed for reference, no PR; stacked on #294's commits). A train-mode `ConvBnAct` off
the CPU runs its bias-free conv through a new `Backend::conv2d_bn_stats` hook. On tensor cores (mixed precision,
unsplit `K`) the coop GEMM's epilogue folds each lane's 8 f32 results per fragment row into a Welford partial
(exact mean and `M2` of the 8, then Chan's merge), merges the lanes of a row through `Cs`, and writes one
`(count, mean, M2)` per (channel, image, column tile) in the layout `bn_finalize` already merges. The BatchNorm then
skips its `bn_stats` pass (`norm::batch_norm_forward_partials`); a `BnStats` handle is used only while it still
describes the input (same storage and version, dims, dense from offset 0). Statistics come from the unrounded f32
results (before the f16 storage rounding the separate pass saw) and in another order, so AMP training numerics
change (not inference). Tests: the partials against f64 statistics
of the written output over six shapes (masked columns, rows past `M`, several column tiles, pointwise and
windowed, strided); stale statistics ignored; f32 and split-K convs produce none. Mutations (no column mask, one
lane of the pair, `fits` always true) fail them.

Sub-gate (`bn-gate2-f16-*.txt`, `bn-gate2-f32-1.txt`: throwaway `bn_gate` bench over the 26 conv shapes of a folo-n
forward with their multiplicities, batch 16; the bench source is kept as `logs/bn_gate.rs`):

| path | epilogue cost / `bn_stats` pass it replaces | conv + BatchNorm per forward |
|---|---|---|
| f16, tensor cores | 38 % (gate: under 50 %, passes) | 4120 → 3850 µs (−270, twice: 4125 → 3844) |
| f32, regular kernels | 67 % (fails) | 8087 → 8060 µs (tie) |

(The ratio is the sum over the shapes, weighted by their count per forward, of the `epilogue` column over the `stats
pass` column of the bench's second table: 38.4 % and 38.3 % in the two f16 runs, 67 % in f32.) So only the tensor-core
path was kept; the regular-kernel epilogue (also built) is archived as `logs/bn-stats-regular-and-coop.diff`.

A/B, AMP, against #294's binary (`glue1`) so only the statistics differ:

| log | rounds | glue1 img/s | bn1 img/s | fastest step | load |
|---|---|---|---|---|---|
| `ab-bn2` | 5 | 605 (494–625) | 588 (452–598) | 25.18 → 25.01 ms | cpu 5–62 % |
| `ab-bn3` | 9 | 580 (566–624) | 612 (593–619) | 25.21 → 25.03 ms | cpu 6–100 % (round 1 under one of my builds) |
| pooled | 14 | 598 | 601.5 (+0.6 %) | −0.7 % | |

(`ab-bn1`, bn1 against main, 5 rounds, cpu 34–100 %: AMP 548 → 568, f32 363 → 372, i.e. #294 and the epilogue together.) The
medians swing ±5 % between runs (`glue1` is bimodal, 566–580 or 604–624 per round); the fastest step moves −0.7 %
in both (25.18 → 25.01 and 25.21 → 25.03 ms, 170 and 180 µs; the sub-gate's 270 µs per forward would be about
1 % of a 26 ms step). **A tie: not shipped.** A summation-order change to AMP
training, and the proxy-P1 pair it needs (about 17 GPU minutes), are not worth a 1 % that the A/B cannot resolve;
the proxy pair was therefore not run.

Where the ceiling's 8 % went: `profile-base-amp.txt` (main, AMP, `FOLO_GPU_PROFILE=1`, GPU time per step) has
`bn_stats` 1.43 ms, `bn_apply` 1.00, `bn_finalize` 0.57 in the forward and `bn_bwd_reduce` 2.17, `bn_bwd_dx` 1.37,
`bn_bwd_finalize` 0.49 in the backward. The gate removed `bn_stats` and `bn_bwd_dx` outright; the epilogue keeps 38 %
of the statistics' cost, and the merge stays in `bn_finalize`.

**BN backward `dx` inside the weight-gradient GEMM: skipped.** The proposal's form computes `dx` in the weight
gradient's `A` prologue from `dy` and `z` and lets column tile 0 store it for the input gradient. Each weight-gradient
GEMM with more than one column tile then reads two tensors per element where it read one, so the saving is the
`bn_bwd_dx` pass minus that extra traffic: about a third to two thirds of 1.37 ms, 0.45–0.9 ms or 1.7–3.4 % of an
AMP step, at or under the tie band, for a `ConvBnAct` that would have to become one custom op with a hand-written
backward (today the conv and the BatchNorm are separate autograd nodes). With 2a measured at about 1 % the two
together would not reach the brief's targets either.

## Results against the targets

Row 9 targets: AMP 599 → ≥ 700, f32 369 → ≥ 420. Proposed: PR #294 only, a tie on both (f32 +2.4 %, AMP +1.9 %,
fastest step −2 % / −3 %), and it was closed, not merged. Not reached. The gates say why: the glue that proposal 1
removes is worth 0 to +0.7 % in AMP at its ceiling (three A/Bs, inside the tie band), BatchNorm statistics in the epilogue keep 38 % of their cost, and the
step's GPU time is dominated by the coop GEMMs: in `profile-base-amp.txt` (main, AMP) the backward's input- and
weight-gradient GEMMs take 12.7 ms (6.46 + 6.22) of the backward's 22.5 ms, in a step of about 33.9 GPU-ms
(forward 9.75, loss 0.32, backward 22.49, optimizer 1.29).

## Leftovers, by expected gain

1. Sub-pixel stride-2 input gradient on tensor cores (about 0.7 ms per AMP step, 2.6 %; `gate3-bwd.txt`), despite
   the named gate.
2. Batched-chunk 1×1 weight gradient on tensor cores (about 0.4 ms before its reduction, 1.5 %).
3. BatchNorm statistics in the epilogue (`gpu/bn-epilogue`, about 1 %), worth shipping only together with 2b or
   item 1, with the proxy-P1 pair the brief asks for.
4. BN backward `dx` in the weight-gradient GEMM (0.45–0.9 ms), after `ConvBnAct` becomes one autograd op.
5. Tests on #294's branch, should it be picked up again: `cat_overwrites_a_recycled_buffer` does not detect a restored clear (it checks the values only); nothing
   tests the declines for a self-overlapping destination or for f16 without `shader-f16`. `ew::add_into` repeats
   about 25 lines of `elementwise`'s parameter and coalescing code, which a shared helper could remove.
