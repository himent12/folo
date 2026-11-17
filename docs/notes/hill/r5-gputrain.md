# Hill-climb 5: GPU training glue, backward GEMM shapes, BatchNorm in the conv (lab notebook)

Lane `gputrain` of hill-climb 5 (2026-11-16). Brief: `D:/folo-build/hill5/briefs/gputrain.md`. Target: the folo-n
b16 training step of `docs/BENCHMARKS.md` row 9 (AMP 599 → ≥ 700 img/s, f32 369 → ≥ 420) without touching
inference numerics. Three proposals, run in the order 1, 3, 2, each behind a ceiling gate measured first:

- proposal 1, training glue: gradient fan-in, `Tensor::cat`'s zero fill, the training concat;
- proposal 3, backward GEMM shapes: the stride-2 input gradient as one sub-pixel conv, 1×1 weight gradients as a
  plain GEMM;
- proposal 2, BatchNorm inside the conv GEMMs: statistics in the conv epilogue, BN backward `dx` in the
  weight-gradient GEMM.

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
samples in `loads.log` (the GPU also read 51–89 % busy before some runs: other lanes).

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
  bit-identical parts ship (below). Where fan-in happens (AMP, `fanin-log-amp.txt`: 24 identical steps of 40
  accumulations): per step the second gradient comes from a conv 23 times, a `narrow` (C2f's chunk) 11, `max_pool`
  3, `upsample` 2, `matmul` 1; 32 of the 40 accumulators are exclusive, 8 are not (5 `narrow`, 2 `max_pool`, 1
  `add`), so an epilogue accumulate would also need strided, shared targets.
- **Gate 2 passes** (+8 % on both): BatchNorm statistics and the BN backward `dx` pass are worth building.

## Proposal 1 lite: cat without the clear, in-place GPU accumulation (PR `gpu/train-glue`)

Two changes, both bit-identical to main:

- `Tensor::uninit_with` (whose one caller is `Tensor::cat`) asked the backend for a zero-filled buffer
  (`full(.., 0.0)`), so every GPU `cat` paid a `clear` it then overwrote. New `Backend::empty` (default:
  `full(.., 0.0)`, so the CPU keeps its zeros); the GPU backend returns a pooled buffer as is. Zeros-then-scatter
  (`zeros_with`) still clears, as it must.
- Gradient fan-in (`autograd::accumulate`) on the GPU was `acc = acc + g` out of place plus a copy back. New
  `Backend::add_into(dst, layout, src)`, a strided elementwise `dst[i] += src[i]` (`ew::add_into`, KIND 5 of the
  elementwise generator), used by `accumulate` for any float dtype off the CPU and by `add_assign` there. The GPU
  declines (`NotSupported`) other dtypes and destinations that overlap themselves or share the source's buffer, and
  the caller then adds out of place. The source is read-locked first (as in every in-place op) and the destination
  only *tried* (`StorageCell::try_write`): a busy destination also goes out of place, so the addition never waits
  for the destination while it holds the source and cannot deadlock against opposing additions or `copy_from`'s
  read-then-write (a first fix took the two locks in address order, which still crossed `copy_from`). As with
  `StorageCell::write`, the version is bumped as the lock is taken, before the data changes, so a `backward()` that
  checks it during the addition sees the modification (a decline after that bump is harmless: both callers then
  write the destination out of place, or replace an accumulator nothing else sees).

Tests (`crates/folo-gpu/src/tests/backend/accumulate.rs`): in-place add bitwise against the CPU over dense (also at an
offset, vec4 and scalar), odd, channel-slice, transposed and broadcast sources; fan-in gradients against the CPU in F32
and F16 (vec4 and odd shapes), and bit for bit against the GPU's own out-of-place add, each asserting that the fan-in
ran the in-place kernel (a per-thread count of the backend's successful `add_into` calls, test builds only); `cat` into
a recycled buffer that was filled with NaN (whole-word and half-word f16 runs); opposing additions on two threads; and
in folo-core (`ops/inplace.rs`), `add_into_` on a destination whose read guard is held returns `Ok(false)` at once,
values and version untouched (a blocking write fails it at its 30 s watchdog). Mutation checks: making KIND 5 a copy (`Y
= X0`) fails two of them, one extra rounding (`· 1.0000001`) fails the bit-for-bit one, a GPU `add_into` that always
declines fails both fan-in tests.

A/B (`ab-glue1`, 5 rounds, base = main cf65f7a5, PR side = the WIP commit 8aad6628, which differs from the PR only
in the precompile bounds and the review fixes, none of which touch a steady-state step; cpu 69–100 %, rounds 3–5
fully loaded and the slowest for AMP):

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
