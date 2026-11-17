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

| gate | toggles | f32 img/s | AMP img/s | rounds | load | log |
|---|---|---|---|---|---|---|
| 1: glue (fan-in + zero fill) | `nofanin,noclear` vs none | 376 (371–377) → 397 (391–397), +5.6 % | 588 (583–598) → 605 (559–631), +2.9 % | 3 | cpu 16–94 % | `ab-gate1` |
| 1, AMP again | `nostep,nofanin,noclear` vs `nostep` | — | 590.5 (545–613) → 591.5 (568–626), tie | 8 | cpu 85–100 % | `ab-gate3` |
| 2: BN in the conv | `nostep,nostats,nodx` vs `nostep` | 376 (370–378) → 407 (402–410), +8.2 % | 573 (421–581) → 620 (434–626), +8.2 % | 5 | cpu 54–80 % | `ab-gate2` |

- **Gate 1 fails for AMP.** The brief asks for ≥ 3 % on both precisions. Removing all fan-in accumulation and every
  zero fill is worth +5.6 % in f32 but a tie in AMP over 8 rounds (the fastest step moves 3–5 %, the median does
  not: AMP's step is bound elsewhere). So the fan-in-in-the-producer's-epilogue machinery (a hint from autograd to
  the producing kernel to accumulate instead of write) was **not built**; only its two cheap, bit-identical parts
  ship (below). Where fan-in happens (AMP, `fanin-log-amp.txt`, one step): the second gradient comes from a conv
  about 32 times, a `narrow` (C2f's chunk) 16, `upsample` 4, `max_pool` 3, `matmul` 2; some accumulators are
  non-exclusive `narrow` views out of the concat backward, so an epilogue accumulate would need a strided target.
- **Gate 2 passes** (+8 % on both): BatchNorm statistics and the BN backward `dx` pass are worth building.

## Proposal 1 lite: cat without the clear, in-place GPU accumulation (PR `gpu/train-glue`)

Two changes, both bit-identical to main:

- `Tensor::uninit_with` asked the backend for a zero-filled buffer (`full(.., 0.0)`), so every GPU `cat`,
  `zeros`-then-scatter and kernel output that was about to be overwritten paid a `clear`. New `Backend::empty`
  (default: `full(.., 0.0)`, so the CPU keeps its zeros); the GPU backend returns a pooled buffer as is.
- Gradient fan-in (`autograd::accumulate`) on the GPU was `acc = acc + g` out of place plus a copy back. New
  `Backend::add_into(dst, layout, src)`, a strided elementwise `dst[i] += src[i]` (`ew::add_into`, KIND 5 of the
  elementwise generator), used by `accumulate` for any float dtype off the CPU and by `add_assign` there; it refuses
  overlapping or aliased destinations, so the out-of-place path stays the fallback.

Tests (`crates/folo-gpu/src/tests/backend/accumulate.rs`): in-place add bitwise against the CPU over dense, odd,
channel-slice, transposed and broadcast sources; fan-in gradients against the CPU in F32 and F16; `cat` into a
recycled buffer that was filled with NaN. Mutation check: making KIND 5 a copy (`Y = X0`) fails two of them.

A/B (`ab-glue1`, 5 rounds, base = main cf65f7a5, cpu 69–73 %):

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
  sum to 2049 against 2766 µs), in f32 it wins most on 16→32 @320 (953 → 385 µs).
- **1×1 weight gradient as a plain `A·Bᵀ` GEMM** over `K = N·H·W`: 3–50× slower without a split-K (one workgroup
  walks 400k columns). A batched GEMM per image and K chunk (partials not reduced in the timing) beats the current
  kernel on tensor cores (256→64 @80: 292 → 81 µs; 48→64 @160: 358 → 270; 192→128 @40: 71 → 40) but barely in f32
  (except 384→128 @40: 278 → 123). Possible tensor-core saving about 0.4 ms per step, before the reduction it needs.
