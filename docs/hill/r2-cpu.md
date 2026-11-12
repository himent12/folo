# Hill-climb 2: FOLO on the CPU (lane `cpu`, 2026-11-11)

Goal: CPU batch-1 latency, the headline row round 1 ([cpu.md](cpu.md)) did not move (folo-n forward 18.3 ms,
end to end 18.6 ms at 28 threads; target 12 ms), then per-layer outliers and the CPU training leftovers.
Baseline: `main` at 647b981f. Machine: i7-14700KF (8 P + 12 E cores, 28 threads, AVX2 + FMA), shared with five
other lanes' builds, benches and training runs.

## Method

- Baseline binaries are the bench binaries of this branch built before any change (the branch started at
  647b981f), copied to `D:/folo-build/hill2-cpu-bin/base-<bench>.exe`; each change's binaries are copied next to
  them (`<label>-<bench>.exe`). Square-sgemm baseline: `D:/folo-wt/hill2-base` (647b981f).
- `tools/bench-ab.ps1` alternates the binaries run by run at High priority, inside the shared `cpu` lock
  (`D:/folo-build/hill/lock.sh acquire cpu cpu`), 3 rounds unless stated. A copy of it (`ab2.ps1`, not committed)
  takes per-run environment variables (`label=exe|K=V,K=V`), for sweeping knobs inside one binary.
- Benches: `cargo bench -p folo-vision --bench cpu_e2e` (`FOLO_BENCH_ITERS=60`; `FOLO_BENCH_SCALE=s|m`;
  `FOLO_BENCH_BATCH=16` for `predict_batch`), `--bench train_step` (folo-n, 320 px, batch 16),
  `cargo bench -p folo-core --bench gemm`. Rows below are medians of the rounds' medians, ranges in brackets.
- "busy" is the CPU time of every other process over the run's wall time, in cores. **The machine was never
  quiet: 8–20 cores busy elsewhere in every run**, 11–16 in most. The 8-pinned-P-core rows (`-Threads 8
  -Affinity 5555`) were the noisiest (17–20 busy), so the claims use 28-thread rows.

### Profiling

samply records nothing on this machine, so, as in round 1, an op timer was patched in for the profile runs and
never committed: a guard on the CPU backend methods (conv label, flops), sub-phase guards inside the conv forward
(shifted-rows copy, A packing, the tile loop), and a per-task record (thread index, start, end) inside every GEMM
tile loop, printed by a temporary `FOLO_PROF=1` section of `cpu_e2e`. Findings on `main`, folo-n, 640², batch 1:

- **28 threads, 16 cores busy elsewhere, forward 17.2 ms.** Before the head (13.4 ms of it): GEMM tile loops 8.6
  ms, shifted-rows copies 1.3, **A packing 1.1** (a serial pass on the calling thread before every conv GEMM:
  `pack_lhs_panels` goes parallel only from 128K floats, and conv weights are 2K–295K floats), small ops 1.6
  (3 max pools 0.40, the PSA attention's two batched matmuls 0.45, softmax 0.17, `mul_scalar` 0.14, adds 0.42).
  The head ran 2.7 ms, all six branches side by side, the P3 box branch (the two largest convs of the model,
  64→64 3×3 at 80², 0.47 GFLOP each) its critical path.
- **Per-task records.** In a typical conv region about 16 workers join within 1–5 µs (spinning) and the rest
  20–130 µs late (woken from sleep); busy / (wall × workers used) over all regions was 0.48. Summed over the
  backbone and neck, task time was 52.8 ms at 1 thread, 68.6 ms at 8 pinned threads and 113.5 ms at 28 threads:
  a task runs about 2.15× slower at 28 threads, which matches hyper-threads sharing a P-core and E-cores at about
  half a P-core's AVX2 rate, not cache misses or tiling.
- **Thread count per layer does not help under this load** (sweeps below): fewer tasks for small layers, larger
  tasks, more tasks per thread, a smaller pool, and running the neck's tail beside the head were all ties or
  losses. With 11–16 other cores busy the forward is bound by the CPU time it needs, not by fork-join waits:
  only less work per forward moved it.

## 1. Conv forward without serial packing (PR 1)

Three changes, all bit for bit the old results:

- **A read in place.** A row-major A (`Operand::rows_in_place`; conv weights are `Strided` with unit K stride)
  whose packed form fits one K super-block (`PACK_BUDGET`, 16 MiB) is no longer packed: two new micro-kernels
  broadcast the six A values of each K step from six row pointers (`rows_a_kernel` for the shifted-rows B,
  `arows_kernel` with an epilogue for packed B and for the ragged last column panel). Same FMAs in the same
  order, same `KC` block boundaries, rows past `mr` repeat row 0 and are never stored. Removes the serial A pass
  from every conv forward and pointwise conv, and the packed copy (peak memory of the train step −30 MB).
  Larger A (a 2048² sgemm) keeps the packed path.
- **Max pooling without indices** when nothing is recorded (`no_grad`, or an input that does not require grad):
  `Backend::max_pool2d_values` (default: `max_pool2d` and drop the indices), on the CPU two passes of a select
  over `-inf`-padded rows (same `beats` rule; `-inf` padding never changes a value, see the doc comment), planes
  grouped into ~4K-output tasks. The indexed version allocated a fresh 410 KB `i64` buffer per SPPF pool (page
  faults) and wrote indices nobody read.
- **Batched `matmul` splits the pool between its matrices** (`threads / batch` each) instead of running small
  ones (< 16 MFLOP) one thread each: the PSA attention's two heads were two single-thread 10 MFLOP GEMMs
  (0.25 → 0.05 ms and 0.20 → 0.09 ms).

Tests: `a_in_place_is_bitwise_the_packed_path` (gemm_ex and gemm_rows, row-major vs column-major A, ragged
blocks, K over one block, the K split, 1 and 8 threads, with and without bias + SiLU),
`a_rows_kernels_are_bitwise_the_packed_kernel` (portable and AVX2 kernels, ragged rows, epilogue),
`max_pool_values_are_the_indexed_values_bit_for_bit` (±0 ties, NaN payloads, `-inf` windows, planes smaller
than the window, strides 1–3, paddings).

| A/B, 28 threads, 3 rounds alternated | `main` | PR 1 | |
|---|---|---|---|
| folo-n forward b1 (busy 9–13) | 16.37 ms (16.19–16.68) | **14.55 ms** (13.99–14.75) | −11% |
| folo-n e2e 640×640 | 16.78 ms (16.58–17.06) | **14.79 ms** (14.27–14.99) | −12% |
| folo-n e2e 1280×720 | 12.43 ms (12.36–13.08) | **11.10 ms** (10.60–11.25) | −11% |
| folo-s forward b1 (busy 11–12) | 36.34 ms (36.21–36.85) | **32.23 ms** (32.05–32.56) | −11% |
| folo-s e2e 640×640 | 35.71 ms (35.59–36.23) | **32.21 ms** (31.97–32.37) | −10% |
| folo-m forward b1 (busy 8–10) | 87.58 ms (85.96–88.17) | **79.31 ms** (78.18–80.70) | −9.4% |
| folo-m e2e 640×640 | 85.81 ms (84.30–88.17) | **79.63 ms** (78.59–80.19) | −7.2% |
| folo-n `predict_batch` 16 (busy 11–12) | 153.9 ms (150.6–156.9) | 153.5 ms (150.9–154.5) | tie |
| folo-n train step 320 px b16 (busy 11–15) | 263.0 / 263.7 / 256.6 ms | 256.0 / 255.8 / 256.2 ms | tie (−2%) |
| folo-n forward b1, 8 pinned P-cores (busy 17–20) | 20.49 ms (19.30–22.96) | 16.18 ms (15.27–18.96) | noisy |

Square sgemm (`--bench gemm`, best of 1.5 s, 3 rounds at 28 threads, 2 at 1 thread on one P-core): 28 threads
256³ 431–436 → 689–723 GFLOP/s, 512³ 883–917 → 1090–1102, 1024³ 1208–1217 → 1248–1304, 2048³ (packed, unchanged
path) 1174–1197 → 1170–1181. One thread: 256³ 116–119 → 121–127, 512³ 102–104 → 108–111, 1024³ and 2048³ within
the noise of a loaded core (2048³ 79–80 vs 64–73 when it was read in place too, which is why A over one
super-block keeps packing).

Steps on the way (forward b1, 28 threads, same session each): in-place A alone 16.98 → 15.47 ms; the first
values-only max pool (a scalar flag scan, planes in 8K-output tasks) made the pools slower (3 × 0.24 ms against
3 × 0.13), the padded select version 15.45 → 15.01 together with the matmul split.

## Negative results

All in one binary with temporary environment knobs, alternated, folo-n forward b1 at 28 threads (11–17 busy):

| knob | forward b1 |
|---|---|
| `main` tiling (3 tasks per thread, ≥ 1 MFLOP each) | 16.84 / 17.00 ms |
| tasks ≥ 2 / 4 MFLOP | 17.79 / 21.41 |
| tasks ≥ 0.5 / 0.25 MFLOP | 16.68 / 17.01 |
| 6 tasks per thread (and ≥ 0.5 MFLOP) | 17.05 (16.90) |
| conv threads capped at flops / (2, 4, 8 MFLOP) | 17.11 / 18.03 / 19.59 |

| pool size (`RAYON_NUM_THREADS`, same binary) | 28 | 24 | 20 | 16 | 12 |
|---|---|---|---|---|---|
| forward b1 | **17.02** | 18.01 | 18.63 | 19.69 | 23.55 |

- **The neck's tail beside the head**: on CPU inference, start each level's head branches as soon as the neck
  has made its feature (`rayon::join` of the P3 branches with layers 17–22, then P4 with 20–22): 14.71 vs 14.68
  ms, a tie. The overlap happens, but the neck's convs, the critical path, then share the pool (down3 0.28 → 1.08
  ms). Giving the P3 / P4 branches a thread budget (a thread-local cap on the tiling of the kernels they start:
  P3 box 2–12, cls 1–4) made it worse: 15.34–17.85 ms against 14.86. Patch kept in the lane's bin folder.
- **Shifted-rows copy in parallel from 8K / 16K / 32K floats** instead of 64K: 15.50 / 14.94 / 15.81 ms against
  15.01, a tie.
- Not tried: pinning workers to P-cores first. Every pool smaller than 28 threads lost under this load, and a
  pinned 8-thread pool was the noisiest configuration measured.

## Leftovers (by expected gain)

- **3×3 stride-1 convs** are 37% of the forward's flops (2.39 of 6.48 GFLOP); Winograd F(2×2, 3×3) would cut
  their multiplies 2.25× (not bit for bit; within an f32 tolerance).
- **Stride-2 convs** run at 41–73 GF/s on one core against 76–88 for stride 1 (16→32 at 320², 64→64 at 160²):
  their phase copies leave L2 and `m = 32` wastes a third of the last micro-panel.
- The shifted-rows copy (1.3 ms of the 28-thread forward) could be written by the conv that produces the input
  (a padded output layout), the head's depthwise convs scale poorly (0.26 ms at 28 threads for 0.5 ms at one),
  and softmax over the PSA's 800 rows of 400 is 0.17 ms.
- On a quiet machine the scheduling questions (fewer threads for small layers, P-cores first, the head beside
  the neck) may come out differently; under the load of this session none won.
