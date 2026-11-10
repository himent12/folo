# Hill-climb: FOLO on the CPU (branch `hill/cpu`, 2026-11-09)

Goal: faster CPU inference and training on the dev machine (i7-14700KF, 8 P + 12 E cores, 28 threads, AVX2 +
FMA). Baseline: `main` at c50211ba.

## Method

- Two builds of the same bench sources, `main` in `D:/folo-wt/hill-cpu-base` (target `D:/folo-build/hill-cpu-base`)
  and this branch, copied to `D:/folo-build/hill-cpu-bin/<label>-<bench>.exe`.
- `tools/bench-ab.ps1 -Runs "base=...;new=..." -Rounds 3 -Threads N [-Affinity 5555] -Env "K=V;..."` alternates
  the binaries run by run at High priority and prints, per run, the cores the rest of the machine used
  ("busy").
- Timing runs held the shared CPU lock (`D:/folo-build/hill/lock.sh`). The machine was never quiet: other
  agents' builds and a training run's loaders kept **5 to 26 cores** busy. 8-thread rows pinned to the P-cores
  were often starved (the 8 pinned threads got 1.4–6 cores), so the claims below use 28-thread rows, where
  rounds agree within a few percent. 1-thread per-core numbers come from `-Affinity 4`.
- Profiling: samply records nothing on this machine, so a temporary op timer (a guard on every CPU backend
  method and every autograd backward node, `FOLO_PROF=1`) was patched in for the profile runs and never
  committed.

## Results

| headline (28 threads, medians of alternated rounds) | `main` | this branch |
|---|---|---|
| folo-n train step, 320 px, batch 16 (`cargo bench -p folo-vision --bench train_step`) | 360.6 / 362.0 / 367.5 ms (43.9 img/s) | **279.7 / 287.9 / 294.5 ms (55.6 img/s), −21%** |
| folo-n `predict_batch` 16 × 640² (`cpu_e2e`, `FOLO_BENCH_BATCH=16`) | 228.2 / 225.1 / 230.7 ms (70 img/s) | **158.8 / 162.9 / 159.2 ms (100 img/s), +43%** |
| folo-s `predict_batch` 16 | 633.1 / 620.4 ms (25.5 img/s) | **474.0 / 468.5 ms (34 img/s), +34%** |
| folo-m `predict_batch` 16 | 1810.9 / 1767.7 ms (9.0 img/s) | **1255.0 / 1229.4 ms (12.9 img/s), +43%** |
| folo-n forward, batch 1 | 17.1–17.9 ms | 17.3–17.9 ms (unchanged) |
| folo-s forward, batch 1 (bar: yolo26s 42.5 ms) | 39.3 / 39.5 ms | 40.9 / 39.3 ms (unchanged; beats the bar) |
| folo-m forward, batch 1 (bar: yolo26m 106.6 ms) | 93.3 / 88.1 ms | 92.6 / 88.7 ms (unchanged; beats the bar) |
| folo-s / folo-m predictor end to end (bars 51.1 / 107.8 ms) | 40.2 / 37.5, 91.0 / 88.1 ms | 39.0 / 38.1, 90.9 / 87.5 ms |

folo-s and folo-m had never been measured on the CPU: both already beat the yolo26 bars at batch 1 (by 8% and
17%). Busy cores elsewhere: 5–11 (train step), 13–14 (folo-n batch), 6–13 (folo-s / folo-m).

### 1. Fused BatchNorm + SiLU on the CPU (train step −12.5%)

The profile of the folo-n train step (355 ms at 28 threads): conv weight gradients 87–105 ms, conv input
gradients 60–77, forward convs 56, BatchNorm forward 21 + SiLU forward 15, BatchNorm backward 23 + SiLU
backward 20, `cat` copies 9, gradient accumulation 8. folo-nn's CPU BatchNorm (`train_cpu`) now applies the
activation in its output pass and its gradient in its backward: per plane, `g' = g · act'(z)` (z recomputed
from x) is written once with a vectorised f32 loop into the dx buffer, summed while it is in cache, and turned
into dx in place. Bit for bit BatchNorm followed by `Tensor::silu` / `Tensor::relu` (test
`fused_activation_is_bitwise_bn_then_act`).

| train step, 28 threads | round 1 | round 2 | round 3 |
|---|---|---|---|
| `main` | 374.6 ms | 380.3 | 375.4 |
| fused BN + SiLU | **330.9** | **328.8** | **328.4** |

Negative result on the way: recomputing `act'(z)` inside the f64 reduction loop (no stored g') made the
backward slower (43 → 70 ms): that loop does not vectorise.

### 2. Conv weight gradient over shifted rows (train step −10% more)

Per core the weight gradient ran at 57–72 GF/s against the forward's ~110 (21 GF/s for an 8-channel layer):
its B operand, the unfolded input, was packed element by element (a transposition with bounds checks), and
each packed element was used only `og` times. Now `dWᵀ = X_unfolded · gyᵀ`: A is the input's shifted rows (the
forward's B operand), read in place by a new 6x16 AVX2 micro-kernel that broadcasts from six row pointers;
B, the output gradient, is transposed once per image into packed panels (AVX2 8x8 transposes, zeros in the
dropped columns). Deterministic: K (images × columns) is cut into parts and C's rows into tiles by shape only
(up to 128 tasks of ≥ 0.5 MFLOP), and the partials are added in order.

1 thread on one P-core, best of a 1 s budget (`cargo bench -p folo-core --bench conv -- n`, batch 8):

| shape | `main` | branch |
|---|---|---|
| 3x3 64→64 at 40² | 13.51 ms (70 GF/s) | **8.81 ms (107)** |
| 3x3 64→64 at 10² | 1.02 (58) | **0.65 (90)** |
| 3x3 16→8 at 80² | 5.99 (20) | **2.95 (40)** |
| 1x1 48→64 at 80² | 5.57 (56) | 4.56 (69) |

At 28 threads in the train step the in-place path lost on strided convs and narrow pointwise convs (stem
3→16 s2: 2.2 → 3.0 ms; 16→32 s2 at 160²: 2.6 → 3.1; 48→64 1x1 at 80²: 1.9 → 2.3): those GEMMs do few flops
per input element, so writing the shifted-rows copy and the transposed gradient and reading them back costs
more than packing on the fly. They keep the packed path (`wgrad_in_place`: stride 1, or pointwise with ≥ 64
input channels). Weight gradients in the profile: 87–105 → 66–69 ms.

| train step, 28 threads | round 1 | round 2 | round 3 |
|---|---|---|---|
| fused BN + SiLU | 328.1 ms | 314.5 | 319.1 |
| + weight gradient | **279.7** | **287.9** | **294.5** |

### 3. CPU batches as concurrent forwards (predict_batch +34–43%)

Convs ran the images of a batch one at a time with the whole pool whenever an image's GEMM exceeded 32 MFLOP,
so a batch of 16 cost 16 batch-1 forwards (folo-s: 38 ms per image at batch 16 and at batch 1). Now convs always
split the pool between the images of a batch, and `Predictor::predict_batch` on the CPU runs chunks of two images
as concurrent forwards inside the pool, so every op, not only the convs, runs nearer one core's efficiency.
Predictions are those of each image alone (test `cpu_predict_batch_matches_single_images`). Chunks of 1, 2 and
4 images measured 165 / 159–172 / 166–169 ms at batch 16 (two picked); convs alone splitting the pool gave
200–204 ms; `main` 229–236.

## Negative results

- **The WIP spin pool** (`cpu/par.rs`, an OpenMP-style fork-join pool whose idle workers spin 1 ms before
  parking): at 28 threads it kept ~21 cores busy spinning and ran the folo-n forward in 22.9 / 26.4 / 33.5 ms
  against rayon's 17.3 / 17.4 / 17.3 (5–14 cores busy elsewhere); at 8 pinned P-cores 26.3 / 26.3 / 29.4 vs
  21.8 / 31.0 / 24.1 (19–24 cores busy): no gain inside the noise. Reverted; its bench harness is kept.
- Recomputing the SiLU gradient inside BatchNorm's f64 sums (above).
- BLIS loop order in `gemm_arows` (each B panel held in L1 while the A rows sweep past it, instead of each A
  micro-panel against every B panel): no difference measured; kept as the standard order.
- Images side by side in convs alone did not change the train step (285 vs 281–286 ms).
- Review fix (live scratch): with every image of a batch side by side, a 640 px batch of 16 held sixteen
  shifted-rows copies / unfolded gradients at once (15 MB each for the 16→32 stride-2 conv), and the buffer cache
  kept them. Convs now run only as many images at once as fit a 16 MiB scratch budget (the weight gradient in
  waves, its K parts continuing across them, bit for bit the same sums). Train step, 28 threads, 3–6 cores busy
  elsewhere, `tools/bench-ab.ps1` alternating, peak working set sampled every 100 ms:

  | folo-n train step, b16 | `main` | before the fix | 64 MiB | **16 MiB (kept)** | 8 MiB |
  |---|---|---|---|---|---|
  | 320 px, ms (5 runs) | 338–348 | 259–265 | | **248–254** | 247–254 |
  | 320 px, peak MB | 1394–1398 | 1371–1393 | 1377–1386 | **1291–1305** | 1247–1248 |
  | 640 px, ms (3 runs) | 1466–1511 | 1140–1148 | 1099–1132 | **1046–1064** | 1060–1097 |
  | 640 px, peak MB | 1662–1695 | 1936–1992 | 1791–1906 | **1623–1706** | 1625–1691 |

  (64 MiB at 320 px ran in a separate session, where it tied the binary before the fix: 64.9 vs 66.1 img/s.)
  One image at a time everywhere (a budget of one float) gave `main`'s peak (1598–1741 MB) at 9.7–10.1 img/s,
  slower than `main`: the extra memory was all scratch of concurrent images. Less scratch is also faster here
  (fewer page faults and cache misses), so the 320 px gain stands.

## Not done / leftovers

- **Batch-1 latency** (folo-n 17 ms at 28 threads, target 12): not moved. With 13–26 cores busy elsewhere the
  parallel-efficiency question (small 20×20 / 40×40 layers, P vs E cores, thread count) could not be measured;
  the spin pool above was the only scheduling experiment. Worth retrying on a quiet machine: fewer threads for
  small layers, and the head's branches side by side with fewer threads each.
- **Strided conv input gradients** (col2im): ~25 ms of the train step at ~200 GF/s; a stride-phase
  decomposition (four stride-1 sub-convs for 3x3 s2) would run them through the forward kernel.
- **1x1 input gradient outlier**: 256→64 at 40² took 11.6 ms per call in the profile (0.84 GFLOP, 72 GF/s at 28
  threads) where similar shapes run at 300+; not investigated.
- BatchNorm statistics still take two passes over x (sum, then squared deviations): a per-plane mean / M2
  combined with Chan's formula would make it one, at the cost of matching the old bits.
- The weight gradient's strided / narrow cases could build the shifted rows per K part (in L2) instead of per
  image, which would let them use the in-place kernel too.

## Files outside folo-core/src/cpu

- `crates/folo-nn/src/layers/batchnorm.rs` (CPU train-mode BatchNorm: fused activation).
- `crates/folo-vision/src/predict.rs` (CPU `predict_batch` chunks), `crates/folo-vision/tests/model.rs` (test).
- `crates/folo-core/tests/reference.rs`, `crates/folo-core/tests/determinism.rs` (new cases).
- `tools/bench-ab.ps1`, `crates/folo-vision/benches/cpu_e2e.rs`, `crates/folo-core/benches/conv_infer.rs`
  (from the WIP commit, kept).
