# Hill-climb: GPU inference (branch `hill/gpu-infer`)

Goal: faster GPU inference for folo-n / folo-s at 640 px against the Ultralytics YOLO26 bars on the same RTX 4080
SUPER (docs/BENCHMARKS.md, FOLO targets). Lane: folo-gpu forward kernels, device, pool, upload / readback, the GPU
path of the predictor, inference benches.

## Method

- **Bench**: `FOLO_GPU_TENSOR_CORES=1 FOLO_BENCH_SECTIONS=forward,predict FOLO_BENCH_PRECISION=f32,f16tc cargo bench
  -p folo-vision --bench gpu` (the same binary file, built from each tree, copied aside). The rows below are
  medians (and min–max) over the rounds, each round one full bench run per binary, binaries alternated.
- **Where the time goes**: `cargo bench -p folo-vision --bench gpu_profile` (added by the first commit on this
  branch) records one fused forward and replays every dispatch on its own; the replay sums are GPU-bound and much
  steadier than wall-clock rows on this machine, so per-kernel claims come from there.
- **Noise.** The GPU was taken with `D:/folo-build/hill/lock.sh` for every reported run, and the A/B script waits
  until `nvidia-smi` reads under 10% before each run. Even so, other agents' jobs started on the GPU inside some
  lock windows (45–89% load seen mid-run) and the CPU sat at 60–90% from builds and training data loaders, which
  makes host-bound rows (batch 1, encode) swing by 2x. Rounds with visible load were rerun; the final table below
  is from a window with the GPU under 6% before every run.

## Final: branch head against `main` (c50211ba)

3 rounds alternated, GPU lock held, GPU 0–6% before every run, 2026-11-09. `forward` = fused model, input on the
GPU, synchronised latency (b1) and pipelined throughput (b16); `e2e` = the predictor on decoded `u8` 640×640
images, boxes on the host; `upload` = the bench's split of upload + letterbox.

| row | main | branch | | bar |
|---|---|---|---|---|
| folo-n f32 forward b1 | 3.01 ms (2.82–3.29) | **2.22 ms** (2.09–2.28) | −26% | 4.95 ms |
| folo-n f32 forward b16 | 1412 img/s (1215–1493) | **1660 img/s** (1348–1673) | +18% | 1454 |
| folo-n f32 e2e b1 | 4.53 ms (4.16–5.65) | **2.73 ms** (2.71–3.94) | −40% | 6.92 ms |
| folo-n f32 e2e b16 | 683 img/s (615–692) | **1207 img/s** (1193–1228) | +77% | 615 / 536 |
| folo-n f32 upload b16 | 9.36 ms | **2.64 ms** | −72% | |
| folo-n f16 TC forward b1 | 2.49 ms (2.45–4.33) | 2.41 ms (1.84–2.80) | tie | 6.25 ms |
| folo-n f16 TC forward b16 | 2780 img/s (2152–2827) | **3091 img/s** (3052–3140) | +11% | 2370 |
| folo-n f16 TC e2e b16 | 720 img/s (621–873) | **1788 img/s** (1771–1902) | +148% | 615 |
| folo-s f32 forward b1 | 6.55 ms (5.21–6.95) | **4.08 ms** (4.05–4.13) | −38% | 4.87 ms |
| folo-s f32 forward b16 | 632 img/s (474–640) | 654 img/s (643–661) | tie | 627 |
| folo-s f32 e2e b16 | 413 img/s (205–425) | **565 img/s** (556–573) | +37% | 446 (fp16) |
| folo-s f16 TC forward b1 | 3.48 ms (3.47–3.79) | **2.62 ms** (2.39–3.92) | −25% | 7.47 ms |
| folo-s f16 TC forward b16 | 1186 img/s (1171–1192) | 1236 img/s (1233–1251) | tie | 1159 |
| folo-s f16 TC e2e b16 | 634 img/s (616–670) | **943 img/s** (937–953) | +49% | 446 |

Every target row of the brief is met in this window: folo-n f16 TC forward b16 3091 > 2370, b1 2.41 < 6.25 ms,
e2e b16 1788 > 615; folo-s f32 forward b1 4.08 < 4.87 ms, e2e b16 565 (f32) / 943 (f16 TC) > 446, f16 TC forward
b16 1236 > 1159; folo-n f32 forward b16 1660 > 1454.

Note that `main`'s f16 TC forward b16 here (2780 img/s) is already over the bar: the 1264 img/s in BENCHMARKS.md
was measured under 68–85% load, never on a quiet GPU. The batch-1 rows of both trees are bimodal on this machine
(medians of rounds 2.3–6.3 ms for the same binary): host scheduling, see Leftovers.

**Accuracy** (`folo val --model D:/folo-weights/folo-n-coco.folo --data D:/folo-data/coco`, val2017, 5000 images,
branch build): f32 mAP50 0.44945 / mAP50-95 0.30999; `--half` 0.44940 / 0.31008; `--half --tensor-cores` 0.44944 /
0.31003. Largest gap 0.00009 (bound 0.002).

## Commits

### 1. `gpu: submit the head of a pass early, add a dispatch trace and the gpu_profile bench` (48f3d0a1)

From the previous agent on this branch; kept. After a wait the open batch is submitted at 16 dispatches, the limit
doubling per submission up to 1024. Its commit message has its A/B (folo-n f32 b1 4.02 → 2.30 ms). Its traced op
enum failed `clippy -D warnings` (`large_enum_variant`); commit 2 boxes the traced `Params`.

### 2. `gpu: upload predict images in parallel through reused mapped staging memory` (53737b1c)

Profile: e2e b16 spent 9–12 ms of a 16–22 ms batch in "upload + letterbox" on the host: each image cloned
(fresh pages), uploaded with `Queue::write_buffer` (a fresh wgpu staging allocation and a serial memcpy) and a
flush per image.

Now `GpuContext::upload_with(len, fill)` writes into a mapped staging buffer that is mapped again after the GPU
copy and reused (the last four per context); the predictor writes all images of a batch into one upload, 64 KiB
chunks in parallel (rayon), and the letterbox kernel reads each image in place at its byte offset.

| A/B vs 48f3d0a1 (3 rounds) | before | after |
|---|---|---|
| upload + letterbox b16, folo-n f32 | 12.0 ms | 3.4 ms |
| upload + letterbox b16, folo-s f32 | 12.6 ms | 3.1 ms |
| upload + letterbox b1 | 1.3–1.8 ms | 0.6–0.7 ms |
| e2e b16 folo-n f32 | 677 img/s | 997 img/s |
| e2e b16 folo-s f32 | 338 img/s | 539 img/s |
| e2e b16 folo-s f16 TC | 577 img/s | 893 img/s |

Tests: `letterbox_on_device_is_exact` runs its cases twice (the second pass reuses the mapped staging buffers;
images of odd byte lengths share one upload), `upload_with_reuses_mapped_staging` in folo-gpu.

### 3. `gpu: depthwise conv computes four adjacent outputs per invocation, window unrolled` (4c4e879c)

Profile: the 7 depthwise convs were 8% of the folo-n f16 TC forward at b16 (0.43 ms; f32 0.27 ms). The f16 kernel
was 1.6x slower than the f32 one.

The generator now unrolls the window and computes four adjacent outputs per invocation (input columns shared
between them load once, index math amortised, no WGSL loops, accumulators indexed by constants).

| A/B (3 rounds alternated) | before | after |
|---|---|---|
| `bench conv` depthwise 3×3 s1 8×128×80×80, f16 | 0.127–0.130 ms | **0.044–0.051 ms** |
| same, f32 | 0.070–0.080 ms | 0.063–0.068 ms (tie) |
| folo-n f16 TC b16, the 7 depthwise convs (replay) | 0.43 ms | 0.14 ms |
| folo-s f16 TC b16, the 7 depthwise convs (replay) | 0.78 ms | 0.24–0.28 ms |

About 5% of the f16 TC forward: inside the noise of the headline rows, so no headline claim from this commit alone.

### 4. `gpu: narrow 32- and 16-row GEMM tiles for forward convs with few output channels` (6f99d795)

Profile (folo-n f32 b16): the stem (M = 16, 64-row tile: 75% of its rows empty), 16 → 32 at stride 2 and the
16-channel 3×3 at 160 px were 1.2 ms of 7.6 ms of GEMM. New configs `ROWS32` (32×128×32) and `ROWS16` (16×128×32),
picked by `pick_config_conv` for the forward convs only (the backward kernels keep `pick_config`).

| A/B vs 4c4e879c (4 rounds, GPU < 2% before each) | before | after |
|---|---|---|
| stem 3 → 16, s2, b16 (replay) | 0.42 ms | 0.30 ms |
| 16 → 32, s2, b16 (replay) | 0.44 ms | 0.29 ms |
| 16-channel 3×3 at 160 px, b16 (replay) | 0.38 ms | 0.19 ms |
| folo-n f32 forward b16 | 11.14 ms (11.00–11.18), 1531 img/s | 10.41 ms (10.31–10.48), 1645 img/s |
| folo-s f32 forward b16 | 25.57 ms | 24.70 ms |

+7% on the headline row: a tie by the 10% rule, kept for the per-kernel gain.

## Negative results

- **Concat / split copies as a compute kernel** (`copy2d`: strided row copy in u32 / vec4 words, replacing the
  generic `scatter` for batched concats, the rank-2 `move` gathers, and the buffer copies at batch 1 that end the
  compute pass). GPU-side a tie with the old kernels at batch 16 (folo-n f32 1.085 vs 1.078 ms, f16 0.563 vs 0.503,
  folo-s f32 2.58 vs 2.89, f16 1.26 vs 1.17), and at batch 1 the extra 48 dispatches cost about what the 48 pass
  breaks saved (bump/submit 4.5 vs 6–7.6 µs per dispatch, 165 instead of 117 dispatches; same-binary toggle,
  inconclusive under CPU load). Dropped.
- **Where host encode goes** (temporary timers, not committed): about 8–12 µs per dispatch for f32, 11–19 µs for
  f16 TC; bind group creation 3–4 µs, recording 0.2 µs, the rest is wgpu's submission work (charged to the
  dispatch that triggers a submit). The f16 TC forward encodes slower than f32 (2.4–2.7 vs 1.4–2.0 ms per b1
  forward); not resolved.

## Leftovers (by expected gain)

1. **Fewer dispatches / cheaper dispatches at batch 1.** Batch-1 forwards are host-bound (pipelined ≈ encode time;
   GPU work alone is 0.8 ms for folo-n f16 TC). Options: cache bind groups (the pool hands out the same buffers in
   the same order every forward, so a cache keyed by pipeline + buffers + uniform slot would hit almost always:
   3–4 µs × 117), fuse the residual adds into the conv epilogue (14 `ew2`), and write conv outputs straight into
   the concat destination (the GEMM epilogue already takes `ldc` / offsets; needs a "conv into a channel slice"
   op in folo-nn's C3k2 / SPPF / neck), which removes the 42–55 copies too (7–10% of the b16 GPU time).
2. **Bimodal batch-1 latency.** The same binary gives b1 medians of 2.3 or 6.3 ms by round; probably GPU clock
   ramp after idle gaps between synchronised calls plus CPU contention. Needs a quiet machine to pin down.
3. **f32 GEMM**: 80% of the f32 forward. The 64×128 3×3 convs reach 10–19 TFLOPS of the 31 peak; stride-2 gathers
   and `CHECK` tiles (N = 20·20, 40·40) are the weak spots. Split-K or stream-K for the 13-tile layers at 40 px
   and below.
4. **Upload pipelining**: the remaining 2.6 ms of batch-16 upload + letterbox is a parallel host fill, then the
   PCIe copy (~0.8 ms), then the letterbox; chunking the upload so the copy of one chunk overlaps the fill of the
   next would save about 0.5–1 ms per batch.
