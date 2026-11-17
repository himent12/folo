# Hill-climb 5, lane `cpuq8`: int8 inference on the CPU

Brief: `D:/folo-build/hill5/briefs/cpuq8.md` (post-training int8 for the backbone and neck: an AP gate, a kernel
speed gate, then an opt-in path). Logs, binaries, calibrations, dumps and scripts: `D:/folo-data/runs-hill5/cpuq8/`
(`logs/`, `bin/`, `cal/`, `dumps/`, `records/`, `scripts/`); every path below is relative to it.

The design is docs/MODEL.md §9.6, and the kernels are in docs/CORE.md under "int8 convs".

## AP gate (train-dev, folo-n-coco-v4)
- Model: `folo-n-coco-v4` (`D:/folo-data/runs-hill4/weights/folo-n-r5-kd-e32-e64.folo`, sha256 d3896e3c…).
- Data: hill 3's train-dev split (5 000 train2017 images, `D:/folo-data/runs-hill3/fusion/lists/traindev.txt`).
- Calibration: 512 other train2017 images (`scripts/make_calib.py`, seed 5, none in train-dev:
  `lists/calib512.txt`). One two-pass calibration gives five range methods (`cal/n-v4/cal-*.json`).
- Scoring: CPU, batch 8, candidate dumps (`scripts/gate_dump.sh`), then bench/fusion's B0 (FOLO's default
  post-process, pycocotools-exact) and `bench/fusion/bootstrap.py` (1 000 paired image resamples).

| run | AP | ΔAP vs f32 | 95% interval |
|---|---|---|---|
| f32 (CPU) | 0.4013 | | |
| fake-quant, percentile 99.99% (`p4`) | 0.4003 | −0.0010 | [−0.0026, +0.0013] |
| fake-quant, MSE | 0.4001 | −0.0011 | [−0.0028, +0.0010] |
| **int8 kernels, `p4`** | **0.3997** | **−0.0015** | **[−0.0030, −0.0001]** |
| int8 − fake-quant (`p4`) | | −0.0005 | [−0.0029, +0.0003] |

Logs: `logs/boot-gate-n.log`, `logs/boot-gate-n-int8.log`, and `logs/score-{1,2,3}.log`. The gate was ΔAP ≥ −0.004 with the
interval above −0.006, so it is a **go**. The default method is the 99.99% percentile: MSE scored the same within
noise (−0.0001) and costs a second histogram walk per boundary. The int8 kernels and the fake-quant reference
agree to −0.0005: the integer sums are exact where the reference rounds in `f32`.

## Speed gate (per-conv, `q8_infer`)
`cargo bench -p folo-core --bench q8_infer` times every folo-n conv in int8 (`q8::conv`, u8 in and out) against
the fused `f32` `conv2d_act` in the same process, 3 alternated rounds; median ms. "Load" is the busy cores out of 28
(other lanes' builds) when the run started; the machine was rarely quiet this afternoon.

| session (binary) | threads | load | 48→64 1×1 at 160² | 64→64 3×3 at 80² | all convs, weighted by calls |
|---|---|---|---|---|---|
| s3 (epilogue v3, old task plan) | 28 | 5 | 0.548 → 0.344 ms, **1.59×** | 0.465 → 0.284, **1.64×** | 11.05 → 7.73 ms, 1.43× |
| s3, identity instead of SiLU | 28 | 3–11 | 2.42× | 2.09× | 1.91× |
| s4 (v5: task plan by input size) | 28 | 18–23 | 0.752 → 0.428, **1.76×** | 0.618 → 0.293, **2.11×** | 14.44 → 8.42, 1.71× |
| s4, identity | 28 | 17 | 2.05× | 1.84× | 1.90× |
| s4, one thread | 1 | 12–16 | 1.73× | 1.95× | 1×1 group 1.99×, 3×3 s1 group 2.40× |
S5_SPEED_ROWS
Logs: `logs/session-s3/`, `logs/session-s4/`, `logs/session-s5/` (`q8-*.txt`).

**The 3×3 gate (≥ 1.5×) passes; the 1×1 gate (≥ 2× on 48 → 64 at 160²) does not** (1.59–1.76× with SiLU). That
conv has K = 48, so each 4 × 24 tile does 12 VNNI steps and then an epilogue on 96 outputs: dequantise, SiLU, then
requantise. With the identity it reaches 2.05–2.42×; the SiLU's `exp` polynomial and its divide are what is left.
Making the epilogue cheaper without breaking the bit-exact scalar/AVX2 agreement (no `rcpps`, whose bits differ
between CPU vendors) is a leftover.

Thread scaling (`scripts/scaling.sh`, `scripts/scaling2.sh`, `logs/scaling-*.txt`): the first task plan (halve
pixel blocks until two tasks per thread) made 1 – 2 µs tasks and pixel blocks of one or two 24-column chunks.
Their output runs (96 bytes per plane) share cache lines with their neighbours on every one of the conv's output
planes. At 20², 1×1 128 → 256 took 0.104 ms on 28 threads against 0.198 on one. The plan now aims at three tasks
per thread, none under 2²⁰ MACs. While the image's input fits in 384 KiB, it splits by channel groups first, so
tasks write whole planes; otherwise by pixel chunks. In the paired run (load 20–23): 128 → 256 at 20² went from
0.107 to 0.072 ms on 28 threads. Splitting by channel groups at 160² instead cost 0.45 → 0.84 ms (every task
re-reads a 1.2 MB input), hence the size switch.

## End to end (rows 8 and 7)
TBD

## val2017 (final config, read once)
The shipped flow: `folo quantize` (512 random train2017 images, seed 0, percentile 99.99%, 62 ranges; folo-n
123 s, folo-s 93 s: `logs/quantize-n.log`, `logs/quantize-s.txt`), then the int8 weights on the CPU against the same weights in `f32` on
the CPU, all 5 000 val2017 images, batch 8 (`scripts/val_dump.sh`), B0 pycocotools-exact (`scripts/score_val.sh`),
1 000 paired bootstrap resamples (`logs/boot-val-*.log`).

| model | f32 AP | int8 AP | ΔAP | 95% interval |
|---|---|---|---|---|
| folo-n-coco-v4 | 0.3559 | 0.3541 | **−0.0019** | [−0.0028, −0.0010] |
| folo-s-coco | 0.4162 | 0.4137 | **−0.0025** | [−0.0039, −0.0014] |

Both are within the brief's 0.004. The `f32` CPU numbers match the published 0.356 and 0.416.

## Negative results
TBD

## Leftovers
TBD
