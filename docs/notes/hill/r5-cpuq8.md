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
TBD

## End to end (rows 8 and 7)
TBD

## val2017 (final config, read once)
TBD

## Negative results
TBD

## Leftovers
TBD
