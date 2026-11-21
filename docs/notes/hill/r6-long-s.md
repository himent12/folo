# Wave 6, lane long: `folo-s-coco` continued on full COCO (lab notebook)

Lane `long` of wave 6 (2026-11-20/21). Goal: a better folo-s (the published `folo-s-coco` trained for 56 epochs from
scratch, val2017 AP 0.416), and with it a better teacher for the next folo-n. **Result: val2017 AP 0.4224 against
0.4162, +0.0062 (paired bootstrap 95% interval [+0.0034, +0.0090]), one training seed.** The fine-tuning guard passes
its bar on `best.folo`, but it is mixed: BCCD is a little lower in all three seeds, and on brain-tumor the `last.folo`
models are lower by 0.0146. **Nothing is published**; that decision is the owner's. Run data:
`D:/folo-data/runs-wave6/long/`; scratch, scripts and evaluation files: `D:/folo-wave6/lanes/long/`.

Terms: **AP** is COCO's mAP50-95 (AP_S, AP_M and AP_L are the same for small, medium and large objects); **AMP** is
mixed-precision training (`--amp`: f16 activations, f32 master weights); **EMA** is the running average of the weights
that FOLO validates and exports; **mosaic** is the four-image training augmentation, which a run turns off ("closes")
for its last epochs.

## Result

COCO val2017, 5000 images, 640 px, read once with `folo val --coco --save-json` and scored by pycocotools
(`bench/fusion/coco.py` through `bin/records.py`; `eval/stats-*.json`). The published row was run again with the same
binary, and its detections are byte-identical to hill 4's `dets-teacher.json`.

| model | AP | AP50 | AP75 | AP_S | AP_M | AP_L | detections |
|---|---:|---:|---:|---:|---:|---:|---:|
| `folo-s-coco` (published, the start) | 0.4162 | 0.5798 | 0.4519 | 0.2289 | 0.4594 | 0.5845 | 679,000 |
| **continued, epoch 48, f32** | **0.4224** | 0.5916 | 0.4572 | 0.2478 | 0.4670 | 0.5850 | 725,952 |
| continued, fused f16 (`--half`) | 0.4225 | 0.5916 | 0.4571 | 0.2479 | 0.4669 | 0.5851 | 726,869 |
| continued, `best.folo` (epoch 40, f32, CPU; **picked on this split**, see below) | 0.4259 | 0.5944 | 0.4608 | 0.2482 | 0.4728 | 0.5896 | 735,834 |

Paired bootstrap over the 5000 images (`bench/fusion/bootstrap.py`, 1,000 resamples, seed 0, `logs/bootstrap.log`):
**continued − published = +0.0062, 95% interval [+0.0034, +0.0090]**, P(diff ≤ 0) 0.000. By object size, from the
unrounded scores: AP_S +0.0189, AP_M +0.0076, AP_L +0.0006. Each size is scored on its own, so these do not say how
much each one contributed to the overall AP.

**`best.folo`, scored once afterwards** (wave 6 lane `consensus`, `D:/folo-wave6/lanes/consensus/best/`; sha256
`bc0a63a4…7c96`, the checkpoint of epoch 40, EMA weights). `folo val --coco --save-json --device cpu --workers 8` at
this notebook's protocol (640 px, f32), pycocotools. All twelve numbers: AP 0.4259, AP50 0.5944, AP75 0.4608, AP_S
0.2482, AP_M 0.4728, AP_L 0.5896, AR1 0.3422, AR10 0.5687, AR100 0.6238, AR_S 0.4345, AR_M 0.6779, AR_L 0.7899.
Paired bootstrap against `eval/dets-cont.json` (the epoch 48 row above; 1,000 resamples, seed 0): **+0.0036, 95%
interval [+0.0019, +0.0048]**. As a control, the epoch 48 weights through the same CPU command give as many
detections as that GPU file (725,952) and all twelve numbers within 4e-7 of it, so the difference is not the device.
**This number is optimistic by an amount not measured:** the trainer picked epoch 40 as the best of 48 by its mAP on
val2017, the split scored here, and the interval covers the sampling of the images, not that choice. Nothing was
published or exported from it.

| file | bytes | sha256 |
|---|---:|---|
| `D:/folo-wave6/lanes/long/weights/folo-s-cont.folo` (EMA, unfused, f32) | 37,998,320 | `b4b10bfed8e90c0dadcf6347e43483b9845947441e5617e304d3fe996e5d556c` |
| `D:/folo-wave6/lanes/long/weights/folo-s-cont-f16.folo` (fused, f16) | 18,906,424 | `18cbe44878162af384056b729ff371aedf17c1ad5f7b3b553337bbc205c476dd` |

**What the number does not show.**
- One training seed: the interval covers the sampling of the evaluation images, not the training seed.
- val2017 was not fully untouched. The early-failure rule read the trainer's val2017 mAP at epoch 6 and that reading
  sent the lane from run 1 to run 2. The rule and its threshold were fixed before any run, and the exported weights
  are epoch 48's, also fixed in advance, never `best.folo`. Run 1 ended at the gate, so what it would have
  scored is unknown, and so are the size and the direction of that choice's effect on the result. The bootstrap covers
  neither it nor the training seed.
- Run 1 was stopped at epoch 6, so "0.3x LR is better than the default LR for this continuation" is not measured.

## Recipe

- **Start:** `folo-s-coco`, sha256 `a9d7f588…2e60`, the value in the release's `SHA256SUMS`; all 417 tensors
  transferred.
- **Binary:** `origin/main` 0e682e28 (#396, the faster loader), release, `bin/folo-long.exe`, sha256
  `4fb339b48c262f5ce405c3939a0834c252e577b56f40befbc713f2b4cde57728`. The same file for the runs, the evaluation and
  the guard.
- **Run 2, the one that counts** (`s-cont-lr03`), its settings: `folo train --weights folo-s-coco --data
  D:/folo-data/coco --format coco --epochs 48 --batch 32 --imgsz 640 --amp --cos-lr --seed 0 --device gpu:0 --workers 8
  --optimizer musgd --lr0 0.003 --set val_batch=32`. The launcher (`bin/long.sh`) passes the weights as the local
  copy `weights/folo-s-coco.folo` and adds `--project D:/folo-data/runs-wave6/long --name s-cont-lr03 --json-events
  epoch`. Resolved: MuSGD, momentum 0.937, Muon / SGD 0.2 / 1.0, bias warm-up LR 0.1, batch 32 ×
  accumulate 2, 1,849 optimizer steps per epoch, 3 warm-up epochs, mosaic closed for the last 10. No teacher.

### Decisions and deviations from the brief

1. **Length: 48 epochs, and the two timing measurements disagreed.** The first 10-minute smoke trained a whole epoch
   at 222.1 img/s (532.7 s, about 9.3 min with validation, which would have meant 64 epochs), then died of a GPU
   out-of-memory at the epoch-end validation. It did not have `val_batch=32`, so I repeated it with that setting
   (`bin/smoke.sh`: the command as then planned, a 64-epoch schedule at the default peak LR; batch, image size, AMP,
   seed and workers as in the final run):
   183.6 img/s, 10.74 min of training plus about 0.4 min of validation per epoch, over the 9.5-minute line, so **48**.
   The machine was busier during the second smoke. In hindsight 48 was conservative: run 2's epochs took a median
   of about 7.2 minutes, so 64 epochs would have taken about 7.7 h.
2. **Workers: 8.** One adjacent pair decided it: 12 workers 173.5 img/s, 8 workers 179.7. Two earlier single readings
   (287.9 and 104.5 img/s) were dominated by other load and are not a comparison.
3. **`--set val_batch=32` (default 2 × batch = 64).** See the out-of-memory note below. On a small probe the internal
   mAP was the same to six digits (0.36402515 against 0.3640252) and the validation peak fell from 14.0 to 10.7 GB.
4. **Early-failure rule fired, relaunch at 0.3x LR.** Below.
5. **The guard runs with `--amp`,** not the protocol's f32. Below.
6. **Not shared with finetune2.** The long run (about 9.2 GB) and a folo-n fine-tune (about 9.6 GB) do not fit under the
   brief's 15 GB limit: a smoke beside one ran at 9.6 img/s with 15.4–15.7 GB in use. As the brief says for this case,
   the run waited for finetune2's queue to empty (21:18Z).

## Run 1: the default peak LR failed the early rule

`s-cont`: the `auto` recipe at `lr0` 0.01. The rule, fixed in the brief: the trainer's mAP50-95 at epoch 6 must be at
least 0.36.

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| mAP50-95 | 0.3858 | 0.3464 | 0.3035 | 0.3219 | 0.3398 | **0.3535** |
| seconds | 516 | 489 | 533 | 741 | 801 | 926 |

0.3535 is under the line, so it was stopped with `folo stop` at 23:00:07Z and its folder kept. It was recovering by
about 0.014–0.018 per epoch, so it might well have finished above the start; the rule does not ask, and that is not
measured. Epochs 4–6 ran slower; CI's GPU tests were seen on the same card during them.

## Run 2: per-epoch curve

`s-cont-lr03`, the one allowed relaunch: `lr0` 0.003 (0.3 × the resolved 0.01), everything else as resolved in run 1
(both `config.json` files compared). Epoch 6: 0.3935, gate passed. The trainer's mAP50-95 reads about 0.004 below
pycocotools here (the published model: 0.4120 internal, 0.4162 pycocotools; the continued model: 0.4180 and 0.4224).
`s` is the epoch's duration; `img/s` is the rate the trainer reported at the epoch's last training step.

| epoch | mAP50-95 | mAP50 | train loss | s | img/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4035 | 0.5651 | 3.3139 | 663 | 190.5 |
| 2 | 0.3932 | 0.5539 | 3.3451 | 457 | 274.1 |
| 3 | 0.3760 | 0.5354 | 3.4679 | 432 | 289.5 |
| 4 | 0.3784 | 0.5350 | 3.6104 | 433 | 290.0 |
| 5 | 0.3853 | 0.5432 | 3.6256 | 418 | 289.8 |
| 6 | 0.3935 | 0.5529 | 3.6221 | 418 | 289.0 |
| 7 | 0.3966 | 0.5560 | 3.6192 | 417 | 289.2 |
| 8 | 0.3995 | 0.5595 | 3.6137 | 417 | 289.3 |
| 9 | 0.4032 | 0.5630 | 3.6046 | 417 | 289.3 |
| 10 | 0.4048 | 0.5647 | 3.5950 | 418 | 288.8 |
| 11 | 0.4067 | 0.5672 | 3.5807 | 417 | 289.0 |
| 12 | 0.4077 | 0.5685 | 3.5763 | 418 | 289.5 |
| 13 | 0.4084 | 0.5693 | 3.5623 | 418 | 289.1 |
| 14 | 0.4094 | 0.5700 | 3.5549 | 418 | 289.0 |
| 15 | 0.4097 | 0.5708 | 3.5405 | 418 | 289.0 |
| 16 | 0.4107 | 0.5718 | 3.5275 | 417 | 289.4 |
| 17 | 0.4119 | 0.5732 | 3.5186 | 418 | 289.0 |
| 18 | 0.4124 | 0.5734 | 3.5050 | 418 | 289.0 |
| 19 | 0.4131 | 0.5739 | 3.4917 | 438 | 275.8 |
| 20 | 0.4139 | 0.5751 | 3.4764 | 440 | 274.4 |
| 21 | 0.4149 | 0.5768 | 3.4592 | 440 | 274.5 |
| 22 | 0.4155 | 0.5777 | 3.4450 | 439 | 274.9 |
| 23 | 0.4157 | 0.5784 | 3.4313 | 439 | 274.7 |
| 24 | 0.4163 | 0.5792 | 3.4138 | 434 | 277.9 |
| 25 | 0.4168 | 0.5799 | 3.3961 | 426 | 283.0 |
| 26 | 0.4175 | 0.5811 | 3.3870 | 437 | 276.0 |
| 27 | 0.4179 | 0.5818 | 3.3638 | 437 | 276.1 |
| 28 | 0.4185 | 0.5822 | 3.3513 | 440 | 273.9 |
| 29 | 0.4190 | 0.5829 | 3.3355 | 437 | 276.2 |
| 30 | 0.4198 | 0.5840 | 3.3162 | 437 | 276.0 |
| 31 | 0.4202 | 0.5848 | 3.2988 | 437 | 276.1 |
| 32 | 0.4200 | 0.5848 | 3.2854 | 437 | 276.2 |
| 33 | 0.4204 | 0.5850 | 3.2671 | 439 | 274.6 |
| 34 | 0.4203 | 0.5850 | 3.2534 | 439 | 275.1 |
| 35 | 0.4209 | 0.5856 | 3.2380 | 440 | 274.2 |
| 36 | 0.4212 | 0.5861 | 3.2233 | 439 | 274.7 |
| 37 | 0.4211 | 0.5863 | 3.2091 | 439 | 274.7 |
| 38 | 0.4215 | 0.5869 | 3.1941 | 436 | 276.8 |
| 39 | 0.4212 | 0.5870 | 3.0431 | 427 | 282.7 |
| 40 | 0.4215 | 0.5875 | 3.0132 | 424 | 284.5 |
| 41 | 0.4215 | 0.5874 | 2.9972 | 425 | 284.1 |
| 42 | 0.4212 | 0.5867 | 2.9819 | 424 | 284.3 |
| 43 | 0.4210 | 0.5864 | 2.9685 | 424 | 284.4 |
| 44 | 0.4203 | 0.5858 | 2.9603 | 425 | 284.2 |
| 45 | 0.4196 | 0.5856 | 2.9488 | 425 | 284.2 |
| 46 | 0.4193 | 0.5855 | 2.9442 | 434 | 278.1 |
| 47 | 0.4185 | 0.5848 | 2.9374 | 436 | 276.4 |
| 48 | 0.4180 | 0.5847 | 2.9342 | 424 | 284.3 |

**After epoch 40 the run lost ground on the trainer's own metric.** It peaked at epoch 40 (0.42154; epochs 38 and 41
round to the same 0.4215) and fell to 0.41796 at epoch 48, a drop of 0.0036. Over the last ten epochs mosaic was off
and the learning rate was annealing. One curve shows when the drop happened, not what caused it. The exported
weights are epoch 48's as the brief fixed; `best.folo` (epoch 40, picked by the trainer's own validation on the scored
split) was not exported. Scored afterwards in the standalone COCO evaluation it reads AP 0.4259, +0.0036 [+0.0019,
+0.0048] over epoch 48, with the selection caveat under [Result](#result).

## Wall-clock and throughput

- **Run 2:** 20,854 s of training (5.79 h; 23:00:31Z to 04:48:08Z). Median epoch 282.8 img/s, range 190.5–290.0.
- **Run 1, stopped:** 21:52:37Z to 23:00:07Z (1 h 7.5 min). From the first launch to the end of run 2: 6 h 55.5 min.
- **What else ran:** CI on the same PC (its GPU tests were seen on the card during run 1), other lanes' builds, and normal desktop
  use. No finetune2 run overlapped (its last log entry is at 21:17:41Z). This is not an unloaded cost.

## Speed

The architecture is the same, so the forward pass does the same work at the same settings. End-to-end time also
includes NMS, whose work depends on the candidates a model emits, so it could differ. Neither was measured: no speed
row of docs/BENCHMARKS.md was run again. The first alternated `folo val` timing (four rounds, always
published then continued) read the continued weights at 2.2x the time (medians about 9.7 s and 21.4 s). Two more
trials, with the order reversed in blocks, were also noisy: most single runs took about 10 s or 15–29 s, and two took
about 11.8 s. A last test with **fresh byte-identical copies of both files**, 24 runs in one order shuffled with
seed 0 inside one GPU lock hold (`eval/ab-val4.txt`):

| arm (12 runs: original + copy) | median s | min | max | runs over 15 s |
|---|---:|---:|---:|---:|
| published | 13.1 | 9.35 | 28.63 | 5 |
| continued | 10.3 | 9.74 | 24.31 | 3 |

Both weights have slow runs, a copy of the published file was the slowest of the four arms (median 17.7 s), and the
first trial's 2.2x did not come back in the shuffled trial. That is all these timings show: with 12 noisy runs per
arm they neither establish nor rule out a speed difference between the weights. **The cause of the slow runs is not
identified.** Timings: `eval/ab-val.txt`, `ab-val2.txt`, `ab-val3.txt`, `ab-val4.txt`.

## Fine-tuning guard

Bar, fixed before the runs and the same as for folo-n v4 and v5: the new weights' mean `best.folo` AP over three seeds
is not more than 0.010 below the published folo-s on each set. BCCD (scored on test) and brain-tumor (scored on val),
seeds 0–2, `bench/finetune/README.md`'s protocol with one change, both arms on the same binary (`bin/ft.sh`,
`eval/guard-table.md`).

**`best.folo`, the benchmark's number: the bar passes on both sets.**

| dataset | pub s0 | pub s1 | pub s2 | pub mean | new s0 | new s1 | new s2 | new mean | new − pub | seeds up | bar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BCCD | 0.6084 | 0.6193 | 0.6227 | 0.6168 | 0.6077 | 0.6095 | 0.6189 | 0.6120 | −0.0048 | 0/3 | pass |
| brain-tumor | 0.4232 | 0.4455 | 0.4572 | 0.4420 | 0.4268 | 0.4587 | 0.4659 | 0.4505 | +0.0085 | 3/3 | pass |

**`last.folo`, for reference (not the bar): brain-tumor is lower by more than 0.010.**

| dataset | pub s0 | pub s1 | pub s2 | pub mean | new s0 | new s1 | new s2 | new mean | new − pub | seeds up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BCCD | 0.5880 | 0.5949 | 0.5895 | 0.5908 | 0.5850 | 0.5948 | 0.5916 | 0.5905 | −0.0004 | 1/3 |
| brain-tumor | 0.3850 | 0.3768 | 0.4044 | 0.3887 | 0.3767 | 0.3687 | 0.3770 | 0.3741 | −0.0146 | 0/3 |

How to read it: brain-tumor's `best.folo` is selected on the split it is then scored on, which lane finetune showed
flatters a noisy curve, so the +0.0085 is weak evidence of a gain. The −0.0146 on `last.folo` has no such selection
and points the other way, on the set where a distilled folo-n was also a worse start. **The guard shows no clear
fine-tuning gain and leaves a possible loss on brain-tumor open.**

**The change: both arms train with `--amp`** (every checkpoint is then scored in f32). folo-s trained in f32 at batch
16 reaches 15.7 GB within seconds and dies of a GPU out-of-memory on this 16 GB card, with nothing else on the GPU, at
validation batch 32 and at 16 (`logs/guard-oom/`). The benchmark's folo-s rows ran on larger rented GPUs. With AMP a
run peaks at 12.6 GB. So the two starts are compared with each other under the same settings, but these absolute
numbers are not the f32 table's in docs/BENCHMARKS.md.

## What went wrong

- **The epoch-end out-of-memory.** The first full-epoch smoke (folo-s, batch 32, AMP, default validation batch 64)
  trained for 8.9 minutes, then failed at validation; because no epoch had finished, FOLO removed the run folder and
  its log. A short probe at the same settings passed, peaking at 14.0 GB, so **the failure was not reproduced and its
  exact cause is not established** (a tight fit, another GPU user at that moment, or both).
- **Early-failure relaunch:** 1 h 7.5 min of GPU spent on run 1.
- **My own slip in the guard queue:** after stopping a failed runner I removed the wrong queue line. I caught it at
  once and restored the two missing runs. Each of the 12 configurations completed once (`logs/runner.out` has 12
  rows with exit 0; the failed f32 attempt of the published BCCD seed 0 is in `logs/guard-oom/`).

## What I would do next

1. **Decide on publication with the guard in view.** The measured val2017 gain is +0.0062 on one training seed, and
   the fine-tuning evidence is mixed. As a *teacher* for the next folo-n these weights need no fine-tuning guard; as the recommended fine-tuning
   start, the brain-tumor `last.folo` result argues for keeping `folo-s-coco` until it is understood.
2. **Fix the out-of-memory path.** A default `folo train` of folo-s at batch 32 can lose a whole epoch at its first
   validation on a 16 GB card. Options: validate at the training batch when memory is tight, or free the training
   buffers first. The same note belongs in `bench/finetune/README.md` for folo-s in f32.
3. **A longer continuation at the lower LR.** Epochs ran at about 7.2 minutes, so 64 would fit overnight, but the
   mAP curve had flattened by epoch 38, so expect little from length alone.
4. **Look at the last ten epochs.** The trainer's mAP fell 0.0036 from epoch 40 to 48, with mosaic off and the
   learning rate annealing. Test shorter or later closing on a split carved from train2017, not on val2017.
