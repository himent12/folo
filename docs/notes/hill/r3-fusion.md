# Hill-climb 3: DFL evidence fusion, the post-processing ladder, landing folo-n v2 (lab notebook)

Lane `fusion` of hill-climb 3 (2026-11-13). Brief: `D:/folo-build/hill3/briefs/fusion.md`. Two goals: take the
accuracy the existing weights leave on the table at inference (the post-processing ladder, with a new idea, DFL
evidence fusion, at its top), and finish the schedule of the published `folo-n-coco-v2` weights.

## Headline

COCO val2017 AP (pycocotools), read once per final configuration, with a paired bootstrap over images (1000
resamples, seed 0, 95% percentile interval of the difference). B0 is today's protocol (MODEL.md §9.3: conf 0.001,
IoU 0.7, max_det 300, multi-label, hard NMS). B* = `--fusion dfl --iou 0.65 --max-det 1000`, chosen on train-dev.

| weights | B0 | B* | B* − B0 [95%] | B* − voting [95%] | voting − B0 [95%] | NMS-only − B0 [95%] |
|---|---:|---:|---|---|---|---|
| folo-n v1 (`folo-n-coco`) | 0.3138 | **0.3181** | **+0.0043** [+0.0036, +0.0049] | +0.0010 [+0.0007, +0.0013] | +0.0033 [+0.0026, +0.0039] | +0.0009 [+0.0005, +0.0012] |
| folo-n v2 (`folo-n-coco-v2`) | 0.3168 | **0.3202** | **+0.0034** [+0.0026, +0.0040] | +0.0007 [+0.0004, +0.0011] | +0.0026 [+0.0018, +0.0033] | +0.0009 [+0.0006, +0.0012] |
| folo-s (`folo-s-coco`) | 0.4162 | **0.4186** | **+0.0025** [+0.0019, +0.0032] | +0.0003 [+0.0001, +0.0006] | +0.0021 [+0.0015, +0.0028] | +0.0000 [−0.0003, +0.0004] |
| folo-n landed (this lane) | 0.3396 | **0.3434** | **+0.0038** [+0.0030, +0.0045] | +0.0005 [+0.0002, +0.0009] | +0.0033 [+0.0025, +0.0040] | +0.0007 [+0.0004, +0.0011] |

"voting" is score-weighted box voting (τ 0.75, a 1) with the same NMS settings as B* (IoU 0.65, max_det 1000), so
"B* − voting" is what the DFL weights add over the known baseline, and "B* − NMS-only" (v1 +0.0033, v2 +0.0025,
folo-s +0.0025, landed +0.0031, every interval above 0) is what fusion as a whole adds on top of the NMS change.
P(diff ≤ 0) over the resamples is 0.000 for every difference except B* − voting on folo-s (0.009) and NMS-only on folo-s (0.454: no effect). Logs
`D:/folo-data/runs-hill3/fusion/logs/boot2-<model>.log`.

**Answer to the lane's question.** On val2017, DFL evidence fusion gives **+0.0025 to +0.0043 AP over hard NMS** (B* against B0: folo-n v1 +0.0043,
v2 +0.0034, landed +0.0038, folo-s +0.0025), of which +0.0025 to +0.0033 comes from the fusion itself (B* against
the same NMS without fusion), and **+0.0003 to +0.0010 over plain score-weighted voting** (v1 +0.0010, v2 +0.0007,
landed +0.0005, folo-s +0.0003): every interval excludes zero, but most of the gain is voting's. The DFL certainties
are worth about a quarter of what fusing the cluster is worth on folo-n, an eighth on folo-s. These are post-processing gains:
the incumbent's weights would gain from some of them too (YOLO11 has DFL; YOLO26 does not, so DFL fusion does not
apply to it). The standard-protocol rows stay the headline; B* is its own row, named as what it is. YOLO11n through
the same post-process was **not measured**.

**Landed folo-n** (Part 0): `D:/folo-data/runs-hill3/fusion/land-n-v2/weights/best.folo` (f16 export
`best-f16.folo` beside it), COCO val2017 AP **0.340** / AP50 **0.487** (v2: 0.317 / 0.458). Not published.

## Part 0: landing folo-n-coco-v2

`folo-n-coco-v2` was epoch 62 of a 300-epoch cosine run (LR still 0.90 · lr0, mosaic on). The landing run gives it
an ending: release `folo` of `origin/main` e5a6ccf0, from the v2 EMA weights (`--weights`, not `--resume`: 417 of 417
tensors transferred, the EMA starts fresh from them):

    folo train --weights folo-n-coco-v2 --data D:/folo-data/coco --project D:/folo-data/runs-hill3/fusion \
      --name land-n-v2 --epochs 16 --batch 16 --imgsz 640 --device gpu:0 --workers 16 --optimizer sgd \
      --lr0 0.005 --lrf 0.01 --amp --warmup-epochs 0 --set warmup_bias_lr=0 --set momentum=0.9 \
      --close-mosaic 8 --seed 0 --patience 1000

Resolved (`config.json`): SGD, lr0 0.005 linear to 0.005 · 0.01, momentum 0.9 (the original run's resolved SGD
momentum; `--optimizer sgd` alone would have given 0.937), weight decay 0.0005, no warmup, warmup_bias_lr 0, AMP,
batch 16, mosaic off for the last 8 epochs, seed 0. 16 epochs in 1.58 h (3.9–8.9 min per epoch with other lanes on
the machine). The training marker was kept for the whole run and removed at the end.

Internal validation (the trainer's evaluator, val2017, EMA):

| epoch | train box / cls / dfl | mAP50 | mAP50-95 |
|---:|---|---:|---:|
| 1 | 1.262 / 1.581 / 1.291 | 0.4288 | 0.2931 |
| 2 | 1.260 / 1.574 / 1.288 | 0.4377 | 0.2970 |
| 3 | 1.258 / 1.562 / 1.285 | 0.4494 | 0.3074 |
| 4 | 1.252 / 1.555 / 1.282 | 0.4567 | 0.3142 |
| 5 | 1.248 / 1.547 / 1.280 | 0.4605 | 0.3186 |
| 6 | 1.244 / 1.536 / 1.275 | 0.4647 | 0.3219 |
| 7 | 1.240 / 1.523 / 1.271 | 0.4690 | 0.3249 |
| 8 | 1.234 / 1.512 / 1.268 | 0.4727 | 0.3270 |
| 9 (mosaic off) | 1.228 / 1.422 / 1.257 | 0.4745 | 0.3287 |
| 10 | 1.217 / 1.396 / 1.251 | 0.4760 | 0.3304 |
| 11 | 1.210 / 1.379 / 1.246 | 0.4767 | 0.3314 |
| 12 | 1.204 / 1.360 / 1.241 | 0.4777 | 0.3324 |
| 13 | 1.195 / 1.341 / 1.235 | 0.4788 | 0.3334 |
| 14 | 1.186 / 1.323 / 1.229 | 0.4800 | 0.3342 |
| 15 | 1.179 / 1.308 / 1.224 | 0.4808 | 0.3351 |
| 16 | 1.173 / 1.290 / 1.220 | **0.4812** | **0.3355** |

Epochs 1–2 sit below v2's 0.313 because the EMA restarts from the transferred weights (an EMA of the first
iterations of a new run lags them); it is past 0.313 at epoch 4, so the brief's lr0 0.002 restart was not needed.
The curve still rises at epoch 16 (+0.0004 in the last epoch): a longer landing would gain a little more.

`folo val --coco` of `best.folo` (epoch 16) on val2017:

| | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR1 | AR10 | AR100 | AR_S | AR_M | AR_L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 (published) | 0.317 | 0.458 | 0.343 | 0.153 | 0.344 | 0.453 | | | 0.550 | | | |
| landed, f32 | **0.340** | **0.487** | 0.367 | 0.163 | 0.368 | 0.494 | 0.301 | 0.507 | 0.565 | 0.331 | 0.618 | 0.762 |
| landed, f16 export | 0.339 | 0.487 | 0.367 | 0.163 | 0.367 | 0.494 | 0.301 | 0.507 | 0.565 | 0.332 | 0.618 | 0.762 |

Files in `D:/folo-data/runs-hill3/fusion/land-n-v2/weights/`: `best.folo` (fused f32), `best-f16.folo` (5.3 MB,
`folo export --format folo-f16`), `best-weights.folo` (unfused EMA f32, 10.6 MB, for fine-tuning or publishing),
`last.folo`. No release was created.

## Part 1: the candidate dump and an exact Python post-process

**Dump tool**: `crates/folo-vision/examples/candidate_dump.rs` (an example of folo-vision, not part of the CLI):

    cargo run --release -p folo-vision --example candidate_dump -- --model <.folo> --data D:/folo-data/coco \
      --split val|train [--list <file names>] --out <dir> [--device gpu:0] [--batch 32]

It runs validation's preprocessing (the loader's letterbox, u8 upload on a GPU, `/255` on the device), the
one-to-many head, the device decode, and writes for every anchor with any class score `> 0.001`: anchor index (the
level follows from it), the decoded box in network pixels and in original-image pixels (`ImageMeta::unmap_xyxy`),
the 64 DFL logits; and every (anchor, class, score) pair with score `> 0.001`. Layout in `bench/fusion/dump.py`.

**Python post-process** (`bench/fusion/`): `post.py` (FOLO's NMS replica plus every rung below), `coco.py`
(pycocotools scoring, per-image match records, the weighted AP the bootstrap uses), `run.py` (configurations in
parallel), `bootstrap.py`, `compare.py` (detection by detection), `ladder.py`, `oracles.py`, `operating.py`,
`censor_stats.py`, `make_traindev.py`, `timing_table.py`.

**Reproduction** (before anything changed): Python B0 against `folo val --coco --save-json` of the same weights,
detection by detection:

| weights | detections | differing images | COCO AP (both) |
|---|---:|---:|---:|
| folo-n-coco (v1) | 970,945 | 0 | 0.3138 |
| folo-n-coco-v2 | 1,038,462 | 0 | 0.3168 |
| folo-s-coco | 679,000 | 0 | 0.4162 |
| landed | 969,059 | 0 | 0.3396 |

The record-based AP the bootstrap uses equals pycocotools' to 1e-16.

**train-dev**: `make_traindev.py D:/folo-data/coco <out> --n 5000 --seed 0`: 5000 train2017 images (36,058
annotations), list and COCO json in `D:/folo-data/runs-hill3/fusion/lists/`. Every choice below was made on it. All
three models trained on these images, so train-dev AP is higher than val AP (v1 0.347 against 0.314); the ranking of
configurations is what carries over, and it did (val numbers below).

## Part 2: the ladder (train-dev, v1 / v2 / folo-s)

Each rung alone against B0. AP is COCO AP@[.5:.95]; full stats (AP50, AP_S/M/L, AR100) per row in
`D:/folo-data/runs-hill3/fusion/logs/td-<model>-<rung>.log`.

#### Rung 1: NMS IoU and max_det

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| B0 | 0.3467 | 0.3391 | 0.5086 | 0.3741 | 0.3668 | 0.5558 | 0.154 / 0.364 / 0.492 |
| iou0.6 | 0.3470 | 0.3398 | 0.5073 | 0.3680 | 0.3620 | 0.5480 | 0.154 / 0.364 / 0.492 |
| iou0.65 | 0.3473 | 0.3399 | 0.5084 | 0.3714 | 0.3647 | 0.5523 | 0.154 / 0.364 / 0.492 |
| iou0.75 | 0.3450 | 0.3372 | 0.5076 | 0.3749 | 0.3672 | 0.5576 | 0.154 / 0.362 / 0.490 |
| maxdet1000 | 0.3470 | 0.3395 | 0.5088 | 0.3744 | 0.3671 | 0.5560 | 0.155 / 0.364 / 0.492 |
| maxdet100 | 0.3449 | 0.3372 | 0.5069 | 0.3724 | 0.3650 | 0.5543 | 0.151 / 0.363 / 0.491 |

#### Rung 2: Soft-NMS (Gaussian, σ) and Matrix NMS

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| soft0.1 | 0.3385 | 0.3309 | 0.4978 | 0.3573 | 0.3506 | 0.5357 | 0.147 / 0.352 / 0.482 |
| soft0.3 | 0.3472 | 0.3399 | 0.5078 | 0.3681 | 0.3619 | 0.5496 | 0.153 / 0.363 / 0.492 |
| soft0.5 | 0.3455 | 0.3386 | 0.5034 | 0.3683 | 0.3622 | 0.5469 | 0.154 / 0.363 / 0.490 |
| soft1.0 | 0.3300 | 0.3247 | 0.4806 | 0.3547 | 0.3499 | 0.5244 | 0.150 / 0.351 / 0.473 |
| matrix-g1.0 | 0.3006 | 0.2962 | 0.4421 | 0.3263 | 0.3221 | 0.4848 | 0.139 / 0.326 / 0.446 |
| matrix-g2.0 | 0.3331 | 0.3262 | 0.4860 | 0.3579 | 0.3517 | 0.5300 | 0.151 / 0.354 / 0.479 |
| matrix-g4.0 | 0.3466 | 0.3393 | 0.5070 | 0.3691 | 0.3627 | 0.5499 | 0.153 / 0.363 / 0.491 |
| matrix-lin | 0.3458 | 0.3386 | 0.5070 | 0.3708 | 0.3646 | 0.5525 | 0.154 / 0.363 / 0.491 |

#### Rung 3: per-level logit offsets (P3, P4, P5 = 0, a, b)

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| off-0.30-0.30 | 0.3461 | 0.3390 | 0.5090 | 0.3731 | 0.3661 | 0.5552 | 0.154 / 0.357 / 0.491 |
| off-0.30+0.00 | 0.3462 | 0.3382 | 0.5087 | 0.3732 | 0.3651 | 0.5548 | 0.154 / 0.359 / 0.497 |
| off-0.30+0.30 | 0.3448 | 0.3362 | 0.5073 | 0.3717 | 0.3630 | 0.5532 | 0.154 / 0.361 / 0.498 |
| off+0.00-0.30 | 0.3461 | 0.3388 | 0.5074 | 0.3736 | 0.3668 | 0.5550 | 0.154 / 0.362 / 0.486 |
| off+0.00+0.30 | 0.3463 | 0.3378 | 0.5078 | 0.3737 | 0.3651 | 0.5549 | 0.154 / 0.366 / 0.497 |
| off+0.30-0.30 | 0.3437 | 0.3364 | 0.5044 | 0.3718 | 0.3646 | 0.5526 | 0.154 / 0.364 / 0.478 |
| off+0.30+0.00 | 0.3456 | 0.3380 | 0.5063 | 0.3737 | 0.3664 | 0.5541 | 0.154 / 0.366 / 0.486 |
| off+0.30+0.30 | 0.3459 | 0.3380 | 0.5072 | 0.3739 | 0.3660 | 0.5546 | 0.154 / 0.368 / 0.492 |

#### Rung 4: DFL temperature and ±w-bin window

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| T0.7 | 0.3426 | 0.3352 | 0.5003 | 0.3704 | 0.3633 | 0.5488 | 0.151 / 0.361 / 0.487 |
| T0.85 | 0.3461 | 0.3385 | 0.5072 | 0.3728 | 0.3656 | 0.5533 | 0.154 / 0.364 / 0.491 |
| T1.15 | 0.3456 | 0.3382 | 0.5065 | 0.3731 | 0.3663 | 0.5529 | 0.154 / 0.363 / 0.490 |
| T1.3 | 0.3438 | 0.3365 | 0.5026 | 0.3722 | 0.3652 | 0.5505 | 0.154 / 0.361 / 0.488 |
| win1 | 0.3465 | 0.3389 | 0.5086 | 0.3726 | 0.3648 | 0.5550 | 0.155 / 0.364 / 0.492 |
| win2 | 0.3471 | 0.3394 | 0.5093 | 0.3740 | 0.3667 | 0.5567 | 0.155 / 0.364 / 0.493 |
| win3 | 0.3472 | 0.3394 | 0.5088 | 0.3740 | 0.3668 | 0.5564 | 0.155 / 0.364 / 0.492 |

#### Rung 5: score-weighted voting (τ, a, IoU kernel σ)

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| vote-t0.55-a0.0 | 0.3214 | 0.3110 | 0.4779 | 0.3558 | 0.3457 | 0.5314 | 0.147 / 0.345 / 0.451 |
| vote-t0.55-a1.0 | 0.3400 | 0.3318 | 0.5011 | 0.3627 | 0.3564 | 0.5429 | 0.150 / 0.358 / 0.485 |
| vote-t0.55-a2.0 | 0.3400 | 0.3319 | 0.5012 | 0.3618 | 0.3555 | 0.5423 | 0.150 / 0.357 / 0.486 |
| vote-t0.65-a0.0 | 0.3375 | 0.3286 | 0.4989 | 0.3717 | 0.3639 | 0.5511 | 0.153 / 0.357 / 0.476 |
| vote-t0.65-a1.0 | 0.3457 | 0.3374 | 0.5067 | 0.3716 | 0.3663 | 0.5514 | 0.153 / 0.363 / 0.492 |
| vote-t0.65-a2.0 | 0.3453 | 0.3371 | 0.5062 | 0.3698 | 0.3639 | 0.5494 | 0.153 / 0.362 / 0.492 |
| vote-t0.75-a0.0 | 0.3456 | 0.3374 | 0.5080 | 0.3773 | 0.3692 | 0.5573 | 0.155 / 0.364 / 0.489 |
| vote-t0.75-a1.0 | 0.3488 | 0.3412 | 0.5102 | 0.3764 | 0.3712 | 0.5568 | 0.156 / 0.366 / 0.495 |
| vote-t0.75-a2.0 | 0.3487 | 0.3410 | 0.5099 | 0.3755 | 0.3703 | 0.5561 | 0.156 / 0.366 / 0.495 |
| vote-t0.85-a0.0 | 0.3475 | 0.3401 | 0.5100 | 0.3743 | 0.3682 | 0.5565 | 0.155 / 0.365 / 0.493 |
| vote-t0.85-a1.0 | 0.3484 | 0.3411 | 0.5104 | 0.3743 | 0.3695 | 0.5566 | 0.155 / 0.366 / 0.495 |
| vote-t0.85-a2.0 | 0.3485 | 0.3411 | 0.5102 | 0.3743 | 0.3697 | 0.5564 | 0.155 / 0.366 / 0.495 |
| vote-t0.55-a1.0-s0.025 | 0.3490 | 0.3413 | 0.5104 | 0.3759 | 0.3698 | 0.5571 | 0.156 / 0.367 / 0.495 |
| vote-t0.55-a1.0-s0.05 | 0.3480 | 0.3402 | 0.5093 | 0.3743 | 0.3691 | 0.5556 | 0.155 / 0.366 / 0.494 |
| vote-t0.55-a1.0-s0.1 | 0.3455 | 0.3380 | 0.5067 | 0.3707 | 0.3661 | 0.5510 | 0.153 / 0.363 / 0.492 |

#### Rung 6a: DFL fusion (b, ε, x = excess variance), voting reference, lattice pooling

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| var-t0.75-b0.5-e0.01 | 0.3493 | 0.3416 | 0.5105 | 0.3770 | 0.3707 | 0.5564 | 0.156 / 0.367 / 0.496 |
| var-t0.75-b0.5-e0.1 | 0.3492 | 0.3416 | 0.5105 | 0.3769 | 0.3709 | 0.5566 | 0.156 / 0.367 / 0.496 |
| var-t0.75-b1.0-e0.01 | 0.3494 | 0.3418 | 0.5105 | 0.3773 | 0.3704 | 0.5565 | 0.156 / 0.367 / 0.496 |
| var-t0.75-b1.0-e0.1 | 0.3493 | 0.3419 | 0.5107 | 0.3773 | 0.3706 | 0.5568 | 0.156 / 0.367 / 0.496 |
| var-t0.75-b0.5-e0.01-x | 0.3495 | 0.3417 | 0.5106 | 0.3766 | 0.3712 | 0.5567 | 0.156 / 0.367 / 0.496 |
| var-t0.75-b0.5-e0.1-x | 0.3494 | 0.3419 | 0.5106 | 0.3770 | 0.3710 | 0.5568 | 0.156 / 0.367 / 0.496 |
| var-t0.75-b1.0-e0.01-x | 0.3496 | 0.3419 | 0.5107 | 0.3765 | 0.3708 | 0.5563 | 0.155 / 0.367 / 0.497 |
| var-t0.75-b1.0-e0.1-x | 0.3497 | 0.3422 | 0.5109 | 0.3770 | 0.3710 | 0.5570 | 0.156 / 0.367 / 0.497 |
| vote-t0.75-a1.0-ref | 0.3488 | 0.3412 | 0.5102 | 0.3764 | 0.3712 | 0.5568 | 0.156 / 0.366 / 0.495 |
| lattice-t0.75 | 0.3458 | 0.3374 | 0.4953 | 0.3715 | 0.3637 | 0.5341 | 0.152 / 0.363 / 0.494 |

#### Rung 6b: around B* (c = censor power, thr = censor drop, τ, b, a, σ, NMS IoU, max_det, window)

| config | v1 AP | v2 AP | s AP | v1 AP75 | v2 AP75 | s AP75 | v1 S / M / L |
|---|---:|---:|---:|---:|---:|---:|---|
| varx | 0.3497 | 0.3422 | 0.5109 | 0.3770 | 0.3710 | 0.5570 | 0.156 / 0.367 / 0.497 |
| varx-c1 | 0.3497 | 0.3422 | 0.5109 | 0.3771 | 0.3713 | 0.5572 | 0.156 / 0.367 / 0.497 |
| varx-c2 | 0.3498 | 0.3423 | 0.5109 | 0.3770 | 0.3711 | 0.5572 | 0.156 / 0.367 / 0.497 |
| varx-c4 | 0.3497 | 0.3421 | 0.5108 | 0.3769 | 0.3703 | 0.5569 | 0.156 / 0.367 / 0.497 |
| varx-thr0.5 | 0.3497 | 0.3422 | 0.5108 | 0.3773 | 0.3709 | 0.5571 | 0.156 / 0.367 / 0.497 |
| varx-thr0.3 | 0.3496 | 0.3422 | 0.5109 | 0.3771 | 0.3705 | 0.5565 | 0.155 / 0.367 / 0.497 |
| varx-t0.7 | 0.3492 | 0.3416 | 0.5100 | 0.3765 | 0.3708 | 0.5560 | 0.155 / 0.366 / 0.497 |
| varx-t0.8 | 0.3493 | 0.3420 | 0.5108 | 0.3754 | 0.3702 | 0.5571 | 0.156 / 0.366 / 0.496 |
| varx-t0.85 | 0.3487 | 0.3416 | 0.5105 | 0.3743 | 0.3690 | 0.5565 | 0.155 / 0.366 / 0.495 |
| varx-b1.5 | 0.3497 | 0.3420 | 0.5110 | 0.3767 | 0.3702 | 0.5563 | 0.156 / 0.367 / 0.497 |
| varx-b2 | 0.3495 | 0.3417 | 0.5109 | 0.3767 | 0.3697 | 0.5562 | 0.155 / 0.367 / 0.496 |
| varx-a2 | 0.3497 | 0.3419 | 0.5107 | 0.3761 | 0.3703 | 0.5567 | 0.156 / 0.367 / 0.497 |
| varx-s0.05 | 0.3497 | 0.3421 | 0.5110 | 0.3762 | 0.3702 | 0.5572 | 0.156 / 0.367 / 0.497 |
| varx-iou0.65 | 0.3505 | 0.3433 | 0.5111 | 0.3755 | 0.3699 | 0.5552 | 0.156 / 0.368 / 0.497 |
| varx-md1000 | 0.3500 | 0.3425 | 0.5111 | 0.3773 | 0.3713 | 0.5571 | 0.156 / 0.367 / 0.497 |
| varx-win2 | 0.3500 | 0.3422 | 0.5111 | 0.3768 | 0.3709 | 0.5575 | 0.156 / 0.367 / 0.496 |

**Rung 1** (NMS IoU, max_det): IoU 0.65 is the best threshold for both folo-n (+0.0006 / +0.0008) and ties for
folo-s (−0.0002); 0.75 loses everywhere. max_det 1000 helps a little (+0.0003 / +0.0004 / +0.0002): the 300 cut in
`nms.rs` is per image over all classes, while COCO's AP reads up to 100 per image **and category**, so an image
with many classes loses low-scoring true positives at 300; 100 loses clearly.

**Rung 2** (Soft-NMS, Matrix NMS): Gaussian Soft-NMS at σ 0.3 ties B0 (+0.0005 / +0.0008 / −0.0008), every other σ
loses; Matrix NMS (one IoU matrix and a column maximum) is at best a tie (Gaussian σ 4: −0.0001 / +0.0002 / −0.0016;
linear: −0.0009). Negative.

**Rung 3** (per-level score offsets): a ±0.3 logit grid on P4 and P5 (P3 fixed): no cell beats B0 on more than one
model (best folo-s −0.3/−0.3: +0.0004, the same cell −0.0006 on v1). Negative: the levels' scores are calibrated
against each other already. Caveat on the positive cells: the dump keeps pairs above 0.001 only, so a +0.3 offset
cannot lift the pairs between 0.00074 and 0.001 over the threshold. That band is worth little: dropping the same
logit band from the top instead (B0 at conf 0.00135 against 0.001) moves train-dev AP by 0.0000 / 0.0000 / −0.0001,
less than every positive cell's loss against B0 (−0.0004 to −0.0042), so the conclusion stands, but those cells are
an approximation of the stated transform, not an exact measurement (`logs/review256.log`).

**Rung 4** (DFL point estimate): a softmax temperature loses in both directions (T 0.85: −0.0006 / −0.0006 /
−0.0014). The expectation in a ±2-bin window around the argmax gains +0.0004 / +0.0003 / +0.0007 on its own and
+0.0003 / 0 / +0.0002 on top of fusion (r6b `win2`); left out of B* because it changes the decode itself, so the GPU
detect kernel, the CPU decode and the full decode would all need it, for 0.0003.

**Rung 5** (score-weighted box voting, the baseline to beat): weights `score^a · exp(−(1 − IoU)² / σ)`, cluster =
same-class candidates with IoU ≥ τ. Chosen as the baseline: τ 0.75, a 1, no IoU kernel (+0.0020 / +0.0021 /
+0.0016). The plateau is τ 0.75–0.85, a 1–2; a wide cluster (τ 0.55) needs the IoU kernel, and with σ 0.025 it is
the highest voting row (+0.0023 / +0.0022 / +0.0019, i.e. +0.0003 / +0.0002 / +0.0002 above the chosen one), but it
adds a parameter for a difference inside the plateau's spread, so the simpler setting is the baseline. Under B*'s
NMS settings the two are as close (v1 0.3500 / 0.3503, v2 0.3428 / 0.3429, folo-s 0.5106 / 0.5107) and B* beats
both (0.3508 / 0.3437 / 0.5112; `logs/review256.log`), so "B* − voting" below does not depend on that choice. Flat
weights (a 0) at τ 0.55 lose 0.025 AP (the cluster pulls boxes into their neighbours).

**Rung 6** (DFL evidence fusion, against rung 5 at the same τ 0.75, a 1): inverse-variance weights per side.

- **Plain variance** (`v = stride² · Var`, `b` 1, ε 0.1 bins²): +0.0006 / +0.0007 / +0.0005 over voting.
- **Excess variance** (`v' = max(v − f(1 − f), 0) + ε`): +0.0003 / +0.0003 / +0.0002 more (v1 0.3493 → 0.3497,
  v2 0.3419 → 0.3422, s 0.5107 → 0.5109). Small, the same sign on all three models, kept.
- **Exponent and floor**: `b` 0.5 / 1 / 1.5 / 2 and ε 0.01 / 0.1 all within 0.0003: not knife-edge.
- **Censoring guard**: `(1 − p_end)^c` for c 1, 2, 4 and hard drops above p_end 0.3 / 0.5: every variant within
  ±0.0001 of no guard. Why: the far end almost never saturates. On v1 val2017 candidates (best class > 0.05), the
  share of sides with more than 0.3 of their mass on the last bin is 0.02% overall, at most 0.09% (P4, large
  objects); mass on bin 0 is common (6.2% of sides above 0.3; 22% for small objects at P3, 49% at P4) but that is
  an anchor at or outside the edge, whose estimate is right and sharp. `censor_stats.py`, log
  `logs/censor-v1-val.log`.
- **Lattice pooling** (`Π p_i^{w_i}`, `Σ w_i = 1`, on the kept anchor's bin lattice): loses to plain voting and to
  B0 (v1 0.3458, v2 0.3374, s 0.4953 against B0 0.3467 / 0.3391 / 0.5086), worst on folo-s. As the lead expected:
  a power ≠ 1 distorts DFL's interpolated sub-bin position, and members of other levels sit on lattices that are
  offset, not just 2× split (below, Design errors).
- **Cluster τ** around 0.75: 0.7 −0.0005, 0.8 −0.0004, 0.85 −0.0010 (v1); same shape on v2 and s.

**Rung 7** (score from cluster consensus) was not run: time went to Part 4. The cluster oracle (Part 3) measures
box-choice headroom, which a score change does not reach; the class oracle is the closer (also greedy) estimate of
what rescoring from cluster features could do.

**Combinations** (train-dev; `nms` = IoU 0.65 + max_det 1000):

| config | v1 | v2 | folo-s | landed |
|---|---:|---:|---:|---:|
| B0 | 0.3467 | 0.3391 | 0.5086 | 0.3770 |
| nms | 0.3476 | 0.3402 | 0.5085 | 0.3779 |
| vote (τ 0.75, a 1) | 0.3488 | 0.3412 | 0.5102 | 0.3791 |
| vote + nms | 0.3500 | 0.3428 | 0.5106 | 0.3804 |
| vote (τ 0.55, a 1, σ 0.025) + nms | 0.3503 | 0.3429 | 0.5107 | |
| dfl (excess, b 1, ε 0.1) | 0.3497 | 0.3422 | 0.5109 | 0.3798 |
| **dfl + nms = B\*** | **0.3508** | **0.3437** | **0.5112** | **0.3809** |
| dfl + IoU 0.6 + max_det 1000 | 0.3507 | 0.3437 | 0.5102 | 0.3807 |

B* = DFL evidence fusion (τ 0.75, a 1, b 1, ε 0.1, excess variance, no guard) after hard NMS at IoU 0.65 with
max_det 1000. Chosen on v1 / v2 / s; the landed weights (last column) were run with it unchanged and rank the
configurations the same way.

### B* on val2017 (read once)

| config | v1 | v2 | folo-s | landed |
|---|---:|---:|---:|---:|
| B0 | 0.3138 | 0.3168 | 0.4162 | 0.3396 |
| nms | 0.3147 | 0.3177 | 0.4162 | 0.3403 |
| vote + nms | 0.3171 | 0.3194 | 0.4183 | 0.3428 |
| **B\*** | **0.3181** | **0.3202** | **0.4186** | **0.3434** |

AP50 / AP75 of B* against B0: v1 0.4612 / 0.3424 (0.4554 / 0.3392), v2 0.4634 / 0.3457 (0.4578 / 0.3428), folo-s
0.5840 / 0.4537 (0.5798 / 0.4519), landed 0.4922 / 0.3692 (0.4874 / 0.3671). The NMS part moves AP50 (fewer
duplicates suppressed wrongly, more detections kept); fusion moves AP75 and the high-IoU end.

**Rust reproduces it**: `folo val --coco --fusion dfl --iou 0.65 --max-det 1000` (vision/box-fusion) gives
the same AP as the Python post-process (v1 0.3181, folo-s 0.4186, landed 0.3434; `--fusion vote` on v1 0.3171),
detection by detection: same number of detections (v1 1,447,744; folo-s 814,656; landed 1,443,104), no image with a
different (class, score) list, boxes within 0.0036 / 0.0051 / 0.0043 px (Rust computes the side moments in f32,
Python in f64; voting matches to the bit). Script `D:/folo-data/runs-hill3/fusion/bin/repro-fusion.sh`, log
`logs/repro-fusion.log`.

### The operating point (conf 0.25, single-label, val2017)

What `folo predict` shows a user: single-label candidates above 0.25, NMS IoU 0.7, 300 detections (`operating.py`;
truths matched greedily by score per class at IoU 0.5; a detection covering a crowd region by half of its own area is
left out, as COCOeval does). P50 / P75: share of the remaining detections that match a truth at IoU ≥ 0.5 / 0.75;
mean IoU over the true positives at 0.5.

| weights | config | detections | P50 | P75 | R50 | mean IoU of TPs |
|---|---|---:|---:|---:|---:|---:|
| v1 | B0 | 22,842 | 0.7136 | 0.5661 | 0.4305 | 0.8424 |
| v1 | vote | 22,842 | 0.7134 | 0.5689 | 0.4304 | 0.8440 |
| v1 | dfl | 22,842 | 0.7132 | 0.5691 | 0.4303 | 0.8447 |
| v1 | B* | 22,181 | 0.7312 | 0.5848 | 0.4294 | 0.8452 |
| v2 | B0 | 24,165 | 0.6890 | 0.5477 | 0.4413 | 0.8425 |
| v2 | vote | 24,165 | 0.6892 | 0.5506 | 0.4414 | 0.8437 |
| v2 | dfl | 24,165 | 0.6891 | 0.5504 | 0.4413 | 0.8445 |
| v2 | B* | 23,510 | 0.7060 | 0.5648 | 0.4405 | 0.8449 |
| folo-s | B0 | 25,091 | 0.7784 | 0.6461 | 0.5191 | 0.8580 |
| folo-s | vote | 25,091 | 0.7790 | 0.6485 | 0.5194 | 0.8587 |
| folo-s | dfl | 25,091 | 0.7790 | 0.6486 | 0.5195 | 0.8590 |
| folo-s | B* | 24,752 | 0.7881 | 0.6569 | 0.5188 | 0.8593 |
| landed | B0 | 25,926 | 0.6895 | 0.5484 | 0.4722 | 0.8437 |
| landed | vote | 25,926 | 0.6901 | 0.5511 | 0.4726 | 0.8445 |
| landed | dfl | 25,926 | 0.6899 | 0.5511 | 0.4725 | 0.8452 |
| landed | B* | 25,242 | 0.7059 | 0.5651 | 0.4715 | 0.8457 |

(An earlier version of this table matched crowd regions by plain IoU, so most detections inside a crowd region
(830–1,040 per row under the COCO rule) counted as false positives and P50 was 0.025–0.029 lower everywhere. Fixed
in `operating.py`; the IoU and recall columns did not change.)

Fusion alone keeps the detection count (scores and the kept set are NMS's) and moves P50 and recall by at most
0.0006, and it tightens the boxes a user sees: mean IoU of true positives +0.0007 to +0.0023 and P75 +0.0024 to
+0.0030 over B0. DFL and voting are within 0.0003 of each other on P50, P75 and recall (either can be ahead); DFL's
mean IoU of true positives is higher on all four weights (+0.0003 to +0.0008). The NMS IoU 0.65 of B* drops 1.3–2.9%
of the detections (duplicates), which raises P50 by 0.010–0.018 at a recall cost of at most 0.0011.

### The lead's three guesses

- **Excess variance helps**: held, but small: +0.0002–0.0004 AP over plain inverse variance on train-dev, the same
  sign on all three models (kept in B*).
- **Censoring guard is needed**: did not hold. No variant moved AP by more than 0.0001: the last bin almost never
  carries much mass (0.02% of candidate sides above 0.3), so there is nothing for the guard to correct. Not shipped.
- **Lattice pooling loses**: held. Below plain voting and below B0 on all three models, worst on folo-s (−0.013).

## Part 3: oracles (val2017 dump, against B*)

`oracles.py`, each oracle scored with pycocotools against B* itself:

| | v1 | folo-s | landed |
|---|---:|---:|---:|
| (a) recall ceiling: truths with a same-class candidate at IoU ≥ 0.5 | 0.859 | 0.889 | 0.871 |
| (a) the same at IoU ≥ 0.75 | 0.698 | 0.745 | 0.713 |
| (a) any class at IoU ≥ 0.5 / ≥ 0.75 | 0.890 / 0.720 | 0.909 / 0.761 | 0.899 / 0.732 |
| B* AP | 0.3181 | 0.4186 | 0.3434 |
| (b) cluster oracle (greedy, see below): best-IoU member's box, the cluster's top score | 0.3637 (+0.046) | 0.4613 (+0.043) | 0.3892 (+0.046) |
| (c) box oracle, 25% of the way to the truth | 0.3580 (+0.040) | 0.4653 (+0.047) | 0.3850 (+0.042) |
| (c) 50% | 0.4061 (+0.088) | 0.5202 (+0.102) | 0.4355 (+0.092) |
| (c) 100% | 0.4996 (+0.182) | 0.6244 (+0.206) | 0.5323 (+0.189) |
| (d) class oracle | 0.3555 (+0.037) | 0.4483 (+0.030) | 0.3804 (+0.037) |

The cluster oracle adds AP75 0.048–0.053 and AP50 only 0.012–0.014: the boxes are the headroom, not the
detections. It is a greedy oracle, not an upper bound on member selection: each detection's target is the truth its
B* (fused, clipped) box overlaps most, its cluster is drawn around that box rather than around the NMS survivor, and
two detections can pick the same truth where a joint assignment would cover two. A true bound would be at least as
high. Logs `logs/oracles.log`, `results/oracles-<model>.json`.

**What this says.** A learned "consensus glance" over cluster features that only picks the best member box has at
least +0.043–0.046 AP of room by the greedy oracle, and DFL fusion, a training-free weighting of the same members,
takes +0.003–0.004 of it: a small learned module has room (ten times what fusion gets) but only if it can tell which
member is right,
which the scores and the DFL widths evidently do not. Box refinement is the larger lever: moving every kept box just
a quarter of the way to its truth is worth +0.040–0.047 AP, as much as a perfect member choice, and the recall
ceiling (86–89% of truths have a candidate at IoU 0.5, 70–75% at 0.75) says the candidates are mostly there; a
second-stage regression that closes a quarter of the residual edge error is the next thing to try, not more NMS.

## Part 4: the Rust implementation (vision/box-fusion)

`NmsConfig::fusion` (`none` default, `vote`, `dfl`) and `NmsConfig::fusion_iou` (0.75), `folo val` / `folo predict`
`--fusion` and `--fusion-iou`, `folo val --max-det`; formulas in MODEL.md §9.4. All three post-process paths carry
the side precisions:

- full decode (`nms::postprocess_with`, `HeadOutput::detections`'s full path, `predict_raw`): precisions of every
  anchor from `box_raw` (CPU) or from the device moments (GPU);
- CPU candidate path: each candidate's precision computed in the same pass as its decode, from the same softmax;
- device candidate path and `folo val`'s device paths (decode rows, candidate pairs): one extra readback of the side
  moments `[B, 8, A]` (an eighth of the logits), only when `fusion = dfl`.

Tests: hand-built clusters (`fusion.rs`: two members sharp and vague per side, cluster rule at exactly τ, zero
weights, a lone candidate unchanged, a cluster spanning stride 8 and 32, a censored side, determinism); the naive
O(n²) reference extended to fusion (40 random cases, all three modes, exact equality); the CPU candidate path
against the full decode (bit for bit, vote and dfl); GPU candidates against the CPU (float rounding); `folo val`'s
host, decode-rows and candidate-pair paths against the serial loop with `dfl`, f32 and f16; the pairs path with
arbitrary precisions against the full-row NMS; and the Rust result against this Python post-process on three val2017
images committed as a fixture (`crates/folo-vision/tests/fixtures/fusion/dfl.json`, written by
`bench/fusion/make_fixture.py`; the test fails if fusion is switched to voting or the excess term is dropped).

**Shapes and BCCD with the flags on** (`folo val`, the trainer's evaluator):

| model / split | B0 | `--fusion vote` | `--fusion dfl` | B* flags | IoU 0.65 + max_det 1000 only |
|---|---|---|---|---|---|
| Shapes gate folo-n (`runs-hill/verify/gate-amp`), val, mAP50 / mAP50-95 | 0.9785 / 0.8954 | 0.9785 / 0.9026 | 0.9785 / 0.9015 | 0.9788 / 0.9010 | 0.9788 / 0.8945 |
| BCCD folo-n fine-tune (from folo-n-coco, 640 px, batch 16, 100 epochs, seed 0, this PC), test | 0.9008 / 0.6088 | 0.9007 / 0.6137 | 0.9007 / 0.6148 | 0.9010 / 0.6145 | 0.9010 / 0.6087 |

No regression: fusion adds 0.006 (Shapes) and 0.005 (BCCD) mAP50-95 and leaves mAP50 as it is. On Shapes plain
voting is slightly ahead of DFL fusion (one model, 200 images; not a measured difference). Script
`bin/noregress.sh`, logs `logs/noreg-*.json`; the BCCD weights are `D:/folo-data/runs-hill3/fusion/bccd-n-s0/`.

**Timing** (A/B, three rounds alternated, median and min–max; `folo predict` on 200 val2017 images, mean per image
after the first; `folo val` on val2017; `base` = main e5a6ccf0, `off` = vision/box-fusion without flags):

Post-process = `folo predict`'s `postprocess_ms` (decode, readback, NMS, fusion, rescale), ms per image at batch 1,
median (min–max) of three rounds:

| device | conf | base | off | `--fusion dfl` | B* flags |
|---|---|---|---|---|---|
| CPU | 0.25 | 0.16 (0.15–0.24) | 0.15 (0.14–0.20) | 0.17 (0.17–0.18) | 0.18 (0.15–0.21) |
| CPU | 0.001 | 0.55 (0.44–0.57) | 0.57 (0.51–0.58) | 0.69 (0.62–0.71) | 0.82 (0.73–0.95) |
| GPU | 0.25 | 0.47 (0.26–0.71) | 0.22 (0.18–0.30) | 0.46 (0.45–0.60) | 0.44 (0.38–0.51) |
| GPU | 0.001 | 0.39 (0.29–0.40) | 0.41 (0.33–0.49) | 0.71 (0.48–0.84) | 0.58 (0.52–0.88) |

`folo val` on val2017 (GPU, 5000 images, batch 32): base 6.2 s (5.2–9.0), off 4.4 s (4.1–7.7), dfl 8.1 s
(7.7–15.7), B* flags 11.3 s (7.1–15.8).

**Read with care**: the machine was shared (CPU load sampled before each run: median 79%, 4–100%), so the
end-to-end totals (logged, `logs/timing/`) swing by 2× between rounds and are not reported; the post-process column
is the stable one. What it shows: with the flags off the branch costs nothing (base and off overlap everywhere).
`--fusion dfl` adds about 0.01–0.02 ms per image on the CPU at conf 0.25 and 0.1–0.15 ms at conf 0.001 (the
cluster sums; the precisions come from the softmax the decode already does), and about 0.25–0.3 ms per image on the
GPU, which is the extra readback of the side moments (one more synchronous download per batch). In `folo val` that
is +2–4 s on 5000 images; B*'s max_det 1000 adds detections for the evaluator to score on top. A cheaper GPU path
(moments only at the candidates, inside the detect kernel's record) is a leftover; at well under a millisecond per
image it was not worth widening the kernel record this round.

## The counsel's flag: the trainer's evaluator scored unclipped boxes

Right: `val.rs` fed the evaluator network-space boxes that can protrude past the letterbox's image area, while the
COCO path clips to the original image. Fixed in PR #253 (vision/val-clip): the evaluator now scores boxes clipped to
the image area in network pixels (the same box the COCO path scores, before the unmap). Effect on the internal
mAP50-95 of the published weights on val2017: v1 0.309991 → 0.310246 (+0.00025), v2 0.313401 → 0.313577 (+0.00018),
folo-s 0.411677 → 0.411954 (+0.00028), and no COCO number changes. Whether it would have changed which epoch is
`best.folo` in a past run is not measured: three released checkpoints do not bound how the clip's benefit differs
between epochs of one run, and runs keep no per-epoch checkpoints.

## Negative results

- Soft-NMS (any σ), Matrix NMS (Gaussian and linear), per-level score offsets, a DFL softmax temperature: ties or
  losses on train-dev.
- The censoring guard: no effect (nothing to guard).
- Lattice pooling of the distributions: loses to B0.
- Wide voting clusters with flat weights (τ 0.55, a 0): −0.025 AP.

## Leftovers

- The ±2-bin window expectation (+0.0003 on top of B* on train-dev) needs the same change in the GPU detect kernel,
  the CPU decode and the full decode; not done.
- A longer landing run: the internal mAP still rose 0.0004 in the last epoch.
- YOLO11n through the same post-process (how much of the gain is FOLO-specific): not measured.
- Moments only at the candidates (in the GPU detect kernel's record) instead of a `[B, 8, A]` readback: would
  remove most of the 0.25–0.3 ms the GPU path pays.
- `--fusion` in the server's predict API, Studio, ONNX export and folo-wasm (where `dfl` falls back to `vote`): not
  wired.

## Design errors (in the brief, found while measuring)

- "Members of other levels split or merge bins 2×": the edge lattices of different levels are also offset. An
  anchor's centre is at `(i + 0.5) · stride`, so P3 edge positions are ≡ 4 (mod 8), P4 ≡ 8 (mod 16), P5 ≡ 16
  (mod 32): a P4 lattice point falls halfway between two P3 points, not on one. Lattice pooling across levels
  therefore needs interpolation, which adds its own distortion.
- The censoring guard assumes the far end saturates; on COCO at 640 px with `reg_max` 16 the last bin covers
  15 · stride (480 px at P5), and the assigner sends objects that large to P5, so it essentially never does.
