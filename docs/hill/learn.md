# Hill-climb: learning (lab notebook)

Goal: more accuracy per epoch and per GPU-hour for FOLO's trainer, measured against Ultralytics YOLO11n trained the
same way on the same images. Branch `hill/learn`. All runs on the dev machine (RTX 4080 SUPER, shared with four other
agents' benchmarks, so wall times are noisy; results are not).

## Protocol

**Proxy P1 (COCO subset).** `bench/learn/make_coco_proxy.py --train 20000 --val 2000 --long-side 320 --seed 0`:
the first 20,000 train2017 and 2,000 val2017 ids of a seeded shuffle, all 80 classes present, images pre-shrunk to a
long side of 320 (INTER_AREA, JPEG q95) so neither trainer resizes them again. Both trainers read the same files: FOLO
the COCO JSON, Ultralytics the YOLO txt labels written from it (crowd regions skipped, as both do). 143,880 train /
14,659 val boxes. Checked: image sizes equal the JSON's, labels equal the JSON boxes (200 random images per split).

Run: folo-n / `yolo11n.yaml` from scratch, 320 px, 20 epochs, batch 64 (real batch, no accumulation, 313 optimizer
steps per epoch), mosaic closed for the last 5, AMP, seed s:

    folo train --data mini320 --format coco --model folo-n --imgsz 320 --batch 64 --epochs 20 --close-mosaic 5 --amp \
      --workers 10 --device gpu:0 --seed s --patience 1000 [--optimizer sgd]
    python bench/learn/ultra_train.py --model yolo11n.yaml --data mini320.yaml --epochs 20 --imgsz 320 --batch 64 \
      --seed s --workers 10 --set close_mosaic=5 [optimizer=SGD]

Score: COCO AP (pycocotools-exact) of `best.folo` / `best.pt` on the 2,000 proxy val images: `folo val --coco` for
FOLO, `bench/learn/coco_eval.py` (pycocotools) on Ultralytics' `save_json` detections. Wrapper scripts live in
`D:/folo-data/runs-hill/learn/bin` (`p1.sh`, `u1.sh`, `job.sh` around the shared GPU lock).

**Secondary gates.** The shapes gate (MODEL.md §10.6; mAP50 ≥ 0.90, mAP50-95 ≥ 0.75) and an R6 fine-tune from
`folo-n-coco` (shortened to 30 epochs, r6-clean test and r6-fresh).

## Runs

AP is pycocotools AP@[.5:.95] of the best checkpoint on the 2,000 proxy val images (all 20 epochs improved, so best =
last in every run below unless noted). Wall time is per run on the shared GPU (other agents' jobs interleave, so it
is not a speed measure).

| run | trainer | seed | settings (beyond P1) | AP | AP50 | AP_S | AP_M | AP_L | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| base-s0 | FOLO main (c50211ba) | 0 | defaults: `auto` → AdamW lr0 0.000119, β1 0.9 | 0.0164 | 0.030 | 0.005 | 0.022 | 0.049 | 13 min |
| ultra-auto-s0 | Ultralytics 8.4.198 | 0 | defaults: `auto` → AdamW lr0 0.000119 | 0.0166 | 0.031 | 0.006 | 0.026 | 0.042 | 13 min |
| base-sgd-s0 | FOLO main | 0 | `--optimizer sgd` (lr0 0.01, momentum 0.937, warmup_bias_lr 0.1) | 0.0860 | 0.147 | 0.041 | 0.131 | 0.187 | 12 min |
| base-sgd-s1 | FOLO main | 1 | `--optimizer sgd` | 0.0855 | 0.144 | 0.047 | 0.134 | 0.182 | 17 min |
| ultra-sgd-s0 | Ultralytics 8.4.198 | 0 | `optimizer=SGD` (lr0 0.01, momentum 0.937, warmup_bias_lr 0.1) | 0.0853 | 0.144 | 0.043 | 0.136 | 0.175 | 14 min |

| adamw1e3-s0 | FOLO hill/learn | 0 | `--optimizer adamw --lr0 0.001 --set momentum=0.9 --set warmup_bias_lr=0` (= `auto` rule 2's resolved settings) | 0.0747 | 0.125 | 0.034 | 0.118 | 0.176 | 15 min |
| adamw2e3-s0 | FOLO hill/learn | 0 | the same at `--lr0 0.002` | 0.0844 | 0.143 | 0.041 | 0.136 | 0.167 | 17 min |

**Noise floor:** the two FOLO SGD seeds differ by 0.0005 AP (0.0860 / 0.0855); per-epoch curves agree to about
0.001 throughout (epoch 9: 0.043 / 0.042). A difference below about 0.002 AP is treated as noise.

**Equal budget, before any change:** FOLO and Ultralytics are level. Under the shared default (`auto` → AdamW
0.000119) 0.0164 vs 0.0166, with SGD 0.0860 / 0.0855 vs 0.0853. The training losses agree epoch by epoch too
(SGD, epoch 9: box 1.841 / cls 2.710 / dfl 1.659 for FOLO, 1.831 / 2.728 / 1.671 for Ultralytics; epoch 20: 1.719 /
2.268 / 1.576 vs 1.724 / 2.290 / 1.597). So FOLO's trainer learns as well as the reference per epoch on this proxy;
the gap to the published nano numbers is schedule and data, not a trainer defect.

## Round 2: after merging hill/integration

**Merge check** (`8e630a7d`, the four speed branches; SGD P1, same flags as base-sgd-*):

| run | seed | AP (pycocotools) | 20 epochs |
|---|---:|---:|---:|
| merged-sgd-s0 | 0 | 0.0845 | 506 s |
| merged-sgd-s1 | 1 | **0.0814** | 383 s |

Seed 0 is within 0.002 of the pre-merge 0.0860 / 0.0855; seed 1 is 0.0041 below its pre-merge twin. Evaluation is
not the cause: each binary scores each checkpoint identically (pre-merge `base-sgd-s1` weights: 0.0855 with either
binary; `merged-sgd-s1`: 0.0814 with either). Per epoch (internal mAP50-95), merged-s1 runs 0.002–0.005 under
base-s1 from epoch 9 on; merged-s0 tracks base-s0 to epoch 13, dips (0.065 vs 0.069 at 15) and ends at 0.084.
The merged kernels sum in other orders, so merged and pre-merge runs of one seed are different random trajectories,
and the four FOLO SGD results span 0.0814–0.0860: the pre-merge two-seed spread (0.0005) understated the noise,
which looks closer to ±0.002–0.003. A real regression of about 0.003 is possible but not established with two seeds
per side; a third merged seed (or bisecting the four branches with one seed each) would settle it.

### MuSGD (`--optimizer musgd`, then `auto` rule 3)

Implemented from `ultralytics/optim/muon.py` and `engine/trainer.py::build_optimizer` (8.4.198): docs/MODEL.md §8.2
has the update. P1, 20 epochs, `--optimizer musgd` / `optimizer=MuSGD` (lr0 0.01, momentum 0.937, warmup_bias_lr
0.1, as the SGD rows), merged code base (`folo-musgd.exe` = `59293a5f`):

| run | trainer | seed | AP | AP50 | AP_S | AP_M | AP_L | 20 epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| merged-sgd-s0 | FOLO, SGD | 0 | 0.0845 | 0.143 | 0.043 | 0.131 | 0.170 | 506 s |
| merged-sgd-s1 | FOLO, SGD | 1 | 0.0814 | 0.138 | 0.042 | 0.126 | 0.158 | 383 s |
| **musgd-s0** | **FOLO, MuSGD** | 0 | **0.1131** | 0.188 | 0.059 | 0.173 | 0.259 | 415 s |
| **musgd-s1** | **FOLO, MuSGD** | 1 | **0.1155** | 0.187 | 0.059 | 0.174 | 0.261 | 534 s |
| ultra-sgd-s0 (round 1) | Ultralytics, SGD | 0 | 0.0853 | 0.144 | 0.043 | 0.136 | 0.175 | 820 s |
| ultra-musgd-full-s0 | Ultralytics, MuSGD | 0 | 0.1133 | 0.188 | 0.060 | 0.172 | 0.244 | 729 s |

- FOLO MuSGD vs FOLO SGD: +0.031 AP on the means (0.1143 vs 0.0830), ten times the noise; better at every epoch
  from epoch 2 (epoch 10: 0.069 / 0.068 vs 0.046 / 0.044 internal mAP50-95). Every size bucket gains, large
  objects most (+0.10 AP_L).
- FOLO MuSGD vs the reference's MuSGD: 0.1131 / 0.1155 vs 0.1133, level (the reference sits between FOLO's seeds;
  epoch-by-epoch Ultralytics mAP50-95 of the reference 0.0094 / 0.0173 / 0.0312 at epochs 3 / 4 / 5 against FOLO's
  internal 0.013 / 0.022 / 0.033 and 0.010 / 0.019 / 0.031).
- A first, aborted reference run (`ultra-musgd-full-s0`, attempt 1) died of CUDA out-of-memory at epoch 3 while
  another job held GPU memory outside the lock; it was rerun from scratch (same seed, same curve).
- **Step cost** (A/B on the shared GPU, `folo-rule3.exe`, 2-epoch runs alternated SGD, MuSGD, SGD, MuSGD, epoch-2
  img/s): SGD 889 / 1081, MuSGD 765 / 866, means 985 vs 816: MuSGD costs about **17 %** of the step (the Muon
  half is about 15 small matmul dispatches per conv weight per step). It still wins per GPU-hour: MuSGD passes
  SGD's 20-epoch AP (0.083) at about epoch 13 (internal mAP50-95 0.084 / 0.087), i.e. 13 × 1.17 ≈ 15 SGD-epochs of
  time instead of 20, and ends 0.03 AP higher. Fusing the Newton–Schulz iteration (one kernel per bucket) is the
  follow-up that would take most of the 17 % back.

**Shipped: `auto` rule 3** (`AUTO_RULE = 3`): long runs (> 10,000 iterations, e.g. full COCO) pick MuSGD at lr0
0.01, momentum 0.9, as the reference does since 8.4; short runs are rule 2's AdamW. Shapes gate with the rule-3
binary: `auto → adamw lr0 0.00125` (short run, unchanged), best epoch 30 mAP50 **0.978**, mAP50-95 **0.893** (pass;
0.898 before the merge, same settings: the merged kernels' rounding). R6 (2,030 frames × 100 epochs at batch 16 =
3,200 iterations) is a short run and resolves exactly as before. Old checkpoints resume with their rule (1 or 2:
SGD); `--set auto_rule=2` keeps SGD for a new long run. The web mock and FOLO Desktop now compute `auto` with the
same rule (Desktop calls `folo_nn::optim::auto_optimizer`; the web mock mirrors it in TypeScript).

## Reference diff (FOLO vs Ultralytics 8.4.198, step by step)

Read against the installed reference (`ultralytics/utils/loss.py`, `utils/tal.py`, `engine/trainer.py`,
`data/augment.py`, `optim/muon.py`, `cfg/default.yaml`); no divergence that changes results was found.
- **Init / BN:** conv weights PyTorch default (`kaiming_uniform(a=√5)`), BN `eps 1e-3`, `momentum 0.03`: equal. Head
  priors: box bias 2.0 vs 1.0 (any constant gives uniform DFL bins: same decode), cls bias `ln(5 / nc / (640/s)²)`:
  equal.
- **Loss:** BCE sum / `max(Σ T, 1)`, CIoU weighted by the target score, DFL with targets clamped to `reg_max − 1.01`,
  gains 7.5 / 0.5 / 1.5, total `× B`: equal.
- **Assigner:** TAL top-k 10, α 0.5, β 6, candidates strictly inside the gt with sides below 16 px enlarged to 16
  (8.4's `stride_val = stride[1]`), conflicts by the larger IoU, soft targets `m · u_max / m_max`: equal. One tie
  detail: when a gt has fewer than 10 candidates, `torch.topk` may pick zero-metric non-candidates (then masked)
  instead of zero-metric candidates; FOLO always keeps the candidates. Affects only zero-metric anchors.
- **Augmentation:** mosaic centre `U[s/2, 3s/2)`, affine `M = T·Sh·R·C`, box filter (w, h > 2, area ratio > 0.1,
  aspect < 100), HSV gains, flips: equal by reading. Measured on the proxy (4,000 transformed samples each,
  `crates/folo-data/examples/aug_stats.rs` vs `bench/learn/ultra_aug_stats.py`): mosaic 12.74 vs 12.76 boxes per
  image, `sqrt(wh)` quantiles (5/25/50/75/95 %) 5.0 / 12.7 / 27.8 / 60.0 / 132.9 px vs 4.7 / 12.0 / 27.4 / 59.4 /
  134.5, 13.3 % vs 14.6 % of boxes below 8 px; without mosaic 6.71 vs 6.64 boxes, quantiles within 2 %. The
  reference draws the 3 other mosaic images from a buffer of recently loaded ones, FOLO uniformly from the set.
- **Optimizer / schedule:** groups (decay / BN weight / bias), `wd · batch · accumulate / nbs`, Nesterov SGD, warmup
  interpolation (bias LR from `warmup_bias_lr`, momentum from 0.8), linear `lf`, EMA `0.9999 · (1 − e^{−t/2000})`
  per optimizer step, clip 10: equal up to clock granularity (the reference steps `lf` per epoch, calls
  `scheduler.step()` at the start of each epoch and ramps accumulation during warmup; FOLO uses one per-iteration
  clock, MODEL.md §8.4). 8.4 also dropped the 100-iteration warmup floor FOLO keeps; irrelevant here (939 warmup
  iterations).
- **`auto` in 8.4.198:** above 10,000 iterations it is now **MuSGD** (Muon-orthogonalised update at 0.2 · lr plus
  SGD at lr, head class branch at 3 × lr), no longer SGD. Below, AdamW at the same `0.002 · 5 / (4 + nc)` fit.
- **Validation:** both conf 0.001, IoU 0.7, max_det 300, multi-label NMS; the reference letterboxes val batches to
  rectangles, FOLO to squares (at 320 px on ≤ 320 px images the difference is small; scores above are each from its
  own trainer's detections through pycocotools).

## Shipped

**`auto` rule 2: AdamW `lr0 = max(0.002 · 5 / (4 + nc), 0.001)`** (commit `vision: floor auto's AdamW lr0 at 0.001`).
The one large, real effect on the proxy. `auto` (shared with the reference) picks AdamW for any run of at most
10,000 iterations, and its LR fit falls with the class count: 0.000119 for 80 classes, which is 8x below the
reference's own nominal Adam LR (`lr0: 0.01 # SGD=1E-2, Adam=1E-3` in its default.yaml). The fit was made for
fine-tuning; from scratch on 80 classes it barely learns in 20 epochs.

| proxy P1, defaults (`auto`) | AP | vs reference |
|---|---:|---:|
| Ultralytics 8.4.198 YOLO11n | 0.0166 | |
| FOLO before (rule 1) | 0.0164 | −0.0002 (noise) |
| **FOLO after (rule 2: AdamW 0.001)** | **0.0747** | **+0.058, 4.5x** |
| (for scale: either trainer with SGD 0.01) | 0.085–0.086 | |

- The floor binds from `nc = 6` on (`fit(6) = 0.001`), so the shapes gate (4 classes, 0.00125) and R6 (2 classes,
  0.001667) resolve to exactly the settings they had: their results cannot change (checked: the shapes-v2 run below
  logs the same `adamw lr0 0.00125`). Long runs (> 10,000 iterations: full COCO) picked SGD under rule 2 (MuSGD since rule 3, below).
- Old checkpoints resume with the rule they started with: `TrainConfig.auto_rule` (serde default 1 when missing),
  validated to `1..=2`; tests: `folo-nn` `recipe_helpers`, `folo-vision` `train::config::tests::auto_rule_versions`.
- Why 0.001 and not 0.002 (0.0844, level with SGD): a 0.002 floor binds for every class count (`fit(1) = 0.002`),
  so it would also raise shapes' and R6's LR, and R6 is a fine-tune where a higher LR can cost transfer. That needs
  its own gate runs (follow-up).

**`--set KEY=VALUE`** (from the handed-over WIP, now tested and in docs/CLI.md): any `train` key of `config.json`,
dotted, e.g. `--set loss.tal.topk=13`. `TrainConfig.loss` (gains and assigner settings) is new with it; configs
without it get the §7 defaults.

**Gates for rule 2.** Shapes (MODEL.md §10.6, `folo-learn2` = this branch, f32 on the GPU): `auto → adamw lr0
0.00125`, as before; best epoch 29 mAP50 **0.978**, mAP50-95 **0.898**, no non-finite loss: the same as main
(0.978 / 0.898). R6: not rerun. Its two classes resolve to 0.001667 under both rules (`fit(2) > 0.001`), so the
fine-tune is unchanged by construction.

## Not shipped (negative or inconclusive)

- **AdamW floor 0.002** (proxy 0.0844, +0.010 over the 0.001 floor, level with SGD): not shipped, see above; needs
  shapes + R6 at the higher LR first.
- **o2o head as auxiliary supervision:** not run. FOLO, like the reference's YOLOv10/YOLO26 head, detaches the
  one-to-one branch's inputs (MODEL.md §6.6), so its loss never reaches the backbone or the o2m branch: as an
  auxiliary signal for an o2m model it is a no-op by construction. Making it non-detached is an architecture change
  (follow-up, opt-in).
- **Augmentation, loss, assigner, schedule fixes:** none needed; see the reference diff. Both trainers land within
  the seed spread of each other with the same settings.
- Planned but dropped for GPU time (the GPU was shared with four agents; about half of each hour went to their
  jobs): `loss.cls_gain = 1.0`, `loss.tal.topk = 13`, `--cos-lr`, batch 16 × accumulate 4 vs real batch 64. All are
  one `--set` flag away on this branch.

## Follow-ups

1. **Rerun the COCO equal-budget check at 640 px** with the published recipe on both trainers (folo-n vs YOLO11n,
   SGD, the 50-epoch schedule of examples/coco) to confirm that the per-epoch parity seen here holds at full scale;
   on this proxy there is nothing left to fix in the trainer.
2. **MuSGD** (Ultralytics 8.4's `auto` choice for runs over 10,000 iterations, so its long COCO runs no longer use
   plain SGD). FOLO's long-run `auto` is still SGD, which on the proxy is level with the reference's SGD; whether
   MuSGD gains anything was probed on the reference only (`ultra-musgd-s0`: `optimizer=MuSGD`, otherwise exactly
   `ultra-sgd-s0`; stopped after epoch 7 to keep the GPU hold under 25 min). Ultralytics' own val mAP50-95 per epoch:

   | epoch | 3 | 4 | 5 | 6 | 7 |
   |---|---:|---:|---:|---:|---:|
   | SGD | 0.0038 | 0.0094 | 0.0178 | 0.0253 | 0.0270 |
   | MuSGD | 0.0094 | 0.0173 | 0.0312 | 0.0378 | 0.0424 |

   MuSGD is about 1.5x ahead of SGD at epoch 7, a large early-training gain (the end-of-run gap is not measured).
   This is the most promising lever left: implement MuSGD in folo-nn as an opt-in optimizer (Newton–Schulz on every
   conv weight per step: about 15 small matmuls per tensor on the GPU; a second momentum buffer in the checkpoint's
   `folo.optim`), confirm on P1 to epoch 20 against both SGD runs, then make it `auto`'s long-run choice under a
   rule 3.
3. **AdamW floor 0.002** with shapes + R6 gates (proxy +0.010 AP over 0.001).
4. The web mock's copy of the `auto` rule (`web/src/lib/api/mock/runs.ts`) still has rule 1; only its offline
   estimates are affected (the real server resolves with folo-vision).
