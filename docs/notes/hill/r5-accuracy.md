# Hill-climb 5: class-wise rank distillation (lab notebook)

Lane `accuracy` of hill-climb 5 (2026-11-17). Goal: a third distillation term, **class-wise rank distillation**
(docs/MODEL.md §8.9, `distill.rank_gain`), gated at full COCO scale on forks of hill 4's control, then the round's
long run (`folo-n-coco-v4` continued for 64 epochs with the teacher). The design is the Fable 5.2 counsel of this
round (not in the repo; summarised here where it matters); hill 4's distillation is
[r4-distill.md](r4-distill.md). Branch prefix `vision/`; run data `D:/folo-data/runs-hill5/accuracy/`.

## The idea in one paragraph

Hill 3's swap put the teacher's scores on the student's own boxes and gained about +0.03 AP. COCO AP is computed per
class from the order of the scores alone, so that pool is ordering. §8.9's class term `L_kc` asks for equal
probabilities, more than AP reads, and against a supervised target that wants something else at the same anchors.
The rank term asks for what AP reads: per class, over the batch's candidates, the student's logits should be an
increasing linear function of the teacher's (one weighted Pearson correlation per class, invariant to the class's
shift and scale), so the supervised loss keeps the calibration and the teacher supplies only the order.

## Gate step 0 (CPU, before any code)

Hill 3's val2017 candidate dumps at 640 px (`D:/folo-data/runs-hill3/fusion/dumps/land-val` = folo-n-coco-v3, AP
0.340, the student; `.../s-val` = folo-s-coco, 0.416, the teacher), default post-processing (hard NMS 0.7, 300),
pycocotools. Script [`bench/distill/rank.py`](../../../bench/distill/rank.py), log
`D:/folo-data/runs-hill5/accuracy/logs/step0.log` (9 min, 12 processes). Every configuration keeps the
**student's** candidates and boxes and changes only scores at **shared pairs**: the same (anchor, class) scored above
0.001 by both models (2,070,908 of the student's 5,252,237 pairs, 39%, but 85% of its pair-score mass).

| configuration | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Δ AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s`: the student | 0.3396 | 0.4874 | 0.3671 | 0.163 | 0.368 | 0.494 | |
| `vswap`: the teacher's scores | 0.3639 | 0.5226 | 0.3927 | 0.189 | 0.407 | 0.529 | +0.0243 |
| **`rswap` (a): the student's own scores in the teacher's within-class order (whole set)** | **0.3631** | 0.5215 | 0.3915 | 0.190 | 0.407 | 0.526 | **+0.0235** |
| `rswap16`: the same, order per 16-image group | 0.3452 | 0.4947 | 0.3719 | 0.165 | 0.378 | 0.503 | +0.0056 |
| `band_hi` (b): teacher's scores where the student's >= 0.1 | 0.3535 | 0.5052 | 0.3824 | 0.168 | 0.390 | 0.508 | +0.0139 |
| `band_lo` (b): teacher's scores where the student's < 0.1 | 0.3563 | 0.5168 | 0.3843 | 0.187 | 0.396 | 0.523 | +0.0167 |
| `rband_hi`: rank swap within the >= 0.1 band | 0.3545 | 0.5068 | 0.3832 | 0.170 | 0.392 | 0.507 | +0.0149 |
| `rband_lo`: rank swap within the < 0.1 band | 0.3484 | 0.5029 | 0.3760 | 0.179 | 0.378 | 0.504 | +0.0088 |

(c) The weighted Pearson correlation of the two models' class logits, as the term computes it (`τ` 0.005, floored
logits, `rank_min` 8, `ε` 1e-4), per class over groups of 16 consecutive val images: W-weighted mean **0.528** at
γ = 0.5 (0.413 at γ = 1); over the whole set per class 0.608 (0.469). On average 42.4 classes per 16 images are used
(of 59.5 with any candidate). Restricted to elements at 0.1 or more in either model, ρ is 0.22.

**Reading against the counsel's bars.**
- (a) The rank swap keeps +0.0235 of the value swap's +0.0243 (97%): the pool is order, as the invariance says.
  Bar: kill under +0.020. **Passes.** (The counsel's reference +0.031 was the swap on the teacher's candidate set,
  `tS` in r4-distill.md; on the student's own candidates the score swap is +0.024.)
- (b) Above 0.1 the value swap gives 0.0139 of the full 0.0243 (57%; the band gains are not additive: 0.0139 +
  0.0167 = 0.0306); the rank version 0.0149 of 0.0235 (63%). Rule: γ = 1 if two thirds of the gain sits above 0.1,
  else 0.5. **γ = 0.5** (the default) by both readings.
- (c) Kill if the mean ρ is 0.97 or more. It is 0.53: **far from learned**, the term has something to teach. The
  failure mode "fewer than 20 classes used per batch" does not apply on val (42 per 16 images; training batches are
  mosaics of 4 images each, so more).

**Not shown by step 0.** `rswap16` is +0.0056 only: ranking each class within 16 images and keeping the student's
own per-group values loses most of the pool, because AP ranks a class over all 5000 images and the student's
per-group score sets are themselves mis-ordered across groups. The term works per batch, but it is applied over
thousands of batches with the supervised loss holding each class's calibration, so `rswap16` is not a bound on what
training gives, only a reminder that the term teaches within-batch order and relies on consistency across batches.
The dumps are v3 at inference on val, not the e32 / e48 students on augmented training batches.

## Setting `rank_gain` (the gradient probe)

The counsel's rule: over the first 100 fork batches, log the L2 norm of the class-logit gradient of the rank term
and of the supervised loss; if their ratio is outside 0.3-1.0, rescale `λr` once and record it as the default.
`FOLO_DISTILL_PROBE=100` on a throwaway copy of hill 4's control at epoch 32, resumed with `--teacher folo-s-coco
--set distill.rank_gain=2.0`, the first 100 batches of epoch 33, inside one GPU lock hold (script
`D:/folo-data/runs-hill5/accuracy/bin/probe.sh`, norms `logs/probe-g2.0.norms.txt`, summary
`logs/probe-g2.0.stats.txt`; binary `folo-rank-c08fbc7c.exe`):

| class-logit gradient, L2, per batch (B and gains included) | median over 100 batches |
|---|---:|
| supervised (BCE at gain 0.5) | 0.1697 |
| class distillation `L_kc` (gain 0.5) | 0.1294 |
| rank term at `λr` 2.0 | 0.2142 |
| rank / supervised | **1.27** (min 1.08, max 1.56) |

1.27 is outside 0.3-1.0. The counsel meant 2.0 as "half the supervised class gradient", so the rescale aims at a
ratio of 0.5: `λr = 2.0 · 0.5 / 1.27 = 0.79`, rounded to **0.8** (the term is linear in `λr`, so the ratio becomes
0.51). The fork arms use `--set distill.rank_gain=0.8`. (A literal reading, `λr` = 0.5, would give a ratio of 0.32,
inside the band at its lower edge; not run.) `rank_gain` stays 0 (off) in `DistillConfig::default`: whether 0.8
becomes the value `--teacher` starts with depends on the gate.
