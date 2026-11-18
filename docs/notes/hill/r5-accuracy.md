# Hill-climb 5: class-wise rank distillation, the fused distillation kernel and the long run (lab notebook)

Lane `accuracy` of hill-climb 5 (2026-11-17). Goal: a third distillation term, **class-wise rank distillation**
(a `distill.rank_gain` beside docs/MODEL.md §8.9's terms), gated at full COCO scale on forks of hill 4's control,
then the round's long run (`folo-n-coco-v4` continued for 64 epochs with the teacher; "Long run" at the end:
val2017 AP 0.3701, weights not published). **The gate killed the term**
(below), so it is not in main: its code, tests and `bench/distill/rank.py` are only in the closed PR #295 (commits
c08fbc7c, 342699d8, e85a2cbe, a8235fd7, ba874d3b). What the round kept is the fused distillation kernel (#297). The design is the Fable 5.2 counsel of this
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
pycocotools. Script `bench/distill/rank.py` (in closed PR #295, not in main), log
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
  mosaics of 4 images each, so probably more: the rank arms' epoch lines report the training count).

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
inside the band at its lower edge; not run.) The term was off by default (`rank_gain` 0) pending the gate.

## Gate step 1: the fork at epoch 32 (killed)

Copies of hill 4's control at epoch 32 (`D:/folo-data/runs-hill4/coco/n-r5-ctl-e32`, never the original) resumed
with `--teacher folo-s-coco` for epochs 33-36, AMP, seed 0, each arm in one GPU lock hold (script
`D:/folo-data/runs-hill5/accuracy/bin/arm2.sh`, logs `logs/f-*-e32.{arm,train,val}.log`, `logs/step1*.log`):
`f-t4` is hill 4's recipe; `f-rank` adds `--set distill.rank_gain=0.8` (γ = 0.5, binary `folo-rank.exe` =
c08fbc7c; both arms ran it, `f-t4` with the term off). Exported EMA weights, `folo val --coco --save-json`,
pycocotools, paired bootstrap over the 5000 images (`bench/fusion/bootstrap.py`, 1000 resamples, seed 0, log
`logs/gate-step1-bootstrap.log`).

| arm | COCO AP | AP50 | AP75 | internal mAP e33 | e34 | e35 | e36 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f-t4` | 0.3249 | 0.4680 | 0.3503 | 0.31664 | 0.31825 | 0.31989 | 0.32171 |
| `f-rank` | 0.3250 | | | 0.31678 | 0.31824 | 0.31991 | 0.32186 |
| f-rank − f-t4 | **+0.0002** [−0.0003, +0.0009], P(≤0) 0.135 | | | +0.00014 | −0.00001 | +0.00002 | +0.00015 |

- `f-t4` reproduces hill 4's logged fork (internal mAP 0.3217 at epoch 36) to 0.00001, so the arms compare with
  hill 4's numbers too.
- Bar: kill under +0.0005 at epoch 36 of step 1. **+0.0002: killed.** Step 2 (the fork at epoch 48) and the γ = 1
  arm were not run. The long run goes without the term, and the term was not merged (#295 closed).
- The term trained: its loss `1 − ρ` fell from 0.357 (epoch 33) to 0.334 (epoch 36), i.e. the batch-weighted
  student-teacher correlation rose from about 0.64 to 0.67, with 68.8 classes used per batch (the "fewer than 20"
  failure mode does not apply). The class and DFL distillation losses matched the control's to within 1-2%. Raising
  the within-batch correlation by 0.02 did not move COCO AP; step 0's `rswap16` (+0.0056 for a perfect per-group
  order) already said that within-batch order is worth little by itself.
- Cost: not measured cleanly. Epoch throughput under the other lanes' load was 172-245 img/s for `f-rank` and
  105-244 for `f-t4` (whose epochs 33-34 shared the machine with a bench and a build); at epoch 36, 231 vs
  244 img/s. Given the kill, no clean A/B was run.

## The fused distillation kernel (item 4)

train.md's proposal 3a: the class and DFL terms of §8.9 as one folo-gpu op (`folo_gpu::distill`), used on a GPU by
`distill_loss_fused` (`loss/distill/fused.rs`). The tensor path formed the teacher's probabilities and the DFL
targets as `[16, 80, 8400]` / `[16, 4, 16, 8400]` f32 tensors per batch and ran about 50 elementwise commands over
them; the fused op reads the raw heads (the teacher's in f16 under AMP; its class head twice, for the class term
and the DFL weight `q`) in two kernels with fixed-order reductions, combines them in a third and writes the
closed-form gradients in two.

**Correctness** (`crates/folo-vision/tests/gpu_distill.rs`): fused against the tensor path on the GPU and on the CPU,
4 x 80 x 2100, `T` 4 and 1, f32 and f16 teachers. Values within 1e-5 relative (measured at most
5e-7), gradients within 1e-5 of the largest element (measured 2e-7 to 6e-7), backpropagated from 0.37 · total so a
backward that ignores the upstream gradient fails (checked by mutation). Teacher box logits near 2^24 (f32) and 2048
(f16) agree within 1e-7: the first version lost the term there (150.6 against 144.4, Sol's review) because it
divided the raw logits by `T` (WGSL's division is not exactly rounded) and added the log-sum to the maximum before
subtracting; the kernel now subtracts the maximum first. The reference there is the same pattern without the
offset (the softmax ignores it): at 2^24 with `T` = 3 the tensor path itself rounds `z / T` to multiples of 0.5 and
is off by a factor 2.2 (Sol, round 2), while the fused value matches the offset-free one. Non-finite inputs give a
NaN term exactly when the tensor path's term is non-finite (13 cases, f32 and f16 teachers; a teacher class logit
of ±inf or a box bin of −inf gives the finite limit, as on the tensor path), and so do logits at the edge of f32
(4 cases: `z / T` overflowing at `T` 0.5, a side of −3.2e38, below the first version's −3e38 starting maximum).
Round 3 (Sol): the student's shift now divides by `T` first, as the tensor path, so student sides spanning ±2e38
overflow only where the tensor path does (a fifth edge case, term about 1e21); f16 teacher values are decoded bit
by bit, since DX12's `unpack2x16float` did not carry infinities (the f16 cases failed there). The final review
(Sol 6.2 and Opus, on the head rebuilt on main) found four more gaps at the edge of f32, each fixed with a test
case that fails when its fix is reverted (`logs/mutate-r5.log`): a student side whose `z / T − max (z / T)`
overflows was not flagged (the tensor path's element is then `0 · inf` or inf; fused gave a finite 5.9e36); `q`
weighed the sum of the four sides instead of each element (that sum overflowed while the term was about 1e17);
a teacher side wider than f32's range overflowed `z − max z` although `z / T` spans only 5 at `T` 8e37 (such
bins now take the tensor path's `z / T − max (z / T)`; 2-6% off before); and the backward's `B · λ / N`
overflowed in f32 at `λ` 1e38 (now rounded once from f64). All 11 tests pass on Vulkan and DX12. A term whose value itself comes near f32's largest value can still overflow in one path's sums
and not the other's. A teacher output modified in place before the backward is an error; an f16
upstream gradient works; more than 65535 image·class rows fold the grid; bad shapes, CPU tensors, mixed teacher
dtypes and sizes that overflow `usize` are errors or rejected by `fused_supported`.

**Speed**, one lock hold (script `D:/folo-data/runs-hill5/accuracy/bin/ab_fused.sh`, logs `logs/ab/`), base =
origin/main cf65f7a5 release, fused = this branch's code release (`bin/folo-fused.txt`):

| | tensor path | fused | |
|---|---:|---:|---:|
| `kd_step` bench, one COCO batch (B 16, A 8400, f16 teacher): forward + readback + backward, median of 5 alternated rounds of 30 calls; first version / round-1 fixes / round-2 fixes / round-3 fixes / final-review fixes (final) | 5.02 (4.99-5.24) / 5.13 (5.02-5.15) / 4.82 (4.81-4.88) / 5.67 (5.49-6.03) / 6.84 ms (6.49-12.24) | 0.85 (0.80-0.88) / 0.98 (0.97-1.02) / 0.90 (0.88-0.93) / 1.12 (1.11-1.14) / 1.23 ms (1.12-1.63) | 5.9x / 5.2x / 5.3x / 5.1x / 5.6x |
| GPU time of one call (`FOLO_GPU_PROFILE=1`), same five | 4.49 / 4.49 / 4.45 / 4.87 / 4.61 ms in 53 commands | 0.65 / 0.80 / 0.76 / 0.79 / 0.83 ms in 7 commands | |
| full-COCO folo-n AMP training with the folo-s teacher, median step time after step 300, 180 s per arm, 3 alternated rounds | 43.5 / 74.4 / 74.3 ms | 41.6 / 68.9 / 65.5 ms | −4% / −7% / −12% |

The kernel saves about 4 ms of GPU time per step (3.84 ms in the first version's profile, 3.79 ms in the
final one, `logs/ab-r2/`; earlier ones in `logs/ab-r2-a3aa285c/`, `logs/ab-r2-be97acd3/` and `logs/ab-r2-9d508d23/`;
the holds' loads differ, the tensor path read 4.45 to 4.87 ms; the final hold ran with the CPU at 100%). End to end the machine was loaded (CPU at
95-100% in every arm, the other lanes' jobs; round 1 ran at 305-335 img/s, rounds 2-3 at 154-213), so the training
rows are not a clean throughput number: the **median** step time favours the fused path in every round (by more than
the ±3% tie band in rounds 2 and 3), but the **mean** step time (throughput) of round 1 favours base, 47.8 against
52.5 ms (335 against 305 img/s); rounds 2 and 3 favour fused by mean as well (154 / 181 against 175 / 213 img/s).
`bin/ab_steps.py` printed `logs/ab/ab_steps.txt`; the arms' run folders were deleted afterwards, so only that summary
and the arms' console logs (`logs/ab/*.out`, `*.smi`, `*.cpu`) remain. The training A/B ran the first version; the
review fixes (the maximum subtracted before dividing by `T`, the folded class grid, the finer non-finite rules,
the per-element `q`) cost about 0.18 ms of GPU time per call (0.65 -> 0.83 ms; `kd_dfl` 0.38 -> 0.53 ms; the
unchanged `kd_cls_bwd` also read 11% slower in round 2's hold, so part of it is the GPU's state). A first A/B that
ran alongside this lane's own workspace tests was discarded (`logs/ab-noisy-tests/`).

## Long run: `folo-n-coco-v4` continued for 64 epochs with the teacher

**Why a continuation.** The round's first counsel proposed carving folo-n out of `folo-s-coco` (one channel subset
per layer group) as the initialisation of a new 64-epoch run. The Fable 5.2 counsel rejected it before any code was
written: `folo-n-coco-v4` (a real folo-n at 0.356) is a better warm start than any carve, needs no code, and the
carve's proxy gate could not answer the question that decides it. The carve was not built. The counsel expected
0.365-0.370 from a 64-epoch continuation of v4 with the teacher; the lane's target was 0.364 or more, claimed only
with its interval.

**Recipe** (`D:/folo-data/runs-hill5/accuracy/bin/LONG-RUN.md`, `bin/long.sh`; run folder
`long/n-v4-cont-kd`, its `config.json` and `metrics.jsonl`): `folo train` on full COCO from `--weights
folo-n-coco-v4` (all 417 tensors transferred) with `--teacher folo-s-coco` (f16, class gain 0.5, DFL gain 1.5,
`T` 4; the fused kernel of #297 on the GPU; no rank term), 64 epochs, batch 16, 640 px, AMP, seed 0, 16 workers,
`auto` optimizer (rule 5 with `--weights`: MuSGD, Muon / SGD 0.2 / 1.0). The run starts a new learning-rate schedule
(3 warm-up epochs, `lr0` 0.01, `lrf` 0.01) and closes mosaic for its last 10 epochs. Binary `bin/folo-long.exe`:
origin/main ac56a0c4 (#297 merged), release (`bin/folo-long.txt` has its sha256). Started 2026-11-17 23:05:21Z,
`trainer exit 0` at 2026-11-18 05:02:46Z (`long/long.log`): 64 epochs in 21,439 s (about 6 h; 294-492 s per epoch),
on the shared machine (the round's `gpu-training.txt` marker kept GPU speed measurements off it while it ran, but
other work did not stop), so this is not an unloaded cost.

**The trainer's own mAP50-95** (its validation after each epoch, EMA weights; `metrics.jsonl`): 0.3397 at epoch 1,
down to 0.3031 at epoch 3 (likely the warm-up back to `lr0` moving v4's converged weights), back above epoch 1 at epoch 28
(0.3398), 0.3664 at epoch 64. It was still rising at the end: +0.0082 over the last ten epochs, +0.0006 in the last.

**val2017, read once** (the EMA weights of epoch 64 exported unfused as `weights/n-v4-cont-kd-e64.folo`, then
`folo val --coco --save-json eval/dets-long.json` on the GPU, `logs/val-long.log`; `bin/records.py` scored the same
detections with pycocotools, `eval/stats-long.json`, and gives the same numbers):

| model | AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---|---:|---:|---:|---:|---:|---:|
| **long run, epoch 64 (NOT PUBLISHED)** | **0.3701** | 0.5250 | 0.4000 | 0.1789 | 0.4052 | 0.5396 |
| `folo-n-coco-v4` (published), its start | 0.3559 | 0.5087 | 0.3842 | 0.1711 | 0.3880 | 0.5128 |
| hill 4's control, 64 epochs from scratch without the teacher (not published) | 0.3507 | 0.4996 | 0.3762 | 0.1672 | 0.3816 | 0.5043 |

Paired bootstrap over the 5000 images (`bench/fusion/bootstrap.py`, 1000 resamples, seed 0, records of
`bin/records.py`; `logs/long-bootstrap.log`):

| difference | AP | 95% interval | P(diff ≤ 0) |
|---|---:|---|---:|
| **long − v4** | **+0.0141** | [+0.0114, +0.0166] | 0.000 |
| long − control | +0.0194 | [+0.0160, +0.0223] | 0.000 |

The target (0.364) is met, and the gain shows in every AP column.

**What is not shown.**
- No teacher-free continuation of v4 was run, so +0.0141 is 64 more epochs and the teacher together; how much of
  it each gives is not known.
- One training seed: the intervals cover the sampling of the evaluation images, not the training seed.
- The teacher was trained on train2017, the student's training set.
- The trainer's mAP was still rising at epoch 64; a longer continuation is not measured.
- Not measured on these weights: the fused f16 file, the opt-in box fusion flags, int8 (`folo quantize`).
- The run's cost against v4's is not measured (loaded machine, no A/B).

**The weights are not published.** The file is local, `D:/folo-data/runs-hill5/accuracy/weights/n-v4-cont-kd-e64.folo`;
whether to publish it (as `folo-n-coco-v5`) is the owner's decision. Until then the published folo-n is v4, and
docs/BENCHMARKS.md's headline numbers stay v4's 0.356.
