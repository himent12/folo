# Wave 6, lane trainfix: four trainer findings from a research report (lab notebook)

Lane `trainfix` of wave 6 (2026-11-21). A research report listed four defects in the trainer (its section 5,
`D:/folo-wave6/research/formula-neon.md`). Each was checked against FOLO's code and against the trainer FOLO was matched
to, Ultralytics 8.4.198 (the copy `bench/finetune` uses). **Result: one finding is a real defect and is fixed (3, off the
default path). Findings 1 and 2 describe behaviour FOLO copied from the reference on purpose; the changes the report
proposes went through a gate on the COCO proxy and did not pass, so they are not shipped. Finding 4's NaN cannot happen
and it got a comment. No default training number changes.** Scratch, scripts and logs:
`D:/folo-wave6/lanes/trainfix/`; run folders: `D:/folo-data/runs-wave6/trainfix/`.

Terms: **AP** is COCO's mAP50-95, scored by `folo val --coco` (pycocotools-exact). **P1** is the 320 px proxy of
[learn.md](learn.md) (`mini320`: 20,000 train and 2,000 val images of COCO; folo-n, 20 epochs, batch 64, close-mosaic 5,
AMP, `auto` optimizer). **o2m** is the default one-to-many head, **o2o** the optional one-to-one head (`--head o2o`).

## Verdicts

| # | the report's claim | verdict | what settles it |
|---|---|---|---|
| 1 | the one-to-many weight anneals from 0.8 to 0.1 although the one-to-one branch is detached | **not a defect**: the reference does exactly this. Only runs with a one-to-one head reach it (`--head o2o`, or a fine-tune of an `o2o` checkpoint); every published COCO run is `o2m`. The proposed weights (1 and 1 while detached) lost | `ultralytics/nn/modules/head.py:196` detaches, `utils/loss.py:1341, 1346` anneals |
| 2a | MuSGD orthogonalises depthwise weights across channels | **not a defect**: the reference routes every rank-2 and rank-4 parameter to Muon. As a candidate improvement: tied alone (`dw`, +0.0003), and +0.0012 with the scale rule (`dws`), under the gate's +0.0030 | `ultralytics/engine/trainer.py:1201` |
| 2b | `muon_scale` should use the flattened matrix | **not a defect**: the reference uses the last two dimensions of the unflattened gradient. As a candidate improvement: lost | `ultralytics/optim/muon.py:106` |
| 3 | the one-to-one branch overwrites the user's `loss.tal.*` | **a defect, fixed**: `--set` promises the key, and the branch silently used the defaults | docs/CLI.md (`--set`), `train/epoch.rs` before this change |
| 4 | AMP clipping overflows to NaN at a loss scale of 2^125 | **no NaN**: the overflow means "no clip", and no logged run came near that scale | below |

## The gate

Written in the brief before any run: a change to default numerics ships only with **at least +0.0030 mean AP over three
seeds and above the control on all three seeds**; a tie is not shipped. All arms were one binary, `origin/main` at
7b8ce7e3 plus environment switches (`exp.patch`, never committed), so the control is today's code. `ofix` is the
branch's build with the fix of finding 3. Each run held the GPU lock for its whole length, so no other lane trained next
to it.

| arm | what changes | AP seed 0 | seed 1 | seed 2 | mean | against the control |
|---|---|---:|---:|---:|---:|---|
| `ctl` | control | 0.1255 | 0.1238 | 0.1249 | 0.1247 | |
| `dw` | depthwise weights out of Muon | 0.1271 | 0.1245 | 0.1235 | 0.1250 | +0.0003, above on 2 of 3: **no pass** |
| `scale` | `muon_scale` on the flattened matrix | 0.1246 | | | | below on seed 0: **no pass** |
| `dws` | both of the above | 0.1270 | 0.1255 | 0.1254 | 0.1260 | +0.0012, above on 3 of 3: **no pass** (needs +0.0030) |
| `octr` | control for `--head o2o` | 0.1072 | | | | |
| `ow11` | `--head o2o`, `w_m = w_o = 1` while detached | 0.1044 | | | | below on seed 0: **no pass** |
| `ofix` | `--head o2o`, finding 3 fixed | 0.1072 | | | | the same loss as `octr` at all 6260 steps |

Runs that could no longer change a verdict were not made. The gate needs the arm above the control on **all three**
seeds, so an arm below it on seed 0 has failed whatever the other two seeds do. That is why some rows have one seed. A
one-seed row says the gate failed; it does not measure how much worse the arm is.

**One control run was disturbed and was replaced.** The first `--head o2o` control, `octl-s0`, scored 0.1068 and left the
path of every other run of that binary and seed at step 2. It ran while this lane's own `cargo test` used the GPU outside
the lock. Its reruns agree to the bit: `repa-s0` and `repb-s0` (2 epochs, 626 steps) with each other and with the fix
build's `repf-s0`, and `octr-s0` (20 epochs) with `ofix-s0`. The contention is the likely cause and was not isolated.
`ow11` is below both controls. The o2m runs were not repeated, so how far a disturbance could have moved them was not
measured. The one seen here moved an o2o run by 0.0005 AP, and `dws` is 0.0018 short of the gate.

Run folders, under `D:/folo-data/runs-wave6/trainfix/`: `ctl-s0..2`, `dw-s0..2`, `dws-s0..2`, `scale-s0`, `octl-s0`,
`octr-s0`, `ow11-s0`, `ofix-s0`, `repa-s0`, `repb-s0`, `repf-s0`.

## Finding 1: the one-to-one weights while the branch is detached

`step_loss` applies `w_m` only when the model has a one-to-one branch. The default head from scratch is `o2m`
(`TrainConfig::default`), for which `out.o2o` is `None` and the loss is the one-to-many loss unweighted. So **no run from
scratch on default flags and no published COCO run reaches this code** (their `config.json` say `"head": "o2m"`). A run
reaches it with `--head o2o`, or by fine-tuning an `o2o` checkpoint without `--head`, because the CLI then takes the
checkpoint's head.

For a run with that head the report's reading of the arithmetic is right: with the branch detached, `w_m` only scales the
gradient of the backbone, the neck and the one-to-many head, and it falls to 0.1. But the reference does the same thing:
its `Detect.forward` detaches the features for the one-to-one head (`head.py:196`) and `E2ELoss` anneals
`o2m` from 0.8 to 0.1 regardless (`loss.py:1341`, `1346`, `1349`). FOLO copied it, so it is not a defect, and the report's
fix (`w_m = w_o = 1` while detached) is a candidate. It went through the gate as arm `ow11` and finished below its control on
seed 0, so it is not shipped. The report's argument did not show up in training.

## Finding 2: MuSGD on depthwise convolutions, and the Muon scale

Both behaviours are the reference's:

- **Routing.** `if param.ndim in {2, 4} and use_muon` (`trainer.py:1201`) sends every conv weight to Muon, depthwise
  ones included. FOLO's `uses_muon` is the same test.
- **Scale.** `scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5` (`muon.py:106`) reads the unflattened gradient,
  so it is 1 for every square kernel, exactly as FOLO's `muon_scale`.

So both are candidates, not defects. Two details the report did not have:

- folo-n has **7 depthwise weights** (`backbone.psa.block0.attn.pe` and six in `head.cls`) and **5 other weights that are
  tall when flattened** (scale 1.12 to 1.41). The report's rule also applies to the depthwise weights, where it is far
  larger: `[256, 1, 3, 3]` flattens to 256 x 9, a factor of 5.33 (2.67 to 5.33 over the seven). So the rule alone
  (`scale`) is a large change to the head's class branch, and the mild version the report describes needs the depthwise
  weights out of Muon first (`dws`).
- Taking a weight out of Muon changes the optimizer's saved groups and buffers, so a shipped version would need a version
  rule for `--resume`, as `auto_rule` has. Nothing here passed, so none was added.

In the `dw` arm the depthwise weights move to their own groups, `dw` and `dw.cls`, with the same LR and weight decay and
only the SGD update at the group's LR.

- `dw` tied the control: +0.0003 on the mean, above it on two seeds of three.
- `scale` alone finished below the control on seed 0, so it failed the gate there.
- `dws` (both) was above the control on all three seeds, by +0.0015, +0.0017 and +0.0005, and **+0.0012 on the mean**. The
  gate asks for +0.0030, so it is not shipped and `auto` is unchanged. With no real effect and independent, fair signs,
  all three seeds land above the control one time in eight, so this is a lead for a longer run and not a result: at the size the
  gate asks for, the report's argument did not show up in training.

## Finding 3: the one-to-one branch and `loss.tal.*`

`train/epoch.rs` built the branch's loss as `LossConfig { tal: TalConfig::o2o(), ..self.loss_cfg }`, so `--set
loss.tal.alpha` (or `beta`, `eps`, `min_box_size`) changed the one-to-many branch and silently not the other, although
docs/CLI.md says `--set` sets the key. The reference has no such setting (`alpha=0.5, beta=6.0` are written into both
assigners, `utils/loss.py:382`), so nothing there decides it; FOLO's own promise does.

The fix (`TalConfig::one_to_one`, `LossConfig::o2o_branch`): the branch takes the run's `alpha`, `beta`, `eps` and
`min_box_size`, and keeps its own `topk = 7`, `topk2 = 1`, which are what makes it one-to-one (MODEL.md §6.6). `loss.tal.topk`
and `topk2` stay the one-to-many branch's.

`--resume` is promised to be bit-exact, and a run started before the fix with a custom `loss.tal` trained this branch on the
defaults. So the behaviour is versioned like `warmup_floor` and `ema_run_fraction`: a new key `o2o_keeps_tal`, on by default for
new runs and false when a stored config does not have it.

Test: `o2o_branch_takes_the_runs_assigner_settings` (`crates/folo-vision/tests/train.rs`). After one optimizer step the
branch's weights depend only on its own loss, so they differ with `loss.tal.beta` exactly when its assigner reads it (before
the fix they did not), and with `o2o_keeps_tal` off they equal the default run's.

With default settings the fix changes nothing: `TalConfig::default().one_to_one()` is `TalConfig::o2o()`. Measured on
the GPU, a 20-epoch `--head o2o` P1 run gives the same loss at all 6260 steps with and without the fix (`octr-s0`,
`ofix-s0`, AP 0.1072 on both). On the CPU, all 1473 saved tensors of a 12-step run are byte-identical.

## Finding 4: AMP clipping at a loss scale of 2^125

`(max_norm / pre_f) as f32` does round to infinity when `max_norm · S` is too far past `f32::MAX` to round back to it.
Three things keep that from being the reported failure:

1. **It is not a NaN.** The threshold is `h · inf = inf` for the positive `h` of a finite gradient, so `n > threshold` is
   false and the step is simply not clipped. The other factor, `k`, is not selected then.
2. **Clipping would need buffers near the top of the `f32` range.** It needs a buffer norm above `max_norm · S`, about
   2^128. The weight gradients are built from f16 input gradients (docs/MODEL.md §8.8), which overflow to infinity at
   65504 and skip the step, and at the scales logged (point 3) the default threshold is far from overflowing. Finite
   buffers above it can be built by hand (two values of 3e38 through `unscale_clip(1 / 65536, 6e33)`), and that step
   stays unclipped. This is a known limit of the GPU path, left alone because this change adds no code to the clipping
   path, which every default run goes through.
3. **No logged run came near that scale.** A fresh default scaler starts at 2^16 and doubles after 2000 finite steps in
   a row, so 2^125 takes at least 218,000 finite optimizer steps (an overflow halves the scale and restarts the count,
   which only adds to that), with every f16 input gradient below about 2^-109 in the original units and a scaled loss
   still finite in `f32`. Zero gradients qualify, so a run with all its loss gains at 0 could get there. As an
   observation, the 720 per-epoch scale logs of this wave's COCO and proxy runs never pass **2^17**.

A very large `--set clip_grad_norm` reaches the same overflow sooner: it happens whenever `(max_norm / pre_f) as f32`
rounds to infinity, as with 1e35 at the fresh AMP scale of 2^16 (and not after five backoffs, at 2^11, or without AMP,
where `S` is 1). Points 1 and 2 hold there too: no NaN, and no clip.

So the fix is a comment at the line (`crates/folo-nn/src/optim/grads.rs`), as the brief asked for this case.

## What was not done

- `scale` and `ow11` have one seed, for the reason under the gate's table.
- The brief asked for a three-seed before/after check that `--head o2o` is not worse with fix 3. It was run on seed 0
  only, because the fix gave the identical run there and on the CPU, and an identical run cannot be worse.
- The gate is one proxy (folo-n, 320 px, 20 epochs). A change that helps only long runs or larger models would not show
  here. Nothing was measured on full COCO.
