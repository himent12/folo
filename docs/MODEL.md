# FOLO detector: model and training specification (v1)

This is the normative spec for the FOLO detector family (`folo-n/s/m/l/x`), its loss, its training recipe and its
inference path. folo-nn implements the blocks, folo-vision implements the model, head, assigner, loss, trainer and
predictor. If code and this file disagree, one of them is a bug: fix the code, or change this file in the same PR.

Conventions used throughout:
- Tensors are `NCHW`, `f32`. Images enter the model as RGB in `[0, 1]` (`u8 / 255`), no mean/std normalisation.
- `Conv(c1→c2, k, s)` always means the composite block of §2.1 (conv + BatchNorm + SiLU). A bare 2-D convolution with
  a bias and no BN is written `Conv2d`.
- Padding is `k / 2` (integer division) zeros on every side, so `H_out = floor((H + 2·(k/2) − k) / s) + 1`. With odd
  `k` and the stride-aligned inputs below, `H_out = H / s`.
- `split(x, [a, b])` splits along channels, the first `a` channels first. `cat(a, b, …)` concatenates along channels in
  the order written.
- `round(x)` is round-half-away-from-zero (Rust `f64::round`). `ln` is the natural log.
- Input height and width must be multiples of 32 (the largest stride). Training and validation use square inputs;
  prediction may use rectangular ones (§9.2).

## 0. Summary of decisions

| topic | FOLO v1 choice |
|---|---|
| topology | YOLO11-class: Conv stem, C3k2 stages, SPPF, C2PSA (attention at P5 only), PAN-FPN neck, 3 levels (P3/P4/P5, strides 8/16/32) |
| scales | `n s m l x` = YOLO11 depth/width/max-channel multipliers; parameter shapes identical to YOLO11 (verified, §4) |
| head | anchor-free, decoupled box/class branches, DFL box regression with `reg_max = 16`, one-to-many + NMS by default; `head = "o2o"` adds the NMS-free one-to-one branch (dual assignment, predicts without NMS, §6.6) |
| assigner | task-aligned (TAL): `top_k = 10`, `alpha = 0.5`, `beta = 6.0`, CIoU overlaps, small-target box enlargement to 16 px, deterministic ties |
| loss | `7.5 · CIoU + 0.5 · BCE + 1.5 · DFL`, each normalised by the target-score sum, times the batch size |
| optimizer | `auto`: MuSGD (SGD + Nesterov plus a Muon step on conv weights) for every run since rule 4; v1 picked AdamW for short runs, SGD for long ones (§8.2) |
| schedule | linear LR decay to `lrf = 0.01`, 3 warmup epochs (bias LR 0.1 → lr0, momentum 0.8 → 0.937), nominal batch 64 via gradient accumulation |
| EMA | `d = 0.9999 · (1 − exp(−updates / 2000))`, over parameters and BN running stats |
| precision | f32 training (no AMP); f32 or f16 inference after Conv+BN fusion |
| distillation | opt-in (`distill`, §8.9): a frozen teacher's class probabilities and DFL distributions at the shared anchors; off by default, training-only |

## 1. Design rationale

**Topology: adopt YOLO11's, deliberately.** FOLO's risk is the from-scratch framework, not the architecture. YOLO11
is the best published accuracy-per-FLOP point among NMS-based YOLOs (YOLO11n: 39.5 COCO mAP50-95 at 6.5 GFLOPs,
against YOLOv8n's 37.3 at 8.7), and reproducing its parameter shapes exactly buys us an oracle: per-layer parameter
counts and shapes are checked against the reference (done for this spec, §4), and an optional local forward-parity
test can compare FOLO against reference weights (§10.5). Every deviation below is in the recipe, where we gain
determinism or simplicity.

**C3k2 over C2f.** C3k2 is built from the same primitives as C2f (1×1/3×3 Conv, channel split, concat, residual add):
no extra kernels, so no extra implementation cost. Its bottlenecks halve their hidden width (`e = 0.5`) and the
larger scales nest a C3k (two more 3×3s), which is where YOLO11's FLOP saving over YOLOv8 comes from.

**C2PSA (attention) at P5 only.** One attention stage at 20×20 tokens (640 px input) gives the backbone a global
receptive field. The whole C2PSA layer costs 4% of the FLOPs at n/s and 1.5% at l/x, and its two attention matmuls
only 0.06 to 0.37 GFLOPs. It needs batched matmul and a row softmax,
which folo-core and folo-gpu need anyway (GEMM, and the DFL softmax). We do not adopt YOLO26's extra attention in
the neck: more cost and more surface for little measured gain.

**DFL (`reg_max = 16`) over YOLO26-style direct regression.** Direct regression (`reg_max = 1`) shrinks the box output
from 64 to 4 channels per anchor and removes one softmax. But the raw distance it regresses is unbounded and can go
negative early in training, and YOLO26 pairs it with a new loss schedule (ProgLoss) and MuSGD to train well. DFL gives
bounded, non-negative distances (`0..15` strides), a well-conditioned cross-entropy signal and a proven recipe. Its
cost is a 1×1 conv to 64 instead of 4 channels plus a 16-way softmax per side, about 1% of folo-n's FLOPs. `reg_max` stays a config
field so direct regression can be added later without a format break.

**One-to-many head + NMS by default; NMS-free head opt-in.** Dual assignment (YOLOv10/YOLO26) doubles head compute
in training and converges more slowly on small data, and our M2 gate is fast convergence on the synthetic shapes
set. NMS costs well under 1 ms on CPU at `conf = 0.25`. So `head = "o2m"` stays the default and `head = "o2o"`
(§6.6, `folo train --head o2o`) adds the one-to-one branch under names reserved from the start, so v1 files are
unchanged and still load.

**Assigner tweaks.** The TAL values (`top_k 10, alpha 0.5, beta 6.0`) are the reference ones. Two changes: (1) ground
truths smaller than 16 px on a side are enlarged to 16 px *for candidate selection only*, so a tiny object always
contains anchor centres (this is YOLO26's small-target-aware assignment, and it matters for the shapes set). (2) Ties
are broken by index and conflicts are resolved only among the ground truths that actually selected the anchor. The
reference leaves ties to the GPU top-k, and resolves conflicts across all ground truths. Ours is deterministic and
easier to test.

**Optimizer: SGD and AdamW in v1, MuSGD since.** v1 left out MuSGD (YOLO26: a batched Newton–Schulz
orthogonalisation for every conv weight, a lot of new kernel surface for a gain reported mainly on long COCO runs)
and had `auto` pick SGD with Nesterov momentum for long runs and AdamW with the reference's class-count-fitted LR for
short runs and small datasets. MuSGD has since been implemented (§8.2) and measured: it beats both on every run
length tried, fine-tunes included, so `auto` rule 4 picks it everywhere (docs/notes/hill/learn.md, docs/notes/hill/r2-learn.md).

**Tricks that pay for themselves:** mosaic (the biggest single gain on small datasets), closing mosaic for the last
epochs (it removes the train/test distribution gap), EMA (smoother, better validation weights for one extra copy of the weights), LR and
momentum warmup with a high bias LR (stability in the first epoch), gradient-norm clipping at 10 (insurance against
spikes), the nominal batch of 64 with gradient accumulation (the recipe works at any batch size that fits in memory),
and the prior-probability bias init (a sane initial classification loss). **Left out of v1:** mixup, copy-paste,
multi-scale training, rectangular training batches, AMP, label smoothing and test-time augmentation. They are
either small or negative for n/s models, or they add pipeline complexity before we can measure them.

## 2. Building blocks

All blocks are deterministic functions of their inputs and parameters. "Residual" means elementwise addition of the
block input to the block output (same shape).

### 2.1 Conv (conv + BN + activation)
```
Conv(c1→c2, k, s, g=1, act=SiLU):
  y = act(BN(conv2d(x, W, stride=s, padding=k/2, groups=g, bias=none)))
  W: [c2, c1/g, k, k]
```
- **BatchNorm2d**: `eps = 1e-3`, `momentum = 0.03` (PyTorch convention:
  `running = (1 − 0.03) · running + 0.03 · batch_stat`). Training normalises with the batch mean and the *biased*
  batch variance over `N·H·W`. The running variance is updated with the *unbiased* variance (`· n / (n − 1)`).
  Evaluation uses the running statistics. Learnable `weight` (γ) and `bias` (β) per channel. Training rejects any
  batch where some BN layer would see `N·H·W < 2` (impossible with the `imgsz ≥ 64` rule of §8.1).
- **SiLU**: `x · sigmoid(x)`. `act=none` means identity (BN output returned).
- **DWConv(c→c, k)**: `Conv(c→c, k, 1, g=c)`, depthwise, with BN and SiLU.

### 2.2 Bottleneck
```
Bottleneck(c1→c2, k=(3,3), e, shortcut=true):
  h = int(c2 · e)
  y = Conv(h→c2, k[1], 1)(Conv(c1→h, k[0], 1)(x))
  out = x + y   if shortcut and c1 == c2,  else y
```
FOLO only instantiates bottlenecks with `c1 == c2` and `shortcut = true`, so every bottleneck is residual.

### 2.3 C3k
```
C3k(c1→c2, n=2, e=0.5):
  h = int(c2 · e)
  a = Bottleneck(h→h, k=(3,3), e=1.0) applied n times in sequence to Conv(c1→h, 1, 1)(x)    # cv1, block{j}
  b = Conv(c1→h, 1, 1)(x)                                                                  # cv2
  out = Conv(2h→c2, 1, 1)(cat(a, b))                                                       # cv3
```

### 2.4 C3k2
```
C3k2(c1→c2, n, c3k, e):
  c = int(c2 · e)
  y0, y1 = split(Conv(c1→2c, 1, 1)(x), [c, c])                                             # cv1
  for i in 0..n:  y_{i+2} = block_i(y_{i+1})                                               # block{i}
  out = Conv((2+n)·c → c2, 1, 1)(cat(y0, y1, y2, …, y_{n+1}))                              # cv2
  block_i = C3k(c→c, n=2, e=0.5)              if c3k
            Bottleneck(c→c, k=(3,3), e=0.5)   otherwise
```
`c3k` is forced to `true` for every C3k2 in scales `m`, `l` and `x` (see the tables).

### 2.5 SPPF
```
SPPF(c1→c2, k=5):
  h = c1 / 2
  y0 = Conv(c1→h, 1, 1)(x)                                                                 # cv1 (with SiLU)
  y1 = maxpool(y0);  y2 = maxpool(y1);  y3 = maxpool(y2)      # maxpool: kernel 5, stride 1, padding 2
  out = Conv(4h→c2, 1, 1)(cat(y0, y1, y2, y3))                                             # cv2
```
Max-pool padding never wins: padded positions count as −∞ (output size equals input size).

### 2.6 Attention (multi-head self-attention over spatial positions)
```
Attention(C, heads):
  hd = C / heads;  kd = hd / 2;  scale = kd^-0.5;  N = H·W
  qkv = Conv(C → C + 2·kd·heads, 1, 1, act=none)(x)                                        # qkv
  view qkv as [B, heads, 2·kd + hd, N]  (channel index = head·(2·kd + hd) + j)
  q = qkv[:, :, 0:kd, :];  k = qkv[:, :, kd:2kd, :];  v = qkv[:, :, 2kd:2kd+hd, :]
  A[b,h,i,j] = softmax_j( scale · Σ_d q[b,h,d,i] · k[b,h,d,j] )                            # [B, heads, N, N]
  o[b,h,d,i] = Σ_j v[b,h,d,j] · A[b,h,i,j]                                                 # [B, heads, hd, N]
  o, v viewed as [B, C, H, W]  (channel index = head·hd + d)
  out = Conv(C→C, 1, 1, act=none)(o + Conv(C→C, 3, 1, g=C, act=none)(v))                   # proj, pe
```
`heads = max(C / 64, 1)` wherever Attention is used. Softmax subtracts the row maximum before `exp`.

### 2.7 PSABlock and C2PSA
```
PSABlock(c):
  x = x + Attention(c, heads = max(c / 64, 1))(x)                                          # attn
  x = x + Conv(2c→c, 1, 1, act=none)(Conv(c→2c, 1, 1)(x))                                  # ffn1, ffn2
C2PSA(c1→c1, n):
  c = c1 / 2
  a, b = split(Conv(c1→2c, 1, 1)(x), [c, c])                                               # cv1
  b = PSABlock(c) applied n times in sequence to b                                         # block{i}
  out = Conv(2c→c1, 1, 1)(cat(a, b))                                                       # cv2
```

### 2.8 Glue
- **Upsample**: nearest neighbour ×2: `out[y][x] = in[y / 2][x / 2]`.
- **Concat**: along channels, operands in the order of the table's `from` column.
- **DFL decode** (no parameters): for each anchor and side, `p = softmax(logits[0..16])`, distance `Σ_i i · p_i`.
  The projection `[0, 1, …, 15]` is a constant, not a stored tensor.

### 2.9 Initialisation
All tensors are initialised in canonical order (the order the name list of §5 enumerates them, which is module
construction order), each drawing from the model's seeded `folo_core::rng` stream in turn, so the same seed gives the
same model on every backend.
- Every conv weight (including depthwise convs and the head's `Conv2d`s): `U(−b, b)` with `b = 1 / sqrt(fan_in)`,
  `fan_in = (c_in / groups) · k · k`. This is PyTorch's default, `kaiming_uniform(a = √5)`.
- `Conv2d` biases (head outputs only) are overwritten by the priors below. Before that they draw from `U(−b, b)` like
  the weights, so the RNG stream stays aligned.
- BatchNorm: `weight = 1`, `bias = 0`, `running_mean = 0`, `running_var = 1`.
- **Head priors** (§6.5) overwrite the output biases after the draws.

## 3. Scaling and model config

Each layer in §4 is written for a base model. A scale `(depth d, width w, max_channels M)` maps the base arguments to
concrete ones:
- channels: `ch(c) = ceil(min(c, M) · w / 8) · 8`
- repeats: `rep(n) = max(round(n · d), 1)` if `n > 1`, else `n`
- C3k2 `c3k` flag: the base flag, forced to `true` for `m`, `l`, `x`

| scale | depth | width | max_channels | C2PSA heads | P3 / P4 / P5 channels | `c_box` | `c_cls` (nc = 80) |
|---|---:|---:|---:|---:|---|---:|---:|
| n | 0.50 | 0.25 | 1024 | 2 | 64 / 128 / 256 | 64 | 80 |
| s | 0.50 | 0.50 | 1024 | 4 | 128 / 256 / 512 | 64 | 128 |
| m | 0.50 | 1.00 | 512 | 4 | 256 / 512 / 512 | 64 | 256 |
| l | 1.00 | 1.00 | 512 | 4 | 256 / 512 / 512 | 64 | 256 |
| x | 1.00 | 1.50 | 512 | 6 | 384 / 768 / 768 | 96 | 384 |

**Model config** (`folo.config` in `.folo` files, see docs/FORMAT.md). v1 readers require `depth`, `width` and
`max_channels` to equal the row for `scale`, `strides = [8, 16, 32]`, `reg_max = 16`, `head` one of `"o2m"` and
`"o2o"` (§6.6), and `in_channels = 3`:
```json
{ "family": "folo", "spec": 1, "scale": "n", "depth": 0.5, "width": 0.25, "max_channels": 1024,
  "num_classes": 80, "in_channels": 3, "strides": [8, 16, 32], "reg_max": 16, "head": "o2m",
  "bn_eps": 0.001, "bn_momentum": 0.03 }
```

## 4. Layer tables

Base layers (index, `from`, block, base arguments). `-1` means the previous layer:
```
 0 -1      Conv  (64, k3, s2)                 P1/2        12 [11,6]  Concat
 1 -1      Conv  (128, k3, s2)                P2/4        13 -1      C3k2 (512, n2, c3k=no, e=0.5)
 2 -1      C3k2  (256, n2, c3k=no, e=0.25)                14 -1      Upsample ×2
 3 -1      Conv  (256, k3, s2)                P3/8        15 [14,4]  Concat
 4 -1      C3k2  (512, n2, c3k=no, e=0.25)                16 -1      C3k2 (256, n2, c3k=no, e=0.5)    → P3 out
 5 -1      Conv  (512, k3, s2)                P4/16       17 -1      Conv (256, k3, s2)
 6 -1      C3k2  (512, n2, c3k=yes, e=0.5)                18 [17,13] Concat
 7 -1      Conv  (1024, k3, s2)               P5/32       19 -1      C3k2 (512, n2, c3k=no, e=0.5)    → P4 out
 8 -1      C3k2  (1024, n2, c3k=yes, e=0.5)               20 -1      Conv (512, k3, s2)
 9 -1      SPPF  (1024, k5)                               21 [20,10] Concat
10 -1      C2PSA (1024, n2)                               22 -1      C3k2 (1024, n2, c3k=yes, e=0.5)  → P5 out
11 -1      Upsample ×2                                    23 [16,19,22] Detect
```

The tables below are computed (not hand-written) for `num_classes = 80` and a 640×640 input. `params` counts
learnable tensors only (conv weights, BN γ/β, head biases); BN running statistics are buffers and not counted. GFLOPs
are `2 ×` the multiply–accumulates of all convolutions plus the two attention matmuls; biases, BN, activations,
pooling, upsampling and adds are excluded. Ultralytics' tool counts slightly more, so it reports 6.6 / 21.7 / 68.5 /
87.6 / 196.0. The per-layer parameter counts and output shapes match Ultralytics' YOLO11 models exactly for every
scale, and the totals equal YOLO11's trainable ("gradients") counts.

| scale | params (nc = 80) | params (nc = 4) | params (nc = 1) | GFLOPs @640 | GFLOPs @320 | named tensors (params + buffers) |
|---|---:|---:|---:|---:|---:|---|
| folo-n | 2,624,064 | 2,590,604 | 2,590,019 | 6.54 | 1.62 | 255 + 162 |
| folo-s | 9,458,736 | 9,429,324 | 9,428,163 | 21.59 | 5.37 | 255 + 162 |
| folo-m | 20,114,672 | 20,056,076 | 20,053,763 | 68.10 | 17.00 | 330 + 212 |
| folo-l | 25,372,144 | 25,313,548 | 25,311,235 | 87.15 | 21.74 | 513 + 334 |
| folo-x | 56,966,160 | 56,878,380 | 56,874,915 | 195.27 | 48.75 | 513 + 334 |

The head's class width `c_cls = max(C_P3, min(nc, 100))` depends on `nc`, so the class count changes more than the
output layer. Only row 23 (head) differs between class counts. Output shapes are `C × H × W`. At 320 px every `H` and
`W` halves.

#### folo-n

| # | name | from | block | arguments | output at 640 | params | GFLOPs |
|---:|---|---|---|---|---|---:|---:|
| 0 | `backbone.stem` | -1 | Conv | 3→16, k3 s2 | 16 × 320 × 320 | 464 | 0.088 |
| 1 | `backbone.down2` | -1 | Conv | 16→32, k3 s2 | 32 × 160 × 160 | 4,672 | 0.236 |
| 2 | `backbone.stage2` | -1 | C3k2 | 32→64, n=1, c3k=no, e=0.25 | 64 × 160 × 160 | 6,640 | 0.328 |
| 3 | `backbone.down3` | -1 | Conv | 64→64, k3 s2 | 64 × 80 × 80 | 36,992 | 0.472 |
| 4 | `backbone.stage3` | -1 | C3k2 | 64→128, n=1, c3k=no, e=0.25 | 128 × 80 × 80 | 26,080 | 0.328 |
| 5 | `backbone.down4` | -1 | Conv | 128→128, k3 s2 | 128 × 40 × 40 | 147,712 | 0.472 |
| 6 | `backbone.stage4` | -1 | C3k2 | 128→128, n=1, c3k=yes, e=0.5 | 128 × 40 × 40 | 87,040 | 0.275 |
| 7 | `backbone.down5` | -1 | Conv | 128→256, k3 s2 | 256 × 20 × 20 | 295,424 | 0.236 |
| 8 | `backbone.stage5` | -1 | C3k2 | 256→256, n=1, c3k=yes, e=0.5 | 256 × 20 × 20 | 346,112 | 0.275 |
| 9 | `backbone.sppf` | -1 | SPPF | 256→256, k5 | 256 × 20 × 20 | 164,608 | 0.131 |
| 10 | `backbone.psa` | -1 | C2PSA | 256→256, n=1, heads=2 | 256 × 20 × 20 | 249,728 | 0.259 |
| 11 | — | 10 | Upsample | ×2 nearest | 256 × 40 × 40 | 0 | 0 |
| 12 | — | 11, 6 | Concat | channels | 384 × 40 × 40 | 0 | 0 |
| 13 | `neck.td4` | -1 | C3k2 | 384→128, n=1, c3k=no, e=0.5 | 128 × 40 × 40 | 111,296 | 0.354 |
| 14 | — | 13 | Upsample | ×2 nearest | 128 × 80 × 80 | 0 | 0 |
| 15 | — | 14, 4 | Concat | channels | 256 × 80 × 80 | 0 | 0 |
| 16 | `neck.out3` | -1 | C3k2 | 256→64, n=1, c3k=no, e=0.5 | 64 × 80 × 80 | 32,096 | 0.406 |
| 17 | `neck.down3` | -1 | Conv | 64→64, k3 s2 | 64 × 40 × 40 | 36,992 | 0.118 |
| 18 | — | 17, 13 | Concat | channels | 192 × 40 × 40 | 0 | 0 |
| 19 | `neck.out4` | -1 | C3k2 | 192→128, n=1, c3k=no, e=0.5 | 128 × 40 × 40 | 86,720 | 0.275 |
| 20 | `neck.down4` | -1 | Conv | 128→128, k3 s2 | 128 × 20 × 20 | 147,712 | 0.118 |
| 21 | — | 20, 10 | Concat | channels | 384 × 20 × 20 | 0 | 0 |
| 22 | `neck.out5` | -1 | C3k2 | 384→256, n=1, c3k=yes, e=0.5 | 256 × 20 × 20 | 378,880 | 0.301 |
| 23 | `head` | 16, 19, 22 | Detect | in 64/128/256, c_box=64, c_cls=80 | 144 × 8400 | 464,896 | 1.869 |
| | **total** | | | | | **2,624,064** | **6.54** |

#### folo-s

| # | name | from | block | arguments | output at 640 | params | GFLOPs |
|---:|---|---|---|---|---|---:|---:|
| 0 | `backbone.stem` | -1 | Conv | 3→32, k3 s2 | 32 × 320 × 320 | 928 | 0.177 |
| 1 | `backbone.down2` | -1 | Conv | 32→64, k3 s2 | 64 × 160 × 160 | 18,560 | 0.944 |
| 2 | `backbone.stage2` | -1 | C3k2 | 64→128, n=1, c3k=no, e=0.25 | 128 × 160 × 160 | 26,080 | 1.311 |
| 3 | `backbone.down3` | -1 | Conv | 128→128, k3 s2 | 128 × 80 × 80 | 147,712 | 1.887 |
| 4 | `backbone.stage3` | -1 | C3k2 | 128→256, n=1, c3k=no, e=0.25 | 256 × 80 × 80 | 103,360 | 1.311 |
| 5 | `backbone.down4` | -1 | Conv | 256→256, k3 s2 | 256 × 40 × 40 | 590,336 | 1.887 |
| 6 | `backbone.stage4` | -1 | C3k2 | 256→256, n=1, c3k=yes, e=0.5 | 256 × 40 × 40 | 346,112 | 1.101 |
| 7 | `backbone.down5` | -1 | Conv | 256→512, k3 s2 | 512 × 20 × 20 | 1,180,672 | 0.944 |
| 8 | `backbone.stage5` | -1 | C3k2 | 512→512, n=1, c3k=yes, e=0.5 | 512 × 20 × 20 | 1,380,352 | 1.101 |
| 9 | `backbone.sppf` | -1 | SPPF | 512→512, k5 | 512 × 20 × 20 | 656,896 | 0.524 |
| 10 | `backbone.psa` | -1 | C2PSA | 512→512, n=1, heads=4 | 512 × 20 × 20 | 990,976 | 0.911 |
| 11 | — | 10 | Upsample | ×2 nearest | 512 × 40 × 40 | 0 | 0 |
| 12 | — | 11, 6 | Concat | channels | 768 × 40 × 40 | 0 | 0 |
| 13 | `neck.td4` | -1 | C3k2 | 768→256, n=1, c3k=no, e=0.5 | 256 × 40 × 40 | 443,776 | 1.416 |
| 14 | — | 13 | Upsample | ×2 nearest | 256 × 80 × 80 | 0 | 0 |
| 15 | — | 14, 4 | Concat | channels | 512 × 80 × 80 | 0 | 0 |
| 16 | `neck.out3` | -1 | C3k2 | 512→128, n=1, c3k=no, e=0.5 | 128 × 80 × 80 | 127,680 | 1.625 |
| 17 | `neck.down3` | -1 | Conv | 128→128, k3 s2 | 128 × 40 × 40 | 147,712 | 0.472 |
| 18 | — | 17, 13 | Concat | channels | 384 × 40 × 40 | 0 | 0 |
| 19 | `neck.out4` | -1 | C3k2 | 384→256, n=1, c3k=no, e=0.5 | 256 × 40 × 40 | 345,472 | 1.101 |
| 20 | `neck.down4` | -1 | Conv | 256→256, k3 s2 | 256 × 20 × 20 | 590,336 | 0.472 |
| 21 | — | 20, 10 | Concat | channels | 768 × 20 × 20 | 0 | 0 |
| 22 | `neck.out5` | -1 | C3k2 | 768→512, n=1, c3k=yes, e=0.5 | 512 × 20 × 20 | 1,511,424 | 1.206 |
| 23 | `head` | 16, 19, 22 | Detect | in 128/256/512, c_box=64, c_cls=128 | 144 × 8400 | 850,352 | 3.199 |
| | **total** | | | | | **9,458,736** | **21.59** |

#### folo-m

| # | name | from | block | arguments | output at 640 | params | GFLOPs |
|---:|---|---|---|---|---|---:|---:|
| 0 | `backbone.stem` | -1 | Conv | 3→64, k3 s2 | 64 × 320 × 320 | 1,856 | 0.354 |
| 1 | `backbone.down2` | -1 | Conv | 64→128, k3 s2 | 128 × 160 × 160 | 73,984 | 3.775 |
| 2 | `backbone.stage2` | -1 | C3k2 | 128→256, n=1, c3k=yes, e=0.25 | 256 × 160 × 160 | 111,872 | 5.662 |
| 3 | `backbone.down3` | -1 | Conv | 256→256, k3 s2 | 256 × 80 × 80 | 590,336 | 7.550 |
| 4 | `backbone.stage3` | -1 | C3k2 | 256→512, n=1, c3k=yes, e=0.25 | 512 × 80 × 80 | 444,928 | 5.662 |
| 5 | `backbone.down4` | -1 | Conv | 512→512, k3 s2 | 512 × 40 × 40 | 2,360,320 | 7.550 |
| 6 | `backbone.stage4` | -1 | C3k2 | 512→512, n=1, c3k=yes, e=0.5 | 512 × 40 × 40 | 1,380,352 | 4.404 |
| 7 | `backbone.down5` | -1 | Conv | 512→512, k3 s2 | 512 × 20 × 20 | 2,360,320 | 1.887 |
| 8 | `backbone.stage5` | -1 | C3k2 | 512→512, n=1, c3k=yes, e=0.5 | 512 × 20 × 20 | 1,380,352 | 1.101 |
| 9 | `backbone.sppf` | -1 | SPPF | 512→512, k5 | 512 × 20 × 20 | 656,896 | 0.524 |
| 10 | `backbone.psa` | -1 | C2PSA | 512→512, n=1, heads=4 | 512 × 20 × 20 | 990,976 | 0.911 |
| 11 | — | 10 | Upsample | ×2 nearest | 512 × 40 × 40 | 0 | 0 |
| 12 | — | 11, 6 | Concat | channels | 1024 × 40 × 40 | 0 | 0 |
| 13 | `neck.td4` | -1 | C3k2 | 1024→512, n=1, c3k=yes, e=0.5 | 512 × 40 × 40 | 1,642,496 | 5.243 |
| 14 | — | 13 | Upsample | ×2 nearest | 512 × 80 × 80 | 0 | 0 |
| 15 | — | 14, 4 | Concat | channels | 1024 × 80 × 80 | 0 | 0 |
| 16 | `neck.out3` | -1 | C3k2 | 1024→256, n=1, c3k=yes, e=0.5 | 256 × 80 × 80 | 542,720 | 6.921 |
| 17 | `neck.down3` | -1 | Conv | 256→256, k3 s2 | 256 × 40 × 40 | 590,336 | 1.887 |
| 18 | — | 17, 13 | Concat | channels | 768 × 40 × 40 | 0 | 0 |
| 19 | `neck.out4` | -1 | C3k2 | 768→512, n=1, c3k=yes, e=0.5 | 512 × 40 × 40 | 1,511,424 | 4.823 |
| 20 | `neck.down4` | -1 | Conv | 512→512, k3 s2 | 512 × 20 × 20 | 2,360,320 | 1.887 |
| 21 | — | 20, 10 | Concat | channels | 1024 × 20 × 20 | 0 | 0 |
| 22 | `neck.out5` | -1 | C3k2 | 1024→512, n=1, c3k=yes, e=0.5 | 512 × 20 × 20 | 1,642,496 | 1.311 |
| 23 | `head` | 16, 19, 22 | Detect | in 256/512/512, c_box=64, c_cls=256 | 144 × 8400 | 1,472,688 | 6.650 |
| | **total** | | | | | **20,114,672** | **68.10** |

#### folo-l

| # | name | from | block | arguments | output at 640 | params | GFLOPs |
|---:|---|---|---|---|---|---:|---:|
| 0 | `backbone.stem` | -1 | Conv | 3→64, k3 s2 | 64 × 320 × 320 | 1,856 | 0.354 |
| 1 | `backbone.down2` | -1 | Conv | 64→128, k3 s2 | 128 × 160 × 160 | 73,984 | 3.775 |
| 2 | `backbone.stage2` | -1 | C3k2 | 128→256, n=2, c3k=yes, e=0.25 | 256 × 160 × 160 | 173,824 | 8.808 |
| 3 | `backbone.down3` | -1 | Conv | 256→256, k3 s2 | 256 × 80 × 80 | 590,336 | 7.550 |
| 4 | `backbone.stage3` | -1 | C3k2 | 256→512, n=2, c3k=yes, e=0.25 | 512 × 80 × 80 | 691,712 | 8.808 |
| 5 | `backbone.down4` | -1 | Conv | 512→512, k3 s2 | 512 × 40 × 40 | 2,360,320 | 7.550 |
| 6 | `backbone.stage4` | -1 | C3k2 | 512→512, n=2, c3k=yes, e=0.5 | 512 × 40 × 40 | 2,234,368 | 7.130 |
| 7 | `backbone.down5` | -1 | Conv | 512→512, k3 s2 | 512 × 20 × 20 | 2,360,320 | 1.887 |
| 8 | `backbone.stage5` | -1 | C3k2 | 512→512, n=2, c3k=yes, e=0.5 | 512 × 20 × 20 | 2,234,368 | 1.783 |
| 9 | `backbone.sppf` | -1 | SPPF | 512→512, k5 | 512 × 20 × 20 | 656,896 | 0.524 |
| 10 | `backbone.psa` | -1 | C2PSA | 512→512, n=2, heads=4 | 512 × 20 × 20 | 1,455,616 | 1.403 |
| 11 | — | 10 | Upsample | ×2 nearest | 512 × 40 × 40 | 0 | 0 |
| 12 | — | 11, 6 | Concat | channels | 1024 × 40 × 40 | 0 | 0 |
| 13 | `neck.td4` | -1 | C3k2 | 1024→512, n=2, c3k=yes, e=0.5 | 512 × 40 × 40 | 2,496,512 | 7.969 |
| 14 | — | 13 | Upsample | ×2 nearest | 512 × 80 × 80 | 0 | 0 |
| 15 | — | 14, 4 | Concat | channels | 1024 × 80 × 80 | 0 | 0 |
| 16 | `neck.out3` | -1 | C3k2 | 1024→256, n=2, c3k=yes, e=0.5 | 256 × 80 × 80 | 756,736 | 9.647 |
| 17 | `neck.down3` | -1 | Conv | 256→256, k3 s2 | 256 × 40 × 40 | 590,336 | 1.887 |
| 18 | — | 17, 13 | Concat | channels | 768 × 40 × 40 | 0 | 0 |
| 19 | `neck.out4` | -1 | C3k2 | 768→512, n=2, c3k=yes, e=0.5 | 512 × 40 × 40 | 2,365,440 | 7.550 |
| 20 | `neck.down4` | -1 | Conv | 512→512, k3 s2 | 512 × 20 × 20 | 2,360,320 | 1.887 |
| 21 | — | 20, 10 | Concat | channels | 1024 × 20 × 20 | 0 | 0 |
| 22 | `neck.out5` | -1 | C3k2 | 1024→512, n=2, c3k=yes, e=0.5 | 512 × 20 × 20 | 2,496,512 | 1.992 |
| 23 | `head` | 16, 19, 22 | Detect | in 256/512/512, c_box=64, c_cls=256 | 144 × 8400 | 1,472,688 | 6.650 |
| | **total** | | | | | **25,372,144** | **87.15** |

#### folo-x

| # | name | from | block | arguments | output at 640 | params | GFLOPs |
|---:|---|---|---|---|---|---:|---:|
| 0 | `backbone.stem` | -1 | Conv | 3→96, k3 s2 | 96 × 320 × 320 | 2,784 | 0.531 |
| 1 | `backbone.down2` | -1 | Conv | 96→192, k3 s2 | 192 × 160 × 160 | 166,272 | 8.493 |
| 2 | `backbone.stage2` | -1 | C3k2 | 192→384, n=2, c3k=yes, e=0.25 | 384 × 160 × 160 | 389,760 | 19.818 |
| 3 | `backbone.down3` | -1 | Conv | 384→384, k3 s2 | 384 × 80 × 80 | 1,327,872 | 16.987 |
| 4 | `backbone.stage3` | -1 | C3k2 | 384→768, n=2, c3k=yes, e=0.25 | 768 × 80 × 80 | 1,553,664 | 19.818 |
| 5 | `backbone.down4` | -1 | Conv | 768→768, k3 s2 | 768 × 40 × 40 | 5,309,952 | 16.987 |
| 6 | `backbone.stage4` | -1 | C3k2 | 768→768, n=2, c3k=yes, e=0.5 | 768 × 40 × 40 | 5,022,720 | 16.043 |
| 7 | `backbone.down5` | -1 | Conv | 768→768, k3 s2 | 768 × 20 × 20 | 5,309,952 | 4.247 |
| 8 | `backbone.stage5` | -1 | C3k2 | 768→768, n=2, c3k=yes, e=0.5 | 768 × 20 × 20 | 5,022,720 | 4.011 |
| 9 | `backbone.sppf` | -1 | SPPF | 768→768, k5 | 768 × 20 × 20 | 1,476,864 | 1.180 |
| 10 | `backbone.psa` | -1 | C2PSA | 768→768, n=2, heads=6 | 768 × 20 × 20 | 3,264,768 | 2.969 |
| 11 | — | 10 | Upsample | ×2 nearest | 768 × 40 × 40 | 0 | 0 |
| 12 | — | 11, 6 | Concat | channels | 1536 × 40 × 40 | 0 | 0 |
| 13 | `neck.td4` | -1 | C3k2 | 1536→768, n=2, c3k=yes, e=0.5 | 768 × 40 × 40 | 5,612,544 | 17.931 |
| 14 | — | 13 | Upsample | ×2 nearest | 768 × 80 × 80 | 0 | 0 |
| 15 | — | 14, 4 | Concat | channels | 1536 × 80 × 80 | 0 | 0 |
| 16 | `neck.out3` | -1 | C3k2 | 1536→384, n=2, c3k=yes, e=0.5 | 384 × 80 × 80 | 1,700,352 | 21.706 |
| 17 | `neck.down3` | -1 | Conv | 384→384, k3 s2 | 384 × 40 × 40 | 1,327,872 | 4.247 |
| 18 | — | 17, 13 | Concat | channels | 1152 × 40 × 40 | 0 | 0 |
| 19 | `neck.out4` | -1 | C3k2 | 1152→768, n=2, c3k=yes, e=0.5 | 768 × 40 × 40 | 5,317,632 | 16.987 |
| 20 | `neck.down4` | -1 | Conv | 768→768, k3 s2 | 768 × 20 × 20 | 5,309,952 | 4.247 |
| 21 | — | 20, 10 | Concat | channels | 1536 × 20 × 20 | 0 | 0 |
| 22 | `neck.out5` | -1 | C3k2 | 1536→768, n=2, c3k=yes, e=0.5 | 768 × 20 × 20 | 5,612,544 | 4.483 |
| 23 | `head` | 16, 19, 22 | Detect | in 384/768/768, c_box=96, c_cls=384 | 144 × 8400 | 3,237,936 | 14.588 |
| | **total** | | | | | **56,966,160** | **195.27** |

## 5. Parameter names (`.folo`)

Names are dotted paths: `<layer prefix>.<sub-block path>.<tensor>`. Layer prefixes are the `name` column of §4 plus
the head names below. Sub-block names come from the block definitions in §2 (the `# name` comments).

| module | tensors |
|---|---|
| Conv | `<p>.conv.weight`, `<p>.bn.weight`, `<p>.bn.bias`, `<p>.bn.running_mean`, `<p>.bn.running_var` (the last two are buffers; `num_batches_tracked` is not stored) |
| Conv2d (head outputs) | `<p>.weight`, `<p>.bias` |
| Bottleneck | `<p>.cv1.*`, `<p>.cv2.*` |
| C3k | `<p>.cv1.*`, `<p>.cv2.*`, `<p>.cv3.*`, `<p>.block{j}.*` (j = 0, 1) |
| C3k2 | `<p>.cv1.*`, `<p>.cv2.*`, `<p>.block{i}.*` (i = 0 .. n−1) |
| SPPF | `<p>.cv1.*`, `<p>.cv2.*` |
| C2PSA | `<p>.cv1.*`, `<p>.cv2.*`, `<p>.block{i}.attn.qkv.*`, `.attn.proj.*`, `.attn.pe.*`, `.ffn1.*`, `.ffn2.*` |
| Detect, per level `L ∈ {p3, p4, p5}` | box: `head.box.L.conv0.*`, `head.box.L.conv1.*` (Conv), `head.box.L.out.*` (Conv2d). Class: `head.cls.L.dw0.*`, `head.cls.L.pw0.*`, `head.cls.L.dw1.*`, `head.cls.L.pw1.*` (Conv), `head.cls.L.out.*` (Conv2d) |
| One-to-one branch (`head = "o2o"`, §6.6), per level `L` | the same tensors under `head.o2o.`: `head.o2o.box.L.{conv0,conv1,out}.*`, `head.o2o.cls.L.{dw0,pw0,dw1,pw1,out}.*` |

Examples: `backbone.stem.conv.weight [16, 3, 3, 3]`, `backbone.stage2.block0.cv1.conv.weight [8, 16, 3, 3]` (folo-n),
`backbone.stage4.block0.block1.cv2.bn.running_var [32]` (folo-n, C3k inside C3k2),
`backbone.psa.block0.attn.pe.conv.weight [128, 1, 3, 3]` (folo-n), `head.cls.p5.out.bias [80]`.

Canonical order (initialisation, `named_parameters()` iteration, golden-file tests) is the table order, and within a
block the order its sub-blocks appear in §2: `cv1, cv2, block0, …` for C3k2 and C2PSA; `cv1, cv2, cv3, block0,
block1` for C3k; `attn.qkv, attn.proj, attn.pe, ffn1, ffn2` for PSABlock; for each head level `p3, p4, p5` in turn,
`box.conv0, box.conv1, box.out, cls.dw0, cls.pw0, cls.dw1, cls.pw1, cls.out`, then (with `head = "o2o"`) the same
for `o2o.box.*` / `o2o.cls.*`, level by level; so the first tensors of an `o2o` model are drawn exactly as those of
the `o2m` model with the same seed. Within a Conv: `conv.weight,
bn.weight, bn.bias, bn.running_mean, bn.running_var`. Within a Conv2d: `weight, bias`. Weight files sort tensors by
the safetensors layout rule of FORMAT.md, which is unrelated to this order.

**Class-dependent tensors.** `head.cls.{p3,p4,p5}.out.{weight,bias}` (and `head.o2o.cls.*.out.*`) are the classification output layers referred to
by FORMAT.md's partial-loading rule. When `nc` changes `c_cls`, `head.cls.*.pw0/dw1/pw1/out` change shape too, and
the shape check of partial loading skips them.

**Fused files** (`folo.fused = "true"`): every Conv becomes `<p>.conv.weight` and `<p>.conv.bias` and has no `bn.*`
tensors. Head `Conv2d`s are unchanged.

Implementation: `folo_vision::model` (`FoloModel`, the blocks in `model::blocks`, the head in `model::head`).
`crates/folo-vision/tests/model.rs` checks §4's per-layer parameter counts and output shapes for every scale and the
§10.1 / §10.2 cases; `tests/golden/folo-n-names.txt` is the reviewed `(name, shape)` list of folo-n (`nc = 80`),
regenerated with `FOLO_BLESS=1` when a rename is intended.

## 6. Head

### 6.1 Branches
For each level `L` with input channels `C_L` (P3, P4, P5 outputs of rows 16, 19, 22), with
`c_box = max(16, C_P3 / 4, 4 · reg_max)` and `c_cls = max(C_P3, min(nc, 100))` (both use `C_P3` at every level):
```
box_L = Conv2d(c_box → 4·reg_max, 1×1, bias)( Conv(c_box→c_box, 3, 1)( Conv(C_L→c_box, 3, 1)(x_L) ) )
cls_L = Conv2d(c_cls → nc, 1×1, bias)( Conv(c_cls→c_cls, 1, 1)( DWConv(c_cls, 3)(
          Conv(C_L→c_cls, 1, 1)( DWConv(C_L, 3)(x_L) ) ) ) )
```

### 6.2 Anchors
Level `l` with stride `s_l ∈ {8, 16, 32}` has a grid `H_l × W_l = (H / s_l) × (W / s_l)`. Anchor points are cell centres
in grid units, `(x + 0.5, y + 0.5)`. Anchors are numbered level-major (P3, then P4, then P5), row-major within a level:
`a = offset_l + y · W_l + x`, `offset_0 = 0`, `offset_{l+1} = offset_l + H_l · W_l`. In pixels an anchor is
`((x + 0.5) · s_l, (y + 0.5) · s_l)`. At 640×640 there are `6400 + 1600 + 400 = 8400` anchors; at 320×320, 2100.
Anchors depend only on the input size and are recomputed whenever it changes.

### 6.3 Raw output layout (training)
- `box_raw: [B, 4·reg_max, A]`: the per-level `box_L` maps flattened (`[B, 64, H_l·W_l]`) and concatenated along the
  anchor axis. Channel `side · 16 + bin`, side order `(left, top, right, bottom)`.
- `cls_raw: [B, nc, A]`: class logits, concatenated the same way.

These two tensors (plus the level shapes) are the model's training output and the loss's input.

### 6.4 Decode (inference)
For anchor `a` at grid point `(a_x, a_y)` with stride `s`:
```
d_side = Σ_{i=0..15} i · softmax(box_raw[side·16 .. side·16+16, a])_i          side ∈ {l, t, r, b}
x1 = (a_x − d_l) · s     y1 = (a_y − d_t) · s     x2 = (a_x + d_r) · s     y2 = (a_y + d_b) · s
score[c] = sigmoid(cls_raw[c, a])
```
The decoded output is `[B, A, 4 + nc]`, anchor-major (`x1, y1, x2, y2, score_0 … score_{nc−1}`) in input-image
pixels. Anchor-major keeps the NMS scan contiguous; a GPU decode kernel writes this layout directly.

### 6.5 Bias priors
After random init (§2.9):
- `head.box.L.out.bias = 2.0` for every element. Any constant gives uniform bins, so the initial distance is 7.5
  strides on every side.
- `head.cls.L.out.bias = ln(5 / nc / (640 / s_L)²)`, which encodes a prior of about 5 objects per 640×640 image spread
  over the level's cells. For `nc = 80`: −11.5366 (P3), −10.1503 (P4), −8.7641 (P5). For `nc = 4`: −8.5409,
  −7.1546, −5.7683. The 640 is a constant, independent of the training image size.

### 6.6 One-to-one (NMS-free) branch, `head = "o2o"`
As in YOLOv10 / YOLO26's end-to-end mode. `head = "o2o"` models (`folo train --head o2o`, `TrainConfig.head`) have
both branches and are trained with dual assignment; `head = "o2m"` files (every v1 file) are unchanged.
- A second copy of both branches, `head.o2o.box.L.*` and `head.o2o.cls.L.*`, with the same structure and priors
  (drawn after the o2m branch, §5). Its input features are detached (stop-gradient) during training, so only the
  o2m branch trains the backbone and neck. Opt-in: `TrainConfig.o2o_attached_epochs = N` leaves them attached for
  the first `N` epochs, so the one-to-one loss trains the backbone and neck too (YOLO26's published run carries a
  `detach_epoch = 10` that Ultralytics 8.4.198 does not have; default 0, always detached). Fine-tuning an `o2o` model
  from `o2m` weights copies the o2m head into the o2o branch (as YOLOv10 initialises it).
- Loss: `L = w_m · L_o2m + w_o · L_o2o`. `L_o2o` is §7 with `top_k = 7` and a second filter `top_k2 = 1` after
  conflict resolution: each ground truth keeps only its highest-metric positive (ties: smaller anchor); the anchors
  it drops become background, and its soft target (§7.3 step 5, over that single positive) is about its IoU.
  Weights after `e` completed epochs: `w_m = max(1 − e / max(E − 1, 1), 0) · (0.8 − 0.1) + 0.1`, `w_o = 1 − w_m`
  (starting at `w_m = 0.8`, ending at 0.1). The reported loss items (§7.4) are the same weighted sums of the two
  branches' items. With `amp` (§8.8) both branches run under autocast and both losses in f32 on f32 copies of their
  outputs, like the o2m loss (`tests/gpu_amp.rs` trains a dual head under amp against the f32 run).
- Inference uses o2o only, with no NMS, and computes only that branch: take the `k = min(max_det, A)` anchors with
  the highest max-class score (ties: smaller anchor), then the top `k` (anchor, class) pairs among them (ties: the
  anchor's rank, then the class), then apply `conf` (validation: `conf = 0.001`, §9.3). Single-label prediction
  (`predict`, §9.3's `multi-label: no`) keeps each of the `k` anchors once, with its best class (first maximum),
  then applies `conf`: the same detections, minus the second class of an anchor, exactly as o2m prediction is
  single-label. `iou` is unused. Anchors whose best score is not `> conf` are dropped before ranking, which changes
  nothing (they contribute no pair above `conf`).
- The branch is a switch on the model (`FoloModel::set_head_mode`, `folo predict/val --head o2m`), so the same
  `o2o` model can also predict with its o2m branch and NMS (to compare them). An `o2m` model has no o2o branch.
- Export: `.folo` exports keep both branches. ONNX exports the predicting branch only; `--decode` of an o2o model
  writes end-to-end `detections: [B, k, 6]` (`x1, y1, x2, y2, score, class`, the multi-label selection above
  without `conf`), docs/CLI.md.
- Implementation: `folo_vision::model::head` (`Detect::forward_o2o`, `HeadMode`), `assign::TalConfig::o2o` /
  `topk2`, `loss::dual_loss` / `o2m_weight`, `nms::topk_detections`; on a GPU the second filter is a step of the
  device assigner (`tal_targets.wgsl`, `K2`). Tests: `crates/folo-vision/tests/o2o.rs`.

## 7. Loss

Inputs: `box_raw`, `cls_raw`, the anchors, and the ground truth of each image `i`: `G_i` boxes, each a class id and
an `xyxy` box in input pixels (converted from the loader's normalised `cx, cy, w, h` by multiplying by the input
width/height). folo-data drops degenerate boxes, so every box has `w, h > 0`. Images may have no boxes.

### 7.1 Predictions used by the assigner and the loss
- `pd_box[a]`: the §6.4 decode in **grid units** (no `· s`), from `box_raw` with gradient.
- For the assigner only: `pd_score = sigmoid(cls_raw)` and `pd_box · s` (pixels), both **detached**.

### 7.2 CIoU
For boxes `b1, b2` in `xyxy`, with `eps = 1e-7`:
```
w1 = x2₁ − x1₁,  h1 = y2₁ − y1₁ + eps,  w2 = x2₂ − x1₂,  h2 = y2₂ − y1₂ + eps
inter = max(min(x2₁, x2₂) − max(x1₁, x1₂), 0) · max(min(y2₁, y2₂) − max(y1₁, y1₂), 0)
union = w1·h1 + w2·h2 − inter + eps;  iou = inter / union
cw = max(x2₁, x2₂) − min(x1₁, x1₂);  ch = max(y2₁, y2₂) − min(y1₁, y1₂);  c2 = cw² + ch² + eps
rho2 = ((x1₂ + x2₂ − x1₁ − x2₁)² + (y1₂ + y2₂ − y1₁ − y2₁)²) / 4
v = (4 / π²) · (atan(w2 / h2) − atan(w1 / h1))²
α = v / (1 − iou + v + eps)                 # treated as a constant: no gradient flows through α
CIoU = iou − (rho2 / c2 + v · α)
```
folo-vision provides exactly this form, including the `eps` placements, as `boxes::ciou_eps`; the assigner uses it.
`boxes::iou(.., IouKind::Ciou)` is a different, eps-free variant (identical boxes give exactly 1) for general use, not
for the assigner or the loss. The loss also needs a differentiable version of `ciou_eps` for `pd_box`.

### 7.3 Task-aligned assigner
Per image, for ground truths `g` and anchors `a` (anchor points in pixels), with `top_k = 10`, `alpha = 0.5`,
`beta = 6.0`, `eps = 1e-9`:
1. **Candidates.** Copy the gt box; if its width or height is below `16` px (the stride of P4), set that side to 16 px
   about the box centre. This enlarged box is used for this step only. `a` is a candidate of `g` iff the anchor point
   is strictly inside it by more than `eps` on all four sides: `ax − x1 > 1e-9`, `ay − y1 > 1e-9`, `x2 − ax > 1e-9`,
   `y2 − ay > 1e-9`.
2. **Metric** (candidates only): `u(g,a) = max(CIoU(gt_g, pd_box_px[a]), 0)` and
   `m(g,a) = pd_score[a, class_g]^0.5 · u(g,a)^6`. Non-candidates have `u = m = 0`.
3. **Top-k.** Each `g` selects the `min(top_k, #candidates)` candidates with the largest `m`, ties broken by smaller
   anchor index. Zero-metric candidates are eligible.
4. **Conflicts.** An anchor selected by several gts is kept only for the one with the largest `u(g,a)` among those
   gts, ties broken by smaller gt index. Now every anchor has at most one gt: `fg[a]`, `gt[a]`.
5. **Soft targets.** For each `g` with final positive set `P_g`: `m_max = max_{a∈P_g} m(g,a)`,
   `u_max = max_{a∈P_g} u(g,a)`. For `a ∈ P_g`: `t[a] = m(g,a) · u_max / (m_max + 1e-9)`. The class target is
   `T[a, c] = t[a]` if `c = class_g`, else 0. Background anchors have `T[a, ·] = 0`.
6. **Box targets.** `tb[a] = gt_g` (original, not enlarged), converted to grid units by dividing by the anchor's
   stride.

An image with no gt has every anchor background. The assigner runs without autograd (`no_grad`).

Implementation: `folo_vision::assign` (`assign`, `TalConfig` with these defaults and `min_box_size = 16`,
`candidate_box`, `dfl_targets`, `dfl_bins`), one image at a time on plain slices, everything in input pixels.
`crates/folo-vision/tests/assign_cases.rs` has a hand-checked test for each step above.

### 7.4 Terms
Over the whole batch (all images, all anchors, all classes):
```
tss    = max(Σ_{i,a,c} T, 1)                                     # target score sum, floored at 1
L_cls  = Σ_{i,a,c} BCE(cls_raw, T) / tss                         # BCE(x, t) = max(x, 0) − x·t + ln(1 + exp(−|x|))
L_box  = Σ_{fg} t[a] · (1 − CIoU(pd_box[a], tb[a])) / tss        # grid units
L_dfl  = Σ_{fg} t[a] · DFL(a) / tss
DFL(a) = (1/4) Σ_{side} −( w_l · log_softmax(box_raw[side, ·, a])[j] + w_r · log_softmax(...)[j + 1] )
         target distance δ = clamp(ltrb_side, 0, reg_max − 1 − 0.01) = clamp(…, 0, 14.99)
         ltrb = (a_x − x1, a_y − y1, x2 − a_x, y2 − a_y) of tb[a] in grid units
         j = floor(δ),  w_l = j + 1 − δ,  w_r = 1 − w_l
L      = B · (7.5 · L_box + 0.5 · L_cls + 1.5 · L_dfl)           # B = images in this batch
```
- `t[a]` is a constant (no gradient). Gradients flow into `cls_raw` (BCE) and `box_raw` (CIoU through the DFL
  expectation, and DFL).
- With no foreground in the batch, `L_box = L_dfl = 0` and `L_cls = Σ BCE(x, 0)`.
- Reported loss items (CLI, API `Step.loss`): `box = 7.5·L_box`, `cls = 0.5·L_cls`, `dfl = 1.5·L_dfl`,
  `total = box + cls + dfl`. These are not multiplied by `B`. Epoch values are running means over the epoch's batches.

Implementation: `folo_vision::loss` (`detection_loss`, `LossConfig` with these weights and `reg_max = 16`).
- `targets_from_normalized` turns the loader's `[batch_idx, class, cx, cy, w, h]` rows into per-image `xyxy` pixel
  boxes (`ImageTargets`).
- `assign_batch` is §7.1 and §7.3: it reads `box_raw` / `cls_raw` once on the host (borrowed on the CPU, downloaded
  from a device: the loss's only full-size device-to-host transfer; CIoU's `atan` also round-trips `N_fg` values and
  the items read back four scalars), decodes boxes to pixels, takes the sigmoid of the score columns of the classes
  present, and runs `assign::assign` per image, images in parallel. The result (`BatchAssignment`) lists the
  foreground anchors in image-then-anchor order with `t[a]`, class and gt box.
- `loss_with_assignment` is §7.4 for that fixed assignment. `Σ BCE(cls_raw, T)` is computed as
  `Σ_bg softplus(x) + Σ_fg (softplus(x) − x · t)` (`T` is zero off the foreground), each foreground term formed
  before summing so confident positives do not cancel against the dense sum. `LossOutput::is_finite` also checks
  the batch-scaled total, which can overflow when the items do not. The foreground box logits are gathered
  once by flat index into `[N_fg, 4, reg_max]`; `pd_box` is the softmax expectation in grid units, `loss::ciou` is
  `boxes::ciou_eps` in tensor ops (α under `no_grad`), and DFL gathers `log_softmax` at bins `j, j + 1` with weights
  `t[a] · w / 4`. On the CPU the BCE sum and the gather are fused custom ops with hand-written backwards; other
  devices use the equivalent tensor-op composite. Large sums are fixed trees of row sums, so results are bitwise
  reproducible.
- If any head output is NaN or infinite, `detection_loss` skips the assigner and returns a non-finite loss
  (`LossOutput::is_finite() == false`, `total` a detached NaN), which the trainer skips (§8.3).
- **On a GPU** (head outputs on a folo-gpu device, `topk <= 64`, `reg_max <= 32`) `detection_loss` runs all of the above as kernels
  (`folo_gpu::detect`, docs/GPU.md) and downloads nothing but the reported items: the decode, §7.3 step by step (each
  gt's candidates scored and ranked in one workgroup, conflicts by atomics, then the soft targets and a dense
  per-anchor target map), the BCE over every logit (each element formed whole), CIoU + DFL per positive with the
  backward written out by hand (α constant, min / max ties split as folo-core's autograd does), and a fixed-order
  reduction. `detection_loss_host` keeps the host assigner on any device. `crates/folo-vision/tests/gpu_loss.rs`
  checks the device assignment against the host one (the same positives, gts and labels; `t` to 1e-5) on several
  sizes, class counts and settings, and the items and both gradients against the host loss. The positives are
  listed per gt slot rather than per anchor and the sums run in another (fixed) order, so values agree to float
  rounding, not bit for bit.
- `crates/folo-vision/tests/loss_cases.rs` checks the terms against an `f64` reference, the loss cases of §10.3, a
  finite-difference gradient check of the whole loss, CIoU against `ciou_eps`, and determinism.

## 8. Training recipe

### 8.1 Defaults
| key | default | notes |
|---|---|---|
| `epochs` | 100 | |
| `batch` | 16 | images per forward/backward |
| `imgsz` | 640 | square, multiple of 32, at least 64 for training (so P5 is at least 2×2 and every BN channel sees several values) |
| `nbs` | 64 | nominal batch: gradients accumulate over `accumulate = max(round(nbs / batch), 1)` batches |
| `optimizer` | `auto` | `sgd`, `adamw`, `musgd` or `auto` (§8.2) |
| `lr0` | 0.01 | base LR (ignored by `auto`) |
| `lrf` | 0.01 | final LR = `lr0 · lrf` |
| `momentum` | 0.937 | SGD momentum / AdamW β1 |
| `weight_decay` | 0.0005 | scaled with the batch (§8.2) |
| `warmup_epochs` | 3.0 | |
| `warmup_floor` | 0 | warmup floor in iterations (§8.4); 100 in configs written before the key |
| `warmup_momentum` | 0.8 | |
| `warmup_bias_lr` | 0.1 | 0.1 under `auto` rule 4; 0.0 when an older `auto` rule chooses |
| `cos_lr` | false | linear decay by default |
| `close_mosaic` | 10 | epochs at the end with mosaic off, at most half of the run (§8.6) |
| `ema` | true | `false`: no EMA (§8.5) |
| `clip_grad_norm` | 10.0 | max global gradient L2 norm; `0` disables clipping |
| `patience` | 100 | early stop after this many epochs without a fitness gain |
| `seed` | 0 | |
| `distill` | none | distillation from a frozen teacher (§8.9); absent: none |

### 8.2 Optimizer and parameter groups
Three groups, assigned by tensor name:
- `decay`: every conv weight (`*.conv.weight` and the head `Conv2d` `*.out.weight`), weight decay `wd`.
- `bn`: BatchNorm weights (`*.bn.weight`), weight decay 0.
- `bias`: BatchNorm biases (`*.bn.bias`) and head `Conv2d` biases (`*.out.bias`), weight decay 0.

Effective weight decay: `wd = weight_decay · batch · accumulate / nbs`.

`auto` (rule 5): MuSGD (below) with `lr0 = 0.01`, `momentum = 0.937` and `warmup_bias_lr = 0.1` for every run,
short or long, from scratch or from pretrained weights; from scratch with the step weights `muon = sgd = 0.5`
(`folo_nn::optim::AUTO_MUSGD_SCRATCH_WEIGHTS`, the defaults of Ultralytics' `MuSGD` class), from pretrained weights
with `musgd.muon` / `musgd.sgd` (default 0.2 / 1.0, the reference trainer's). Whether a run starts from weights is
`TrainConfig.transfer`, which the trainer sets from `--weights` and stores with the run, so a resume resolves the
same. An explicit `sgd`/`adamw`/`musgd` uses `lr0`, `momentum`, `warmup_bias_lr` and `musgd.*` as given. Rule 4
is rule 5 with `musgd.*` for every run.

Why the step weights (docs/notes/hill/r2-learn.md): from scratch 0.5 / 0.5 trains folo-n on the COCO proxy to AP 0.1234
against 0.1135 with 0.2 / 1.0 (three seeds each, every seed above every other; Ultralytics 8.4.198's own MuSGD on
the same images, one run in round 1: 0.1133), and the shapes gate to mAP50-95 0.934 against 0.926 (one seed); fine-tuning
`folo-n-coco-v2` on BCCD it does worse (test AP 0.616 against 0.625, three seeds), so fine-tunes keep 0.2 / 1.0.

Why (docs/notes/hill/r2-learn.md; three seeds per side unless marked "one seed"): the reference trainer (Ultralytics
8.4.198, and FOLO's rule 3) zeroes the bias warmup LR whenever `auto` chooses, which suits AdamW at `lr0 ≈ 0.001` but
costs MuSGD about 10% on the COCO proxy (MuSGD at momentum 0.9: AP 0.1021 with `warmup_bias_lr = 0`, 0.1118 with
0.1); and below 10,000 iterations it picks AdamW, which MuSGD beats in every short case measured: the COCO proxy
(6,260 iterations from scratch, AP 0.0747 for AdamW, one seed, from docs/notes/hill/learn.md, against 0.112), the shapes
gate (§10.6, mAP50-95 0.892 against 0.926, one seed) and fine-tuning `folo-n-coco-v2` on BCCD (test AP 0.607
against 0.625). On the other fine-tuning sets of docs/BENCHMARKS.md it is level: medical-pills 0.807 against
0.804, signature 0.9885 against 0.9892 (a saturated set); brain-tumor: see docs/notes/hill/r2-learn.md.

Rule 3: `iterations = ceil(N_train / max(batch, nbs)) · epochs`. If `iterations > 10000`: MuSGD with `lr0 = 0.01`,
`momentum = 0.9`, as the reference trainer picks since Ultralytics 8.4. Otherwise: AdamW with
`lr0 = max(round(0.002 · 5 / (4 + nc), 6), 0.001)` and `β1 = 0.9` (e.g. 0.00125 for `nc = 4`, 0.001 from `nc = 6`
on, so 0.001 for `nc = 80`). In both cases `warmup_bias_lr = 0.0`.

`TrainConfig.auto_rule` (current: 5) versions this rule: rules 5, 4 and 3 are the above; rule 2 picks SGD (`lr0 = 0.01`,
`momentum = 0.9`) instead of MuSGD for long runs; rule 1 is rule 2 without the AdamW floor. On the COCO proxy of
docs/notes/hill/learn.md (20 epochs, SGD regime forced) MuSGD reaches 1.38x the AP of SGD (0.1143 vs 0.0830), as the
reference's own MuSGD does. That was measured with `--optimizer musgd` at momentum 0.937 and warmup_bias_lr 0.1;
rule 3 ships momentum 0.9 and warmup_bias_lr 0 (the proxy's 6,260 iterations and the shapes gate resolve AdamW). The
shipped rule-3 recipe has run once at full scale: folo-s on COCO (examples/coco, folo-s; 56 epochs, 103,544
optimizer steps, AMP) resolved `auto` to MuSGD at lr0 0.01, momentum 0.9, warmup_bias_lr 0, trained without a
non-finite batch and reached val2017 AP 0.416. That shows the recipe works at scale; with no SGD twin at that scale
it does not measure MuSGD's gain over SGD on full COCO. The 0.001 floor is FOLO's (rule 2 on). The reference trainer's fit alone (rule 1)
gives 80 classes 0.000119, and a from-scratch run on a 20k-image COCO subset (20 epochs, 6,260 iterations) then
reaches about a fifth of the AP of SGD on the same budget, in Ultralytics as in FOLO (docs/notes/hill/learn.md). Rule 2
changes nothing below 6 classes. Settings saved before the key existed (a checkpoint's `folo.train`, a run's
`config.json`) mean rule 1, so a run resumes with the rule it started with; `--set auto_rule=1` (or `2` to `4`;
CLI) selects an older rule for a new run.

**SGD** (PyTorch semantics, Nesterov, no dampening), per parameter `p` with gradient `g`:
```
g = g + wd · p                     (decay group only)
buf = g  on the first step,  else  buf = μ · buf + g
p = p − lr · (g + μ · buf)
```
**AdamW** (decoupled decay), `β2 = 0.999`, `eps = 1e-8`, step counter `t` starting at 1:
```
p = p · (1 − lr · wd)              (decay group only)
m = β1·m + (1 − β1)·g;   v = β2·v + (1 − β2)·g²
p = p − lr · (m / (1 − β1^t)) / (sqrt(v / (1 − β2^t)) + eps)
```
**MuSGD** (`optimizer = musgd`; Ultralytics 8.4's `MuSGD` with its trainer's groups, `folo_nn::optim::MuSgd`). Six
groups: the three above, each split into the head's class branches (`head.cls.*`, `head.o2o.cls.*`: groups
`decay.cls`, `bn.cls`, `bias.cls`) at `3 · lr0` and the rest at `lr0`; the schedule (§8.4) treats `bias.cls` as a
bias group. Every group takes the SGD update above (Nesterov, `momentum_buffer`). Before it, every parameter of
rank 2 or 4 in a `decay` group (every conv weight) also takes a Muon step with the group's current `lr` and `μ`:
```
m = μ · m + (1 − μ) · g                      (muon_buffer, starts at zero)
u = μ · m + (1 − μ) · g                      (Nesterov)
X = u as [d0, d1·d2·d3] / (‖u‖_F + 1e-7), transposed if it has more rows than columns
5 times:  A = X·Xᵀ;  X = 3.4445·X + (−4.7750·A + 2.0315·A·A)·X      (then transposed back)
p = p − 0.2 · lr · sqrt(max(1, d₋₂ / d₋₁)) · X        (the factor is 1 for square kernels)
```
then the SGD step reads the updated `p` (weight decay included), as the reference's in-place updates do. The two
weights and the class-branch factor are `TrainConfig.musgd` (`muon = 0.2`, `sgd = 1.0`: the SGD half of a Muon
parameter runs at `sgd · lr`; `cls_lr = 3.0`), the reference trainer's (`auto` rule 5 uses 0.5 / 0.5 for a run from
scratch, above); other step weights are saved in the
checkpoint's `folo.optim` (FORMAT.md), so a resume continues with them; YOLO26's published COCO run used 0.528 /
0.674 / 2.737 (the `muon_w`, `sgd_w`, `cls_w` of `yolo26n.pt`'s `train_args`), which fine-tuned BCCD worse here
(docs/notes/hill/r2-learn.md). Same
`lr0` / `momentum` / `warmup_bias_lr` handling as `sgd`. FOLO runs the Newton–Schulz iteration in `f32` (the
reference: `bfloat16`); matrices are batched by column count with zero-padded rows, which changes no value. On a
GPU the momentum and both parameter updates are folo-gpu's multi-tensor optimizer kernels, and the rest of the
Muon step is one fused call for all matrices (`Tensor::muon_step_many_`, `kernels::muon`: the Nesterov mix, the
normalisation, each Newton–Schulz step as three grouped GEMMs over every matrix at once with no zero padding, and
`p −= 0.2 · lr · O`); only the summation order of the products and of the norm differs from the tensor ops. Tests: `crates/folo-nn/tests/optim.rs` (two steps against an `f64` re-implementation of the reference,
Newton–Schulz against `f64`), `tests/gpu/` (GPU against CPU; the fused kernels against the tensor ops on wide,
tall, multi-tile and single-row matrices, zero, near-zero, NaN and empty ones), `tests/folo_io.rs` (bit-exact resume).

### 8.3 Batches, accumulation and the step loop
**Batches.** At the start of epoch `e` (0-based), the training indices `0 .. N_train` are shuffled with a permutation
drawn from an RNG seeded by `(seed, e)`, so resuming at an epoch boundary reproduces the order. Batches are
consecutive chunks of `batch` indices; the last one may be smaller and is kept (its `B` is its actual size).
`nb = ceil(N_train / batch)`. Augmentation randomness for a sample comes from an RNG seeded by
`(seed, e, sample index)`, so results do not depend on the number of loader threads.

**Accumulation.** `accumulate` is fixed for the whole run. An accumulation window is `accumulate` consecutive
batches of one epoch; the epoch's last window may be shorter. Gradients never cross an epoch boundary, so the
checkpoints FORMAT.md writes at epoch boundaries hold the complete training state. A short window steps with the
gradients it has (no rescaling). Optimizer steps per epoch: `spe = ceil(nb / accumulate)`.

```
t = 0                                       # completed optimizer iterations (= sched.__step)
for e in 0 .. epochs:
  for k, batch in enumerate(batches(e)):
    loss = L(batch)                         # §7.4
    if loss is finite: backward, adding into the gradient buffers
    else: skip backward, emit a warning     # the batch still occupies its slot in the window
    if (k + 1) % accumulate == 0 or k == nb − 1:          # end of a window
      if the window had at least one finite batch:
        set every group's lr and momentum for iteration t (§8.4)
        if clip_grad_norm > 0:
          n = sqrt(Σ over all parameters of Σ g²)          # one global norm
          if n > clip_grad_norm: scale every gradient by clip_grad_norm / (n + 1e-6)
        optimizer step; EMA update (§8.5); t += 1         # optim.__step = sched.__step = t
      zero gradients
```
`t` advances only with an optimizer step, as FORMAT.md requires. A window made only of non-finite batches is
skipped without a step, so the run can end with `t < T` (the remaining iterations are not made up).

**Non-finite batches.** The train-mode forward updates BN running statistics before the loss is known, so the
trainer snapshots every `running_mean` / `running_var` before each forward (well under 100k floats even for folo-x).
If the loss is not finite, or any running statistic became non-finite, it restores the snapshot, so the next EMA
update never sees poisoned buffers. Ten consecutive non-finite batches, or an epoch without a single finite batch,
abort the run with an error (`status: failed`).

**Epoch boundaries.** The loader of epoch `e + 1` starts as soon as epoch `e` has taken its last batch, so it works
while validation and the checkpoints run and the next epoch's first step does not wait for it (not after the last
epoch or once a stop is requested; if it cannot start, a warning is logged and the epoch starts its loader itself).
Its batches are the same whenever it starts (everything is seeded by the epoch and the sample position). When the
run ends (finished, stopped, stopped early or failed) a loader started for an epoch that will not train is dropped,
joining its workers and the feeder. The epoch's checkpoint is snapshotted on the training thread (every tensor
downloaded to the host) and its files (`weights/last.folo`, `best.folo` when it improved, the samples) are written by a
writer thread, each to a temporary name, synced and renamed, while the next epoch trains; the `epoch` event and every
event after it wait until the files are on disk (docs/API.md), and the next epoch's end, a stop, a failure and the
end of the run wait for the write (one at a time; a wait at the next epoch's end counts in that epoch's
`duration_s`). A write that fails fails the run without that epoch's event. A trainer dropped with a write in
flight (a panic) waits for it without delivering any event.

**On a GPU** (`--device gpu:0`, or `auto` with an adapter) the same loop runs with everything on the device: the model,
the EMA, the optimizer state and the accumulation buffers live there; the loader delivers `u8` pixels and a feeder
thread stages the next batch in a mapped buffer while the current step runs, so a step starts with one buffer copy
and a `u8 → f32 / 255` conversion on the device; the loss is folo-gpu's (§7.4); global-norm clipping computes the norm
and the factor on the device. A step waits for the GPU twice, both after the loss: the reported items and the
finiteness check of the running statistics (one readback for all of them). Validation uploads `u8` batches too and
runs the predictor's forward + decode on the device (NMS on the host); with the one-to-many head, multi-label NMS (the default) and a positive normal `conf` (zero or a subnormal threshold uses the full decode, since a GPU may flush subnormal scores to zero) a device kernel (`folo_gpu::detect::multi_label_candidates`) also picks out the `(anchor, class, score)` pairs above `conf`, so a batch downloads those pairs and the boxes instead of every anchor's scores (an image with more pairs than the buffer holds downloads its whole decode; the buffer makes room for a quarter more than the most pairs an image had so far, and is never larger than half the decode: a batch whose pairs would not fit in that, e.g. many classes at a low `conf`, downloads its whole decode instead), with exactly the detections of the full decode; the download, NMS and scoring of a batch run on a second thread while the next batches' forward passes run, with the same result as one after the other. The numbers are the same as on the CPU up to
float rounding (sums in other orders), so a run is reproducible on one device (a GPU resume is bit-exact) but not
bit-identical across devices. If the GPU is lost mid-run (a driver reset / TDR), the run fails with `status: failed`
and a message naming the lost device and the `--resume` command: checkpoints are written only at epoch boundaries, so
`weights/last.folo` still holds the last finished epoch.

**Progress reporting (API `Run` / `Step`).** `Run.step` and `Step.step` count training **batches** completed in the
run (`e · nb + k + 1`), and `Run.steps_per_epoch = nb`. The optimizer iteration `t` is internal (checkpoints only).
`Step.lr` is the current LR of the `decay` group. `Step.loss` holds the batch's reported items (§7.4), and
`Epoch.train_loss` their mean over the epoch's finite batches. A non-finite batch emits no `step` event, only a
`log` warning (`level: warn`). The counters still advance, so every emitted number is finite JSON.

### 8.4 LR schedule and warmup
The schedule is a function of the iteration `t` (optimizer steps, FORMAT.md's `unit: "iteration"`), with
`T = epochs · spe` iterations in total:
```
warmup_iters W = 0                                                     if warmup_epochs <= 0
               = min(max(round(warmup_epochs · spe), F), max(epochs − 1, 0) · spe)     otherwise
             F = min(warmup_floor, floor(0.3 · T))                     the warmup floor (`warmup_floor`, default 0)
lf(t) = (1 − t / max(T − 1, 1)) · (1 − lrf) + lrf                      linear (default)
lf(t) = lrf + (1 − lrf) · (1 + cos(π · t / max(T − 1, 1))) / 2         cosine (`cos_lr`)
target_g = lr0 · lf(t)                                                 every group
t <  W:  lr_g = interp(t, [0, W], [start_g, target_g]),   start_g = warmup_bias_lr for `bias`, 0 otherwise
         momentum = interp(t, [0, W], [warmup_momentum, momentum])     (SGD μ, or AdamW β1)
t >= W:  lr_g = target_g,  momentum = momentum
```
`warmup_floor` defaults to 0: the warmup is `warmup_epochs` of optimizer steps, as Ultralytics 8.4 counts it (it
dropped the older 100-iteration floor). Runs made before the key existed had a floor of 100 (their configs and
checkpoints keep it, and a resume uses the stored `warmup_iters` anyway); with it a 205-image fine-tune (4 steps per
epoch, 400 in all) spent 25 of its 100 epochs in warmup instead of 3, and fine-tuned BCCD about 0.013 AP worse
(docs/notes/hill/r2-learn.md). A floor is never more than 30% of the run, and `warmup_epochs · spe` is never cut below
what it asks for (only by `(epochs − 1) · spe`). The trainer warns when the run is short anyway: a run from scratch with `T < 200`, or
`W > T / 2` (`folo_vision::train::schedule_warnings`; `warn` log lines after the first one, and `warnings` in the
run's `config.json`).

`interp` is linear interpolation. When `T ≥ 2`, the first iteration's target is `lr0` and the last one's exactly
`lr0 · lrf`. A run with `T = 1` takes its single step at `lr0` (and `W = 0`, since `epochs − 1 = 0`). This
fills FORMAT.md's `folo.sched` as `type`, `epochs`, `steps_per_epoch = spe`, `lrf`, `warmup_iters = W`,
`warmup_bias_lr`, `warmup_momentum` and `step = t`. Compared with the reference trainer (per-batch warmup with an
accumulation ramp, per-epoch decay), this is one clock instead of three, and the run is otherwise equivalent.

### 8.5 EMA
With `ema = true`: an f32 copy of every parameter and BN running statistic, initialised from the model at the start
of training. After every optimizer step: `updates += 1`, `d = 0.9999 · (1 − exp(−updates / 2000))`, then for every
tracked tensor `ema = d · ema + (1 − d) · model`. The EMA is always in eval mode. Validation, the `best`/`last`
weights and export use it (FORMAT.md: `ema.*`, `ema.__updates`, `folo.ema = {"decay": 0.9999, "tau": 2000}`).
With `ema = false`, those uses take the live model in eval mode, and checkpoints carry no `ema.` tensors.

### 8.6 Data and augmentation (folo-data implements; the trainer configures)
`s = imgsz`. Gray means 114 in every channel. Boxes are `xyxy` pixels throughout, and every geometric step below
applies the same map to the boxes as to the pixels.
- **Resize** (bilinear): an `h0 × w0` image with `r = s / max(h0, w0)` becomes `h1 × w1` with
  `w1 = clamp(round(w0 · r), 1, s)` and `h1 = clamp(round(h0 · r), 1, s)` (skipped when `r = 1`). Boxes scale by the
  actual per-axis ratios: `x · w1 / w0` and `y · h1 / h0`.
- **Pad** to `S_w × S_h`: `left = floor((S_w − w1) / 2)`, `top = floor((S_h − h1) / 2)`, gray. Boxes shift by
  `(left, top)`.

Training pipeline per sample, in order:

1. **Mosaic** (probability `mosaic = 1.0`). Take this image and 3 others drawn uniformly with replacement from the
   training set, all resized as above. Canvas `2s × 2s`, gray. Centre `xc, yc = floor(U[s/2, 3s/2))`, drawn
   independently. Image `i` of size `h × w` goes to the canvas region `a` and is cut from the source region `b`:
   ```
   i=0 top-left:     a = (max(xc−w,0), max(yc−h,0), xc, yc)            b = (w−(a.x2−a.x1), h−(a.y2−a.y1), w, h)
   i=1 top-right:    a = (xc, max(yc−h,0), min(xc+w,2s), yc)           b = (0, h−(a.y2−a.y1), min(w, a.x2−a.x1), h)
   i=2 bottom-left:  a = (max(xc−w,0), yc, xc, min(2s,yc+h))           b = (w−(a.x2−a.x1), 0, w, min(a.y2−a.y1, h))
   i=3 bottom-right: a = (xc, yc, min(xc+w,2s), min(2s,yc+h))          b = (0, 0, min(w, a.x2−a.x1), min(a.y2−a.y1, h))
   ```
   Boxes (already resized) shift by `(a.x1 − b.x1, a.y1 − b.y1)` and are clipped to the canvas. Without mosaic
   (closed, or not drawn): resize, then pad to `s × s`, both as above.
2. **Random affine.** Input canvas `w_c × h_c` (`2s` for a mosaic, `s` otherwise), output `s × s`. Draw
   `a ~ U(−degrees, degrees)`, `g ~ U(1 − scale, 1 + scale)`, `sh_x, sh_y ~ U(−shear, shear)`,
   `t_x, t_y ~ U(0.5 − translate, 0.5 + translate)`, and form `M = T · Sh · R · C` (3×3):
   `C` translates by `(−w_c/2, −h_c/2)`; `R` rotates by `a` degrees and scales by `g` about the origin;
   `Sh = [[1, tan(sh_x°), 0], [tan(sh_y°), 1, 0], [0, 0, 1]]`; `T` translates by `(t_x · s, t_y · s)`. Warp the canvas
   by `M` (bilinear, gray fill). Each box is mapped by transforming its 4 corners and taking their min/max, then
   clipped to `[0, s]²`. A box is kept iff `w' > 2`, `h' > 2`, `w'·h' / (w·g · h·g + 1e-16) > 0.1` and
   `max(w'/(h'+1e-16), h'/(w'+1e-16)) < 100`, where `w, h` are its size before the warp. Defaults: `degrees = 0`,
   `translate = 0.1`, `scale = 0.5`, `shear = 0`, `perspective = 0` (no perspective term in v1).
3. **Mixup** `0.0` (off in v1).
4. **HSV jitter** (gains `h = 0.015`, `s = 0.7`, `v = 0.4`): draw `d_k = U(−1, 1) · gain_k` for `k ∈ {h, s, v}`.
   In 8-bit HSV (`H ∈ [0, 180)`, `S, V ∈ [0, 255]`, OpenCV's convention):
   `H' = trunc((H + d_h · 180) mod 180)` (`mod` with a non-negative result), `S' = trunc(clamp(S · (1 + d_s), 0, 255))`,
   `V' = trunc(clamp(V · (1 + d_v), 0, 255))`, then back to RGB. The hue shift is additive, as in the reference
   (Ultralytics ≥ 8.3.79). Any standard RGB↔HSV conversion with these ranges is acceptable.
5. **Flips**: vertical with `flipud = 0.0`, horizontal with `fliplr = 0.5` (`x1' = s − x2`, `x2' = s − x1`).

Finally boxes are converted to the loader's normalised `cx, cy, w, h`. For the last `c = min(close_mosaic,
floor(epochs / 2))` epochs (`e >= epochs − c`; `close_mosaic = 10`), mosaic and mixup are 0 and steps 2, 4 and 5
stay on. The cap keeps mosaic on for at least half of a short run (with `close_mosaic >= epochs` a run would never
see a mosaic); runs of at least `2 · close_mosaic` epochs are unaffected. To train without mosaic, set
`mosaic = 0`. Checkpoints record the closing epochs in `folo.train` (`close_mosaic_epochs`), and a resume keeps
them; a checkpoint without the key predates the cap and resumes with `min(close_mosaic, epochs)`. Validation uses no
augmentation: resize as above and pad to a square `s × s` (§9.2) only.

### 8.7 Validation schedule and checkpoints
- Validate after every epoch and always after the last, with the EMA weights (or the live model if `ema = false`),
  at the training `imgsz`, with batch `2 · batch`, `conf = 0.001`, `iou = 0.7`, `max_det = 300`, multi-label NMS
  (§9.3).
- Metrics: precision, recall, mAP@0.5 and mAP@0.5:0.95 exactly as docs/METRICS.md defines them (pycocotools mAP,
  101-point, at most 100 detections per class per image out of the 300 NMS keeps; precision and recall at the
  confidence that maximises the class-mean F1), computed in the network input space: ground-truth boxes go through
  the same resize and pad as the image, so predictions need no inverse map; they are clipped to the letterboxed
  image (not to the padded input), as the COCO path clips them to the original image. Fitness is mAP@0.5:0.95.
- Best and early stopping: `best_fitness` starts at −∞ and `best_epoch` at −1. After validating epoch `e`, if
  `fitness > best_fitness`, set both and write `best`. Write `last` every epoch (on a GPU its tensors come back in one download per dtype and 128 MiB chunk, concatenated on the device, not one wait per tensor). Stop after epoch `e` when
  `e − best_epoch >= patience`. Checkpoints persist `{ "epoch": best_epoch, "fitness": best_fitness }` as
  `folo.best` (FORMAT.md; `fitness` is `null` while it is −∞), so a resume continues the same patience count.
- v1 reports training losses only (API `Epoch.train_loss`). Validation loss is not computed.

### 8.8 Mixed precision (`amp`)
Opt-in (`amp = true`, `folo train --amp`; default `false`), as `torch.autocast` + `torch.cuda.amp.GradScaler`:
- **Forward** under folo-core's `autocast`: every conv and matrix product runs in f16 on the GPU's tensor cores
  (f16 operands, f32 accumulation) with f16 copies of the weights, so activations are f16; softmax runs in f32.
  BatchNorm takes the f16 activations with f32 γ, β and running statistics; its batch statistics and every sum are
  f32 (§2.1 unchanged otherwise). The head outputs are cast to f32 and the loss (§7, assigner included) is f32.
- **Backward** of `L · S`, where `S` is the loss scale (a power of two): the input gradients of the convs are f16
  (tensor cores), their weight gradients are written straight to f32, so the **master weights, the optimizer state
  (§8.2), the gradient accumulation (§8.3) and the EMA (§8.5) stay f32**.
- **At the end of a window**, before clipping: the gradients are unscaled (`· 1 / S`; with clipping on, the factor is
  folded into the clip factor) and checked: if any is not finite, the window takes **no optimizer step** (as a window
  of non-finite batches, §8.3: `t`, `optim.__step`, `sched.__step` and `ema.__updates` unchanged) and the step is
  counted and logged. The check is done on the GPU; the step waits for one flag.
- **Dynamic loss scaling**: `S = 2^16` at the start; a skipped step multiplies it by 0.5, and 2000 finite steps in a
  row multiply it by 2 (PyTorch's defaults). `S` stays in `[2^-126, 2^126]`, where it and `1 / S` are normal f32
  values (the loss is scaled and the gradients unscaled in f32): a growth or backoff past the range leaves it. The
  scaler state (`S`, the growth count, the skipped steps and the current run of them) is the checkpoint's
  `folo.amp` (FORMAT.md), so a resume is bit-exact. 50 skipped steps in a row (the scale has backed off by `2^50`)
  fail the run (`training_failed`), counted across resumes. A non-finite *loss* (an f16 forward overflow) is a non-finite batch (§8.3), not an
  overflow: the scale does not change the forward.
- **Hardware**: tensor cores (folo-gpu's cooperative matrices: `shader-f16`, 16×16×16 f16 → f32, 32-wide subgroups;
  the RTX 4080 SUPER on Vulkan) are required, and `folo train --amp` requests them. Anything else (the CPU, a GPU
  without them, a `folo serve` or `folo mcp` started without `--tensor-cores`) is `not_supported` before training
  starts (the server refuses the run when it is submitted):
  f16 without tensor cores computes no faster than f32 on current GPUs, so AMP does not silently fall back.
- **Accuracy**: on the §10.6 shapes run, AMP reaches the same mAP as f32 (docs/BENCHMARKS.md has the curves). The
  first steps of a run usually overflow at `2^16` and are skipped while the scale backs off (2–5 steps).

### 8.9 Distillation (`distill`, opt-in)
Dense distillation from a frozen teacher: `TrainConfig.distill = { teacher, cls_gain, dfl_gain, temperature }`
(`folo train --teacher <file or published name>`, `--set distill.<key>=<value>`). Absent (the
default, and not written to `folo.train` or `config.json`), training is §8.3 exactly. Inference, export, `.folo` files
and the HTTP API do not change: the teacher exists only in the trainer.

| key | default | notes |
|---|---|---|
| `teacher` | (required) | a `.folo` file or a published name (`folo weights pull`); same `nc`, `reg_max` and strides as the student |
| `cls_gain` | 0.5 | `λc`, finite, `>= 0` |
| `dfl_gain` | 1.5 | `λd`, finite, `>= 0` |
| `temperature` | 4.0 | `T`, finite, `> 0` (DFL term only; 4 beat 1 on the proxy, docs/notes/hill/r4-distill.md) |

**Teacher.** Loaded once when the trainer is built (a new run and a resume alike; a missing file fails the run with
an error naming `distill.teacher`; a resume may also turn distillation on or change it, `ResumeOverrides::distill`,
`folo train --resume --teacher`, from the resumed epoch on, recorded in the run's config: the teacher holds no
optimizer state), fused (§9.1), eval mode, on the trainer's device, in f16 on a GPU when `amp` is on
and in f32 otherwise; its one-to-many branch is used whatever its `head`. A teacher with another `nc`, `reg_max` or
strides is refused (both heads must have the same anchors, §6.2, and channels, §6.3, so the raw outputs align element
by element). Class names that differ are only warned about: classes match by index. A teacher that is the run's
own `weights/last.folo` or `best.folo` (after resolving links and `..`) is refused, also as a resume override: every
epoch overwrites them, so the next resume would load another teacher; a copy outside `weights/` is accepted.

**Targets (constants, no autograd).** Each training batch `x` (`[B, 3, H, W]`, the augmented batch the student sees)
goes through the teacher before the student's forward, under `no_grad`; its outputs are cast to f32 and:
- `p~ = sigmoid(cls_raw_T)`, `[B, nc, A]`; `s~ = softmax(box_raw_T / T)` over the `reg_max` bins of each side,
  `[B, 4, reg_max, A]` (§6.3 layout, side order `l t r b`).
- `q_a = max_c p~[b, c, a]`, the teacher's confidence that anchor `a` matters.
- Logarithms of the targets are taken at `max(v, f32::MIN_POSITIVE)`, so a probability of 0 or 1 contributes 0.

**Loss.** With the student's one-to-many raw outputs `x = cls_raw`, `z = box_raw` (the f32 copies §7 uses) and
`N = max(tss, 1)`, `tss` the unfloored target-score sum of the same batch's supervised one-to-many loss (§7.4; its
normaliser):
```
L_kc = (1 / N) · Σ_{b,c,a} [ softplus(x) − x · p~ − H(p~) ],   H(p) = −p ln p − (1 − p) ln(1 − p)
L_kd = (T² / N) · Σ_{b,a} q_a · (1/4) Σ_side Σ_i s~_i · (ln s~_i − log_softmax(z / T)_i)
L    = B · (7.5 L_box + 0.5 L_cls + 1.5 L_dfl + λc · L_kc + λd · L_kd)
```
`L_kc` is the Bernoulli KL from `p~` to `sigmoid(x)`: the BCE against `p~` minus the teacher's entropy, a constant,
so its gradient is the BCE's and it is 0 when the student equals the teacher. The gradients are
`dL/dx = B · λc · (sigmoid(x) − p~) / N` and `dL/dz = B · λd · T · q_a · (softmax(z / T) − s~) / (4 N)`. Because the
class term shares the supervised normaliser, supervised BCE at class gain `g` plus `λc · L_kc` is exactly BCE against
the blended target `(g · T + λc · p~) / (g + λc)` at gain `g + λc` (background pairs where both models say about 0
give no gradient); the fair control of a distillation run is the same total class gain without a teacher.

Every element (`softplus(x) − x · p~ − H(p~)`, and `q_a · s~_i · (ln s~_i − log_softmax(z / T)_i)`) is formed before
it is summed; the class elements are summed over `a` per `(b, c)` and the DFL elements over `a` per `(b, side, bin)`,
then those row sums in the fixed-order tree of §7 (rows of 4096), so the term is bit-reproducible on the CPU.
`T = 1` divides by nothing. For a `head = "o2o"` student the term is computed on the one-to-many outputs and is part
of the one-to-many loss: `L = w_m · (L_o2m + B · (λc · L_kc + λd · L_kd)) + (1 − w_m) · L_o2o` (§6.6, `B` and the gains
inside `L_o2m` as above), so the teacher never outweighs the schedule that moves the emphasis to the one-to-one
branch. For an `o2m` student `L` is the line above.

**Reported.** `λc · L_kc` and `λd · L_kd` (without `B`, like §7.4's items) are read back once per batch; the epoch
logs one line, `epoch E: distillation cls … dfl … over n batches (first quarter …, last quarter …)`, the means over
its finite batches and over their first and last quarter; for an `o2o` student they are not weighted by `w_m`. `LossItems`, the `step` and `epoch` events and the API's
loss object do not change. A batch whose distillation terms are not finite is a non-finite batch (§8.3).

**Mixed precision.** The teacher runs in f16 (its own fused f16 weights, f16 input; no autocast needed) and its
outputs are cast to f32 before the sigmoid and softmax; the targets and the term are f32, computed outside autocast
on the f32 copies of the student's head (§8.8).

## 9. Inference

### 9.1 Fusion
Fuse each Conv's BN into its conv, in f32: `k = γ / sqrt(running_var + eps)`, `W' = W · k` (per output channel) and
`b' = β − running_mean · k`. The head `Conv2d`s are unchanged. Then optionally cast to f16 (FORMAT.md export).
Fused inference must match unfused eval-mode inference to `1e-4` relative on the decoded output (f32).

Half precision (`Predictor::new_f16`, `folo predict --half`): the fused model is converted to f16 and runs with f16
weights and activations on a GPU with `shader-f16`; every GEMM, convolution, reduction and softmax accumulates in
f32, the letterbox writes the f16 input, and the head decode (DFL softmax, sigmoid, boxes) runs in f32 on the f16
outputs. Its mAP must be within 0.002 of the f32 model's (docs/BENCHMARKS.md), with and without tensor cores
(`--tensor-cores`, docs/GPU.md).

Inside a `Predictor` (or after `FoloModel::set_forward_plans(true)`), a fused eval-mode GPU forward under
`no_grad`, and the device side of candidate-first prediction (§9.5; keyed also by `conf` and DFL fusion), is
recorded at the second call of an input shape (and head mode) and replayed from the third, with the same bits
(forward plans, docs/GPU.md); any change to the model's tensors retires the plan. A plan does not see the
layers' settings (activation, stride, `residual`, SPPF `k`), so a model on its own has plans off; turn them off
before changing one. `FOLO_GPU_NO_PLAN=1` turns them off for the process.

### 9.2 Letterbox
For an `h0 × w0` image and target `S` (default `imgsz = 640`): `r = min(S / h0, S / w0)` (upscaling allowed),
`nw = clamp(round(w0 · r), 1, S)`, `nh = clamp(round(h0 · r), 1, S)`, bilinear resize, then pad with gray `114` to:
- **square** (validation; batches of mixed canvases; prediction with `rect` off): `S × S`;
- **rect** (prediction, the default): the smallest size `≥ (nh, nw)` whose sides are multiples of 32
  (`dw = (S − nw) mod 32`, `dh = (S − nh) mod 32`). A batch (`Predictor::predict_batch`) whose images all get the
  same such canvas runs on it, so every image gets the input single-image prediction gives it (the same boxes on
  the CPU; on a GPU a kernel may sum in another order at another batch size, last bits). Until 2026-11-13 such
  batches were padded square, which moved scores and boxes slightly, in the third digit. A batch of mixed canvases
  is padded square (one pass per canvas measured slower on the CPU).

Padding is split `left = floor(dw / 2)`, `right = dw − left` (same for top/bottom). This is the same resize and
pad as §8.6. Predicted boxes map back with the per-axis ratios, `x = (x' − left) · w0 / nw` and
`y = (y' − top) · h0 / nh`, then are clipped to `[0, w0] × [0, h0]`.

The resize (`folo_data::resize`, used by training, `val`, `predict`, the server and the browser) aligns pixel
centres (`src = (dst + 0.5) · scale − 0.5`) with 11-bit fixed-point weights, and when an axis shrinks by 2× or more
it first averages `fx × fy` blocks, `fx = floor(w0 / nw)` and `fy = floor(h0 / nh)` (an integer box filter), then
resamples bilinearly. So a 1920×1080 frame at 640 is an exact 3×3 block average. Another runtime fed by an ONNX
export should resize the same way: PIL's antialiased `BILINEAR` gives the same boxes on 1080p frames but scores up to
0.15 apart, and OpenCV's `INTER_LINEAR` (no prefilter) aliases on large reductions.

### 9.3 Post-processing (NMS)
| | predict | val |
|---|---|---|
| `conf` | 0.25 | 0.001 |
| `iou` | 0.7 | 0.7 |
| `max_det` | 300 | 300 |
| multi-label | no: one class per anchor, its argmax | yes: every (anchor, class) pair with score > conf |
| class-agnostic | no | no |

From the `[A, 4 + nc]` decode, keep the candidates whose score is `> conf`, sort by score (descending, ties by anchor
index then class), keep at most 30000, run greedy class-aware NMS (a candidate is suppressed when a kept box of the
same class has plain IoU `> iou`), keep the first `max_det`, then map boxes back (§9.2). Output per image:
`[x1, y1, x2, y2, score, class]`, sorted by score.

### 9.4 Box fusion (opt-in)
Off by default (`NmsConfig::fusion = none`; every published number is without it). `folo val` / `folo predict`
`--fusion vote|dfl [--fusion-iou 0.75]` refine each box NMS keeps from the candidates it suppressed
(`crates/folo-vision/src/fusion.rs`). Scores, classes, which candidates are kept and their order stay NMS's; only the
four edges move.

**Cluster.** For a kept candidate `k`, `C(k)` is every candidate `j` that entered NMS (score `> conf`, class filter,
the 30000 cap) with `class_j = class_k` (any class with `agnostic`) and `IoU(b_k, b_j) >= fusion_iou`, computed in
f32 as NMS computes it; `k ∈ C(k)`.

**Fused edge.** For each side `e ∈ {x1, y1, x2, y2}`, with member weights `w_{j,e}`:

    x̂_{k,e} = Σ_{j ∈ C(k), w_{j,e} > 0} w_{j,e} · x_{j,e}  /  Σ_{j ∈ C(k), w_{j,e} > 0} w_{j,e}

summed in f64 in candidate order (anchor, then class), rounded to f32; a side with no positive weight keeps `x_{k,e}`.

- `vote` (score-weighted box voting): `w_{j,e} = score_j`.
- `dfl` (DFL evidence fusion): `w_{j,e} = score_j · π_{j,e}`, the member's precision on that side, read from its DFL
  distribution (§6.4, `reg_max = R = 16` bins of `stride` pixels). With `p_i = softmax(logits_{j,e})_i`,
  `μ = Σ_i p_i · i`, `σ² = Σ_i p_i · i² − μ²` (bins²) and `f = μ − ⌊μ⌋`:

      π_{j,e} = 1 / ( s_j² · ( max(σ² − f·(1 − f), 0) + ε ) ),   ε = 0.1 bins²,  s_j = the anchor's stride

  `f·(1 − f)` is the variance DFL's own two-bin target has for a distance `f` of the way between two bins (§7.4), so
  a perfectly certain side between bins has excess variance 0, like one on a bin; `s_j²` converts bins² to pixels², so
  across pyramid levels a stride-8 member is trusted 16 times a stride-32 member with the same distribution. An
  anchor near an object's left edge typically knows that edge sharply and the far edge vaguely, so its left edge
  counts for more than its right. There is no censoring guard: a side piled on bin `R − 1` (or 0) has `σ² ≈ 0` and
  the highest precision `1 / (s² · ε)`, like any certain side (a guard was measured and changed nothing; such
  sides are rare, docs/notes/hill/r3-fusion.md).

**Ties and edge cases.** Membership is `>=` on the f32 IoU (a member at exactly `fusion_iou` is in). A member side whose
weight is not `> 0` (a zero score, a NaN, a zero precision) or whose edge is not finite is left out of that side's
sums. Sums run in candidate
order, so the result does not depend on threads or on other classes' candidates, and identical inputs give
identical boxes.

**Where the moments come from.** On the CPU, from the logits, in f32, with the same accumulation as the decode's
expectation (the mean is the decoded distance bit for bit). On a GPU, the device computes `Σ p_i · i` and
`Σ p_i · i²` per anchor and side (`HeadOutput::moments_on_device`, `[B, 8, A]`, an eighth of the logits) and the host
takes `σ² = max(m₂ − μ², 0)`; the full decode, the CPU candidate path and the device candidate path give the same
detections up to that float rounding. Callers with boxes but no logits (`nms_detections`, `Predictor::finish` as
folo-wasm uses it) weigh every side alike, which is `vote`. The NMS-free `o2o` selection has no clusters and ignores
`fusion`. ONNX export does not include it.

**Chosen how.** Hill-climb 3's lane `fusion` (docs/notes/hill/r3-fusion.md) compared hard-NMS settings, Soft-NMS,
Matrix NMS, per-level score offsets, DFL temperature and windowed expectations, box voting variants, lattice pooling
of the distributions and this fusion on 5000 seeded train2017 images (folo-n v1, v2 and folo-s), then read val2017
once. The best combination for COCO AP is `--fusion dfl --iou 0.65 --max-det 1000` (`fusion_iou` 0.75). On
val2017 it adds +0.0043 (folo-n-coco), +0.0034 (folo-n-coco-v2) and +0.0025 (folo-s-coco) AP over §9.3, of which
voting alone gives most; the DFL weights add +0.0003 to +0.0010 over voting (paired bootstrap intervals in the note).
Cost per image at batch 1: ~0.01–0.15 ms on the CPU (conf 0.25 / 0.001), ~0.3 ms on a GPU (the moments readback).

### 9.5 Candidate-first evaluation
Single-label prediction needs a box only for the anchors whose best class scores `> conf` (tens per image at
`conf = 0.25`; `o2o`: at most `max_det` of them, the best by score then anchor, at any `conf`), and §6.1's box and
class branches are independent functions of the same level feature. So an implementation may run the class
branches densely, take the candidates straight from each level's class map (§9.3's extraction), and evaluate the
box branch (`conv0`, `conv1`, `out`, then §6.4's decode) only at the candidates: `out` and `conv1` at the
candidate's pixel, `conv0` on that pixel's 3×3 receptive field (zero padding at the map border is `conv1`'s, as in
the dense branch). The detections are the dense path's when every conv gives each evaluated output the value the
dense conv gives it. FOLO's CPU path (fused f32 models in eval mode, single-label: `Predictor`,
`FoloModel::predict_detections`) does so bit for bit: its 3×3 box convs run Winograd F(2×2, 3×3) over the list of
2×2 output tiles that hold a needed site, where each output reads only its own 3×3 window, and the 1×1 `out` runs
the dense GEMM over the sites' columns. FOLO's GPU path (fused models in eval mode, f32 or f16, single-label)
does so bit for bit too: a class pass lists each image's candidates and the pixels of their 3×3 windows into
fixed-capacity lists on the device, the 3×3 box convs run the dense implicit GEMM (regular or tensor-core) with
one column per listed pixel and the dense conv's split-K chunks, `out` runs the dense GEMM over the candidates'
columns, and one decode appends the boxes to one record list per image, read back once; an image with more
candidates on a level than the list holds (1024, 512, 256 for P3, P4, P5) takes the dense head instead. The dense
raw outputs (§6.3, `HeadOutput`) are not made on these paths; training, validation (multi-label at
`conf = 0.001`), export and every other consumer of them use the dense forward. Box fusion (§9.4) is served as on
the dense path: `vote` needs only the boxes and scores, and `dfl` takes each candidate's side precisions from its
own box logits, which are the dense logits bit for bit, with the dense path's functions: on the CPU the same
softmax moments per candidate; on a GPU the dense path's moment ops (`HeadOutput::moments_on_device`) over the
candidates' columns, which compute each column on its own, so the same bits. With forward plans on (§9.1), the
GPU path from the input to the records and moments is one recorded plan; the readbacks follow the replay.

### 9.6 int8 inference (opt-in, CPU)
Post-training quantisation of the backbone and neck (layers 0–22); the head (§6) stays `f32`. Opt-in:
`folo quantize` calibrates a fused `f32` model and stores the ranges with the weights (`folo.quant`,
docs/FORMAT.md), and `folo val` / `folo predict --precision int8` (or `FoloModel::set_int8`) run it on the CPU.
GPU forwards, training, export and the `f32` default are unchanged.

- **Boundaries.** An activation is quantised where a layer writes it: each conv's output after its activation (and
  after the residual add of a bottleneck, PSA `proj` and `ffn2`, which the conv's epilogue does), the stem's output,
  and the input of each attention `proj` (`o + pe(v)`). Each boundary has one affine u8 grid
  `x ≈ scale · (q − zero)`, `q ∈ [0, 255]`, from its calibrated range `[lo, hi]` widened to hold 0:
  `scale = (hi − lo) / 255`, `zero = round(−lo / scale)` clamped to `[0, 255]`. Concatenation, split
  (channel slices), max pooling and nearest upsampling keep their input's grid (exact on codes). The attention's
  `qkv` conv writes `f32`; the attention products, softmax and the depthwise `pe` run in `f32`. P3, P4 and P5
  are dequantised for the head.
- **Weights.** Per output channel, symmetric s8: the input grids' scales are folded into the weight columns
  first (`w'[o, c, ·] = w[o, c, ·] · scale(c)`), then `s_o = max|w'[o]| / 127` and
  `q = round(w' / s_o)` (ties away from zero), clamped to `±127`. A conv over a concatenation folds each part's
  scale into its columns. The input zero points become one constant per output channel,
  `corr[o] = Σ q[o, c, ky, kx] · zero(c)`, and zero padding uses each part's zero point, so it is exact.
- **Conv.** `acc[o] = Σ q · u` in i32 (u8 × s8; exact: no product or partial sum saturates), then
  `y = (acc − corr) · s_o + bias`, SiLU, the residual `+ scale_r · (u_r − zero_r)` if any, and the output code
  `clamp(round_half_even(clamp(y / scale_out, −512, 512)) + zero_out, 0, 255)` (multiplying by the `f32`
  reciprocal of `scale_out`). Every step is `f32` arithmetic in a fixed order, so the result does not depend on
  the CPU, the kernel or the thread count.
- **Calibration.** The `f32` model runs on sample images (default: 512 training images, letterboxed as for
  validation) twice: the first pass records each boundary's minimum and maximum, the second a 2048-bin histogram
  between them. The range is then the `p`, `1 − p` quantiles (default, `p = 10⁻⁴`), the minimum and maximum, or
  the histogram-edge range of least expected squared error (rounding `Δ²/12` inside, clipping outside).
- **Mixed precision.** A conv named in `f32_convs`, or one the int8 kernel does not take (grouped, dilated,
  channels not a multiple of 4, a reduction `c · k²` over 2¹⁵, stride over 64, padding over 4096, or a kernel
  larger than its padded input), dequantises its inputs, runs in `f32` and quantises its output. The fake-quant
  reference keeps such a conv's `f32` weights too.
- **CPU.** The fast kernel needs AVX-VNNI (`vpdpbusd` on 256-bit registers, with AVX2 and FMA). Without it, AVX-512
  VNNI CPUs included (that form is not used), the int8 convs run a portable kernel with the same integers that is
  several times slower than the `f32` path: one folo-n row-8 run in review took 66.3 ms against 17.0 ms in `f32`.
  `folo val` / `folo predict --precision int8` print a `note:` on stderr then.

FOLO's implementation: folo-core's `cpu::q8` kernels (docs/CORE.md, int8 convs) under folo-vision's
`model::quant` (the graph walk is shared by calibration, the int8 executor and a fake-quant reference that runs the
same grids in `f32`). On COCO val2017, with `folo quantize`'s defaults against the same weights in `f32` on the CPU
(paired bootstrap, 95% intervals), folo-n-coco-v4 loses 0.0019 AP (0.3559 → 0.3541, −0.0028 to −0.0010) and
folo-s-coco 0.0025 (0.4162 → 0.4137, −0.0039 to −0.0014): docs/notes/hill/r5-cpuq8.md.

## 10. Verification plan

### 10.1 Block unit tests (folo-nn / folo-vision)
For each block, build it with the stated configuration, check the parameter count, run a forward on a zero input and
check the output shape:

| block | configuration | params |
|---|---|---:|
| Conv | 3→16, k3 s2, input 640×640 → output 16×320×320 | 464 |
| Bottleneck | 16→16, e=0.5 | 2,352 |
| Bottleneck | 32→32, e=1.0 | 18,560 |
| C3k | 64→64, n=2 | 45,568 |
| C3k2 | 32→64, n=1, c3k=no, e=0.25 | 6,640 |
| C3k2 | 64→64, n=2, c3k=no, e=0.5 | 31,168 |
| C3k2 | 128→128, n=1, c3k=yes, e=0.5 | 87,040 |
| SPPF | 256→256, k5 | 164,608 |
| Attention | C=128, heads=2 | 51,328 |
| C2PSA | 256, n=1 | 249,728 |
| C2PSA | 256, n=2 | 367,360 |

Every differentiable block also gets a finite-difference gradient check at a tiny size (for example 8 channels, 6×6,
batch 2, in train mode so BN uses batch statistics).

### 10.2 Whole-model tests
- For each scale and `nc ∈ {80, 4, 1}`: the total parameter count equals §4, and the per-layer counts equal the
  `params` column (a test helper sums `named_parameters()` by layer prefix).
- Named-tensor counts equal §4's table, and the first and last names in canonical order are
  `backbone.stem.conv.weight` and `head.cls.p5.out.bias`.
- Forward at 640×640 (folo-n, batch 1, eval) yields P3/P4/P5 features of shapes `64×80×80`, `128×40×40`,
  `256×20×20`, and raw outputs `box_raw [1, 64, 8400]`, `cls_raw [1, 80, 8400]`. At 320: 2100 anchors.
  Rectangular 384×640 (H × W) input: 5,040 anchors (48·80 + 24·40 + 12·20).
- A golden file of `(name, shape)` for folo-n (`nc = 80`), generated once from the implementation and reviewed
  against §5, guards against accidental renames.
- Priors: with the head output weights zeroed (a test helper), `cls_raw` equals the prior bias of §6.5 at every
  anchor and every decoded box is exactly 15 strides wide and tall (7.5 strides per side).

### 10.3 Assigner and loss tests (hand-built cases)
- One 100×100 px gt centred on a P3 cell: the candidates are exactly the anchor centres strictly inside it (all
  levels), at most 10 positives, `max t = u_max`.
- A 6×6 px gt: the enlargement to 16×16 gives it P3 candidates. Without enlargement it might have none.
- Two overlapping gts competing for an anchor: the larger-IoU gt wins; equal IoU, the lower index wins.
- DFL target `δ = 3.3`: weights 0.7 on bin 3 and 0.3 on bin 4; `δ = 15.2` is clamped to 14.99.
- CIoU of identical boxes ≈ 1 (exactly `1 −` the eps effects); disjoint boxes give a negative CIoU (assigner `u = 0`).
- An empty batch: `L_box = L_dfl = 0`, `L_cls = Σ softplus(cls_raw)` (with `tss = 1`).
- Loss scaling: for a batch whose unfloored target-score sum `Σ T` is at least 1, duplicating it (B → 2B, the same
  images) keeps the three reported items and doubles `L`. Below the floor `tss = 1` stays fixed: an all-background
  batch duplicated doubles the reported `cls` and quadruples `L`.

### 10.4 Training-loop tests
- Batching and schedule: for `N_train = 800`, `batch = 16`, `E = 30`: `nb = 50`, `accumulate = 4`,
  `spe = 13` (the last window of each epoch has 2 batches), `T = 390`, `W = 39` (`round(3 · 13)`; no floor by
  default). With `warmup_floor = 100` (configs written before the key) the floor applies and `W = 100`; on that
  schedule check the LR and momentum of every group at `t ∈ {0, 50, 99, 100, 389}`, that `lf(0) = 1` and
  `lf(389) = lrf`, and that `warmup_epochs = 0` gives `W = 0`.
- Short runs: the same data for `E = 10` (`T = 130`) gives `W = 39` and a short-run warning (with the 100 floor
  too: it is cut to `floor(0.3 · 130) = 39`). With the 100 floor, `E = 12` of `spe = 3` (`T = 36`) gives `W = 10`
  (no floor: 9), `T = 333` has floor 99 and `T = 334` the full 100. `close_mosaic = 10` closes mosaic for 10 epochs
  of 20 or 30, 6 of 12, 4 of 8 and none of 1.
- Accumulation never crosses an epoch boundary: after each epoch the gradient buffers are zero.
- A forward that produces NaN in a BN layer (inject it), followed by a valid batch: the BN running statistics,
  the EMA and eval-mode outputs stay finite, and no `step` event carries a non-finite value.
- A window whose batches are all non-finite (inject NaN) takes no optimizer step and leaves `t`, `optim.__step`,
  `sched.__step` and `ema.__updates` unchanged. A checkpoint written after it resumes to the same state as an
  uninterrupted run.
- Early stopping: with `patience = 2` and fitness `[0.1, 0.3, 0.2, 0.2]`, training stops after epoch 3
  (`best_epoch = 1`). A resume from the epoch-2 checkpoint stops at the same epoch.
- `auto` (rules 4 and 5) picks MuSGD at `lr0 = 0.01`, momentum 0.937 and `warmup_bias_lr = 0.1` for the run above,
  with step weights 0.5 / 0.5 under rule 5 (a run from scratch); `auto_rule = 3` picks AdamW at `lr0 = 0.00125` for
  `nc = 4`, 2000 images and 30 epochs.
- EMA: after one update `d = 0.9999 · (1 − e^{−1/2000}) ≈ 4.99825e-4`.
- **Overfit** (`#[ignore]`, about 1–2 min on CPU): folo-n at 320 px, 8 synthetic images, augmentation off, AdamW
  `lr0 = 0.002`, 300 steps, batch 8, no accumulation. The total loss must fall below 25% of its first value, and
  mAP@0.5 on the same 8 images must exceed 0.95.

### 10.5 Optional reference parity (local only)
Because FOLO's shapes equal YOLO11's (with the SPPF `cv1` SiLU of the original YOLO11 release), a developer may
convert locally obtained YOLO11 weights to `.folo` and compare eval-mode raw outputs on a fixed input (tolerance
`1e-4` relative). Name map: `model.{0,1,2,…,22}` → the layer prefixes of §4; `m.{i}` → `block{i}`; C2PSA
`m.{i}.ffn.0/1` → `ffn1/ffn2`; Detect `cv2.{l}.{0,1,2}` → `head.box.L.{conv0,conv1,out}`,
`cv3.{l}.0.0/0.1/1.0/1.1/2` → `head.cls.L.{dw0,pw0,dw1,pw1,out}`; drop `dfl.conv.weight`. The reference weights are
AGPL-3.0: never commit them, any derived file, or the conversion script; gate the test on an environment variable
(`FOLO_PARITY_WEIGHTS`) and keep it `#[ignore]`.

### 10.6 Convergence smoke test (M2 gate)
- Dataset: folo-data's synthetic shapes, 4 classes (circle, square, triangle, star), 2000 train and 200 val images
  at 320×320, 1 to 8 objects per image, object size 6 to 35% of the image side, textured backgrounds, seed 0.
- Run: `folo train --arch folo-n --imgsz 320 --epochs 30 --batch 16 --close-mosaic 5 --seed 0`, other settings at
  default (so `auto` rule 5 chooses MuSGD at `lr0 = 0.01`, momentum 0.937, `warmup_bias_lr = 0.1` and step weights
  0.5 / 0.5; `nb = 125`, accumulate 4, `spe = 32`, warmup `W = 96` iterations: 3 epochs, no floor).
- **Pass:** EMA-model validation mAP@0.5 ≥ 0.90 and mAP@0.5:0.95 ≥ 0.75 by epoch 30, with no non-finite loss.
- Measured (seed 0, RTX 4080 SUPER, docs/notes/hill/r2-learn.md): rule 5's settings give final mAP@0.5 0.988 and
  mAP@0.5:0.95 0.934, the bar first met at epoch 8. Rule 4 (step weights 0.2 / 1.0) measured 0.985 / 0.926, first
  met at epoch 8; the `auto` of rule 3 (AdamW at `lr0 = 0.00125`, `warmup_bias_lr = 0`, `W = 100` with the old floor)
  0.980 / 0.892, first met at epoch 9.
- Calibration (historical: made when `auto` chose AdamW here): the reference implementation (Ultralytics 8.4
  YOLO11n: the same topology, trained with AdamW, its `auto` at 0.00125, from scratch with this run configuration on
  a generator matching this description; its warmup is 375 batches ≈ 94 optimizer steps instead of the 100 FOLO had
  then, with an accumulation ramp) reached
  mAP@0.5 0.854 / 0.904 / 0.933 / 0.948 at epochs 6 / 11 / 16 / 30, and mAP@0.5:0.95 0.752 at epoch 9 and 0.873 at
  epoch 30. Its losses dropped visibly when mosaic closed at epoch 26. The pass bar leaves margin for generator and
  implementation differences. A FOLO run far below it points to a bug, not to tuning.
- Cost: folo-n at 320 px is 1.62 GFLOPs forward, so 30 epochs of 2000 images are roughly 0.3 PFLOP of training
  compute. That is minutes on the GPU backend and about an hour on a desktop CPU at 100 GFLOP/s effective, so the
  CPU variant of this test is `#[ignore]` and run on demand.
