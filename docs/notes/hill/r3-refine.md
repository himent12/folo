# Hill-climb 3: a learned box refiner that probes the predicted edges (lab notebook)

Lane `refine` of hill-climb 3 (2026-11-13). Brief: `D:/folo-build/hill3/briefs/refine.md`. Question: does a small
second stage on the kept boxes, sampling the pyramid at probes placed across each predicted edge (spacing set by
the detector's own DFL width for that side), move the B* boxes toward their truths enough to pay in COCO AP?

## Headline

**No. The gate (+0.005 AP over B* on dev and val-A) is not passed; nothing goes to Rust.**

| | dev AP | Δ vs B* (95%) |
|---|---:|---|
| B* (lane fusion) | 0.3809 | |
| the brief's refiner (P3–P5 edge probes, σ spacing), best of 9 runs | 0.3806 | −0.0002 [−0.0006, +0.0002] |
| same, trained on all matched boxes | 0.3709 | −0.0100 [−0.0111, −0.0091] |
| experiment H1 (probes on the input and the stem) | 0.3799 | −0.0010 [−0.0015, −0.0004] |
| experiment H2 (input strip CNN) | 0.3811 | +0.0002 [−0.0004, +0.0008] |
| H2 with 1 px strip spacing (dev diagnostic) | 0.3815 | +0.0007 [+0.0001, +0.0013] |

(The best P3–P5 row is c-prim-nofeat, the no-features control; every feature variant is below it.)

Why: the top-scoring box per truth (a proxy for the box AP sees) is already close, with a median edge error of
1.2 / 1.9 / 2.7 px for small / medium / large objects, 3–4.5× below the DFL width that sets the probe spacing. None
of the refiners tried (on P3–P5, on the stem, on the input image down to 1 px spacing) moves medium and large
primaries toward their truths; small ones gain 1.7% of their edge error from the input image. A refiner trained on
all boxes at IoU ≥ 0.3 does close 19% of the edge error of the boxes at IoU ≥ 0.5, but mostly on lower-ranked
duplicates, and costs the tight top boxes more AP than it gains. (COCO's annotation noise, often put at 1–2 px per
edge, may set a floor for the small objects; it is not measured here.)

## Setup

- Detector: the landed folo-n (`D:/folo-data/runs-hill3/fusion/land-n-v2/weights/best.folo`), frozen.
- B*: lane fusion's post-process (DFL evidence fusion after NMS IoU 0.65, max_det 1000, conf 0.001),
  `bench/fusion/post.py`, reused unchanged through `bench/refine/boxes.py`.
- Splits: dev = lane fusion's train-dev list (5000 train2017 images, never trained on by the refiner); val-A = the
  1000 val2017 images with the smallest ids (`make_val_split.py`); val-B = the other 4000; training pool = 24000 other
  train2017 images (`make_traindev.py --n 24000 --seed 1 --exclude traindev.txt`). Lists and truth files:
  `D:/folo-data/runs-hill3/refine/lists/`.
- val-A was dumped but never scored (no configuration reached the gate on dev); val-B was never dumped.

## Part 1: features and box tables

**Dump.** `candidate_dump` (lane fusion's example) gained `--features` (the head's three inputs, rows 16, 19, 22 of
MODEL.md §4, through `FoloModel::forward_traced`, stored f16 `[C, H, W]` per level: 1.43 MB per folo-n image) and
`--aug <seed>` (per image: horizontal flip with p 1/2 and a zoom about the input's centre drawn from 0.75..1.25,
applied to the letterboxed u8 input before the forward; the loader's boxes go through the same transform).

| dump | images | time | size |
|---|---:|---:|---:|
| dev (`land-dev`) | 5000 | 102 s | 7.2 GB features |
| val-A (`land-valA`) | 1000 | 15 s | 1.4 GB |
| training views (`land-tr24k-a1`, `--aug 1`) | 24000 | 506 s | 34.4 GB |

**Box tables** (`boxes.py`): B* per image (lane fusion's `batched_nms` + `fuse`, bit for bit), and per kept box
what the refiner reads: the kept candidate's level and 4 × 16 DFL probabilities, σ per side
(`stride · sqrt(max(Var − f(1 − f), 0) + 0.1)`, MODEL.md §9.4's precision as a width), the cluster's
score-weighted mean precision as a width, the members' weighted spread around the fused edge, the cluster size,
the image's area in the view, and the same-class truth the box overlaps most. dev: 1,430,547 kept boxes (286 per
image at conf 0.001), 149,943 with a truth at IoU ≥ 0.3; training views: 6,611,147 and 707,693.

**Checks.**
- B* from the dev table: AP **0.3809**, AP50 0.5366, AP75 0.4121, exactly lane fusion's train-dev number for the
  landed weights (`results/dev-bstar.json`).
- The dumped features against the detector (`check_features.py`): the head's box branch (BN folded, PyTorch) run on
  the dumped P3 / P4 / P5 gives the dumped DFL logits of every candidate anchor, max |logit diff| 0.006 (relative
  ~1e-4, the f16 storage), decoded distances within 0.0002 bins on average and 0.004 at worst; a wrong level order,
  layout or a one-cell shift would be off by whole bins (`logs/check-features.log`). A first version read the
  file's raw weights and got 0.02–0.06 bins: `best.folo` also holds an `ema.` copy, which is what FOLO infers with
  (`read_folo` now prefers it). The refiner never used these weights (it reads the dumped features), so no result
  changes.
- `check_sample.py`: the refiner's gather-based bilinear sampling equals `grid_sample` (border padding,
  `align_corners = False`) to 3e-14 in f64, and an untrained refiner returns its input boxes exactly.

## Part 2: the refiner (`bench/refine/refiner.py`)

As the brief, in one canonical frame per side: outward normal `n_e`, probes at `x_e + t · s_e · n_e`, `t ∈ {−2..2}`,
at `u ∈ {1/8, 3/8, 5/8, 7/8}` along the edge; `s_e = max(1.5 σ_e, 0.04 · size across the edge)`; bilinear samples
of the level whose stride is closest to `s_e` after a per-level 1×1 projection to `d = 64`; a 3 × 3 context grid
inside the box (level by the box's short side / 3) through a linear map to `d`; 32 scalars per side (σ/size,
σ_fused/σ, s/size, log w, log h, score, the anchor's level, the probe level, the 16 DFL bins, spread/s,
log(1 + members), the outward distance to the image border / s and a border flag); a 16-d side embedding; one
shared MLP (2 × 256, SiLU) to 17 bins over `[−2, 2]`; refined edge `x_e + n_e · s_e · E[t]`. Loss: DFL's two-bin
cross-entropy on `clamp((x_gt − x_e) · n_e / s_e, −2, 2)` plus (1 − CIoU) of the refined box. The last layer starts
at zero, so training starts from B*. AdamW 1e-3, cosine, 16 images per step, 4–6 epochs over the 24k views
(~270 steps / min on the 4080).

## Part 3: the refiner on dev

Every row is a full training run on the 24k views, scored on dev through `evaluate.py` (refined boxes replace the
B* boxes, scores unchanged). Δ and the 95% interval are the paired image bootstrap of `bench/fusion/bootstrap.py`
(1000 resamples, `logs/boot-dev.log`). The first three runs (base, sw1, sw1-nofeat) trained on labels with a bug (see
"Design errors"); the rest on the fixed tables.

| run | trains on | changes | dev AP | Δ vs B* | 95% |
|---|---|---|---:|---:|---|
| B* | | | 0.3809 | | |
| base | boxes at IoU ≥ 0.3 | 6 epochs, unclipped labels | 0.3703 | | |
| sw1 | " | + loss weighted by score | 0.3703 | | |
| sw1-nofeat | " | + no features | 0.3751 | | |
| c-base | boxes at IoU ≥ 0.3 | 4 epochs | 0.3709 | −0.0100 | [−0.0111, −0.0091] |
| c-sw1 | " | loss weighted by score | 0.3705 | | |
| c-nofeat | " | no features (scalars only) | 0.3763 | −0.0046 | [−0.0053, −0.0040] |
| c-prim | primary boxes | | 0.3794 | −0.0014 | [−0.0020, −0.0009] |
| c-prim-nofeat | primary boxes | no features | 0.3806 | −0.0002 | [−0.0006, +0.0002] |
| p-wiou5 | primary boxes | CIoU weight 5 | 0.3805 | −0.0004 | [−0.0009, +0.0002] |
| p-fixed | primary boxes | fixed probe spacing | 0.3803 | −0.0005 | [−0.0011, −0.0001] |
| p-all | primary boxes | all three levels at every probe | 0.3788 | −0.0021 | [−0.0028, −0.0015] |
| p-big | primary boxes | d 128, hidden 512 | 0.3786 | −0.0023 | [−0.0030, −0.0016] |
| p-ovf | primaries of 800 training views | 40 epochs, CIoU weight 5 (capacity check) | 0.3445 | | |

The features hurt: c-base − c-nofeat −0.0054 [−0.0064, −0.0045], c-prim − c-prim-nofeat −0.0012 [−0.0017,
−0.0007]. No run reaches B*, so the gate (+0.005 on dev) is far away, and val-A was not scored.

**The refiner does learn something.** c-base closes 19% of the mean edge error of the dev boxes at IoU ≥ 0.5 with
a truth (71,957 boxes, `evaluate.py`'s diagnostics; it trained on IoU ≥ 0.3), mean IoU with the truth 0.695 →
0.740; by size S +1.3%, M +13%, L +22.5% (`logs/dev-c-base.log`). But AP weighs most the top-scoring box per
truth, and most of the gain is on the many lower-ranked duplicates around it.

**Attribution (c-base, `attribution.py`).** Refining only the boxes that have a truth at IoU ≥ 0.3 gives 0.3709, the
unmatched only 0.3809 (untouched). By the true IoU of the B* box: refining only [0.3, 0.6) gives 0.3829 (+0.0020),
[0.6, 0.8) 0.3789, [0.8, 1] 0.3704. The tight boxes are what is lost, and they carry AP75 and above. A truth-assisted
gate that keeps the refined box only where its IoU with its assigned truth went up gives 0.3920 (+0.011,
`logs/attrib-dev-c-base-oracle.log`): an illustration of how much of the refiner's movement is useful, not a
ceiling (COCO's matching between competing boxes and truths can make another subset score higher).

**Gates and shrinkage** (c-base; trained on the training views, picked there): moving only part of the way
(λ 0.25: 0.3801; 0.5: 0.3780), a z gate on the predicted shift over its spread (z 1: 0.3778, 2: 0.3803, 3: 0.3805)
and a James–Stein-like shrink (js 1: 0.3797, 4: 0.3806). Every gate lands at or below B*: what the refiner can tell
apart from its own outputs is not where it helps.

**Not overfitting.** `bandcheck.py`: the tight boxes are hurt by the same amount on the training views as on dev,
and the residual distribution (`residuals.py`) of dev, val-A and the training views agree once the labels were
fixed.

**The primary boxes.** For each truth its primary box is the top-scoring same-class B* box at IoU ≥ 0.5
(`primary.py`; 30,949 on dev, mean |edge error| 5.29 px: S 2.02, M 4.09, L 8.50). This is a proxy for the box AP
sees: COCO matches greedily by score at each IoU threshold, so the primary is the first box tried for its truth
at IoU 0.5, unless another truth takes it first. Fraction of the primaries' mean |edge error| closed
(`primary_error.py`; negative = worse):

| run | all | S | M | L |
|---|---:|---:|---:|---:|
| c-base | −0.162 | −0.228 | −0.192 | −0.137 |
| c-sw1 | −0.147 | −0.212 | −0.169 | −0.126 |
| c-nofeat | −0.097 | −0.172 | −0.118 | −0.077 |
| c-prim | −0.018 | −0.005 | −0.016 | −0.021 |
| c-prim-nofeat | −0.006 | −0.002 | −0.005 | −0.008 |
| p-wiou5 | −0.007 | +0.002 | −0.010 | −0.007 |
| p-fixed | −0.010 | +0.005 | −0.005 | −0.015 |
| p-all | −0.020 | −0.003 | −0.019 | −0.023 |
| p-big | −0.025 | −0.006 | −0.023 | −0.028 |
| p-ovf | −0.264 | −0.191 | −0.274 | −0.270 |

(`logs/primary-error-dev.log`.)

Every refiner makes the primaries worse. Trained on primaries only, the training CIoU loss stays flat (0.2096 at
the start, 0.2107 at the end of c-prim): the network cannot find a direction that helps those boxes, and goes
nearly to identity. It can fit a smaller set (p-ovf: training CIoU 0.209 → 0.079 on the primaries of 800 views, with
CIoU weight 5), so the flat loss is not a plain inability to fit; it does not rule out capacity or optimisation
limits at 24k views and weight 1. With P3–P5 (strides 8–32) the remaining error of a primary box is below what these probes
resolved.

## Part 4: experiment H, high-resolution probes

Question: is the primaries' remaining error resolvable at all from finer inputs than P3 (stride 8)? Mean |edge
error| of a primary is 2.0 px for small objects, 4.1 for medium, 8.5 for large: under one P3 cell for S and M.

**Inputs.** `candidate_dump --input` writes the network input (u8 `[3, 640, 640]`, after `--aug`'s flip and zoom) of
the same views (anchors.bin identical to the feature dumps': dev, val-A and the training views; 1.2 MB per image,
29.5 GB for the training views, 723 s). Layers 0 and 1 (the stem: Conv 3→16 k3 s2 and Conv 16→32 k3 s2, BN folded,
SiLU) run online in PyTorch (`stem.py`). Checked against FOLO's own trace (`--stem`, `forward_traced` rows 0 and 1)
on 8 dev images: max |diff| 4e-5, mean 7e-7 (`logs/check-stem.log`).

**H1.** The refiner of Part 2 unchanged, except that the edge probes sample `[RGB / 255, layer 0, layer 1]`
(3 + 16 + 32 = 51 channels at strides 1, 2, 4; bilinear, border clamped) at the same 5 × 4 points (spacing s_e),
through a linear map to d = 64. The 3 × 3 context grid stays on P3–P5.

**H2.** Per side, a strip of the input image in the side's canonical frame: 32 samples across the edge at spacing
s_e / 4 (±4 s_e, outward increasing) × 8 along it, through a small CNN (4 convs 3×3, 16–32 channels, stride 2
across twice and along once, mean over the along axis, a linear map to 128), concatenated with the context and the
scalars.

Both train on the primary boxes (as c-prim), 4 epochs, same loss and schedule (8 min each on the 4080).

**Results on dev.** First the primaries (`primary_error.py`, `logs/primary-error-dev.log`): the fraction of the
mean |edge error| closed, and per edge the share moved closer to its truth by more than 0.1 px / farther by more
than 0.1 px.

| run | closed S | M | L | closer / farther S | M | L | dev AP | Δ vs B* | 95% |
|---|---:|---:|---:|---|---|---|---:|---:|---|
| c-prim (P3–P5) | −0.005 | −0.016 | −0.021 | 0.25 / 0.27 | 0.36 / 0.39 | 0.40 / 0.44 | 0.3794 | −0.0014 | [−0.0020, −0.0009] |
| H1 | −0.001 | −0.015 | −0.017 | 0.38 / 0.36 | 0.38 / 0.42 | 0.41 / 0.45 | 0.3799 | −0.0010 | [−0.0015, −0.0004] |
| H2 | **+0.018** | −0.008 | −0.014 | 0.40 / 0.33 | 0.39 / 0.40 | 0.42 / 0.44 | 0.3811 | +0.0002 | [−0.0004, +0.0008] |
| H2, strip at 1 px | +0.017 | −0.007 | −0.011 | 0.40 / 0.33 | 0.40 / 0.40 | 0.42 / 0.44 | 0.3815 | **+0.0007** | [+0.0001, +0.0013] |

The last row (`--strip-px 1`: the strip samples 1 input pixel apart for every box, ±16 px across the edge) was
added after the diagnosis below, to test it directly; it is a dev diagnostic, not a choice made on training data.
h2px − H2: +0.0005 [+0.0001, +0.0009].

H2 − c-prim: +0.0016 [+0.0011, +0.0021] (`logs/boot-dev-h.log`); H1 − c-prim +0.0005 [−0.0001, +0.0010]. H2's AP by
size: S 0.1740 (B* 0.1728), M 0.4043 (0.4045), L 0.5422 (0.5436); h2px: S 0.1747, M 0.4044, L 0.5427. The input
strip is the first source that carries something the pyramid does not, and only for small objects. M and L
primaries do not move toward their truths in any variant; by the stop rule, experiment H ends here, and the gate
(+0.005) is out of reach: the best run, h2px, is significantly above B* but at 14% of the gate.

**Gates on H2** (`gate.py`, scored on dev as an upper bound, since a gate picked on dev is optimistic): z 1, 2, 3
and js 1, 4 all leave AP at 0.3809. H2's predicted shifts are small next to their own spread
(|E[t]| / std[t]: 99th percentile 0.54, the same for h2px, against 3.6 for c-base), so no gate finds a confident
subset; the network moves every edge a little and none with conviction.

## Part 5: diagnosis

Where the primaries' error lives (`logs/primary-scale-dev.log`, dev primaries, network-input pixels):

| size | primaries | median \|edge error\| | mean | median σ_e | median s_e | median error / s_e | edges under 1 px |
|---|---:|---:|---:|---:|---:|---:|---:|
| S | 7438 | 1.17 px | 2.02 | 3.45 | 5.17 | 0.22 | 44% |
| M | 11591 | 1.88 px | 4.09 | 5.97 | 8.96 | 0.21 | 32% |
| L | 11920 | 2.74 px | 8.50 | 12.52 | 19.00 | 0.15 | 23% |

(In original-image pixels the medians are 1.11, 1.81 and 2.65.)

1. **The error is below the probe spacing.** The median primary edge is 0.15–0.22 s_e from its truth, under one of
   the 17 output bins (0.25 s_e) and under one probe step. The DFL width σ_e (the spread of the head's own
   distribution for that side) is 3–4.5× the actual median error of the top-scoring box: the primaries are much
   more accurate than their distributions say. So the brief's central idea (probe spacing from σ) places
   the probes too far apart for the boxes that count. P3 (stride 8) is coarser still.
2. **Resolution helps small objects only.** H2's strip spacing s_e / 4 is ≈ 1.3 px for small objects (about their
   median error) and 2.2 px / 4.75 px for medium / large (coarser than theirs); only the small primaries improved.
   That suggested M and L only lacked resolution, so h2px samples every strip 1 px apart (finer than every size's
   median error). S stays where it was (+1.7%), M and L are hurt a little less (−0.7%, −1.0%) but still move
   farther as often as closer. (h2px also narrows the strip's coverage, to ±16 px.) So for M and L, none of these
   architectures, losses and sampling patterns found in finer image evidence which way the truth lies; other
   designs are not ruled out.
3. **The mean is the tail.** Mean / median is 1.7 (S), 2.2 (M), 3.1 (L): a few edges with large errors carry the mean
   (not inspected one by one; presumably occluded or truncated extents and part-whole ambiguity, which a local edge
   probe cannot settle).
4. **Annotation noise (a caveat, not measured here).** COCO's boxes are often said to carry about 1–2 px of noise
   per edge. If so, the small primaries' median error (1.1 px in the original image) is at that level and little
   of it is reducible against these labels; the medium ones (1.8 px) are near it. This lane has no measurement
   that separates label noise from prediction error, so it is a possible floor, not an explanation.

So the lane's answer: B*'s top boxes are already within about 1–3 px of their truths at the median, well inside
the DFL width the probes were spaced by; no refiner tried here moved the medium and large ones toward their truths
from any source (stride 8 down to 1 px); small boxes gain a little from the input image (+1.7% of their edge error,
+0.0019 AP_S); and what a pyramid refiner gains on lower-ranked boxes (refining only the boxes at true IoU 0.3–0.6
gives +0.0020) did not offset its losses on the tight top boxes under any gate tried.

## Negative results

- Edge-probe refiner on P3–P5 (the brief's design), trained on all matched boxes: −0.0100 on dev; on primary
  boxes: −0.0014. No ablation reaches B*: score-weighted loss, CIoU weight 5, fixed probe spacing, all levels at
  every probe, double width.
- The no-features control (scalars only: σ, DFL bins, score, spread, borders) is better than every feature variant
  (−0.0046 and −0.0002): the pyramid samples add harm, not signal.
- Shrinking the move (λ 0.25, 0.5) and z / js confidence gates: at or below B*.
- High-resolution probes (H1: input + stem at the same probe points): −0.0010.
- Input strip CNN (H2): +0.0002 [−0.0004, +0.0008]; with 1 px spacing +0.0007 [+0.0001, +0.0013]; small objects
  only; gates do nothing.

## Leftovers

- `bench/refine/` stays as a prototype (feature and input dumps, B* tables, the refiner, training, scoring and the
  diagnostics). `candidate_dump` keeps `--features`, `--aug`, `--input` and `--stem` (examples only, no library
  change).
- Data on D: (can be deleted): feature dumps 43 GB, input dumps 37 GB, under `D:/folo-data/runs-hill3/refine/`.
- `D:/folo-build/hill/lock.sh`: a zeroed owner file (after a hard reboot) is never treated as stale, and `acquire`
  is re-entrant per owner, so a lane's second job takes the lock its first job holds.
- If anyone retries: judge an edge refiner on the primaries first (the boxes AP sees), by size. The one lead left is
  small objects from the input image (h2px: +1.7% of their edge error); it is worth +0.0019 AP_S and +0.0007 AP,
  and in Rust it would need the u8 input kept until after NMS (1.2 MB per image) plus the strip CNN and MLP at
  5.8 M multiply-adds per box (four sides), so about 1.2 GFLOP for 100 refined boxes per image: some 18% of
  folo-n's 6.54 GFLOPs at 640, for +0.0007 AP. Not worth it.
- val-A and val-B were not scored (no configuration reached the dev gate); val-B was never dumped.

## Design errors

- The first two runs trained on `--aug` truths that were not clipped to the visible view, which taught the refiner
  to grow boxes past the image (fixed in `boxes.py`; the runs are kept, marked).
- Training on every box at IoU ≥ 0.3 optimised the many duplicates around a truth, not the top-scoring box AP sees.
  Most of the first round went into that; the primary-box view should have come first.
- The box-branch check read the raw weights instead of the `ema.` copy FOLO infers with (fixed, no result
  affected).
- The brief's spacing rule (s_e from σ_e) was taken as given; checking σ_e against the primaries' actual error
  (Part 5) would have shown it 3–4.5× too wide before any training.
