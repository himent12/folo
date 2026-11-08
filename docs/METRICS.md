# How FOLO computes detection metrics

Implementation: `crates/folo-vision/src/metrics.rs` (`Evaluator`). Tests with hand-computed answers:
`crates/folo-vision/tests/metrics_known.rs`; parity with pycocotools on 24 seeded scenarios:
`crates/folo-vision/tests/coco_parity.rs` (see [Parity with pycocotools](#parity-with-pycocotools)).

**Short version:** mAP follows **pycocotools** (`COCOeval`, `iouType="bbox"`, area range `all`, `maxDets=100`, no
crowd boxes), and matches it to f32 rounding (≤ 3e-8) for every box with area in `[0, 1e10]`. The one intentional
difference, pycocotools' upper area bound, is switched on by `EvalConfig::coco()` (`coco_compatible`), which then
matches it for every input. Precision, recall and F1 are reported the way Ultralytics reports them: at the one
confidence threshold that maximises the class-mean F1 at IoU 0.5. The confusion matrix follows Ultralytics'
conventions.

## Inputs

Per image: predictions `(box xyxy, score, class)` and ground truths `(box xyxy, class)` in the same coordinates (any
units; the evaluator only compares boxes with each other). Predictions normally come from the post-process with the
validation settings (`NmsConfig::val()`: `conf > 0.001`, `iou 0.7`, multi-label, `max_det 300`). Images can be added
in any order, but score ties across images are broken by insertion order (see step 4), exactly like pycocotools
breaks them by image id order.

## 1. IoU

`IoU = intersection / union` of two boxes in `f64` (as pycocotools' `bbIou`); `0` if the boxes do not overlap. No
`+1` pixel convention.

IoU thresholds are `t ∈ {0.50, 0.55, …, 0.95}` computed exactly like `numpy.linspace(0.5, 0.95, 10)` (so the 9th one
is `0.8999999999999999`, not `0.9`; this matters only for IoUs that land exactly on a threshold).

## 2. Per-image matching (for each class and each threshold)

For one image and one class `k`:

1. Take the image's predictions of class `k`, sorted by descending score (stable: equal scores keep input order),
   and keep the first `max_dets_per_class` (default 100, COCO's `maxDets`; extra predictions are *dropped*, not
   counted as false positives).
2. For each threshold `t`, walk the predictions in that order. Each one is matched to the **not yet matched** ground
   truth of class `k` with the highest IoU, provided that IoU is `>= t` (ties: the later ground truth, as in
   pycocotools' loop). A matched prediction is a true positive (TP) at `t`, otherwise a false positive (FP).

Consequences: a duplicate detection of an already matched object is a FP; a prediction of the wrong class is a FP for
its class and the object is a miss for its own class; there is no "ignore" region (FOLO datasets have no crowd
boxes), so every unmatched prediction is a FP. (Except with `coco_compatible`, below.)

### Area range and `coco_compatible`

pycocotools' area range `all` is `[0, 1e5²]`, inclusive, so even "all" ignores some boxes:

- a ground truth whose `area` is outside it is *ignored*: it does not count in `npig`, ignored truths are scanned
  after the regular ones (a prediction that already has a regular match stops there, even if an ignored truth
  overlaps it better), and a prediction matched to an ignored truth is neither a TP nor a FP;
- an *unmatched* prediction whose area (`w * h`) is outside it is ignored rather than a FP. That includes a box
  inverted in exactly one axis (negative area).

FOLO's default leaves this out on purpose: its boxes can be in any unit (`Inputs` above), and a cut-off at a box area
of 10¹⁰ only means something in pixels. For pixel boxes the rule never fires (it needs a side over 100,000 px, or an
inverted box, which FOLO's post-process never makes), so both settings give identical reports.
`EvalConfig::coco()` (`coco_compatible: true`) applies the rule exactly as pycocotools does, using the box area
`(x2 - x1) * (y2 - y1)` in f64. (The original COCO annotation files store the *segment* area as `area`; FOLO only
has boxes, and for the "all" range the difference does not matter in practice.) `ClassReport::truths` then counts
only the regular truths, and a class whose truths are all ignored is not evaluated.

## 3. Counts

For class `k`, `npig` is the number of ground truths of class `k` over all images. Classes with `npig = 0` are
**excluded** from every average (pycocotools gives them AP `-1` and skips them); their predictions still appear in
the confusion matrix. A class with ground truth but no predictions has AP 0. If *no* class has ground truth,
pycocotools reports `-1` for every stat; FOLO reports `0` with an empty `per_class`.

## 4. AP: 101-point interpolation (pycocotools)

For class `k` and threshold `t`:

1. Concatenate all images' kept predictions of class `k` and sort by descending score, **stable** (pycocotools uses
   `mergesort`), so ties stay in image order.
2. Cumulative sums over that order: `tp_i`, `fp_i`.
   `recall_i = tp_i / npig`, `precision_i = tp_i / (tp_i + fp_i + ε)` with `ε = 2.22e-16` (`np.spacing(1)`).
3. Precision envelope, right to left: `precision_{i-1} = max(precision_{i-1}, precision_i)`.
4. For each recall point `r ∈ {0.00, 0.01, …, 1.00}` (101 points, `numpy.linspace(0, 1, 101)`), find the first `i`
   with `recall_i >= r` (`searchsorted(..., side="left")`); the sample is `precision_i`, or `0` if recall never
   reaches `r`.
5. `AP(k, t)` = mean of the 101 samples.

Then:

- `per_class[k].map50 = AP(k, 0.50)`, `per_class[k].map50_95 = mean_t AP(k, t)`, `per_class[k].ap = [AP(k, t)]`.
- `map50 = mean_k AP(k, 0.50)`, `map50_95 = mean_k mean_t AP(k, t)`, `map_per_iou[t] = mean_k AP(k, t)`.

### Worked example

One class, two ground truths, three predictions in score order: TP, FP, TP (IoU 1 each).
`recall = [0.5, 0.5, 1.0]`, `precision = [1, 0.5, 0.667]`, envelope `[1, 0.667, 0.667]`.
Recall points `0.00 … 0.50` (51 points) find `i = 0` → `1`; `0.51 … 1.00` (50 points) find `i = 2` → `0.667`.
`AP = (51 · 1 + 50 · 2/3) / 101 = 0.83498`.

## 5. Precision, recall, F1

At IoU 0.5 only. For a confidence threshold `c`, class `k` counts the kept predictions with `score >= c`:
`P_k(c) = TP / (TP + FP)` (`0` when there are none), `R_k(c) = TP / npig`, `F1_k(c) = 2PR / (P + R)` (`0` when
`P + R = 0`).

`c` runs over a grid `0.000, 0.001, …, 1.000`. `best_conf` is the grid value that maximises `mean_k F1_k(c)`; on ties
the highest such `c` is chosen. The report's `precision`, `recall` and `f1` (and each `per_class` entry's) are the
values at `best_conf`. (Ultralytics does the same selection but interpolates P and R linearly between prediction
scores and smooths the mean F1 curve before the argmax; FOLO uses the exact step functions and no smoothing.)

## 6. Curves (for Studio charts)

`report.curves`:

- `recall`: the 101 recall points; `pr` in each curve set is the interpolated precision from step 4 at IoU 0.5 (the
  PR curve whose mean is AP50).
- `confidence`: `0.00, 0.01, …, 1.00`; `precision`, `recall`, `f1` in each curve set are `P_k(c)`, `R_k(c)`,
  `F1_k(c)` from step 5 at those points.
- `all` is the mean over classes with ground truth; `per_class` has one set per such class.

## 7. Confusion matrix

Ultralytics' convention, `(num_classes + 1)²`, indexed `matrix[predicted][true]`, with index `num_classes` meaning
background:

- Only predictions with `score > 0.25` take part (`EvalConfig::confusion_conf`).
- Per image, prediction–truth pairs with IoU `> 0.45` (`confusion_iou`), **regardless of class**, are matched
  greedily one-to-one in descending IoU order. A match adds 1 at `[pred class][true class]` (the diagonal when the
  classes agree).
- An unmatched ground truth adds 1 at `[background][true class]` (a miss); an unmatched prediction adds 1 at
  `[pred class][background]` (a false alarm).

(Ultralytics removes duplicate matches with two `np.unique` passes rather than a true greedy assignment; the result
is the same except in rare chains of overlaps.)

## Parity with pycocotools

`bench/parity/coco_reference.py` generates 24 seeded scenarios and runs pycocotools 2.0.11 `COCOeval` (bbox, area
`all`, `maxDets` 100) on each. It writes the inputs, the 12 `stats`, mAP per IoU threshold and per-class AP (per
threshold, AP50, AP50-95) to `crates/folo-vision/tests/fixtures/coco/*.json`. The scenarios: 1, 3, 5 and 80 classes;
a long low-confidence tail; images with no truths; images with no predictions; near and exact duplicates; tiny
(1/16 px) and huge (90,000 px) boxes; scores tied on 1 or 3 levels; a class with truths and no predictions and the
reverse; over 100 predictions per image, including a cap that cuts through a score tie; IoUs exactly on the
thresholds; crowded overlapping truths; perfect and all-wrong predictions; zero-area boxes; no truths at all; and
boxes outside the area range. Coordinates are multiples of a power of two and scores have three decimals, so both
sides see bit-identical inputs.

`coco_parity.rs` evaluates every fixture in both modes and compares mAP50-95, mAP50, AP75, mAP at each threshold and
every class's APs with a tolerance of 1e-7. The report's fields are f32, and rounding a value in `[0, 1]` to f32
costs at most 3e-8. Result: the largest difference in any scenario is **3.0e-8**, in both modes, except
`outside_area_range`, where the default is 0.07 off (as intended) and `coco_compatible` is again within 2e-8.
`no_truths_at_all` checks the `-1` / empty convention from step 3. FOLO does not compute the area-split stats
(`APs/m/l`) or AR, so those are only stored in the fixtures. Regenerate the fixtures with
`D:/folo-venvs/parity/Scripts/python bench/parity/coco_reference.py` (a venv with `pycocotools` and `numpy`).

## Differences from Ultralytics' mAP

Ultralytics' `ap_per_class` differs from pycocotools, and so from FOLO, in these ways, so the two usually give
different numbers on the same predictions:

1. **Matching** is by IoU, not by score: all (prediction, truth) pairs of the same class above `t` are taken in
   descending IoU order, so a lower-scored but tighter box can take an object from a higher-scored loose one.
2. **AP integration** is the trapezoidal integral of the precision envelope interpolated at 101 recall points
   (`np.trapz(np.interp(x, mrec, mpre), x)`), with the recall curve extended by `(0, 1)` and `(1, 0)`, instead of
   the mean of 101 left-searched samples. Two side effects follow. First, past the highest recall the curve falls
   linearly to `(1, 0)`, so a class that tops out at recall `r` with precision `p` gets about `p · (1 - r) / 2` of
   extra area that COCO scores as 0. Second, a perfect detector scores 0.995, not 1, because the last trapezoid
   averages 1 and 0.
3. **No per-image cap**: all predictions count (Ultralytics' `max_det = 300` per image applies before evaluation).
   Set `EvalConfig::max_dets_per_class = None` to match that part.
4. **Precision details**: IoU is computed in f32 with `+1e-7` in the denominator, and the thresholds are f32, so an
   IoU exactly on a threshold falls just below it. Score ties are ordered by an unstable `np.argsort`.

How close it is, measured on the 23 parity scenarios that have truths. The reference script includes a numpy port
of `ap_per_class` and `match_predictions`. Ultralytics minus COCO:

| | median | mean | range |
|---|---|---|---|
| mAP50-95 | +0.026 | +0.017 | −0.183 … +0.054 |
| mAP50 | +0.012 | +0.000 | −0.340 … +0.079 |

On ordinary detector output (`five_classes`, `eighty_classes`, `mixed_scales`, `crowded`) Ultralytics' mAP50-95 is
0.02–0.04 higher, mostly from the triangle in (2). The gap grows when recall is low (`empty_pred_images` +0.054)
and flips sign when duplicates' scores don't track their IoU (`duplicates` −0.18 / −0.34 at AP50), because of (1).
So Ultralytics and COCO numbers are not interchangeable, even on identical predictions.

COCO numbers (what papers report) are the pycocotools ones, which is why FOLO matches pycocotools.
