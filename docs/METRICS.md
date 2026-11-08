# How FOLO computes detection metrics

Implementation: `crates/folo-vision/src/metrics.rs` (`Evaluator`). Tests with hand-computed answers:
`crates/folo-vision/tests/metrics_known.rs`.

**Short version:** mAP follows **pycocotools** (`COCOeval`, `iouType="bbox"`, area range `all`, `maxDets=100`, no
crowd boxes). Precision, recall and F1 are reported the way Ultralytics reports them: at the one confidence threshold
that maximises the class-mean F1 at IoU 0.5. The confusion matrix follows Ultralytics' conventions.

## Inputs

Per image: predictions `(box xyxy, score, class)` and ground truths `(box xyxy, class)` in the same coordinates (any
units; the evaluator only compares boxes with each other). Predictions normally come from the post-process with the
validation settings of `docs/MODEL.md` §8.7 / §9.3 (`NmsConfig::val()`: `conf > 0.001`, `iou 0.7`, multi-label,
class-aware, `max_det 300`); the trainer validates after every epoch in network-input pixels and uses `map50_95` as
its fitness. Of the up to 300 detections per image, step 2 keeps at most 100 per class. Images can be added
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
boxes), so every unmatched prediction is a FP.

## 3. Counts

For class `k`, `npig` is the number of ground truths of class `k` over all images. Classes with `npig = 0` are
**excluded** from every average (pycocotools gives them AP `-1` and skips them); their predictions still appear in
the confusion matrix. A class with ground truth but no predictions has AP 0.

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

## Differences from Ultralytics' mAP

Ultralytics' `ap_per_class` differs from pycocotools, and so from FOLO, in three ways, so the two usually give
slightly different numbers on the same predictions:

1. **Matching** is by IoU, not by score: all (prediction, truth) pairs of the same class above `t` are taken in
   descending IoU order, so a lower-scored but tighter box can take an object from a higher-scored loose one.
2. **AP integration** is the trapezoidal integral of the precision envelope interpolated at 101 recall points
   (`np.trapz(np.interp(x, mrec, mpre), x)`), with the recall curve extended by `(0, 1)` and `(1, 0)`, instead of
   the mean of 101 left-searched samples.
3. **No per-image cap**: all predictions count (Ultralytics' `max_det = 300` per image applies before evaluation).
   Set `EvalConfig::max_dets_per_class = None` to match that part.

COCO numbers (what papers report) are the pycocotools ones, which is why FOLO matches pycocotools.
