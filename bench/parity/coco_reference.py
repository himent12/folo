"""Reference fixtures for FOLO's COCO mAP parity test (crates/folo-vision/tests/coco_parity.rs).

Generates seeded random detection scenarios, evaluates each with pycocotools' COCOeval (bbox, area range "all",
maxDets 100) and writes the inputs plus the reference numbers to crates/folo-vision/tests/fixtures/coco/<name>.json.
The scenarios with crowd regions and annotation areas for the full COCOeval port (crates/folo-vision/src/cocoeval.rs,
all twelve summary numbers) go to crates/folo-vision/tests/fixtures/coco_full/<name>.json.
It also computes Ultralytics' AP (a numpy port of ultralytics 8.4.198 and of the older 8.x algorithm, see below) so
docs/METRICS.md can say how far that method is from COCO's; the Rust test does not check those.

Setup (once, outside the repo):

    python -m venv D:/folo-venvs/parity
    D:/folo-venvs/parity/Scripts/python -m pip install pycocotools numpy

Run from the repo root:

    D:/folo-venvs/parity/Scripts/python bench/parity/coco_reference.py

Exactness: box coordinates are multiples of a power-of-two step and so exactly representable in f32, which FOLO
stores; scores have three decimals, so parsing them to f32 (FOLO) or f64 (pycocotools) keeps their order and ties.
Both sides then compute IoU in f64 from identical inputs and agree bit for bit.
"""

import contextlib
import io
import json
import os
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics_port import ULTRALYTICS_VERSION, run_ultralytics  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "folo-vision", "tests", "fixtures", "coco")

STAT_NAMES = [
    "AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
]


# --------------------------------------------------------------------------------------------------------------------
# Scenario generation
# --------------------------------------------------------------------------------------------------------------------


class Gen:
    """Box and score helpers on a seeded RNG. Every coordinate is snapped to a multiple of `q` (a power of two)."""

    def __init__(self, seed, q=0.25):
        self.rng = np.random.default_rng(seed)
        self.q = q

    def snap(self, v):
        return float(np.round(v / self.q) * self.q)

    def box(self, size, lo, hi):
        """A box inside `size x size` with sides log-uniform in [lo, hi]."""
        w, h = np.exp(self.rng.uniform(np.log(lo), np.log(hi), 2))
        x = self.rng.uniform(0, max(size - w, 0))
        y = self.rng.uniform(0, max(size - h, 0))
        return self.fix([x, y, x + w, y + h])

    def fix(self, b):
        x1, y1, x2, y2 = (self.snap(v) for v in b)
        # Keep boxes non-inverted (and at least one step wide) unless a scenario builds degenerate ones on purpose.
        x2 = max(x2, x1 + self.q)
        y2 = max(y2, y1 + self.q)
        return [x1, y1, x2, y2]

    def jitter(self, b, frac):
        w, h = b[2] - b[0], b[3] - b[1]
        d = self.rng.uniform(-frac, frac, 4) * np.array([w, h, w, h])
        return self.fix([b[0] + d[0], b[1] + d[1], b[2] + d[2], b[3] + d[3]])

    def score(self, lo=0.001, hi=1.0):
        return round(float(self.rng.uniform(lo, hi)), 3)


def detector(g, truths, nc, *, size=640, p_detect=0.85, jitter=0.15, p_dup=0.1, p_wrong=0.05, n_fp=(0, 6),
             lo=8, hi=300, score_tp=(0.3, 1.0), score_fp=(0.001, 0.6)):
    """A plausible detector output for one image: jittered hits, some duplicates, wrong classes and background FPs."""
    preds = []
    for t in truths:
        if g.rng.random() < p_detect:
            c = t[4] if g.rng.random() >= p_wrong else int(g.rng.integers(nc))
            preds.append(g.jitter(t[:4], jitter) + [g.score(*score_tp), c])
            if g.rng.random() < p_dup:
                preds.append(g.jitter(t[:4], jitter * 2) + [g.score(*score_fp), c])
    for _ in range(int(g.rng.integers(n_fp[0], n_fp[1] + 1))):
        preds.append(g.box(size, lo, hi) + [g.score(*score_fp), int(g.rng.integers(nc))])
    g.rng.shuffle(preds)
    return preds


def random_truths(g, nc, n, *, size=640, lo=8, hi=300, classes=None):
    classes = classes if classes is not None else list(range(nc))
    return [g.box(size, lo, hi) + [int(g.rng.choice(classes))] for _ in range(n)]


def generic(seed, nc, n_images, *, gts=(0, 8), q=0.25, size=640, lo=8, hi=300, **det):
    g = Gen(seed, q)
    images = []
    for _ in range(n_images):
        truths = random_truths(g, nc, int(g.rng.integers(gts[0], gts[1] + 1)), size=size, lo=lo, hi=hi)
        images.append({"truths": truths, "preds": detector(g, truths, nc, size=size, lo=lo, hi=hi, **det)})
    return images


def sc_empty_gt_images(seed):
    # Half the images have no ground truth at all; their predictions are pure false positives.
    g = Gen(seed)
    images = []
    for i in range(40):
        truths = random_truths(g, 3, int(g.rng.integers(1, 6))) if i % 2 == 0 else []
        images.append({"truths": truths, "preds": detector(g, truths, 3, n_fp=(1, 5))})
    return images


def sc_empty_pred_images(seed):
    # Every third image has ground truth but no predictions (pure misses).
    g = Gen(seed)
    images = []
    for i in range(45):
        truths = random_truths(g, 3, int(g.rng.integers(1, 6)))
        preds = [] if i % 3 == 0 else detector(g, truths, 3)
        images.append({"truths": truths, "preds": preds})
    return images


def sc_duplicates(seed):
    # Every hit comes with 1-4 near-duplicates at lower and higher scores (duplicates of a matched object are FPs).
    g = Gen(seed)
    images = []
    for _ in range(30):
        truths = random_truths(g, 2, int(g.rng.integers(1, 6)))
        preds = []
        for t in truths:
            for _ in range(int(g.rng.integers(1, 5))):
                preds.append(g.jitter(t[:4], 0.12) + [g.score(0.2, 1.0), t[4]])
        images.append({"truths": truths, "preds": preds})
    return images


def sc_exact_duplicates(seed):
    # Identical boxes with identical scores inside an image: the tie is broken by input order on both sides.
    g = Gen(seed)
    images = []
    for _ in range(25):
        truths = random_truths(g, 2, int(g.rng.integers(1, 5)))
        preds = []
        for t in truths:
            b = g.jitter(t[:4], 0.2)
            s = g.score(0.1, 1.0)
            preds += [b + [s, t[4]] for _ in range(int(g.rng.integers(1, 4)))]
        preds += [g.box(640, 8, 200) + [g.score(), int(g.rng.integers(2))] for _ in range(2)]
        g.rng.shuffle(preds)
        images.append({"truths": truths, "preds": preds})
    return images


def sc_score_ties(seed, levels):
    # Scores drawn from a handful of levels: massive ties within and across images.
    g = Gen(seed)
    images = []
    for _ in range(40):
        truths = random_truths(g, 3, int(g.rng.integers(0, 7)))
        preds = detector(g, truths, 3, n_fp=(0, 5))
        for p in preds:
            p[4] = float(g.rng.choice(levels))
        images.append({"truths": truths, "preds": preds})
    return images


def sc_class_without_preds(seed):
    # Class 2 has ground truth but is never predicted (AP 0, counted in the mean).
    g = Gen(seed)
    images = []
    for _ in range(30):
        truths = random_truths(g, 3, int(g.rng.integers(1, 6)))
        preds = [p for p in detector(g, truths, 3, p_wrong=0) if p[5] != 2]
        images.append({"truths": truths, "preds": preds})
    return images


def sc_class_without_gts(seed):
    # Classes 3 and 4 are predicted but never annotated (excluded from the mean, like pycocotools' -1).
    g = Gen(seed)
    images = []
    for _ in range(30):
        truths = random_truths(g, 5, int(g.rng.integers(1, 6)), classes=[0, 1, 2])
        preds = detector(g, truths, 5, n_fp=(1, 6))
        images.append({"truths": truths, "preds": preds})
    return images


def sc_over_max_dets(seed):
    # 130-180 predictions of one class per image, true positives spread over all ranks: the 100-per-image cap drops
    # the lowest-scored ones (including some true positives) before matching.
    g = Gen(seed)
    images = []
    for _ in range(4):
        truths = random_truths(g, 2, 25, lo=10, hi=120)
        preds = [g.jitter(t[:4], 0.1) + [g.score(0.05, 1.0), t[4]] for t in truths]
        for _ in range(int(g.rng.integers(130, 181))):
            preds.append(g.box(640, 5, 80) + [g.score(0.05, 1.0), 0])
        g.rng.shuffle(preds)
        images.append({"truths": truths, "preds": preds})
    return images


def sc_max_dets_tie_cut(seed):
    # 150 predictions per image all at score 0.5: which 100 survive the cap is decided by input order.
    g = Gen(seed)
    images = []
    for _ in range(3):
        truths = random_truths(g, 1, 40, lo=10, hi=100)
        preds = [g.box(640, 5, 60) + [0.5, 0] for _ in range(110)]
        preds += [g.jitter(t[:4], 0.1) + [0.5, 0] for t in truths]
        g.rng.shuffle(preds)
        images.append({"truths": truths, "preds": preds})
    return images


def sc_iou_boundaries(seed):
    # IoUs that land exactly on the thresholds: a 100x100 truth and predictions [0, 0, w, 100] have IoU w / 100.
    # w = 50, 55, ..., 95 hits every threshold exactly (and 90 / 100 = 0.9 > linspace's 0.8999999999999999).
    g = Gen(seed)
    images = []
    for i in range(30):
        ox, oy = 200.0 * (i % 3), 200.0 * (i // 3 % 3)
        truths = [[ox, oy, ox + 100.0, oy + 100.0, 0]]
        w = float(g.rng.choice([45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]))
        preds = [[ox, oy, ox + w, oy + 100.0, g.score(0.2, 1.0), 0]]
        if g.rng.random() < 0.4:
            w2 = float(g.rng.choice([50, 75, 95]))
            preds.append([ox, oy, ox + 100.0, oy + w2, g.score(0.2, 1.0), 0])
        images.append({"truths": truths, "preds": preds})
    return images


def sc_crowded(seed):
    # Heavily overlapping truths: a prediction often has several candidate truths and the greedy choice matters.
    g = Gen(seed)
    images = []
    for _ in range(25):
        cx, cy = g.rng.uniform(100, 500, 2)
        truths = []
        for _ in range(int(g.rng.integers(3, 9))):
            w, h = g.rng.uniform(30, 80, 2)
            dx, dy = g.rng.uniform(-25, 25, 2)
            truths.append(g.fix([cx + dx - w / 2, cy + dy - h / 2, cx + dx + w / 2, cy + dy + h / 2]) + [0])
        images.append({"truths": truths, "preds": detector(g, truths, 1, jitter=0.25, p_dup=0.3, n_fp=(0, 3))})
    return images


def sc_perfect(seed):
    g = Gen(seed)
    images = []
    for _ in range(20):
        truths = random_truths(g, 4, int(g.rng.integers(0, 6)))
        images.append({"truths": truths, "preds": [t[:4] + [g.score(0.5, 1.0), t[4]] for t in truths]})
    return images


def sc_all_wrong(seed):
    # Every prediction is in the right place with the wrong class, or of the right class in the wrong place.
    g = Gen(seed)
    images = []
    for _ in range(20):
        truths = random_truths(g, 3, int(g.rng.integers(1, 5)))
        preds = []
        for t in truths:
            preds.append(t[:4] + [g.score(), (t[4] + 1) % 3])
            b = t[:4]
            preds.append([b[0] + 640, b[1], b[2] + 640, b[3], g.score(), t[4]])
        images.append({"truths": truths, "preds": preds})
    return images


def sc_degenerate(seed):
    # Zero-width and zero-height boxes (area 0, inside COCO's "all" range): they never overlap anything, so a
    # degenerate prediction is a FP and a degenerate truth is a miss.
    g = Gen(seed)
    images = []
    for _ in range(25):
        truths = random_truths(g, 2, int(g.rng.integers(1, 5)))
        if g.rng.random() < 0.4:
            b = g.box(640, 8, 100)
            truths.append([b[0], b[1], b[0], b[3], int(g.rng.integers(2))])
        preds = detector(g, truths, 2)
        for _ in range(int(g.rng.integers(0, 3))):
            b = g.box(640, 8, 100)
            preds.append([b[0], b[1], b[2], b[1], g.score(), int(g.rng.integers(2))])
        images.append({"truths": truths, "preds": preds})
    return images


def sc_no_truths_at_all(seed):
    # No ground truth anywhere: pycocotools reports -1 everywhere, FOLO an empty per_class and 0 summaries.
    g = Gen(seed)
    return [{"truths": [], "preds": detector(g, [], 2, n_fp=(0, 4))} for _ in range(10)]


def sc_outside_area_range(seed):
    # Boxes beyond COCO's "all" area range (area > 1e5²) and inverted predictions (negative area). pycocotools
    # ignores those truths and the unmatched ones of those predictions; FOLO only does so with `coco_compatible`.
    g = Gen(seed, q=1.0)
    images = []
    for _ in range(20):
        size = 400000
        truths = [g.box(size, 50, 1000) + [int(g.rng.integers(2))] for _ in range(int(g.rng.integers(1, 4)))]
        truths += [g.box(size, 60000, 200000) + [int(g.rng.integers(2))] for _ in range(int(g.rng.integers(0, 3)))]
        preds = detector(g, truths, 2, size=size, lo=50, hi=200000, n_fp=(0, 3))
        for _ in range(int(g.rng.integers(0, 3))):
            b = g.box(size, 50, 1000)
            if g.rng.random() < 0.5:
                b = [b[2], b[1], b[0], b[3]]  # inverted in x only: negative area
            else:
                b = [b[2], b[3], b[0], b[1]]  # inverted in both: positive area, still never overlaps
            preds.append(b + [g.score(), int(g.rng.integers(2))])
        images.append({"truths": truths, "preds": preds})
    # Hand-made: a prediction of area exactly 1e10 (inside the range) overlaps an ignored truth (area 1.00001e10, IoU
    # 0.99999) better than a regular one (IoU 0.9999). pycocotools still matches it to the regular truth (it stops
    # scanning at the ignored ones once it has a regular match); the next, identical prediction takes the ignored
    # truth and is ignored, not a FP. The truth order is swapped in the second image (ignored ones sort last).
    reg, ign, big = [0, 0, 99990, 100000], [0, 0, 100000, 100001], [0, 0, 100000, 100000]
    images.append({"truths": [reg + [0], ign + [0]], "preds": [big + [0.7, 0], big + [0.6, 0]]})
    images.append({"truths": [ign + [1], reg + [1]], "preds": [big + [0.95, 1], big + [0.3, 1], big + [0.2, 1]]})
    return images


SCENARIOS = [
    ("single_class", "1 class, 40 images, jittered hits, duplicates and background FPs",
     lambda: generic(101, 1, 40)),
    ("five_classes", "5 classes, 60 images, typical detector output",
     lambda: generic(102, 5, 60)),
    ("eighty_classes", "80 classes (COCO-sized), 60 sparse images: many classes with few truths",
     lambda: generic(103, 80, 60, gts=(1, 6), p_wrong=0.1)),
    ("low_conf_tail", "3 classes, 60 images, a long tail of conf 0.001-0.05 predictions like NmsConfig::val()",
     lambda: generic(104, 3, 60, n_fp=(5, 20), score_fp=(0.001, 0.05))),
    ("empty_gt_images", "half the images have no truths (their predictions are all FPs)",
     lambda: sc_empty_gt_images(105)),
    ("empty_pred_images", "every third image has truths and no predictions",
     lambda: sc_empty_pred_images(106)),
    ("duplicates", "every hit has 1-4 near-duplicates at random scores",
     lambda: sc_duplicates(107)),
    ("exact_duplicates", "identical boxes with identical scores within an image",
     lambda: sc_exact_duplicates(108)),
    ("tiny_boxes", "sides 0.0625-4 px on a 1/16 px grid",
     lambda: generic(109, 2, 40, q=1 / 16, size=64, lo=0.0625, hi=4, jitter=0.2)),
    ("huge_boxes", "sides 2,000-90,000 px (area up to 8.1e9, inside COCO's 1e10 'all' range)",
     lambda: generic(110, 2, 40, q=1.0, size=100000, lo=2000, hi=90000)),
    ("mixed_scales", "sides 1-600 px, small / medium / large all present",
     lambda: generic(111, 4, 50, lo=1, hi=600)),
    ("ties_three_levels", "scores drawn from {0.1, 0.5, 0.9}: ties within and across images",
     lambda: sc_score_ties(112, [0.1, 0.5, 0.9])),
    ("ties_single_level", "every score is 0.5: ranking is pure image / input order",
     lambda: sc_score_ties(113, [0.5])),
    ("class_without_preds", "class 2 has truths but no predictions (AP 0, in the mean)",
     lambda: sc_class_without_preds(114)),
    ("class_without_truths", "classes 3, 4 are predicted but never annotated (excluded from the mean)",
     lambda: sc_class_without_gts(115)),
    ("over_max_dets", "130-180 predictions per image: the top-100 cap drops low-scored TPs",
     lambda: sc_over_max_dets(116)),
    ("max_dets_tie_cut", "135-150 predictions per image all at score 0.5: the cap cuts through a tie",
     lambda: sc_max_dets_tie_cut(117)),
    ("iou_boundaries", "IoUs exactly 0.45, 0.50, ..., 1.00 (on the thresholds)",
     lambda: sc_iou_boundaries(118)),
    ("crowded", "heavily overlapping truths; greedy matching choices matter",
     lambda: sc_crowded(119)),
    ("perfect", "predictions equal the truths (AP 1)",
     lambda: sc_perfect(120)),
    ("all_wrong", "right place wrong class, or right class wrong place (AP 0)",
     lambda: sc_all_wrong(121)),
    ("degenerate_boxes", "zero-width / zero-height truths and predictions (area 0)",
     lambda: sc_degenerate(122)),
    ("no_truths_at_all", "no truths anywhere (pycocotools: all -1)",
     lambda: sc_no_truths_at_all(123)),
    ("outside_area_range", "boxes with area > 1e10 and inverted predictions: needs coco_compatible",
     lambda: sc_outside_area_range(124)),
]

# Scenarios where FOLO's default mode is expected to differ from pycocotools (documented in docs/METRICS.md).
NEEDS_COCO_COMPATIBLE = {"outside_area_range"}


# Full-COCOeval scenarios (crates/folo-vision/tests/coco_full.rs, fixtures/coco_full/): truths are
# [x1, y1, x2, y2, class, iscrowd, area], where `area` is an annotation area that differs from the box area (as a
# segmentation area does) and buckets small / medium / large by it.


def seg_area(g, b, lo=0.4, hi=1.0):
    """An annotation area: the box area times U(lo, hi), on a 1/4 grid (exact in f64 on both sides)."""
    a = (b[2] - b[0]) * (b[3] - b[1]) * float(g.rng.uniform(lo, hi))
    return float(np.round(a * 4) / 4)


def sc_crowd_regions(seed):
    # Regular objects plus 0-2 crowd regions per image; the detector also fires inside the crowds (those
    # detections are ignored, not FPs, and crowds can absorb any number of them).
    g = Gen(seed)
    images = []
    for _ in range(40):
        truths = [t + [0, seg_area(g, t)] for t in random_truths(g, 3, int(g.rng.integers(0, 7)), lo=6, hi=250)]
        preds = detector(g, truths, 3, lo=6, hi=250)
        for _ in range(int(g.rng.integers(0, 3))):
            c = int(g.rng.integers(3))
            crowd = g.box(640, 120, 400)
            truths.append(crowd + [c, 1, seg_area(g, crowd)])
            for _ in range(int(g.rng.integers(1, 6))):
                w, h = g.rng.uniform(10, 80, 2)
                x = g.rng.uniform(crowd[0], max(crowd[2] - w, crowd[0]))
                y = g.rng.uniform(crowd[1], max(crowd[3] - h, crowd[1]))
                cls = c if g.rng.random() < 0.8 else int(g.rng.integers(3))
                preds.append(g.fix([x, y, x + w, y + h]) + [g.score(), cls])
        g.rng.shuffle(truths)
        images.append({"truths": truths, "preds": preds})
    return images


def sc_bucket_edges(seed):
    # Sides 4-300 px; annotation areas include the bucket edges 32^2 and 96^2 exactly (inclusive on both sides,
    # so such a truth counts in two buckets), and predictions whose box area straddles them.
    g = Gen(seed)
    images = []
    for _ in range(50):
        truths = []
        for t in random_truths(g, 4, int(g.rng.integers(1, 9)), lo=4, hi=300):
            r = g.rng.random()
            area = 1024.0 if r < 0.15 else 9216.0 if r < 0.3 else seg_area(g, t, 0.3, 1.2)
            truths.append(t + [0, area])
        images.append({"truths": truths, "preds": detector(g, truths, 4, lo=4, hi=300, n_fp=(0, 8))})
    return images


def sc_crowd_only_class(seed):
    # Class 1 only ever appears as a crowd: it has no regular truth, so it is left out of every mean even though
    # it is predicted; class 0 is ordinary.
    g = Gen(seed)
    images = []
    for _ in range(30):
        truths = [t + [0, seg_area(g, t)] for t in random_truths(g, 1, int(g.rng.integers(1, 5)))]
        preds = detector(g, truths, 1)
        if g.rng.random() < 0.5:
            crowd = g.box(640, 100, 300)
            truths.append(crowd + [1, 1, seg_area(g, crowd)])
            preds.append(g.jitter(crowd, 0.1) + [g.score(), 1])
        images.append({"truths": truths, "preds": preds})
    return images


SCENARIOS_FULL = [
    ("crowd_regions", "regular truths plus crowd regions that absorb detections", lambda: sc_crowd_regions(201)),
    ("bucket_edges", "annotation areas on and around the 32^2 / 96^2 bucket edges", lambda: sc_bucket_edges(202)),
    ("crowd_only_class", "a class with only crowd truths (excluded from the means)",
     lambda: sc_crowd_only_class(203)),
]
OUT_DIR_FULL = os.path.join(os.path.dirname(OUT_DIR), "coco_full")


# --------------------------------------------------------------------------------------------------------------------
# pycocotools
# --------------------------------------------------------------------------------------------------------------------


def run_cocoeval(images, nc):
    """COCOeval on xyxy truths / predictions. Class k is category id k + 1, image i is id i + 1 (ids must be >= 1:
    COCOeval stores matched truth ids in `dtMatches` and treats 0 as unmatched)."""
    gt = {
        "images": [{"id": i + 1, "width": 1, "height": 1} for i in range(len(images))],
        "categories": [{"id": k + 1, "name": str(k)} for k in range(nc)],
        "annotations": [],
    }
    dets = []
    for i, im in enumerate(images):
        for t in im["truths"]:
            w, h = t[2] - t[0], t[3] - t[1]
            crowd, area = (t[5], t[6]) if len(t) > 5 else (0, w * h)
            gt["annotations"].append({
                "id": len(gt["annotations"]) + 1, "image_id": i + 1, "category_id": t[4] + 1,
                "bbox": [t[0], t[1], w, h], "area": area, "iscrowd": crowd,
            })
        for p in im["preds"]:
            dets.append({"image_id": i + 1, "category_id": p[5] + 1, "bbox": [p[0], p[1], p[2] - p[0], p[3] - p[1]],
                         "score": p[4]})
    assert dets, "loadRes needs at least one detection"
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    prec = ev.eval["precision"][:, :, :, 0, 2]  # [T, R, K], area "all", maxDets 100

    def mean_valid(a):
        a = a[a > -1]
        return float(np.mean(a)) if a.size else -1.0

    per_class = []
    for k in range(nc):
        if (prec[:, :, k] > -1).any():
            per_class.append({
                "class": k,
                "ap": [mean_valid(prec[t, :, k]) for t in range(prec.shape[0])],
                "ap50_95": mean_valid(prec[:, :, k]),
                "ap50": mean_valid(prec[0, :, k]),
            })
    return {
        "stats": [float(s) for s in ev.stats],
        "map50_95": mean_valid(prec),
        "map50": mean_valid(prec[0]),
        "map_per_iou": [mean_valid(prec[t]) for t in range(prec.shape[0])],
        "per_class": per_class,
    }


# --------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------


def num(v):
    """Shortest JSON for a coordinate or score: integers without '.0'."""
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def write_fixture(path, fx):
    """Compact JSON with one image per line, so the fixtures stay small and diffable."""
    images = fx.pop("images")
    head = json.dumps(fx, indent=1)
    lines = []
    for im in images:
        t = ",".join("[" + ",".join(num(v) for v in b) + "]" for b in im["truths"])
        p = ",".join("[" + ",".join(num(v) for v in b) + "]" for b in im["preds"])
        lines.append('{"truths":[' + t + '],"preds":[' + p + "]}")
    with open(path, "w", newline="\n") as f:
        f.write(head[:-2] + ',\n "images": [\n  ' + ",\n  ".join(lines) + "\n ]\n}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    total = 0
    for name, desc, make in SCENARIOS:
        images = make()
        nc = 1 + max([b[4] for im in images for b in im["truths"]] + [b[5] for im in images for b in im["preds"]])
        coco = run_cocoeval(images, nc)
        ul = run_ultralytics(images)
        ul_legacy = run_ultralytics(images, legacy=True)
        fx = {
            "name": name,
            "description": desc,
            "num_classes": nc,
            "coco_compatible_only": name in NEEDS_COCO_COMPATIBLE,
            "coco": {"stat_names": STAT_NAMES, **coco},
            "ultralytics": ul and {"version": ULTRALYTICS_VERSION, **ul},
            "ultralytics_legacy": ul_legacy,
            "images": images,
        }
        path = os.path.join(OUT_DIR, name + ".json")
        write_fixture(path, fx)
        size = os.path.getsize(path)
        total += size
        n_t = sum(len(im["truths"]) for im in images)
        n_p = sum(len(im["preds"]) for im in images)
        rows.append((name, len(images), nc, n_t, n_p, coco["map50_95"], coco["map50"],
                     *(u[k] if u else None for u in (ul, ul_legacy) for k in ("map50_95", "map50")), size))
    print(f"{'scenario':22} {'imgs':>4} {'nc':>3} {'gts':>5} {'preds':>5} {'AP':>8} {'AP50':>8} {'UL AP':>8} "
          f"{'UL AP50':>8} {'old AP':>8} {'old AP50':>8} {'bytes':>7}")
    for r in rows:
        fmt = lambda v: f"{v:8.4f}" if v is not None else "       -"
        print(f"{r[0]:22} {r[1]:4} {r[2]:3} {r[3]:5} {r[4]:5} " + " ".join(fmt(v) for v in r[5:11]) + f" {r[11]:7}")
    print(f"total {total} bytes in {len(rows)} fixtures", file=sys.stderr)

    os.makedirs(OUT_DIR_FULL, exist_ok=True)
    for name, desc, make in SCENARIOS_FULL:
        images = make()
        nc = 1 + max([b[4] for im in images for b in im["truths"]] + [b[5] for im in images for b in im["preds"]])
        coco = run_cocoeval(images, nc)
        fx = {"name": name, "description": desc, "num_classes": nc, "coco": {"stat_names": STAT_NAMES, **coco},
              "images": images}
        write_fixture(os.path.join(OUT_DIR_FULL, name + ".json"), fx)
        print(f"{name:22} " + " ".join(f"{v:.4f}" for v in coco["stats"]))


if __name__ == "__main__":
    main()
