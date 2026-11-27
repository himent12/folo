// Client-side per-image error-analysis diagnostics, filtering, and sorting for Benchmark Entry images.
// Computes TP / FP / FN tallies, per-box IoU, class-mismatch / localization / duplicate / background FP causes,
// and diagnostic sort orders directly from existing GT and prediction payloads without any backend API changes.
import type { BenchEntryImage } from '$lib/api/types';
import { iou, MATCH_IOU, matchImage } from '$lib/bench/match';
import { num } from '$lib/format';
import type { Outcome } from './MatchView';

export type ErrorFilter = 'all' | 'fn' | 'fp' | 'errors';

export type ErrorSort = 'worst' | 'order' | 'fn' | 'fp' | 'lowest-iou' | 'lowest-score';

export type BoxCause = 'tp' | 'class-mismatch' | 'localization' | 'duplicate' | 'background' | 'missed';

export interface BoxDiagnostic {
  /** Key matching MatchView (`d<i>` for detection i, `t<i>` for true box i). */
  key: string;
  outcome: Outcome;
  cause: BoxCause;
  cls: number;
  /** Overlapping GT or prediction class index when a class mismatch occurred, else null. */
  otherCls: number | null;
  score: number | null;
  /** Matched IoU for TP, or highest overlapping IoU for FP/FN when overlap >= 0.1, else null. */
  iou: number | null;
  /** Concise human-readable diagnosis shown in the inspector sidebar. */
  detail: string;
}

export interface ImageDiagnostics {
  tp: number;
  fp: number;
  fn: number;
  errors: number;
  truthCount: number;
  meanIou: number | null;
  minIou: number | null;
  minScore: number | null;
  meanScore: number | null;
  boxes: BoxDiagnostic[];
}

export interface AnalyzedImage {
  item: BenchEntryImage;
  /** Index in the loaded images array. */
  index: number;
  diag: ImageDiagnostics;
}

export interface GalleryErrorSummary {
  total: number;
  withFn: number;
  withFp: number;
  withErrors: number;
  tp: number;
  fp: number;
  fn: number;
}

const OUTCOME_ORDER: Record<Outcome, number> = { wrong: 0, missed: 1, right: 2 };
const OVERLAP_FLOOR = 0.1;

function clsName(classes: readonly string[], c: number): string {
  return classes[c] ?? `class ${c}`;
}

/** Diagnoses every detection and missed ground-truth box in an evaluated benchmark image. */
export function diagnoseImage(im: BenchEntryImage, classes: readonly string[]): ImageDiagnostics {
  const truth = im.image.boxes;
  const dets = im.detections;
  const match = matchImage(truth, dets);

  let iouSum = 0;
  let iouCount = 0;
  let minIou = Infinity;
  let scoreSum = 0;
  let scoreCount = 0;
  let minScore = Infinity;

  const boxes: BoxDiagnostic[] = [];

  for (const d of match.detections) {
    const sc = d.det.score ?? null;
    if (sc !== null && Number.isFinite(sc)) {
      scoreSum += sc;
      scoreCount++;
      if (sc < minScore) minScore = sc;
    }

    if (d.truth !== null) {
      iouSum += d.iou;
      iouCount++;
      if (d.iou < minIou) minIou = d.iou;
      boxes.push({
        key: `d${d.index}`,
        outcome: 'right',
        cause: 'tp',
        cls: d.det.class,
        otherCls: null,
        score: sc,
        iou: d.iou,
        detail: `TP · IoU ${num(d.iou, 2)}`,
      });
      continue;
    }

    // False positive: classify why it failed to match a GT box (duplicate, class mismatch, localization, or background).
    let bestSameIou = 0;
    let bestDiffIou = 0;
    let bestDiffCls: number | null = null;
    for (const gt of truth) {
      const v = iou(gt, d.det);
      if (gt.class === d.det.class) {
        if (v > bestSameIou) bestSameIou = v;
      } else if (v > bestDiffIou) {
        bestDiffIou = v;
        bestDiffCls = gt.class;
      }
    }

    if (bestSameIou >= MATCH_IOU) {
      boxes.push({
        key: `d${d.index}`,
        outcome: 'wrong',
        cause: 'duplicate',
        cls: d.det.class,
        otherCls: null,
        score: sc,
        iou: bestSameIou,
        detail: `Duplicate · IoU ${num(bestSameIou, 2)}`,
      });
    } else if (bestDiffIou >= OVERLAP_FLOOR && bestDiffCls !== null && bestDiffIou >= bestSameIou) {
      boxes.push({
        key: `d${d.index}`,
        outcome: 'wrong',
        cause: 'class-mismatch',
        cls: d.det.class,
        otherCls: bestDiffCls,
        score: sc,
        iou: bestDiffIou,
        detail: `Class mismatch (GT: ${clsName(classes, bestDiffCls)}) · IoU ${num(bestDiffIou, 2)}`,
      });
    } else if (bestSameIou >= OVERLAP_FLOOR) {
      boxes.push({
        key: `d${d.index}`,
        outcome: 'wrong',
        cause: 'localization',
        cls: d.det.class,
        otherCls: null,
        score: sc,
        iou: bestSameIou,
        detail: `Localization · IoU ${num(bestSameIou, 2)}`,
      });
    } else {
      boxes.push({
        key: `d${d.index}`,
        outcome: 'wrong',
        cause: 'background',
        cls: d.det.class,
        otherCls: null,
        score: sc,
        iou: null,
        detail: 'Background FP',
      });
    }
  }

  // Missed ground-truth boxes (FN): check if an unmatched FP overlapped them for richer diagnosis.
  const wrongDets = match.detections.filter((d) => d.truth === null);
  truth.forEach((gt, idx) => {
    if (match.found[idx] !== null) return;
    let bestSameIou = 0;
    let bestDiffIou = 0;
    let bestDiffCls: number | null = null;
    for (const fp of wrongDets) {
      const v = iou(gt, fp.det);
      if (fp.det.class === gt.class && v > bestSameIou) bestSameIou = v;
      if (fp.det.class !== gt.class && v > bestDiffIou) {
        bestDiffIou = v;
        bestDiffCls = fp.det.class;
      }
    }
    if (bestDiffIou >= OVERLAP_FLOOR && bestDiffCls !== null && bestDiffIou >= bestSameIou) {
      boxes.push({
        key: `t${idx}`,
        outcome: 'missed',
        cause: 'class-mismatch',
        cls: gt.class,
        otherCls: bestDiffCls,
        score: null,
        iou: bestDiffIou,
        detail: `Missed · pred ${clsName(classes, bestDiffCls)} (IoU ${num(bestDiffIou, 2)})`,
      });
    } else if (bestSameIou >= OVERLAP_FLOOR) {
      boxes.push({
        key: `t${idx}`,
        outcome: 'missed',
        cause: 'localization',
        cls: gt.class,
        otherCls: null,
        score: null,
        iou: bestSameIou,
        detail: `Missed · low IoU (${num(bestSameIou, 2)})`,
      });
    } else {
      boxes.push({
        key: `t${idx}`,
        outcome: 'missed',
        cause: 'missed',
        cls: gt.class,
        otherCls: null,
        score: null,
        iou: null,
        detail: 'Missed · no detection',
      });
    }
  });

  boxes.sort((a, b) => OUTCOME_ORDER[a.outcome] - OUTCOME_ORDER[b.outcome]);

  const tp = truth.length - match.missed;
  const fp = match.wrong;
  const fn = match.missed;

  return {
    tp,
    fp,
    fn,
    errors: fp + fn,
    truthCount: truth.length,
    meanIou: iouCount > 0 ? iouSum / iouCount : null,
    minIou: iouCount > 0 ? minIou : null,
    minScore: scoreCount > 0 ? minScore : null,
    meanScore: scoreCount > 0 ? scoreSum / scoreCount : null,
    boxes,
  };
}

/** Summarizes error counts across loaded gallery images for filter chip counts. */
export function summarizeGalleryErrors(images: readonly BenchEntryImage[]): GalleryErrorSummary {
  let withFn = 0;
  let withFp = 0;
  let withErrors = 0;
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (const im of images) {
    const imTp = Math.max(0, im.image.boxes.length - im.missed);
    tp += imTp;
    fp += im.wrong;
    fn += im.missed;
    if (im.missed > 0) withFn++;
    if (im.wrong > 0) withFp++;
    if (im.missed + im.wrong > 0) withErrors++;
  }
  return { total: images.length, withFn, withFp, withErrors, tp, fp, fn };
}

/** Filters and sorts loaded benchmark entry images by error focus and diagnostic metric. */
export function filterAndSortImages(
  images: readonly BenchEntryImage[],
  filter: ErrorFilter,
  sort: ErrorSort,
  classes: readonly string[],
): AnalyzedImage[] {
  const analyzed: AnalyzedImage[] = images.map((item, index) => ({
    item,
    index,
    diag: diagnoseImage(item, classes),
  }));

  const filtered = analyzed.filter(({ diag }) => {
    if (filter === 'fn') return diag.fn > 0;
    if (filter === 'fp') return diag.fp > 0;
    if (filter === 'errors') return diag.errors > 0;
    return true;
  });

  if (sort === 'order') {
    return filtered.sort((a, b) => a.index - b.index);
  }
  if (sort === 'worst') {
    return filtered.sort((a, b) => b.diag.errors - a.diag.errors || a.index - b.index);
  }
  if (sort === 'fn') {
    return filtered.sort((a, b) => b.diag.fn - a.diag.fn || b.diag.errors - a.diag.errors || a.index - b.index);
  }
  if (sort === 'fp') {
    return filtered.sort((a, b) => b.diag.fp - a.diag.fp || b.diag.errors - a.diag.errors || a.index - b.index);
  }
  if (sort === 'lowest-iou') {
    return filtered.sort((a, b) => {
      const ai = a.diag.minIou ?? Infinity;
      const bi = b.diag.minIou ?? Infinity;
      if (ai !== bi) return ai - bi;
      const am = a.diag.meanIou ?? Infinity;
      const bm = b.diag.meanIou ?? Infinity;
      return am - bm || b.diag.errors - a.diag.errors || a.index - b.index;
    });
  }
  // 'lowest-score'
  return filtered.sort((a, b) => {
    const as = a.diag.minScore ?? Infinity;
    const bs = b.diag.minScore ?? Infinity;
    if (as !== bs) return as - bs;
    const am = a.diag.meanScore ?? Infinity;
    const bm = b.diag.meanScore ?? Infinity;
    return am - bm || b.diag.errors - a.diag.errors || a.index - b.index;
  });
}
