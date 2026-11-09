// Turns `GET /api/models/{id}/eval` into what the Model view draws: chart points per class, a mean curve and
// per-class rows with instance counts (the confusion matrix row sums: every true object lands in exactly one cell).

import { evalSource, matrixAt, type EvalSource, type MatrixAt } from '$lib/evaluated';
import type { ModelEval } from './types';

export interface ClassEvaluation {
  class: number;
  ap50: number;
  ap50_95: number;
  precision: number;
  recall: number;
  /** Ground-truth boxes of this class in the evaluated split. */
  instances: number;
}

export interface Evaluation {
  perClass: ClassEvaluation[];
  /** Precision–recall curve per class at IoU 0.5 (x = recall, y = precision). */
  curves: { x: number; y: number }[][];
  /** Mean precision over classes at each recall point (area ≈ mAP50). */
  mean: { x: number; y: number }[];
  /** (n+1)×(n+1) counts; rows true class, columns predicted, last row/column background. */
  confusion: number[][];
  images: number;
  /** The confidence per-class precision and recall are taken at (`/eval`'s `best_f1_conf`); null when unknown. */
  bestF1Conf: number | null;
  /** What was evaluated: the dataset and the split actually used (`$lib/evaluated` words it). */
  source: EvalSource;
  /** The confusion matrix's own thresholds (not `bestF1Conf`, not the NMS settings); null when the server does not say. */
  matrixAt: MatrixAt | null;
}

export function fromModelEval(e: ModelEval): Evaluation {
  const n = e.pr.length;
  const curves = e.pr.map((c) => c.recall.map((x, i) => ({ x, y: c.precision[i] ?? 0 })));
  const points = e.pr[0]?.recall.length ?? 0;
  const mean = Array.from({ length: points }, (_, i) => ({
    x: e.pr[0]?.recall[i] ?? i / Math.max(1, points - 1),
    y: e.pr.reduce((a, c) => a + (c.precision[i] ?? 0), 0) / Math.max(1, n),
  }));
  const perClass = e.per_class.map((c) => ({
    class: c.class,
    ap50: c.map50,
    ap50_95: c.map50_95,
    precision: c.precision,
    recall: c.recall,
    instances: (e.confusion[c.class] ?? []).reduce((a, b) => a + b, 0),
  }));
  return {
    perClass,
    curves,
    mean,
    confusion: e.confusion,
    images: e.images,
    bestF1Conf: e.best_f1_conf ?? null,
    source: evalSource(e),
    matrixAt: matrixAt(e),
  };
}

/**
 * Shrinks an (n+1)² confusion matrix to the classes in `keep` plus an "other classes" row/column that aggregates the
 * rest, keeping background last. Every count stays in the matrix, so row-normalised percentages are unchanged for
 * the kept classes (confusions with omitted classes land in "other" instead of disappearing).
 */
export function reduceConfusion(confusion: number[][], keep: number[]): number[][] {
  const n = confusion.length - 1;
  const kept = new Set(keep);
  const k = keep.length;
  // Output index of each input index: kept classes in order, then other (k), then background (k + 1).
  const map = (i: number) => (i === n ? k + 1 : kept.has(i) ? keep.indexOf(i) : k);
  const out = Array.from({ length: k + 2 }, () => new Array<number>(k + 2).fill(0));
  for (let r = 0; r <= n; r++) for (let c = 0; c <= n; c++) out[map(r)]![map(c)]! += confusion[r]?.[c] ?? 0;
  return out;
}
