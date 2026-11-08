// Turns `GET /api/models/{id}/eval` into what the Model view draws: chart points per class, a mean curve and
// per-class rows with instance counts (the confusion matrix row sums: every true object lands in exactly one cell).

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
  return { perClass, curves, mean, confusion: e.confusion, images: e.images };
}
