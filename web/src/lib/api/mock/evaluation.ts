// Deterministic evaluation data for the mock's `GET /api/models/{id}/eval`: plausible PR curves and a confusion
// matrix built from what a model has (summary metrics, classes and, when it came from a run, the run's per-class
// AP). Everything is a pure function of its inputs (seeded by the model id), so screenshots are stable.

import type { ClassMetrics, ModelEval, ModelMetrics } from '../types';
import { hash, rng } from './rng';

export interface ClassEvaluation {
  class: number;
  ap50: number;
  ap50_95: number;
  precision: number;
  recall: number;
  /** Ground-truth boxes of this class in the validation split. */
  instances: number;
}

export interface Evaluation {
  perClass: ClassEvaluation[];
  /** Precision-recall curve per class at IoU 0.5: recall ascending (x), precision (y). Area ≈ that class's AP50. */
  curves: { x: number; y: number }[][];
  /** Mean precision over classes on a common recall grid 0..1 (area ≈ mAP50). */
  mean: { x: number; y: number }[];
  /** (n+1)×(n+1) counts, rows = true class, columns = predicted; the last row/column is background. */
  confusion: number[][];
}

export interface EvaluationInput {
  /** Seeds the synthesis (the model id). */
  seed: string;
  classes: readonly string[];
  metrics: ModelMetrics | null;
  /** Per-class metrics from the source run's validation, when known. */
  perClass?: readonly ClassMetrics[] | null;
  /** Validation instances per class, when known (e.g. from dataset stats). */
  instances?: readonly number[] | null;
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

/** Classes that look alike and so get confused with each other (names in the mock datasets and COCO). */
const LOOKALIKES: string[][] = [
  ['bolt', 'screw', 'pin', 'rivet'],
  ['nut', 'washer', 'bearing'],
  ['circle', 'hexagon'],
  ['square', 'triangle', 'star', 'cross'],
  ['car', 'truck', 'bus', 'train'],
  ['bicycle', 'motorcycle'],
  ['cat', 'dog', 'teddy bear'],
  ['horse', 'cow', 'sheep', 'elephant'],
  ['cup', 'bowl', 'wine glass', 'vase', 'bottle'],
  ['chair', 'couch', 'bench', 'bed'],
  ['laptop', 'tv', 'keyboard', 'cell phone', 'remote'],
  ['handbag', 'backpack', 'suitcase'],
  ['orange', 'apple', 'banana', 'donut', 'cake'],
];

function group(name: string): number {
  return LOOKALIKES.findIndex((g) => g.includes(name));
}

/**
 * A PR curve with the given area: precision = pmax·(1 − s·t)·(1 − t^k) for t = r/rmax, a gentle decline that falls off
 * at the curve's maximum recall. `k` is solved by bisection so the area matches; the result is made monotone.
 */
function prCurve(ap: number, recall: number, r: () => number): { x: number; y: number }[] {
  const pmax = clamp(0.985 + (r() - 0.5) * 0.02, 0.9, 1);
  // The curve reaches a little past the operating-point recall (low-confidence detections still find a few more).
  let rmax = clamp(recall + (1 - recall) * (0.35 + r() * 0.3), 0.05, 0.995);
  // Need ap / (pmax·rmax) < ~0.97 for a valid shape; extend recall first if the AP demands it.
  rmax = Math.max(rmax, Math.min(0.999, ap / (0.97 * pmax)));
  const a = clamp(ap / (pmax * rmax), 0.02, 0.97);
  const s = clamp(1.2 * (1 - a), 0.02, 0.3) * (0.7 + 0.6 * r());
  // ∫₀¹ (1 − s·t)(1 − t^k) dt = 1 − s/2 − 1/(k+1) + s/(k+2), increasing in k.
  const integral = (k: number) => 1 - s / 2 - 1 / (k + 1) + s / (k + 2);
  let lo = 0.05;
  let hi = 400;
  for (let i = 0; i < 60; i++) {
    const mid = Math.sqrt(lo * hi);
    if (integral(mid) < a) lo = mid;
    else hi = mid;
  }
  const k = Math.sqrt(lo * hi);
  const n = 100;
  const raw: number[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const wiggle = 1 + (r() - 0.5) * 0.03 * Math.sin(t * Math.PI);
    raw.push(clamp(pmax * (1 - s * t) * (1 - t ** k) * wiggle, 0, 1));
  }
  // Interpolated precision (max to the right), as PR curves are drawn: monotone non-increasing.
  for (let i = n - 1; i >= 0; i--) raw[i] = Math.max(raw[i]!, raw[i + 1]!);
  return raw.map((p, i) => ({ x: r4((i / n) * rmax), y: r4(p) }));
}

/** Precision of a curve at recall `x` (step interpolation; 0 past the curve's end). */
function at(curve: { x: number; y: number }[], x: number): number {
  const last = curve[curve.length - 1];
  if (!last || x > last.x) return 0;
  let lo = 0;
  let hi = curve.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (curve[mid]!.x < x) lo = mid + 1;
    else hi = mid;
  }
  return curve[lo]!.y;
}

/** Area under a curve by the trapezoid rule (for tests and sanity checks). */
export function area(curve: { x: number; y: number }[]): number {
  let s = 0;
  for (let i = 1; i < curve.length; i++) s += ((curve[i]!.x - curve[i - 1]!.x) * (curve[i]!.y + curve[i - 1]!.y)) / 2;
  return s;
}

/** Synthesises evaluation data for a model. See the note at the top of this file. */
export function synthEvaluation(input: EvaluationInput): Evaluation {
  const n = input.classes.length;
  const seed = hash(input.seed);
  const m = input.metrics ?? { map50: 0.5, map50_95: 0.35, precision: 0.6, recall: 0.5 };

  // Per-class metrics: the run's when we have them, else spread around the model's means (re-centred on them).
  let perClass: Omit<ClassEvaluation, 'instances'>[];
  if (input.perClass && input.perClass.length === n) {
    perClass = input.perClass.map((c) => ({ class: c.class, ap50: c.map50, ap50_95: c.map50_95, precision: c.precision, recall: c.recall }));
  } else {
    const r = rng(seed + 17);
    const ease = Array.from({ length: n }, () => 1 + (r() + r() + r() - 1.5) * 0.28);
    const mean = ease.reduce((a, b) => a + b, 0) / Math.max(1, n);
    perClass = ease.map((e0, c) => {
      const e = e0 / mean;
      return {
        class: c,
        ap50: r4(clamp(m.map50 * e, 0.02, 0.995)),
        ap50_95: r4(clamp(m.map50_95 * e, 0.01, 0.98)),
        precision: r4(clamp(m.precision * (0.6 + 0.4 * e) + (r() - 0.5) * 0.03, 0.05, 0.995)),
        recall: r4(clamp(m.recall * (0.55 + 0.45 * e) + (r() - 0.5) * 0.03, 0.05, 0.99)),
      };
    });
  }

  // Instances: given, or a long-tailed split of a plausible validation set.
  const instances: number[] = [];
  {
    const r = rng(seed + 29);
    for (let c = 0; c < n; c++) {
      const given = input.instances?.[c];
      instances.push(given !== undefined ? Math.max(0, Math.round(given)) : Math.round((n > 20 ? 2600 : 600) / (1 + c * (n > 20 ? 0.18 : 0.12)) * (0.7 + 0.6 * r())));
    }
  }

  const curves = perClass.map((c, i) => prCurve(c.ap50, c.recall, rng(seed + 101 + i * 7919)));
  const mean: { x: number; y: number }[] = [];
  for (let i = 0; i <= 100; i++) {
    const x = i / 100;
    const ys = curves.map((cv) => at(cv, x));
    mean.push({ x, y: r4(ys.reduce((a, b) => a + b, 0) / Math.max(1, ys.length)) });
  }

  // Confusion at the operating point: TP on the diagonal; of the misses, a share is confused with lookalike classes
  // and the rest goes to the background column. The background row holds false positives per predicted class.
  const confusion = Array.from({ length: n + 1 }, () => new Array<number>(n + 1).fill(0));
  const r = rng(seed + 43);
  const groups = input.classes.map(group);
  for (let t = 0; t < n; t++) {
    const N = instances[t]!;
    const tp = Math.round(N * perClass[t]!.recall);
    confusion[t]![t] = tp;
    const missed = N - tp;
    // Similar classes: same lookalike group, else two seeded neighbours with a small weight.
    const weights = input.classes.map((_, p) => {
      if (p === t) return 0;
      if (groups[t]! >= 0 && groups[t] === groups[p]) return 1 + r();
      return r() < 2.5 / Math.max(2, n) ? 0.25 * r() : 0;
    });
    const wsum = weights.reduce((a, b) => a + b, 0);
    const confusedShare = wsum > 0 ? (groups[t]! >= 0 ? 0.42 : 0.18) : 0;
    let confused = 0;
    if (wsum > 0) {
      for (let p = 0; p < n; p++) {
        if (p === t) continue;
        const k =Math.round((missed * confusedShare * weights[p]!) / wsum);
        confusion[t]![p] = k;
        confused += k;
      }
    }
    confusion[t]![n] = Math.max(0, missed - confused);
  }
  for (let p = 0; p < n; p++) {
    let predicted = 0;
    for (let t = 0; t < n; t++) predicted += confusion[t]![p]!;
    const prec = perClass[p]!.precision;
    const fp = Math.round((predicted * (1 - prec)) / Math.max(0.05, prec));
    let wrongClass = 0;
    for (let t = 0; t < n; t++) if (t !== p) wrongClass += confusion[t]![p]!;
    confusion[n]![p] = Math.max(0, fp - wrongClass);
  }

  return {
    perClass: perClass.map((c, i) => ({ ...c, instances: instances[i]! })),
    curves,
    mean,
    confusion,
  };
}

/** The API shape (`ModelEval`): curves resampled at 101 recall points as a non-increasing precision envelope. */
export function toModelEval(ev: Evaluation, images: number): ModelEval {
  const pr = ev.curves.map((curve, c) => {
    const recall: number[] = [];
    const precision: number[] = [];
    for (let i = 0; i <= 100; i++) {
      recall.push(i / 100);
      precision.push(at(curve, i / 100));
    }
    // COCO interpolation: precision at recall r is the max precision at any recall >= r.
    for (let i = 99; i >= 0; i--) precision[i] = Math.max(precision[i]!, precision[i + 1]!);
    const rounded = precision.map(r4);
    return { class: c, recall, precision: rounded, ap50: r4(rounded.reduce((a, b) => a + b, 0) / rounded.length) };
  });
  return {
    pr,
    confusion: ev.confusion.map((row, r) => row.map((v, c) => (r === row.length - 1 && c === row.length - 1 ? 0 : Math.round(v)))),
    per_class: ev.perClass.map((c) => ({ class: c.class, map50: pr[c.class]?.ap50 ?? c.ap50, map50_95: c.ap50_95, precision: c.precision, recall: c.recall })),
    images,
  };
}
