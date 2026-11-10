// The numbers of a vision benchmark as docs/BENCH.md defines them (its Definitions), as pure functions: the mock builds
// its results with them, the tests hold those results to them, and the result screens can check what they draw.

import type {
  BenchAxis,
  BenchAxisScore,
  BenchCalibrationBin,
  BenchConditionResult,
  BenchLabelPoint,
  BenchReadiness,
  BenchSeverityPoint,
} from '../api/types';
import { AXES, AXIS_WEIGHT, TARGET_PRECISION } from './rules';

/** `kept(c, s) = min(1, mAP(c, s) / mAP(unchanged))`; 0 when the unchanged mAP is 0. */
export function retention(map: number, clean: number): number {
  return clean > 0 ? Math.min(1, map / clean) : 0;
}

const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

/** A condition's value: the mean of `kept` over its severities. */
export function conditionRetention(points: readonly Pick<BenchSeverityPoint, 'retention'>[]): number {
  return mean(points.map((p) => p.retention));
}

/** The robustness axis: the conditions' values weighted by the plan's weights; null when no condition ran. */
export function robustnessAxis(conditions: readonly Pick<BenchConditionResult, 'weight' | 'retention'>[]): number | null {
  const w = conditions.reduce((a, c) => a + c.weight, 0);
  if (!conditions.length || w <= 0) return null;
  return conditions.reduce((a, c) => a + c.weight * c.retention, 0) / w;
}

/** `ECE = Σ (n_bin / n) · |mean confidence − share right|` over the bins; 0 with no detections. */
export function expectedCalibrationError(bins: readonly BenchCalibrationBin[]): number {
  const n = bins.reduce((a, b) => a + b.count, 0);
  if (!n) return 0;
  return bins.reduce((a, b) => (b.count && b.confidence !== null && b.precision !== null ? a + (b.count / n) * Math.abs(b.confidence - b.precision) : a), 0);
}

/** The localization axis: mean IoU of the correct detections rescaled from 0.5..1 to 0..1; 0 when nothing is right. */
export function localizationAxis(meanIou: number | null): number {
  return meanIou === null ? 0 : Math.max(0, Math.min(1, (meanIou - 0.5) / 0.5));
}

/** The mean IoU a localization histogram stands for: each bin at its centre (bins of 0.05 from 0.5). */
export function histogramMeanIou(histogram: readonly number[]): number | null {
  const n = histogram.reduce((a, b) => a + b, 0);
  if (!n) return null;
  return histogram.reduce((a, c, i) => a + c * (0.525 + i * 0.05), 0) / n;
}

/** The fewest labels a self-labeling threshold must keep (docs/BENCH.md, Self-labeling): a precision measured on fewer is
 * no support for using it (a weak model can be right on its single most confident detection). */
export const MIN_LABELS = 10;

/**
 * The self-labeling operating point on a curve: the lowest confidence on the grid whose kept detections are at least
 * `target` precise on at least MIN_LABELS of them. Null when no confidence is. A curve of one point is a contestant
 * without confidences: that point is its operating point (`threshold` null), counted when precise enough on at least
 * MIN_LABELS detections.
 */
export function labelOperatingPoint(curve: readonly BenchLabelPoint[], target = TARGET_PRECISION): { threshold: number | null; point: BenchLabelPoint | null } {
  const supported = (p: BenchLabelPoint) => p.labels >= MIN_LABELS && p.precision !== null && p.precision >= target - 1e-9;
  if (curve.length === 1) {
    const p = curve[0]!;
    return { threshold: null, point: supported(p) ? p : null };
  }
  for (const p of [...curve].sort((a, b) => a.confidence - b.confidence)) {
    if (supported(p)) return { threshold: p.confidence, point: p };
  }
  return { threshold: null, point: null };
}

/** The verdict on a self-labeling recall: ready from 0.5, partial from 0.25. */
export function readiness(recall: number): BenchReadiness {
  return recall >= 0.5 ? 'ready' : recall >= 0.25 ? 'partial' : 'not_ready';
}

/** `score = 100 · Σ weight · axis / Σ weight` over the axes that were measured (0 when none was). */
export function scoreTotal(axes: readonly Pick<BenchAxisScore, 'value' | 'weight'>[]): number {
  const measured = axes.filter((a) => a.value !== null);
  const w = measured.reduce((s, a) => s + a.weight, 0);
  return w > 0 ? (100 * measured.reduce((s, a) => s + a.weight * (a.value ?? 0), 0)) / w : 0;
}

/** Each axis's share of the score as it was computed: its weight over the measured weights (0 for one not measured). */
export function axisShares(axes: readonly Pick<BenchAxisScore, 'axis' | 'value' | 'weight'>[]): Record<BenchAxis, number> {
  const w = axes.filter((a) => a.value !== null).reduce((s, a) => s + a.weight, 0);
  const out = Object.fromEntries(AXES.map((a) => [a, 0])) as Record<BenchAxis, number>;
  for (const a of axes) if (a.value !== null && w > 0) out[a.axis] = a.weight / w;
  return out;
}

/** The five axes in display order with the catalog's weights, from their values (null: not measured). */
export function axisScores(values: Partial<Record<BenchAxis, number | null>>): BenchAxisScore[] {
  return AXES.map((axis) => ({ axis, value: values[axis] ?? null, weight: AXIS_WEIGHT[axis] }));
}

/** F1 of a precision and a recall (0 when both are 0). */
export function f1(precision: number, recall: number): number {
  return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
}
