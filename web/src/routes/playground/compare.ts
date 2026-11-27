// Pure comparison helpers for Playground Model A/B visual compare: default Model B selection, count/latency deltas,
// per-class breakdowns, and structured JSON payloads.
import type { Model, Prediction } from '$lib/api/types';
import { ms } from '$lib/format';
import { predictionJson, type Det } from './viewState';

export type CompareViewMode = 'side-by-side' | 'split';

export interface CompareClassRow {
  name: string;
  /** Class index used for palette swatch color (prefers Model A's class index, then Model B's). */
  colorClass: number;
  countA: number;
  countB: number;
  delta: number;
}

export interface CompareSummary {
  countA: number;
  countB: number;
  countDelta: number;
  latencyA: number | null;
  latencyB: number | null;
  latencyDeltaMs: number | null;
  /** `latencyA / latencyB` (> 1 means B is faster than A; < 1 means B is slower than A). */
  speedupRatio: number | null;
  classes: CompareClassRow[];
}

/**
 * Picks a sensible default Model B for Model A: prefers another model with the same class list (e.g. folo-n vs
 * folo-s on the same dataset), then any distinct model, falling back to Model A in single-model workspaces.
 */
export function pickDefaultModelB(modelA: Model | null, ranked: readonly Model[]): Model | null {
  if (!ranked.length) return null;
  if (!modelA) return ranked[1] ?? ranked[0] ?? null;
  const key = modelA.classes.join('|');
  return (
    ranked.find((m) => m.id !== modelA.id && m.classes.join('|') === key) ??
    ranked.find((m) => m.id !== modelA.id) ??
    modelA
  );
}

/** Summarizes detection counts, per-class deltas (matched by class name), and latency between Model A and Model B. */
export function summarizeComparison(opts: {
  boxesA: readonly Det[];
  boxesB: readonly Det[];
  predA: Prediction | null;
  predB: Prediction | null;
}): CompareSummary {
  const { boxesA, boxesB, predA, predB } = opts;
  const byName = new Map<string, CompareClassRow>();

  for (const b of boxesA) {
    const key = b.name || `class ${b.class}`;
    const existing = byName.get(key);
    if (existing) {
      existing.countA++;
    } else {
      byName.set(key, { name: key, colorClass: b.class, countA: 1, countB: 0, delta: 0 });
    }
  }

  for (const b of boxesB) {
    const key = b.name || `class ${b.class}`;
    const existing = byName.get(key);
    if (existing) {
      existing.countB++;
    } else {
      byName.set(key, { name: key, colorClass: b.class, countA: 0, countB: 1, delta: 0 });
    }
  }

  const classes = [...byName.values()]
    .map((row) => ({ ...row, delta: row.countB - row.countA }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || b.countA + b.countB - (a.countA + a.countB) || a.name.localeCompare(b.name));

  const latencyA = predA ? predA.timing.total_ms : null;
  const latencyB = predB ? predB.timing.total_ms : null;
  const latencyDeltaMs = latencyA !== null && latencyB !== null ? latencyB - latencyA : null;
  const speedupRatio = latencyA !== null && latencyB !== null && latencyA > 0 && latencyB > 0 ? latencyA / latencyB : null;

  return {
    countA: boxesA.length,
    countB: boxesB.length,
    countDelta: boxesB.length - boxesA.length,
    latencyA,
    latencyB,
    latencyDeltaMs,
    speedupRatio,
    classes,
  };
}

/** Formats a signed integer delta (`+3`, `-2`, `0`). */
export function formatSignedDelta(delta: number): string {
  if (delta > 0) return `+${delta}`;
  if (delta < 0) return String(delta);
  return '0';
}

/** Formats latency difference between Model A and Model B (`B faster by 12 ms (2.1×)` or `+8.4 ms`). */
export function formatLatencyComparison(summary: Pick<CompareSummary, 'latencyA' | 'latencyB' | 'latencyDeltaMs' | 'speedupRatio'>): string {
  const { latencyDeltaMs, speedupRatio } = summary;
  if (latencyDeltaMs === null) return '—';
  if (Math.abs(latencyDeltaMs) < 0.05) return 'Same latency';
  const sign = latencyDeltaMs > 0 ? '+' : '-';
  const diff = `${sign}${ms(Math.abs(latencyDeltaMs))}`;
  if (speedupRatio !== null) {
    if (speedupRatio >= 1.15) return `${diff} · B ${speedupRatio.toFixed(1)}× faster`;
    if (speedupRatio <= 1 / 1.15) return `${diff} · A ${(1 / speedupRatio).toFixed(1)}× faster`;
  }
  return diff;
}

/** Builds a structured JSON payload for Compare A/B view and JSON export. */
export function comparePredictionJson(opts: {
  view: CompareViewMode;
  predA: Prediction | null;
  predB: Prediction | null;
  hasResultA: boolean;
  hasResultB: boolean;
  boxesA: readonly Det[];
  boxesB: readonly Det[];
  modelAName: string;
  modelBName: string;
  runOnA: string;
  runOnB: string;
  o2oA: boolean;
  o2oB: boolean;
  filter: { conf: number; iou: number; maxDet: number; hiddenClasses?: readonly string[] };
  summary: CompareSummary;
}) {
  const a = predictionJson(opts.predA, opts.hasResultA, opts.boxesA, {
    conf: opts.filter.conf,
    iou: opts.filter.iou,
    maxDet: opts.filter.maxDet,
    o2o: opts.o2oA,
    hiddenClasses: opts.filter.hiddenClasses,
  });
  const b = predictionJson(opts.predB, opts.hasResultB, opts.boxesB, {
    conf: opts.filter.conf,
    iou: opts.filter.iou,
    maxDet: opts.filter.maxDet,
    o2o: opts.o2oB,
    hiddenClasses: opts.filter.hiddenClasses,
  });
  if (!a && !b) return null;
  const image = opts.predA?.image ?? opts.predB?.image ?? null;
  const filter = opts.o2oA && opts.o2oB
    ? { conf: opts.filter.conf, max_det: opts.filter.maxDet }
    : { conf: opts.filter.conf, iou: opts.filter.iou, max_det: opts.filter.maxDet };
  return {
    mode: 'compare' as const,
    view: opts.view,
    image,
    filter: opts.filter.hiddenClasses?.length ? { ...filter, hidden_classes: [...opts.filter.hiddenClasses] } : filter,
    comparison: {
      count_a: opts.summary.countA,
      count_b: opts.summary.countB,
      count_delta: opts.summary.countDelta,
      latency_a_ms: opts.summary.latencyA,
      latency_b_ms: opts.summary.latencyB,
      latency_delta_ms: opts.summary.latencyDeltaMs,
      speedup_ratio: opts.summary.speedupRatio,
      classes: opts.summary.classes.map((c) => ({ name: c.name, count_a: c.countA, count_b: c.countB, delta: c.delta })),
    },
    a: a ? { ...a, model_name: opts.modelAName, run_on: opts.runOnA } : null,
    b: b ? { ...b, model_name: opts.modelBName, run_on: opts.runOnB } : null,
  };
}
