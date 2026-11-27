// Pure comparison helpers for Playground Model A/B visual compare: default Model B selection, count/latency deltas,
// per-class breakdowns, and structured JSON payloads. Availability of Model A and Model B is tracked explicitly so
// pending or failed predictions never report fabricated zero counts or misleading deltas.
import type { Model, Prediction } from '$lib/api/types';
import { ms } from '$lib/format';
import { predictionJson, type Det } from './viewState';

export type CompareViewMode = 'side-by-side' | 'split';

export interface CompareClassRow {
  name: string;
  /** Class index used for palette swatch color (prefers Model A's class index, then Model B's). */
  colorClass: number;
  countA: number | null;
  countB: number | null;
  delta: number | null;
}

export interface CompareSummary {
  countA: number | null;
  countB: number | null;
  countDelta: number | null;
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
  const hasA = predA !== null;
  const hasB = predB !== null;
  const byName = new Map<string, { name: string; colorClass: number; a: number; b: number }>();

  if (hasA) {
    for (const b of boxesA) {
      const key = b.name || `class ${b.class}`;
      const existing = byName.get(key);
      if (existing) existing.a++;
      else byName.set(key, { name: key, colorClass: b.class, a: 1, b: 0 });
    }
  }

  if (hasB) {
    for (const b of boxesB) {
      const key = b.name || `class ${b.class}`;
      const existing = byName.get(key);
      if (existing) existing.b++;
      else byName.set(key, { name: key, colorClass: b.class, a: 0, b: 1 });
    }
  }

  const classes: CompareClassRow[] = [...byName.values()]
    .map((row) => ({
      name: row.name,
      colorClass: row.colorClass,
      countA: hasA ? row.a : null,
      countB: hasB ? row.b : null,
      delta: hasA && hasB ? row.b - row.a : null,
    }))
    .sort((a, b) => {
      const da = a.delta === null ? -1 : Math.abs(a.delta);
      const db = b.delta === null ? -1 : Math.abs(b.delta);
      const sumA = (a.countA ?? 0) + (a.countB ?? 0);
      const sumB = (b.countA ?? 0) + (b.countB ?? 0);
      return db - da || sumB - sumA || a.name.localeCompare(b.name);
    });

  const countA = hasA ? boxesA.length : null;
  const countB = hasB ? boxesB.length : null;
  const countDelta = countA !== null && countB !== null ? countB - countA : null;

  const latencyA = hasA ? predA.timing.total_ms : null;
  const latencyB = hasB ? predB.timing.total_ms : null;
  const latencyDeltaMs = latencyA !== null && latencyB !== null ? latencyB - latencyA : null;
  const speedupRatio = latencyA !== null && latencyB !== null && latencyA > 0 && latencyB > 0 ? latencyA / latencyB : null;

  return {
    countA,
    countB,
    countDelta,
    latencyA,
    latencyB,
    latencyDeltaMs,
    speedupRatio,
    classes,
  };
}

/** Formats a signed integer delta (`+3`, `-2`, `0`, or `—` when unavailable). */
export function formatSignedDelta(delta: number | null): string {
  if (delta === null) return '—';
  if (delta > 0) return `+${delta}`;
  if (delta < 0) return String(delta);
  return '0';
}

/** Formats latency difference between Model A and Model B (`-10.0 ms · B 2.0× faster` or `+8.4 ms`). */
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
