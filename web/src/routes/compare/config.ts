// Constants, metric definitions, and shared helpers for Compare runs (`#/runs/compare`).
import type { Epoch, Run } from '$lib/api/types';
import { origin } from '$lib/lineage';

/** How often the compared runs are read again (and a metrics read that failed is tried again). */
export const COMPARE_POLL_MS = 3000;

/** Maximum runs compared side by side at once. */
export const MAX = 4;

export type Axis = 'epoch' | 'progress';
export type MetricId = 'm95' | 'm50' | 'loss' | 'prec' | 'rec';
export type YAxis = 'fit' | 'zero';

/**
 * The weights a run started from, by name: a model id resolves through the models list, a path shows its file name,
 * none is null ("From scratch"). Fine-tuned vs from scratch often explains most of a gap between two runs.
 */
export function startFrom(weights: string | null, models: readonly { id: string; name: string }[] | undefined): string | null {
  const o = origin(weights, models ?? []);
  return o.kind === 'scratch' ? null : o.label;
}

export const isActiveStatus = (s: Run['status']) => s === 'running' || s === 'stopping' || s === 'queued';

export const X_OPTIONS: { value: Axis; label: string }[] = [
  { value: 'epoch', label: 'By epoch' },
  { value: 'progress', label: 'By progress' },
];

/** Phones: the x axis as a compact select ("X axis": Epoch, Progress), so the metric's select keeps its whole name. */
export const X_PHONE: { value: Axis; label: string }[] = [
  { value: 'epoch', label: 'Epoch' },
  { value: 'progress', label: 'Progress' },
];

export const PROGRESS_HOW = 'Each run is drawn over its own planned epochs, so a short plan and a long one compare by the shape of their curves.';

// Run slots: sky and amber first (the most distinct pair, for the common two-run comparison), then magenta and teal.
// No four hues stay apart for every colour-vision type, so slots 3 and 4 also get a line pattern.
export const RUN_SLOTS = [1, 2, 3, 4];
export const SLOT_PATTERN = [undefined, undefined, 'dash', 'dot'] as const;

/** The one caveat that changes how to read the plot: one wording on every screen. */
export const DATA_CAVEAT = 'Different datasets: compare the shape of the curves, not their values.';

export interface CompareMetricDef {
  id: MetricId;
  label: string;
  f: (e: Epoch) => number;
  unit?: boolean;
}

export const METRICS: CompareMetricDef[] = [
  { id: 'm95', label: 'mAP50-95', f: (e) => e.val.map50_95, unit: true },
  { id: 'm50', label: 'mAP50', f: (e) => e.val.map50, unit: true },
  { id: 'loss', label: 'Train loss', f: (e) => e.train_loss.total },
  { id: 'prec', label: 'Precision', f: (e) => e.val.precision, unit: true },
  { id: 'rec', label: 'Recall', f: (e) => e.val.recall, unit: true },
];

export const METRIC_OPTIONS = METRICS.map((m) => ({ value: m.id, label: m.label }));

export const Y_OPTIONS: { value: YAxis; label: string }[] = [
  { value: 'fit', label: 'Axis: Fit' },
  { value: 'zero', label: 'Axis: From 0' },
];
