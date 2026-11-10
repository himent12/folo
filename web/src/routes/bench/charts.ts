// The geometry of the result screens' charts, apart from their markup: which points each draws from a result, how big a
// point is for what it stands on, the scales they share, and how the Compare page lines entries up. Pure, so every mark
// can be traced to a number of the result (./charts.test.ts).

import { overlaps } from './leaderboard';
import type { BenchCalibration, BenchConditionResult, BenchEntry, BenchGeneration, BenchLabelPoint, BenchResult } from '$lib/api/types';
import { unitDomain, unitFitDomain, type Fit } from '$lib/charts/domain';
import type { XYPoint } from '$lib/charts/xy';

// ---------- reliability ----------

/** A bin holding less than this share of the detections is drawn hollow: too few to trust. */
export const THIN_BIN = 0.05;
/** The dot radius of the fullest bin, and of the emptiest one drawn. */
const R_MAX = 6;
const R_MIN = 2.5;

/**
 * The reliability diagram's points: one per bin that holds detections (the `[0, 0.1)` bin is always empty, as
 * detections enter from 0.1, so it is not drawn and the axis starts at 0.1), at the bin's mean confidence (x) and the
 * share of it that is right (y). A point's area follows its count; a bin under THIN_BIN of the detections is hollow.
 */
export function reliabilityPoints(c: Pick<BenchCalibration, 'bins'>): XYPoint[] {
  const filled = c.bins.filter((b) => b.count > 0 && b.confidence !== null && b.precision !== null);
  const n = filled.reduce((s, b) => s + b.count, 0);
  const most = Math.max(1, ...filled.map((b) => b.count));
  return filled.map((b) => ({
    x: b.confidence!,
    y: b.precision!,
    r: R_MIN + (R_MAX - R_MIN) * Math.sqrt(b.count / most),
    hollow: b.count < THIN_BIN * n,
  }));
}

/** The counts under the diagram, one bar per bin from 0.1 (on the plot's x scale, the bin's own span). */
export const reliabilityStrip = (c: Pick<BenchCalibration, 'bins'>) => c.bins.filter((b) => b.lo >= 0.1 - 1e-9).map((b) => ({ x0: b.lo, x1: b.hi, v: b.count }));

/** The bin a cursor at mean confidence `x` reads. */
export const binAtConfidence = (c: Pick<BenchCalibration, 'bins'>, x: number) => c.bins.find((b) => b.confidence !== null && Math.abs(b.confidence - x) < 1e-9) ?? null;

/** The reliability diagram's x domain: from 0.1, where detections start entering it, to 1. */
export const RELIABILITY_X: [number, number] = [0.1, 1];

// ---------- self-labeling ----------

/** The label curve's two lines: the share of the labels kept that is right, and the share of all objects they find. */
export function labelCurves(curve: readonly BenchLabelPoint[]): { precision: XYPoint[]; recall: XYPoint[] } {
  return {
    precision: curve.map((p) => ({ x: p.confidence, y: p.labels > 0 ? p.precision : null })),
    recall: curve.map((p) => ({ x: p.confidence, y: p.labels > 0 ? p.recall : null })),
  };
}

/** The point of a label curve at a confidence on its 0.01 grid. */
export const labelPointAt = (curve: readonly BenchLabelPoint[], x: number) => curve.find((p) => Math.abs(p.confidence - x) < 5e-4) ?? null;

/**
 * Compare's form of the label curve, one line per entry: the precision of the labels kept against the share of all
 * objects they find, as the threshold moves (recall on x). The furthest right a line gets above 0.9 is that entry's
 * self-labeling axis; a contestant without confidences is one dot.
 */
export function labelTradeoff(curve: readonly BenchLabelPoint[]): XYPoint[] {
  return curve
    .filter((p) => p.labels > 0 && p.precision !== null)
    .map((p) => ({ x: p.recall, y: p.precision }))
    .sort((a, b) => a.x - b.x);
}

// ---------- robustness ----------

/**
 * One condition's accuracy against severity, from the unchanged images (severity 0, its mAP) through the severities it
 * ran: the shape of the fall.
 */
export function severityPoints(c: Pick<BenchConditionResult, 'points'>, clean: number): XYPoint[] {
  return [{ x: 0, y: clean }, ...c.points.map((p) => ({ x: p.severity, y: p.map50_95 }))];
}

/** The severity chart's y axis: 0 up to the highest mAP drawn (the unchanged level, or a condition that beat it). */
export function severityDomain(clean: number, conditions: readonly Pick<BenchConditionResult, 'points'>[], plotPx: number): Fit {
  const max = Math.max(clean, ...conditions.flatMap((c) => c.points.map((p) => p.map50_95)));
  return unitDomain(max, plotPx);
}

/** The plan's conditions as the heat table orders them: worst kept first, so where it breaks leads. */
export const byWorstKept = <C extends Pick<BenchConditionResult, 'retention' | 'condition'>>(conditions: readonly C[]) =>
  [...conditions].sort((a, b) => a.retention - b.retention || a.condition.localeCompare(b.condition));

// ---------- loops ----------

/** A loop's generations as the chart draws them: mAP@0.5:0.95 with its interval, for those that have one. */
export function generationPoints(gens: readonly BenchGeneration[]): XYPoint[] {
  return gens.map((g) => ({ x: g.generation, y: g.map50_95, ...(g.ci95 ? { lo: g.ci95[0], hi: g.ci95[1] } : {}) }));
}

/**
 * The generations chart's y axis: fitted to the intervals (a loop's story is a few hundredths of mAP, which 0–1 flattens
 * into a line), always holding generation 0's whole interval; the chart says where it starts ("y from 0.70").
 */
export function generationDomain(gens: readonly BenchGeneration[], plotPx: number): Fit {
  const vals = gens.flatMap((g) => [g.map50_95, ...(g.ci95 ?? [])]).filter((v): v is number => v !== null && Number.isFinite(v));
  if (!vals.length) return unitDomain(1, plotPx);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  // Every value counts as "the data": the floor sits under the lowest whisker, never cutting one.
  return unitFitDomain([lo, hi], [lo], hi, plotPx);
}

// ---------- compare ----------

/** A series slot per entry, in the order picked: sky, amber, magenta, teal; the last two also patterned (RunCompare). */
export const SLOTS = [1, 2, 3, 4] as const;
export const SLOT_PATTERN = [undefined, undefined, 'dash', 'dot'] as const;

/**
 * Slots that follow the entry, not its position: removing one never repaints the others (RunCompare's rule). `prev`
 * maps ids to slots 0–3; ids keep theirs, new ones take the lowest free.
 */
export function assignSlots(ids: readonly string[], prev: Readonly<Record<string, number>>): Record<string, number> {
  const next: Record<string, number> = {};
  for (const id of ids) if (prev[id] !== undefined) next[id] = prev[id]!;
  for (const id of ids) {
    if (next[id] !== undefined) continue;
    let k = 0;
    while (Object.values(next).includes(k)) k++;
    next[id] = k;
  }
  return next;
}

/** A row of an aligned dot plot: one thing (a condition, a class), each entry's value on it (null: it has none). */
export interface AlignedRow {
  key: string;
  values: (number | null)[];
}

/**
 * The plan's conditions as Compare's rows, in plan order, each entry's kept accuracy (its condition's mean of `kept`)
 * or null where it did not run it. Conditions no entry ran are left out.
 */
export function conditionRows(planOrder: readonly string[], results: readonly (BenchResult | undefined)[]): AlignedRow[] {
  return planOrder
    .map((key) => ({ key, values: results.map((r) => r?.robustness?.conditions.find((c) => c.condition === key)?.retention ?? null) }))
    .filter((row) => row.values.some((v) => v !== null));
}

/**
 * The benchmark's classes with true boxes as Compare's rows, each entry's AP@0.5:0.95 (0 for a class it does not know,
 * as scored). `truths` is the benchmark's own count of the class (`Bench.class_boxes`), never one entry's; `graded` is
 * what each entry was graded on, which differs for an entry that ran a part of the images. Sorted by the entries' mean
 * AP, lowest first: where they struggle leads.
 */
export function classRows(
  classes: readonly string[],
  classBoxes: readonly number[],
  results: readonly (BenchResult | undefined)[],
): (AlignedRow & { name: string; truths: number; graded: (number | null)[] })[] {
  const present = new Set<number>();
  for (const r of results) for (const c of r?.accuracy.per_class ?? []) present.add(c.class);
  const rows = [...present].map((k) => ({
    key: String(k),
    name: classes[k] ?? `class ${k}`,
    truths: classBoxes[k] ?? 0,
    graded: results.map((r) => (r ? (r.accuracy.per_class.find((c) => c.class === k)?.truths ?? 0) : null)),
    values: results.map((r) => (r ? (r.accuracy.per_class.find((c) => c.class === k)?.map50_95 ?? null) : null)),
  }));
  const mean = (v: (number | null)[]) => {
    const xs = v.filter((x): x is number => x !== null);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 1;
  };
  return rows.sort((a, b) => mean(a.values) - mean(b.values) || a.name.localeCompare(b.name));
}

/**
 * The scale of an aligned dot plot of 0–1 values: from 0 when the values spread over most of it, else fitted to them
 * on a 0.05 grid (the plot names where it starts), at least 0.2 wide so close dots still separate honestly.
 */
export function dotDomain(values: readonly (number | null)[]): [number, number] {
  const v = values.filter((x): x is number => x !== null && Number.isFinite(x));
  if (!v.length) return [0, 1];
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  let a = Math.max(0, Math.floor((lo - 0.02) * 20) / 20);
  const b = Math.min(1, Math.ceil((hi + 0.02) * 20) / 20);
  if (a < 0.3) a = 0;
  if (b - a < 0.2) a = Math.max(0, +(b - 0.2).toFixed(2));
  return [+a.toFixed(2), +b.toFixed(2)];
}

/** Which compared entries are finished, in the order the URL names them; the ids that are not (unknown or not done). */
export function compareShown(ids: readonly string[], entries: readonly BenchEntry[]): { shown: BenchEntry[]; missing: string[] } {
  const shown: BenchEntry[] = [];
  const missing: string[] = [];
  for (const id of ids) {
    const e = entries.find((x) => x.id === id);
    if (e && e.status === 'done' && e.score) shown.push(e);
    else missing.push(id);
  }
  return { shown, missing };
}

/**
 * The best of a row of Compare's table as printed (`d` decimals): one entry, never a tie at that precision (a "best"
 * beside an equal number says nothing). With `intervals` (the mAP row), only an entry whose interval clears every
 * other's as printed: beside "not clearly different", a "best" would claim what the intervals deny.
 */
export function bestAsPrinted(values: readonly (number | null)[], d: number, intervals: readonly ((readonly [number, number]) | null)[] | null = null): Set<number> {
  const r = values.map((x) => (x === null ? null : Math.round(x * 10 ** d)));
  const v = r.filter((x): x is number => x !== null);
  if (v.length < 2) return new Set();
  const top = Math.max(...v);
  const at = r.flatMap((x, i) => (x === top ? [i] : []));
  if (at.length !== 1) return new Set();
  if (intervals) {
    const mine = intervals[at[0]!];
    if (!mine || intervals.some((c, i) => i !== at[0] && values[i] !== null && (!c || overlaps(c, mine)))) return new Set();
  }
  return new Set(at);
}

/** A value's difference from the baseline as printed: of the two numbers shown, so "0.68 | 0.68" is ±0, never −0.01. */
export function diffAsPrinted(v: number | null, base: number | null, d: number): string | null {
  if (v === null || base === null) return null;
  const x = (Math.round(v * 10 ** d) - Math.round(base * 10 ** d)) / 10 ** d;
  return x === 0 ? '±0' : `${x > 0 ? '+' : '−'}${Math.abs(x).toFixed(d)}`;
}
