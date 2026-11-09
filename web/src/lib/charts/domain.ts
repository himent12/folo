// Fitted y domains for run charts. A domain fits its data (web/DESIGN_DIRECTION.md §2.4: never two-thirds of a plot
// empty) and is re-fitted at most once per epoch, so a live chart never jitters as steps arrive (§2.9). Values are
// never tweened. Every y axis holds 3–5 ticks (§2.4): more gridlines read as a texture rather than a scale. A log axis
// is the exception: its 1-2-5 values sit unevenly, so up to six of them still read as a scale (MAX_LOG_TICKS).
// **One tick rule at every width**: a domain's step and decimals come from the domain alone, never from the plot's
// height, so the same 0–0.86 reads 0.0 · 0.2 · 0.4 · 0.6 · 0.8 on a desktop and on a phone. A plot too short for every
// tick keeps every second (or fifth) one of the same step, with the same decimals (0.0 · 0.4 · 0.8), never a new step.

export interface Fit {
  domain: [number, number];
  /** Tick values inside the domain: niceDomain's end exactly on its top; unitDomain's top may sit a little above. */
  ticks: number[];
  /** Decimals of the domain's tick step: every label of this axis takes them, however many ticks a plot shows. */
  decimals: number;
}

/** Gridlines closer than this (px) read as a texture rather than a scale. */
const MIN_GAP = 32;
/** The most ticks a y axis takes. */
export const MAX_TICKS = 5;
const MIN_TICKS = 3;
/** Nice steps within a power of ten. */
const MANTISSAS = [1, 2, 2.5, 5];

/** Decimals a tick step needs: "0.25" two, "0.2" one, "5" none. */
export function stepDecimals(step: number): number {
  if (!(step > 0) || !Number.isFinite(step)) return 0;
  // From the step's own magnitude, with a relative tolerance: a tiny step (a learning rate of 2e-7) needs 7 decimals,
  // never 0, which would collapse every tick onto the same value.
  const first = Math.max(0, -Math.floor(Math.log10(step)));
  for (let d = first; d <= first + 8; d++) {
    const x = step * 10 ** d;
    if (Math.abs(x - Math.round(x)) < 1e-6 * Math.max(1, Math.abs(x))) return d;
  }
  return first + 8;
}

/** Nice steps around `span / MAX_TICKS`, smallest first. */
function candidates(span: number): number[] {
  const e = Math.floor(Math.log10(span / MAX_TICKS)) - 1;
  const out: number[] = [];
  for (let k = e; k <= e + 2; k++) for (const m of MANTISSAS) out.push(+(m * 10 ** k).toPrecision(6));
  return out;
}

/**
 * The tightest whole-step domain around [lo, hi] that takes 3–5 ticks (failing that, at most 5), the smaller step
 * on a tie, so the plot fills its height and the gridlines stay a scale. `gapOk` vetoes steps too dense for the plot.
 */
function fitSteps(lo: number, hi: number, steps: readonly number[], gapOk: (gaps: number) => boolean = () => true): Fit {
  type C = { a: number; b: number; step: number };
  let best: C | null = null;
  let loose: C | null = null;
  const span = (c: C) => (c.b - c.a) * c.step;
  for (const step of steps) {
    const a = Math.floor(lo / step + 1e-9);
    const b = Math.ceil(hi / step - 1e-9);
    const n = b - a + 1;
    if (n > MAX_TICKS || n < 2 || !gapOk(n - 1)) continue;
    const c = { a, b, step };
    if (n >= MIN_TICKS && (!best || span(c) < span(best) - 1e-12)) best = c;
    if (!loose || span(c) < span(loose) - 1e-12) loose = c;
  }
  const f = best ?? loose;
  if (!f) return { domain: [lo, hi], ticks: [lo, hi], decimals: stepDecimals(hi - lo) };
  const d = stepDecimals(f.step);
  const t: number[] = [];
  for (let k = f.a; k <= f.b; k++) t.push(+(k * f.step).toFixed(d));
  return { domain: [t[0]!, t[t.length - 1]!], ticks: t, decimals: d };
}

/**
 * Below this top a 0–1 metric's axis is zoomed in, and the chart says so (`unitZoomed`, `UNIT_ZOOM_NOTE`): a run stuck
 * near 0 must still read as low, so its plot names the full 0–1 scale instead of leaving two-thirds of itself empty.
 */
export const UNIT_ZOOM_BELOW = 0.25;

/** The y-axis caption of a zoomed 0–1 axis: the scale it was cut from. */
export const UNIT_ZOOM_NOTE = 'Zoomed in · full scale 0–1';

/** Nice tick steps for a 0–1 metric, smallest first. */
const UNIT_STEPS = [0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5];

/**
 * The top of a 0–1 metric's axis: the data's maximum plus 5 % headroom, rounded up to a fifth of its power of ten
 * (0.02 from 0.1, 0.002 from 0.01), capped at 1, so the data reaches at least 80 % of the plot. It moves in those
 * steps, never with every value.
 */
export function unitTop(max: number): number {
  if (!Number.isFinite(max) || max <= 0) return 1;
  const h = max * 1.05;
  const q = 10 ** Math.floor(Math.log10(h)) / 5;
  return Math.min(1, +(Math.ceil(h / q - 1e-9) * q).toPrecision(6));
}

/** Whether a fitted 0–1 axis is zoomed in (its top under UNIT_ZOOM_BELOW). */
export const unitZoomed = (fit: Fit | null | undefined) => !!fit && fit.domain[1] < UNIT_ZOOM_BELOW - 1e-9;

/** How many of a step's ticks a short plot skips to keep its gridlines MIN_GAP apart: every 2nd, else every 5th. */
const THIN = [1, 2, 5];

/**
 * A 0–1 metric (mAP, precision, recall): from 0 (which keeps the scale honest) up to `unitTop(max)`, so the data fills
 * the plot. The gridlines take the nice step whose highest tick comes nearest that top in 3–5 ticks (the smaller step
 * on a tie); the top itself need not be a tick (0–0.65 reads 0 · 0.2 · 0.4 · 0.6). The step depends on the domain
 * only: a plot too short for its ticks `MIN_GAP` apart keeps every 2nd (else 5th) one of them, same step, same
 * decimals, so a phone and a desktop read one scale.
 */
export function unitDomain(max: number, plotPx: number): Fit {
  const top = unitTop(max);
  let best: { step: number; n: number } | null = null;
  for (const step of UNIT_STEPS) {
    const n = Math.floor(top / step + 1e-9) + 1;
    if (n < MIN_TICKS || n > MAX_TICKS) continue;
    if (!best || (n - 1) * step > (best.n - 1) * best.step + 1e-9) best = { step, n };
  }
  // A top too small for three nice ticks: its base and its top.
  if (!best) return { domain: [0, top], ticks: [0, top], decimals: stepDecimals(top) };
  const { step, n } = best;
  const d = stepDecimals(step);
  const all = Array.from({ length: n }, (_, k) => +(k * step).toFixed(d));
  const every = THIN.find((k) => k < n && (plotPx * step * k) / top >= MIN_GAP);
  // Not even every fifth tick fits: the base and the highest tick, still at this step's decimals.
  const ticks = every ? all.filter((_, k) => k % every === 0) : [0, all[n - 1]!];
  return { domain: [0, top], ticks, decimals: d };
}

/**
 * Any linear quantity (loss, learning rate, throughput): the extent of the values widened to whole tick steps, so the
 * plot starts and ends on a gridline, in 3–5 ticks.
 */
export function niceDomain(lo: number, hi: number): Fit {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], decimals: 2 };
  if (hi <= lo) hi = lo + (Math.abs(lo) || 1);
  return fitSteps(lo, hi, candidates(hi - lo));
}

/** Ticks inside a domain the caller fixed: the smallest nice step that leaves 3 to `max` (at most 5) of them. */
export function capTicks(lo: number, hi: number, max = MAX_TICKS): number[] {
  // A non-finite end has no ticks to space (and would never end the loop below).
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (!(hi > lo)) return [lo];
  let fallback: number[] = [];
  for (const step of candidates(hi - lo)) {
    const d = stepDecimals(step);
    const t: number[] = [];
    for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) t.push(+(k * step).toFixed(d));
    if (t.length <= max && t.length >= Math.min(MIN_TICKS, max)) return t;
    if (t.length <= max && t.length > fallback.length) fallback = t;
  }
  return fallback.length ? fallback : [lo, hi];
}

/**
 * The most labels a log axis takes: its 1-2-5 values sit at uneven spacing, so six of them (0.2 · 0.5 · 1 · 2 · 5 · 10)
 * still read as a scale where six linear gridlines would read as a texture.
 */
export const MAX_LOG_TICKS = 6;

/**
 * Log ticks: the 1-2-5 values in the domain when at most MAX_LOG_TICKS of them fit; otherwise the outermost of them and
 * the powers of ten between them (every other power if those are still too many), so the scale reads in decades and
 * both ends are named.
 */
export function logTicks(lo: number, hi: number): number[] {
  // A log axis needs a positive, finite range.
  if (!(lo > 0) || !Number.isFinite(hi) || !(hi >= lo)) return [];
  const all: number[] = [];
  const decades: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) {
      const v = +(m * 10 ** e).toPrecision(6);
      if (v < lo * (1 - 1e-9) || v > hi * (1 + 1e-9)) continue;
      all.push(v);
      if (m === 1) decades.push(v);
    }
  if (all.length <= MAX_LOG_TICKS) return all;
  const ends = (inner: number[]) => [...new Set([all[0]!, ...inner, all[all.length - 1]!])].sort((a, b) => a - b);
  let t = ends(decades);
  for (let k = 2; t.length > MAX_LOG_TICKS; k++) t = ends(decades.filter((_, i) => i % k === 0));
  return t;
}

/**
 * The unlabelled gridlines of a log axis: the 1-2-5 values in the domain that `logTicks` leaves without a label, so a
 * decade-labelled axis (0.01, 0.1, 1, 10) still shows its 2s and 5s and reads as a scale, not a few uneven ticks.
 */
export function logMinors(lo: number, hi: number): number[] {
  const labelled = new Set(logTicks(lo, hi));
  if (!labelled.size) return [];
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) {
      const v = +(m * 10 ** e).toPrecision(6);
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9) && !labelled.has(v)) out.push(v);
    }
  return out;
}

/** A log domain from the 1-2-5 values just outside the data, so it moves in steps, not with every value. */
export function logDomain(lo: number, hi: number): [number, number] {
  if (!(lo > 0) || !(hi > 0)) return [0.1, 1];
  const down = (v: number) => {
    const e = 10 ** Math.floor(Math.log10(v));
    return ([5, 2, 1].map((m) => m * e).find((c) => c <= v * (1 + 1e-9)) ?? e);
  };
  const up = (v: number) => {
    const e = 10 ** Math.floor(Math.log10(v));
    return [1, 2, 5, 10].map((m) => m * e).find((c) => c >= v * (1 - 1e-9)) ?? 10 * e;
  };
  const a = down(lo);
  const b = up(hi);
  return b > a ? [a, b] : [a, a * 2];
}

export interface Held<T> {
  key: string;
  value: T;
}

/**
 * Holds a fitted domain until `key` changes (the epoch, or a control the user set), so a live chart re-fits at most
 * once per epoch. The one exception: a new fit reaching past the held one is taken at once, since a held domain must
 * never cut off data.
 */
export function holdDomain<T extends { domain: [number, number] }>(prev: Held<T> | null, key: string, next: T): Held<T> {
  if (!prev || prev.key !== key) return { key, value: next };
  const [a, b] = prev.value.domain;
  return next.domain[0] < a || next.domain[1] > b ? { key, value: next } : prev;
}
