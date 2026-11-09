// Fitted y domains for run charts. A domain is "nice" (it ends on a tick) and is re-fitted at most once per epoch, so
// a live chart never jitters as steps arrive (web/DESIGN_DIRECTION.md §2.9). Values are never tweened. Every y axis
// holds 3–5 ticks (§2.4): more gridlines read as a texture rather than a scale.

export interface Fit {
  domain: [number, number];
  /** Tick values, ending exactly on the domain's top. */
  ticks: number[];
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
  if (!f) return { domain: [lo, hi], ticks: [lo, hi] };
  const d = stepDecimals(f.step);
  const t: number[] = [];
  for (let k = f.a; k <= f.b; k++) t.push(+(k * f.step).toFixed(d));
  return { domain: [t[0]!, t[t.length - 1]!], ticks: t };
}

/**
 * A 0–1 metric (mAP, precision, recall): from 0 up to the first nice step at or above `max` plus `headroom`, capped
 * at 1. Steps of 0.01, 0.02, 0.05, 0.1, 0.2, 0.25 or 0.5, whichever gives the tightest top in 3–5 ticks at least MIN_GAP apart;
 * the top is always a tick.
 */
export function unitDomain(max: number, plotPx: number, headroom = 0.02): Fit {
  const hi = Number.isFinite(max) && max > 0 ? Math.min(1, max + headroom) : 1;
  return fitSteps(0, hi, [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1], (gaps) => plotPx / gaps >= MIN_GAP);
}

/**
 * Any linear quantity (loss, learning rate, throughput): the extent of the values widened to whole tick steps, so the
 * plot starts and ends on a gridline, in 3–5 ticks.
 */
export function niceDomain(lo: number, hi: number): Fit {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1] };
  if (hi <= lo) hi = lo + (Math.abs(lo) || 1);
  return fitSteps(lo, hi, candidates(hi - lo));
}

/** Ticks inside a domain the caller fixed: the smallest nice step that leaves 3 to `max` (at most 5) of them. */
export function capTicks(lo: number, hi: number, max = MAX_TICKS): number[] {
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
 * Log ticks: the 1-2-5 values in the domain when at most MAX_TICKS of them fit; otherwise the outermost of them and the powers of ten
 * between them (every other power if those are still too many), so the scale reads in decades and both ends are named.
 */
export function logTicks(lo: number, hi: number): number[] {
  const all: number[] = [];
  const decades: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) {
      const v = +(m * 10 ** e).toPrecision(6);
      if (v < lo * (1 - 1e-9) || v > hi * (1 + 1e-9)) continue;
      all.push(v);
      if (m === 1) decades.push(v);
    }
  if (all.length <= MAX_TICKS) return all;
  const ends = (inner: number[]) => [...new Set([all[0]!, ...inner, all[all.length - 1]!])].sort((a, b) => a - b);
  let t = ends(decades);
  for (let k = 2; t.length > MAX_TICKS; k++) t = ends(decades.filter((_, i) => i % k === 0));
  return t;
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
