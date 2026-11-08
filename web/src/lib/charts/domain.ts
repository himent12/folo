// Fitted y domains for run charts. A domain is "nice" (it ends on a tick) and is re-fitted at most once per epoch, so
// a live chart never jitters as steps arrive (web/DESIGN_DIRECTION.md §2.9). Values are never tweened.

import { ticks as niceTicks } from './downsample';

export interface Fit {
  domain: [number, number];
  /** Tick values, ending exactly on the domain's top. */
  ticks: number[];
}

/** Gridlines closer than this (px) read as a texture rather than a scale. */
const MIN_GAP = 32;

/**
 * A 0–1 metric (mAP, precision, recall): from 0 up to the next 0.1 step above `max` plus `headroom`, capped at 1. When
 * the plot is too short for a gridline every 0.1, the step is 0.2 and the top rounds up to it, so the top is a tick.
 */
export function unitDomain(max: number, plotPx: number, headroom = 0.02): Fit {
  const hi = Number.isFinite(max) && max > 0 ? Math.min(1, max + headroom) : 1;
  const room = Math.max(2, Math.floor(plotPx / MIN_GAP));
  const fine = Math.ceil(hi * 10 - 1e-9);
  const step = fine <= room ? 0.1 : 0.2;
  const n = Math.ceil(hi / step - 1e-9);
  const t: number[] = [];
  for (let k = 0; k <= n; k++) t.push(+(k * step).toFixed(1));
  return { domain: [0, t[n]!], ticks: t };
}

/**
 * Any linear quantity (loss, learning rate, throughput): the extent of the values widened to whole tick steps, so the
 * plot starts and ends on a gridline.
 */
export function niceDomain(lo: number, hi: number, count = 4): Fit {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { domain: [0, 1], ticks: niceTicks(0, 1, count) };
  if (hi <= lo) hi = lo + (Math.abs(lo) || 1);
  const t0 = niceTicks(lo, hi, count);
  const step = t0.length > 1 ? t0[1]! - t0[0]! : hi - lo;
  const d = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  const a = Math.floor(lo / step + 1e-9);
  const b = Math.ceil(hi / step - 1e-9);
  const t: number[] = [];
  for (let k = a; k <= b; k++) t.push(+(k * step).toFixed(d));
  return { domain: [t[0]!, t[t.length - 1]!], ticks: t };
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
