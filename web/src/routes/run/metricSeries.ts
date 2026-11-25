// A live run's chart points, kept up to date from its append logs (`live.stepsLog`, `live.epochsLog`): each update
// maps and smooths only the steps added since the last one. The Svelte run page re-mapped every step into points and
// re-smoothed every series on each update (O(history) several times a second); this does the same arithmetic, in the
// same order, on the new points only, so the result is bit-identical to `smooth(perStep(...), alpha)`
// (metricSeries.test.ts compares the two on long random series).
//
// The lists held here are internal and grow in place: hand a chart a copy (`slice()`), whose identity says it changed.

import type { AppendLog, LogCursor } from '$lib/api/appendLog';
import type { Epoch, Step } from '$lib/api/types';
import type { Pt } from '$lib/charts/downsample';
import { runHorizon } from '$lib/charts/horizon';

/** Running bounds of a point list's finite values, and of its positive ones (what a log axis draws). */
export interface Extent {
  lo: number;
  hi: number;
  loPos: number;
  hiPos: number;
  /** Count of finite non-positive values (y <= 0), which cannot be plotted on a log axis. */
  nonPos: number;
}

const noExtent = (): Extent => ({ lo: Infinity, hi: -Infinity, loPos: Infinity, hiPos: -Infinity, nonPos: 0 });

function widen(e: Extent, y: number) {
  if (!Number.isFinite(y)) return;
  if (y < e.lo) e.lo = y;
  if (y > e.hi) e.hi = y;
  if (y > 0) {
    if (y < e.loPos) e.loPos = y;
    if (y > e.hiPos) e.hiPos = y;
  } else {
    e.nonPos++;
  }
}

/** Total count of finite non-positive (`y <= 0`) values across the given track extents. */
export function countNonPositive(list: readonly Extent[]): number {
  let total = 0;
  for (const e of list) total += e.nonPos;
  return total;
}

/**
 * The bounds of several lists together, in order (strict comparisons, so a tie keeps the earlier value, as one loop
 * over all of their points would): `[lo, hi]`, or null when none has a value. `positive`: only values above 0.
 */
export function joinExtents(list: readonly Extent[], positive: boolean): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const e of list) {
    const a = positive ? e.loPos : e.lo;
    const b = positive ? e.hiPos : e.hi;
    if (a < lo) lo = a;
    if (b > hi) hi = b;
  }
  return Number.isFinite(lo) ? [lo, hi] : null;
}

/** A point list as it grows, with its bounds. */
export interface Track {
  pts: Pt[];
  ext: Extent;
}

/**
 * `smooth` (lib/charts/downsample: an EMA with bias correction) applied as a list grows: the same operations in the
 * same order on each new point, so every value is bit-identical to smoothing the whole list again.
 */
export class Smoother implements Track {
  pts: Pt[] = [];
  ext = noExtent();
  #avg = 0;
  #n = 0;
  #done = 0;
  constructor(readonly alpha: number) {}

  /** Catches up with `src`, a list that only grew since the last call. */
  sync(src: readonly Pt[]): void {
    const alpha = this.alpha;
    for (let i = this.#done; i < src.length; i++) {
      const p = src[i]!;
      if (!Number.isFinite(p.y)) {
        this.pts.push(p);
        continue;
      }
      this.#n++;
      this.#avg = alpha * this.#avg + (1 - alpha) * p.y;
      const q = { x: p.x, y: this.#avg / (1 - alpha ** this.#n) };
      this.pts.push(q);
      widen(this.ext, q.y);
    }
    this.#done = src.length;
  }
}

export type StepField = 'total' | 'box' | 'cls' | 'dfl' | 'lr' | 'ips';
const STEP: Record<StepField, (s: Step) => number> = {
  total: (s) => s.loss.total,
  box: (s) => s.loss.box,
  cls: (s) => s.loss.cls,
  dfl: (s) => s.loss.dfl,
  lr: (s) => s.lr,
  ips: (s) => s.imgs_per_s,
};

interface Mapped extends Track {
  done: number;
}

/**
 * A run's per-step series (x in epochs: step / steps per epoch), each mapped and smoothed on demand and only for the
 * steps it has not seen. `sync` first, after the log's `version` moved; `rev` changes whenever the points did.
 */
export class StepSeries {
  #log: AppendLog<Step>;
  #cursor: LogCursor | null = null;
  /** The steps as of the last sync (the log may grow, or be replaced, before the next one). */
  #items: readonly Step[] = [];
  #spe = 1;
  /** Steps per epoch from the history (lib/api/progress `stepsPerEpoch`), found once: the first step of epoch 2. */
  #scanned = 0;
  #found = 0;
  #mapped = new Map<StepField, Mapped>();
  #smoothed = new Map<string, Smoother>();
  #lrMax = 0;
  #lrDone = 0;
  /** Bumps whenever any point changed (steps added, the list replaced, steps per epoch learned). */
  rev = 0;

  constructor(log: AppendLog<Step>) {
    this.#log = log;
  }

  /** The steps, oldest first (the log's list). */
  get steps(): readonly Step[] {
    return this.#items;
  }
  /** `max(1, stepsPerEpoch(run, steps))`, as of the last sync. */
  get spe(): number {
    return this.#spe;
  }

  /** Reads what the log added since the last sync. `runSpe`: the run's own steps per epoch (0 while unknown). */
  sync(runSpe: number): void {
    let r = this.#log.read(this.#cursor);
    // Steps are never trimmed at the front; should they be, start over.
    if (!r.reset && r.dropped) r = this.#log.read(null);
    this.#cursor = r.cursor;
    this.#items = this.#log.items;
    let changed = r.added.length > 0;
    if (r.reset) {
      this.#scanned = 0;
      this.#found = 0;
      this.#lrMax = 0;
      this.#lrDone = 0;
      this.#clear();
      changed = true;
    }
    const spe = Math.max(1, runSpe > 0 ? runSpe : this.#historySpe());
    if (spe !== this.#spe) {
      this.#spe = spe;
      this.#clear();
      changed = true;
    }
    if (changed) this.rev++;
  }

  #clear() {
    this.#mapped.clear();
    this.#smoothed.clear();
  }

  /** `stepsPerEpoch`'s fallback: the steps from the first one to the first of a later epoch (0 until there is one). */
  #historySpe(): number {
    const steps = this.#items;
    const first = steps[0];
    if (!first) return 0;
    if (this.#found) return this.#found;
    for (; this.#scanned < steps.length; this.#scanned++) {
      const s = steps[this.#scanned]!;
      if (s.epoch > first.epoch) {
        this.#found = s.step - first.step;
        // A step count of 0 (two epochs at one step) is "not found" for the cache; the next sync scans on from here.
        if (this.#found) return this.#found;
        return 0;
      }
    }
    return 0;
  }

  /** The raw points of a field (internal: copy before handing out). */
  raw(field: StepField): Track {
    let m = this.#mapped.get(field);
    if (!m) {
      m = { pts: [], ext: noExtent(), done: 0 };
      this.#mapped.set(field, m);
    }
    const steps = this.#items;
    const get = STEP[field];
    const spe = this.#spe;
    for (let i = m.done; i < steps.length; i++) {
      const s = steps[i]!;
      const p = { x: s.step / spe, y: get(s) };
      m.pts.push(p);
      widen(m.ext, p.y);
    }
    m.done = steps.length;
    return m;
  }

  /** A field smoothed at `alpha` (`smooth(raw, alpha)`; the raw points themselves for alpha 0). */
  smoothed(field: StepField, alpha: number): Track {
    const raw = this.raw(field);
    if (alpha <= 0) return raw;
    const key = `${field}|${alpha}`;
    let s = this.#smoothed.get(key);
    if (!s) {
      s = new Smoother(alpha);
      this.#smoothed.set(key, s);
    }
    s.sync(raw.pts);
    return s;
  }

  /** The largest learning rate so far (0 when none is above 0), as a loop over every step would find it. */
  lrMax(): number {
    const steps = this.#items;
    let m = this.#lrMax;
    for (let i = this.#lrDone; i < steps.length; i++) if (steps[i]!.lr > m) m = steps[i]!.lr;
    this.#lrMax = m;
    this.#lrDone = steps.length;
    return m;
  }
}

export type EpochField = 'map50_95' | 'map50' | 'precision' | 'recall';

/** A run's per-epoch series (x: the epoch), appended as epochs finish. `sync` first; `rev` moves with the points. */
export class EpochSeries {
  #log: AppendLog<Epoch>;
  #cursor: LogCursor | null = null;
  #pts: Record<EpochField, Pt[]> = { map50_95: [], map50: [], precision: [], recall: [] };
  rev = 0;

  constructor(log: AppendLog<Epoch>) {
    this.#log = log;
  }

  get epochs(): readonly Epoch[] {
    return this.#log.items;
  }

  sync(): void {
    let r = this.#log.read(this.#cursor);
    if (!r.reset && r.dropped) r = this.#log.read(null);
    this.#cursor = r.cursor;
    if (r.reset) this.#pts = { map50_95: [], map50: [], precision: [], recall: [] };
    else if (!r.added.length) return;
    for (const e of r.added) {
      this.#pts.map50_95.push({ x: e.epoch, y: e.val.map50_95 });
      this.#pts.map50.push({ x: e.epoch, y: e.val.map50 });
      this.#pts.precision.push({ x: e.epoch, y: e.val.precision });
      this.#pts.recall.push({ x: e.epoch, y: e.val.recall });
    }
    this.rev++;
  }

  /** A field's points (internal: copy before handing out). */
  points(field: EpochField): Pt[] {
    return this.#pts[field];
  }
}

/** The "So far" range's ends while a run trains: 5, 10, 20, 50, 100, 200, 500, … epochs. */
const STEPS = [1, 2, 5];

/**
 * The x range of a live run's charts (lib/charts/horizon `runHorizon` for the rest): "So far" grows in steps (5, 10, 20,
 * 50, … epochs, never past the plan) rather than to the next whole epoch, so the axis holds still for several epochs
 * (it snapped from the plan to 0–1 at the first point, then 0–2, 0–3…) and a short run's "mosaic off" mark stays in
 * view from the start.
 */
export function liveHorizon(observed: number, planned: number, mode: 'observed' | 'full'): [number, number] {
  if (mode === 'full' || planned <= 0) return runHorizon(observed, planned, mode);
  const need = Math.max(1, Math.ceil(observed));
  for (let p = 1; ; p *= 10)
    for (const s of STEPS) {
      const end = s * p;
      if (end >= 5 && end >= need) return [0, Math.min(planned, end)];
    }
}
