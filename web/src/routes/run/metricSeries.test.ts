// The run chart's incremental series (./metricSeries) against what the Svelte run page computed on every update:
// `smooth(perStep(f), alpha)` over all steps, the loop over every point for the y extent, and the largest LR. Long
// random runs streamed in random batches (with resets, steps per epoch learned late, non-finite values) must give the
// same numbers, bit for bit (Object.is), after every batch.
import { describe, expect, it } from 'vitest';
import { AppendLog } from '$lib/api/appendLog';
import { stepsPerEpoch } from '$lib/api/progress';
import type { Epoch, Step } from '$lib/api/types';
import { smooth, type Pt } from '$lib/charts/downsample';
import { EpochSeries, joinExtents, Smoother, StepSeries, type StepField } from './metricSeries';

/** A seeded PRNG (mulberry32): the same series every run. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A value, now and then not finite (a diverged step) or not positive (what a log axis skips). */
function value(r: () => number, scale: number): number {
  const u = r();
  if (u < 0.004) return NaN;
  if (u < 0.006) return Infinity;
  if (u < 0.01) return 0;
  if (u < 0.012) return -r();
  return r() * scale * (1 + 10 * r() ** 8);
}

function step(r: () => number, i: number, spe: number): Step {
  return {
    step: i + 1,
    epoch: Math.floor(i / spe) + 1,
    loss: { total: value(r, 8), box: value(r, 3), cls: value(r, 2), dfl: value(r, 1.5) },
    lr: value(r, 0.01),
    imgs_per_s: value(r, 400),
    eta_s: null,
  } as unknown as Step;
}

const FIELDS: Record<StepField, (s: Step) => number> = {
  total: (s) => s.loss.total,
  box: (s) => s.loss.box,
  cls: (s) => s.loss.cls,
  dfl: (s) => s.loss.dfl,
  lr: (s) => s.lr,
  ips: (s) => s.imgs_per_s,
};

/** The Svelte page's own computation, over every step. */
function full(steps: Step[], runSpe: number, f: StepField, alpha: number): Pt[] {
  const spe = Math.max(1, stepsPerEpoch({ steps_per_epoch: runSpe }, steps));
  return smooth(
    steps.map((s) => ({ x: s.step / spe, y: FIELDS[f](s) })),
    alpha,
  );
}
function fullExtent(lists: Pt[][], positive: boolean): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const pts of lists)
    for (const p of pts) {
      if (!Number.isFinite(p.y) || (positive && p.y <= 0)) continue;
      if (p.y < lo) lo = p.y;
      if (p.y > hi) hi = p.y;
    }
  return Number.isFinite(lo) ? [lo, hi] : null;
}
function same(a: readonly Pt[], b: readonly Pt[]) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i]!.x, b[i]!.x) || !Object.is(a[i]!.y, b[i]!.y)) throw new Error(`point ${i}: ${a[i]!.x},${a[i]!.y} vs ${b[i]!.x},${b[i]!.y}`);
  }
}

describe('Smoother', () => {
  it('is smooth(), bit for bit, however the points arrive', () => {
    for (const alpha of [0.6, 0.9, 0.98, 0.999]) {
      const r = rng(Math.round(alpha * 1000));
      const pts: Pt[] = Array.from({ length: 20_000 }, (_, i) => ({ x: i / 7, y: value(r, 5) }));
      const s = new Smoother(alpha);
      for (let n = 0; n < pts.length; ) {
        n = Math.min(pts.length, n + 1 + Math.floor(r() * 40));
        s.sync(pts.slice(0, n));
      }
      same(s.pts, smooth(pts, alpha));
    }
  });
});

describe('StepSeries', () => {
  it('gives the full recomputation after every batch of a long live run', () => {
    const r = rng(7);
    const SPE = 37;
    const log = new AppendLog<Step>();
    const series = new StepSeries(log);
    const all: Step[] = [];
    let batches = 0;
    // The run's steps per epoch is unknown (0) for its first 500 steps, as on a server before the trainer is built.
    for (let i = 0; i < 12_000; ) {
      const n = 1 + Math.floor(r() * 25);
      for (let k = 0; k < n; k++, i++) {
        const s = step(r, i, SPE);
        all.push(s);
        log.push(s);
      }
      const runSpe = i < 500 ? 0 : SPE;
      series.sync(runSpe);
      // Check often early, then now and then (the full recomputation is the slow part of this test).
      if (batches++ % (i < 2000 ? 1 : 40) !== 0) continue;
      for (const alpha of [0, 0.6, 0.98]) {
        for (const f of ['total', 'box', 'cls', 'dfl'] as const) same(series.smoothed(f, alpha).pts, full(all, runSpe, f, alpha));
        // The loss tab's extent: the raw total (when smoothed), the smoothed total and the components, linear and log.
        const lists = [...(alpha > 0 ? [full(all, runSpe, 'total', 0)] : []), ...(['total', 'box', 'cls', 'dfl'] as const).map((f) => full(all, runSpe, f, alpha))];
        const tracks = [...(alpha > 0 ? [series.raw('total').ext] : []), ...(['total', 'box', 'cls', 'dfl'] as const).map((f) => series.smoothed(f, alpha).ext)];
        for (const positive of [false, true]) expect(joinExtents(tracks, positive)).toEqual(fullExtent(lists, positive));
      }
      same(series.smoothed('ips', 0.9).pts, full(all, runSpe, 'ips', 0.9));
      same(series.raw('lr').pts, full(all, runSpe, 'lr', 0));
      let m = 0;
      for (const s of all) if (s.lr > m) m = s.lr;
      expect(series.lrMax()).toBe(m);
    }
  });

  it('starts over when the steps are replaced (a resume rewinds them) and when steps per epoch is learned', () => {
    const r = rng(11);
    const log = new AppendLog<Step>();
    const series = new StepSeries(log);
    let steps = Array.from({ length: 900 }, (_, i) => step(r, i, 50));
    log.replace(steps);
    series.sync(0);
    const rev = series.rev;
    same(series.smoothed('total', 0.98).pts, full(steps, 0, 'total', 0.98));
    // Rewound to epoch 10 (a new list): everything after it goes.
    steps = steps.filter((s) => s.epoch <= 10);
    log.replace(steps);
    series.sync(0);
    expect(series.rev).toBeGreaterThan(rev);
    same(series.smoothed('total', 0.98).pts, full(steps, 0, 'total', 0.98));
    // The run's own figure arrives and differs from the history's: every x moves.
    series.sync(40);
    same(series.smoothed('box', 0.9).pts, full(steps, 40, 'box', 0.9));
    // Nothing new: the points stay as they were.
    const before = series.rev;
    series.sync(40);
    expect(series.rev).toBe(before);
  });
});

describe('EpochSeries', () => {
  it('appends the finished epochs and starts over on a rewind', () => {
    const log = new AppendLog<Epoch>();
    const series = new EpochSeries(log);
    const ep = (n: number) => ({ epoch: n, val: { map50_95: n / 100, map50: n / 50, precision: 0.5, recall: 0.4 } }) as unknown as Epoch;
    for (let n = 1; n <= 30; n++) {
      log.push(ep(n));
      series.sync();
    }
    expect(series.points('map50_95').map((p) => p.x)).toEqual(Array.from({ length: 30 }, (_, i) => i + 1));
    log.replace([ep(1), ep(2)]);
    series.sync();
    expect(series.points('map50').map((p) => p.y)).toEqual([1 / 50, 2 / 50]);
  });
});
