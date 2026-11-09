import { describe, expect, it } from 'vitest';
import { capTicks, FIT_BELOW, fitFloorNote, holdDomain, prDomain, PR_ZOOM, logDomain, logMinors, logTicks, MAX_TICKS, niceDomain, stepDecimals, unitDomain, unitFitDomain, unitTop, unitZoomed, withLabelRow } from './domain';

describe('unitDomain', () => {
  it('fits a 0–1 metric to its data plus a little headroom, so the plot is never mostly empty', () => {
    // A failed run's mAP50-95 (0.494), a comparison topping out at 0.606, a completed run's mAP50 (0.803), a run
    // that learned almost nothing (0.081, run-short-warning).
    expect(unitDomain(0.494, 290).domain).toEqual([0, 0.52]);
    expect(unitDomain(0.606, 360).domain).toEqual([0, 0.64]);
    expect(unitDomain(0.803, 122).domain).toEqual([0, 0.86]);
    expect(unitDomain(0.081, 170).domain).toEqual([0, 0.086]);
    // Whatever the value, the data reaches at least 80 % of the plot's height (rounding on top of 5 % headroom).
    for (const max of [0.004, 0.0099, 0.01, 0.037, 0.081, 0.1, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.88, 0.97])
      expect(max / unitDomain(max, 290).domain[1], `${max}`).toBeGreaterThan(0.8);
    expect(unitTop(0.99)).toBe(1);
  });

  it('draws 3–5 nice gridlines, the highest near the top', () => {
    expect(unitDomain(0.494, 290).ticks).toEqual([0, 0.25, 0.5]);
    expect(unitDomain(0.606, 360).ticks).toEqual([0, 0.2, 0.4, 0.6]);
    expect(unitDomain(0.803, 290)).toEqual({ domain: [0, 0.86], ticks: [0, 0.2, 0.4, 0.6, 0.8], decimals: 1 });
    expect(unitDomain(0.081, 170).ticks).toEqual([0, 0.02, 0.04, 0.06, 0.08]);
    for (const max of [0.004, 0.01, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.88, 0.97, 1]) {
      const { ticks: t, domain } = unitDomain(max, 360);
      expect(t.length, `${max}`).toBeGreaterThanOrEqual(3);
      expect(t.length, `${max}`).toBeLessThanOrEqual(MAX_TICKS);
      expect(t[t.length - 1]! / domain[1], `${max} top tick`).toBeGreaterThanOrEqual(0.7);
      expect(t[t.length - 1]!, `${max}`).toBeLessThanOrEqual(domain[1] + 1e-9);
    }
  });

  it('keeps one step and one set of decimals at every plot height; a short plot shows every second tick', () => {
    // Round 22: the same 0–0.86 read 0.00 · 0.25 · 0.50 · 0.75 on a phone and 0.0 · 0.2 · … · 0.8 on a desktop.
    expect(unitDomain(0.803, 122)).toEqual({ domain: [0, 0.86], ticks: [0, 0.4, 0.8], decimals: 1 });
    for (const max of [0.004, 0.0099, 0.037, 0.081, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.803, 0.88, 0.97, 1]) {
      const wide = unitDomain(max, 360);
      const step = wide.ticks[1]!;
      for (const px of [60, 96, 110, 122, 138, 170, 210, 290]) {
        const f = unitDomain(max, px);
        const at = `${max} @ ${px}px`;
        expect(f.domain, at).toEqual(wide.domain);
        expect(f.decimals, at).toBe(wide.decimals);
        // Every tick a phone shows is one of the desktop's: the same step family, never a new step.
        for (const t of f.ticks) expect(wide.ticks, at).toContain(t);
        const gap = f.ticks[1]! - f.ticks[0]!;
        expect(Math.round(gap / step) * step, at).toBeCloseTo(gap, 9);
        // Where the plot has room, the gridlines keep 32 px apart.
        if ((px * 2 * step) / wide.domain[1] >= 32) expect((px * gap) / wide.domain[1], `${at} gap`).toBeGreaterThanOrEqual(32);
        expect(f.ticks.length, at).toBeGreaterThanOrEqual(2);
      }
    }
  });

  it('says when it is zoomed in, so a run stuck near 0 still reads as low', () => {
    expect(unitZoomed(unitDomain(0.081, 170))).toBe(true);
    expect(unitZoomed(unitDomain(0.2, 170))).toBe(true);
    expect(unitZoomed(unitDomain(0.3, 170))).toBe(false);
    expect(unitZoomed(null)).toBe(false);
  });

  it('never goes past 1, and falls back to 0–1 without data', () => {
    expect(unitDomain(1, 290)).toEqual({ domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], decimals: 2 });
    expect(unitDomain(NaN, 290).domain).toEqual([0, 1]);
    expect(unitDomain(0, 290).domain).toEqual([0, 1]);
  });

  it('is stable: a later, lower value never shrinks it, a higher one moves it in steps', () => {
    const seen = [0.31, 0.42, 0.48, 0.47, 0.5, 0.55, 0.57].map((v, i, a) => unitDomain(Math.max(...a.slice(0, i + 1)), 290).domain[1]);
    expect(seen).toEqual([0.34, 0.46, 0.52, 0.52, 0.54, 0.58, 0.6]);
  });
});

describe('niceDomain', () => {
  it('widens the extent to whole tick steps, in at most five ticks', () => {
    expect(niceDomain(0.42, 3.1)).toEqual({ domain: [0, 4], ticks: [0, 1, 2, 3, 4], decimals: 0 });
    expect(niceDomain(0, 318 * 1.08)).toEqual({ domain: [0, 400], ticks: [0, 100, 200, 300, 400], decimals: 0 });
    expect(niceDomain(1.9, 2.6).ticks).toEqual([1.8, 2, 2.2, 2.4, 2.6]);
  });

  it('keeps small values readable (learning rates)', () => {
    const f = niceDomain(0, 0.0098);
    expect(f.domain).toEqual([0, 0.01]);
    expect(f.ticks).toEqual([0, 0.0025, 0.005, 0.0075, 0.01]);
    expect(f.decimals).toBe(4);
  });

  it('handles a flat or empty series', () => {
    expect(niceDomain(2, 2).domain[0]).toBeLessThanOrEqual(2);
    expect(niceDomain(2, 2).domain[1]).toBeGreaterThan(2);
    expect(niceDomain(Infinity, -Infinity).domain).toEqual([0, 1]);
  });
});

describe('capTicks', () => {
  it('puts 3–5 nice ticks inside a fixed domain', () => {
    expect(capTicks(0, 1)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(capTicks(0, 0.7)).toEqual([0, 0.2, 0.4, 0.6]);
    expect(capTicks(0, 150)).toEqual([0, 50, 100, 150]);
    expect(capTicks(3, 3)).toEqual([3]);
  });
});

describe('logTicks', () => {
  it('keeps the 1-2-5 ticks when they fit, else the ends and the decades', () => {
    expect(logTicks(1, 10)).toEqual([1, 2, 5, 10]);
    // The loss chart's 0.2–10: all six 1-2-5 values are labelled, each with its gridline.
    expect(logTicks(0.2, 10)).toEqual([0.2, 0.5, 1, 2, 5, 10]);
    // Seven or more: both ends and the decades between.
    expect(logTicks(0.1, 100)).toEqual([0.1, 1, 10, 100]);
    expect(logTicks(0.001, 1000)).toEqual([0.001, 0.1, 10, 1000]);
    // A domain that is not on a 1-2-5 value is labelled at the ones inside it.
    expect(logTicks(0.15, 12)).toEqual([0.2, 0.5, 1, 2, 5, 10]);
  });

  it('keeps the unlabelled 1-2-5 values as minor gridlines', () => {
    expect(logMinors(0.2, 10)).toEqual([]);
    expect(logMinors(0.01, 10)).toEqual([0.02, 0.05, 0.2, 0.5, 2, 5]);
    expect(logMinors(1, 10)).toEqual([]);
    expect(logMinors(0.1, 100)).toEqual([0.2, 0.5, 2, 5, 20, 50]);
    expect(logMinors(0, 1)).toEqual([]);
  });
});

describe('stepDecimals', () => {
  it('counts the decimals a step needs', () => {
    expect([5, 0.5, 0.2, 0.25, 0.05, 0.0025].map(stepDecimals)).toEqual([0, 1, 1, 2, 2, 4]);
  });
});

describe('logDomain', () => {
  it('snaps to the 1-2-5 values around the data', () => {
    expect(logDomain(0.43, 3.2)).toEqual([0.2, 5]);
    expect(logDomain(1, 2)).toEqual([1, 2]);
    expect(logDomain(0, 3)).toEqual([0.1, 1]);
  });
});
describe('holdDomain', () => {
  it('keeps the domain within an epoch and re-fits when the epoch changes', () => {
    const d = (a: number, b: number) => ({ domain: [a, b] as [number, number] });
    let h = holdDomain(null, 'ep3', d(0, 4));
    // Loss falls during the epoch: the fit would shrink, the held domain stays.
    h = holdDomain(h, 'ep3', d(0, 3.5));
    expect(h.value.domain).toEqual([0, 4]);
    h = holdDomain(h, 'ep4', d(0, 3.5));
    expect(h.value.domain).toEqual([0, 3.5]);
  });

  it('widens at once rather than cut data off', () => {
    let h = holdDomain(null, 'ep3', { domain: [0, 300] as [number, number] });
    h = holdDomain(h, 'ep3', { domain: [0, 400] as [number, number] });
    expect(h.value.domain).toEqual([0, 400]);
  });
});

describe('tiny ranges', () => {
  it('keeps sub-micro ticks distinct (a small learning rate)', () => {
    const f = niceDomain(0, 9e-7);
    expect(f.domain[1]).toBeGreaterThan(f.domain[0]);
    expect(new Set(f.ticks).size).toBe(f.ticks.length);
    expect(f.ticks.length).toBeGreaterThanOrEqual(3);
    expect(f.ticks[f.ticks.length - 1]).toBeGreaterThanOrEqual(9e-7);
    const t = capTicks(0, 9e-7);
    expect(new Set(t).size).toBe(t.length);
    expect(stepDecimals(2e-7)).toBe(7);
    expect(stepDecimals(0.25)).toBe(2);
    expect(stepDecimals(5)).toBe(0);
  });
});

describe('invalid ranges', () => {
  it('returns no ticks for a non-finite or non-positive log range, and never hangs', () => {
    expect(capTicks(0, Infinity)).toEqual([]);
    expect(capTicks(NaN, 1)).toEqual([]);
    expect(logTicks(0, 1)).toEqual([]);
    expect(logTicks(1, Infinity)).toEqual([]);
  });
});

describe('withLabelRow', () => {
  it('leaves a label row over the highest line end on a labelled chart', () => {
    // Compare at 360 px: runs up to 0.606 used to fit 0–0.64, the labels on the 0.6 gridline.
    const fit = unitDomain(withLabelRow(0.606, 360), 360);
    const below = (1 - 0.606 / fit.domain[1]) * 360;
    expect(below).toBeGreaterThanOrEqual(24);
  });

  it('changes nothing it cannot help: no data, or a plot too short', () => {
    expect(withLabelRow(0, 360)).toBe(0);
    expect(withLabelRow(0.5, 40)).toBe(0.5);
  });
});

describe('unitFitDomain: the Fit axis of Compare', () => {
  /** A run's climb: `n` epochs rising fast from 0 and levelling off at `end`. */
  const climb = (n: number, end: number) => Array.from({ length: n }, (_, i) => end * (1 - Math.exp(-(i + 1) / (n * 0.12))));
  // Round 26's comparison: four runs (37, 150, 61 and 23 epochs) that spend most of their length between 0.45 and 0.61.
  const runs = [climb(37, 0.597), climb(150, 0.604), climb(61, 0.595), climb(23, 0.494)];
  const values = runs.flat();
  const ends = runs.map((r) => r.at(-1)!);

  it('starts the axis under the values most of the lines spend their length in, not at 0', () => {
    const f = unitFitDomain(values, ends, 0.606, 360, true);
    expect(f.domain[0]).toBeGreaterThan(0.3);
    // Every line's end shows, with room under the lowest.
    for (const e of ends) expect(e).toBeGreaterThan(f.domain[0]);
    // At most FIT_BELOW of the values fall under the floor, give or take the rounding down to a half step.
    const under = values.filter((v) => v < f.domain[0]).length / values.length;
    expect(under).toBeLessThanOrEqual(FIT_BELOW + 0.02);
    expect(fitFloorNote(f)).toBe(`y from ${f.domain[0].toFixed(2)}`);
  });

  it('hugs the data: the top at most one minor step over the highest value, the floor one under the percentile', () => {
    // Round 27, Compare at 1440 and 1024: data peaking at 0.61 on a 0.40–0.70 axis, a quarter of the plot empty. The
    // top is 0.625 (half of the 0.05 step), the plot's edge rather than a tick; the floor a tick, as it is written out.
    const vs = [...Array(30).fill(0.2), ...Array(70).fill(0.425), 0.61];
    const f = unitFitDomain(vs, [0.5, 0.604], 0.61, 360, true);
    expect(f.domain).toEqual([0.4, 0.625]);
    expect(f.ticks).toEqual([0.4, 0.45, 0.5, 0.55, 0.6]);
    // Under a 0.1 step the half step is 0.05: data peaking at 0.53 from 0.31 reads on 0.30–0.55.
    expect(unitFitDomain([...Array(30).fill(0.1), ...Array(70).fill(0.31), 0.53], [0.53], 0.53, 360)).toMatchObject({ domain: [0.3, 0.55], ticks: [0.3, 0.4, 0.5] });
    expect(fitFloorNote(f)).toBe('y from 0.40');
    // The highest end keeps its label row (24 px of 360) under the top.
    expect(((f.domain[1] - 0.604) / (f.domain[1] - f.domain[0])) * 360).toBeGreaterThanOrEqual(24);
    // A floor on a half step: runs whose 30th percentile is 0.47 start at 0.45, not 0.40.
    const g = unitFitDomain([...Array(30).fill(0.2), ...Array(70).fill(0.47), 0.61], [0.5, 0.604], 0.61, 360, true);
    expect(g.domain).toEqual([0.45, 0.625]);
    expect(fitFloorNote(g)).toBe('y from 0.45');
    // Generally: each end within one minor step (half the gridlines' step) of the data's, and never cutting it.
    for (const lo of [0.31, 0.42, 0.47, 0.55, 0.68, 0.81])
      for (const max of [lo + 0.06, lo + 0.11, lo + 0.17, Math.min(0.97, lo + 0.33)]) {
        const w = [...Array(30).fill(lo * 0.5), ...Array(70).fill(lo), max];
        const h = unitFitDomain(w, [max], max, 360);
        if (!(h.domain[0] > 0)) continue;
        const step = h.ticks[1]! - h.ticks[0]!;
        // The floor's minor step: half the step, or the step where a half would need a third decimal (0.025 of 0.05).
        const minor = stepDecimals(step / 2) <= Math.max(2, h.decimals) ? step / 2 : step;
        const at = `${lo}–${max}`;
        expect(h.domain[1], at).toBeGreaterThan(max);
        expect(h.domain[1] - max, at).toBeLessThanOrEqual(step / 2 + 0.02 * (max - lo) + 1e-9);
        expect(h.domain[0], at).toBeLessThan(lo);
        expect(lo - h.domain[0], at).toBeLessThanOrEqual(minor + 0.02 * (max - lo) + 1e-9);
      }
  });

  it("keeps round 23's tick rule: 3–5 ticks of the same nice steps, the same step at every height", () => {
    for (const lo of [0.31, 0.42, 0.55, 0.68, 0.81, 0.9])
      for (const max of [lo + 0.05, lo + 0.1, lo + 0.2, Math.min(1, lo + 0.35)]) {
        const vs = [...Array(30).fill(lo * 0.5), ...Array(70).fill(lo), max];
        const wide = unitFitDomain(vs, [max], max, 360);
        const at = `${lo}–${max}`;
        expect(wide.ticks.length, at).toBeGreaterThanOrEqual(3);
        expect(wide.ticks.length, at).toBeLessThanOrEqual(5);
        expect(wide.ticks[0], at).toBeGreaterThanOrEqual(wide.domain[0]);
        expect(wide.ticks.at(-1), at).toBeLessThanOrEqual(wide.domain[1]);
        expect(wide.domain[0], at).toBeLessThanOrEqual(lo);
        const step = wide.ticks[1]! - wide.ticks[0]!;
        expect([0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5].some((s) => Math.abs(s - step) < 1e-9), at).toBe(true);
        // The floor, which is written out, never has more decimals than two or the ticks' own.
        expect(stepDecimals(wide.domain[0]), at).toBeLessThanOrEqual(Math.max(2, wide.decimals));
        for (const px of [96, 122, 210, 290]) {
          const f = unitFitDomain(vs, [max], max, px);
          expect(f.domain, `${at} @ ${px}`).toEqual(wide.domain);
          expect(f.decimals, `${at} @ ${px}`).toBe(wide.decimals);
          for (const t of f.ticks) expect(wide.ticks, `${at} @ ${px}`).toContain(t);
        }
      }
  });

  it('starts at 0 when the floor would sit near it, or a line ends low', () => {
    // Runs still climbing: their 30th percentile is near 0.
    const early = [10, 12].flatMap((n) => Array.from({ length: n }, (_, i) => (0.35 * (i + 1)) / n));
    expect(unitFitDomain(early, [0.3, 0.35], 0.35, 360)).toEqual(unitDomain(0.35, 360));
    // A run that ended at 0.08 keeps its end marker on the plot: from 0.
    expect(unitFitDomain(values, [...ends, 0.08], 0.606, 360)).toEqual(unitDomain(0.606, 360));
    expect(unitFitDomain([], [], 0.606, 360)).toEqual(unitDomain(0.606, 360));
    expect(fitFloorNote(unitDomain(0.606, 360))).toBeNull();
  });
});

describe('prDomain', () => {
  const curve = (pts: [number, number][]) => pts.map(([x, y]) => ({ x, y }));
  it('zooms a near-perfect model to 0.8–1 and says so; any class dipping keeps the 0–1 frame', () => {
    // Round 26: model-o2o, three classes at AP50 0.995, a flat line on the top edge of a 300 px 0–1 plot.
    const good = [curve([[0, 1], [0.5, 0.998], [0.9, 0.99], [0.97, 0.93], [0.99, 0.6]]), curve([[0, 1], [0.94, 0.95], [0.98, 0.4]])];
    const z = prDomain(good)!;
    expect(z.domain).toEqual([PR_ZOOM.floor, 1]);
    expect(z.ticks).toEqual([0.8, 0.85, 0.9, 0.95, 1]);
    expect(z.decimals).toBe(2);
    expect(PR_ZOOM.plotPx).toBe(200);
    // One class under 0.9 before recall 0.95: the whole frame.
    expect(prDomain([...good, curve([[0, 1], [0.6, 0.88], [0.9, 0.5]])])).toBeNull();
    // No data: the whole frame.
    expect(prDomain([[], []])).toBeNull();
  });
});
