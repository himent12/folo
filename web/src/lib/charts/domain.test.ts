import { describe, expect, it } from 'vitest';
import { capTicks, holdDomain, logDomain, logTicks, MAX_TICKS, niceDomain, stepDecimals, unitDomain } from './domain';

describe('unitDomain', () => {
  it('fits a 0–1 metric to the tightest nice top above its maximum, with a little headroom', () => {
    // A failed run's mAP50-95 (0.494), a comparison topping out at 0.606, a completed run's mAP50 (0.86).
    expect(unitDomain(0.494, 290).domain).toEqual([0, 0.6]);
    expect(unitDomain(0.606, 290).domain).toEqual([0, 0.75]);
    expect(unitDomain(0.86, 290).domain).toEqual([0, 1]);
    // Headroom: a value just under a step does not touch the top.
    expect(unitDomain(0.39, 290).domain).toEqual([0, 0.5]);
  });

  it('takes 3–5 ticks in nice steps, ending on its top, at any plot height', () => {
    expect(unitDomain(0.494, 290).ticks).toEqual([0, 0.2, 0.4, 0.6]);
    expect(unitDomain(0.606, 290).ticks).toEqual([0, 0.25, 0.5, 0.75]);
    expect(unitDomain(0.86, 290).ticks).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(unitDomain(0.33, 290).ticks).toEqual([0, 0.1, 0.2, 0.3, 0.4]);
    // A low score keeps a fitted axis instead of a flat line at the bottom of 0–1.
    expect(unitDomain(0.08, 290).ticks).toEqual([0, 0.05, 0.1]);
    for (const max of [0.01, 0.12, 0.27, 0.45, 0.5, 0.63, 0.71, 0.88, 0.97, 1])
      for (const px of [138, 290]) {
        const t = unitDomain(max, px).ticks;
        expect(t.length, `${max} @ ${px}px`).toBeGreaterThanOrEqual(3);
        expect(t.length, `${max} @ ${px}px`).toBeLessThanOrEqual(MAX_TICKS);
        expect(t[t.length - 1]).toBeGreaterThanOrEqual(Math.min(1, max));
        expect((px / (t.length - 1)) >= 32, `${max} @ ${px}px gap`).toBe(true);
      }
  });

  it('never goes past 1, and falls back to 0–1 without data', () => {
    expect(unitDomain(0.99, 290).domain).toEqual([0, 1]);
    expect(unitDomain(1, 290).ticks).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(unitDomain(NaN, 290).domain).toEqual([0, 1]);
    expect(unitDomain(0, 290).domain).toEqual([0, 1]);
  });

  it('is stable: a later, lower value never shrinks it, a higher one moves it by whole steps', () => {
    const seen = [0.31, 0.42, 0.48, 0.47, 0.5, 0.55, 0.57].map((v, i, a) => unitDomain(Math.max(...a.slice(0, i + 1)), 290).domain[1]);
    expect(seen).toEqual([0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]);
  });
});

describe('niceDomain', () => {
  it('widens the extent to whole tick steps, in at most five ticks', () => {
    expect(niceDomain(0.42, 3.1)).toEqual({ domain: [0, 4], ticks: [0, 1, 2, 3, 4] });
    expect(niceDomain(0, 318 * 1.08)).toEqual({ domain: [0, 400], ticks: [0, 100, 200, 300, 400] });
    expect(niceDomain(1.9, 2.6).ticks).toEqual([1.8, 2, 2.2, 2.4, 2.6]);
  });

  it('keeps small values readable (learning rates)', () => {
    const f = niceDomain(0, 0.0098);
    expect(f.domain).toEqual([0, 0.01]);
    expect(f.ticks).toEqual([0, 0.0025, 0.005, 0.0075, 0.01]);
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
    // The loss chart's 0.2–10: six 1-2-5 values would be six gridlines; both ends and the decade between stay.
    expect(logTicks(0.2, 10)).toEqual([0.2, 1, 10]);
    expect(logTicks(0.1, 100)).toEqual([0.1, 1, 10, 100]);
    expect(logTicks(0.001, 1000)).toEqual([0.001, 0.1, 10, 1000]);
    // A domain that is not on a 1-2-5 value is labelled at the outermost ones inside it.
    expect(logTicks(0.15, 12)).toEqual([0.2, 1, 10]);
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
