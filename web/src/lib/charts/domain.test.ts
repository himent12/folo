import { describe, expect, it } from 'vitest';
import { holdDomain, logDomain, niceDomain, unitDomain } from './domain';

describe('unitDomain', () => {
  it('fits a 0–1 metric to the next 0.1 above its maximum, with a little headroom', () => {
    // A failed run's mAP50-95 (0.494), a comparison topping out at 0.606, a completed run's mAP50 (0.86).
    expect(unitDomain(0.494, 290).domain).toEqual([0, 0.6]);
    expect(unitDomain(0.606, 290).domain).toEqual([0, 0.7]);
    expect(unitDomain(0.86, 290).domain).toEqual([0, 0.9]);
    // Headroom: a value just under a step does not touch the top.
    expect(unitDomain(0.59, 290).domain).toEqual([0, 0.7]);
  });

  it('ends on a tick: 0.1 steps when the plot has room, 0.2 when it is short', () => {
    expect(unitDomain(0.606, 290).ticks).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
    // 140 px holds four gridline gaps: 0.7 rounds up to 0.8 in 0.2 steps.
    expect(unitDomain(0.606, 140)).toEqual({ domain: [0, 0.8], ticks: [0, 0.2, 0.4, 0.6, 0.8] });
  });

  it('never goes past 1, and falls back to 0–1 without data', () => {
    expect(unitDomain(0.99, 290).domain).toEqual([0, 1]);
    expect(unitDomain(1, 290).ticks).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(unitDomain(NaN, 290).domain).toEqual([0, 1]);
    expect(unitDomain(0, 290).domain).toEqual([0, 1]);
  });

  it('is stable: a later, lower value never shrinks it, a higher one moves it by whole steps', () => {
    const seen = [0.31, 0.42, 0.48, 0.47, 0.5, 0.55, 0.57].map((v, i, a) => unitDomain(Math.max(...a.slice(0, i + 1)), 290).domain[1]);
    expect(seen).toEqual([0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]);
  });
});

describe('niceDomain', () => {
  it('widens the extent to whole tick steps', () => {
    expect(niceDomain(0.42, 3.1)).toEqual({ domain: [0, 3.5], ticks: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5] });
    expect(niceDomain(0, 318 * 1.08).domain).toEqual([0, 400]);
  });

  it('keeps small values readable (learning rates)', () => {
    const f = niceDomain(0, 0.0098);
    expect(f.domain).toEqual([0, 0.01]);
    expect(f.ticks[f.ticks.length - 1]).toBe(0.01);
  });

  it('handles a flat or empty series', () => {
    expect(niceDomain(2, 2).domain[0]).toBeLessThanOrEqual(2);
    expect(niceDomain(2, 2).domain[1]).toBeGreaterThan(2);
    expect(niceDomain(Infinity, -Infinity).domain).toEqual([0, 1]);
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
