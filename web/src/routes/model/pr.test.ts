import { describe, expect, it } from 'vitest';
import { matchesClass, trimPrTail } from './pr';

const pts = (ys: number[]) => ys.map((y, i) => ({ x: i / Math.max(1, ys.length - 1), y }));

describe('trimPrTail', () => {
  it('ends the curve at its last point with precision above zero, keeping zeros inside it', () => {
    expect(trimPrTail(pts([1, 0.8, 0.4, 0, 0, 0])).map((p) => p.y)).toEqual([1, 0.8, 0.4]);
    expect(trimPrTail(pts([1, 0, 0.5])).length).toBe(3);
    expect(trimPrTail(pts([1, 0.9])).length).toBe(2);
    // Zero throughout: an empty curve.
    expect(trimPrTail(pts([0, 0]))).toEqual([]);
    expect(trimPrTail([])).toEqual([]);
  });
});

describe('matchesClass', () => {
  it('matches substrings case-insensitively and everything for an empty query', () => {
    expect(matchesClass('Traffic light', 'LIGHT')).toBe(true);
    expect(matchesClass('person', ' ')).toBe(true);
    expect(matchesClass('person', 'car')).toBe(false);
  });
});

describe('prEnvelope', () => {
  it('takes per-recall quantiles of interpolated precision', async () => {
    const { prEnvelope, precisionAt } = await import('./pr');
    const flat = (p: number, end: number) => [
      { x: 0, y: p },
      { x: end, y: p },
    ];
    expect(precisionAt(flat(0.8, 0.5), 0.6)).toBe(0);
    expect(precisionAt(flat(0.8, 0.5), 0.2)).toBe(0.8);
    const curves = Array.from({ length: 11 }, (_, i) => flat(i / 10, 1));
    const env = prEnvelope(curves, 0.1, 0.9, 3);
    expect(env.lower.map((p) => p.y)).toEqual([0.1, 0.1, 0.1]);
    expect(env.upper.map((p) => p.y)).toEqual([0.9, 0.9, 0.9]);
  });
});
