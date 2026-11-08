import { describe, expect, it } from 'vitest';
import { classMetrics, matchesClass, trimPrTail } from './pr';

const pts = (ys: number[]) => ys.map((y, i) => ({ x: i / Math.max(1, ys.length - 1), y }));

describe('trimPrTail', () => {
  it('ends the curve at its last point with precision above zero', () => {
    expect(trimPrTail(pts([1, 0.8, 0.4, 0, 0, 0])).map((p) => p.y)).toEqual([1, 0.8, 0.4]);
  });
  it('keeps zeros inside the curve and full curves as they are', () => {
    expect(trimPrTail(pts([1, 0, 0.5])).length).toBe(3);
    expect(trimPrTail(pts([1, 0.9])).length).toBe(2);
  });
  it('returns an empty curve when precision is zero throughout', () => {
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

describe('classMetrics', () => {
  it('reveals only precision and recall, which the narrow row has no column for, at 3 decimals', () => {
    const c = { class: 3, ap50: 0.8971, ap50_95: 0.678, precision: 0.9174, recall: 0.8449, instances: 2793 };
    expect(classMetrics(c)).toEqual([
      { label: 'Precision', value: '0.917' },
      { label: 'Recall', value: '0.845' },
    ]);
  });
});
