import { describe, expect, it } from 'vitest';
import { matchesClass, trimPrTail } from './pr';

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
