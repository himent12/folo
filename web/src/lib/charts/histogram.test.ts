import { describe, expect, it } from 'vitest';
import { binAt, binDensities } from './histogram';

describe('binDensities', () => {
  it('divides each count by its bin width', () => {
    const d = binDensities([0, 0.01, 0.03], [10, 10], Infinity, 0.01);
    expect(d[0]).toBeCloseTo(10);
    expect(d[1]).toBeCloseTo(5);
  });

  it('uses the clipped width for a tail bin past xMax, so area still equals count', () => {
    // [0.5, 1.0] drawn up to 0.6: 0.1 wide on screen, so 40 boxes read as 4 per 1 % step (not 0.8).
    const d = binDensities([0.4, 0.5, 1.0], [20, 40], 0.6, 0.01);
    expect(d[0]).toBeCloseTo(2);
    expect(d[1]).toBeCloseTo(4);
    const area = d[1]! * (0.6 - 0.5) / 0.01;
    expect(area).toBeCloseTo(40);
  });
});

describe('binAt', () => {
  const bars = [
    { x: 40, w: 10 },
    { x: 51, w: 30 },
    { x: 82, w: 4 },
  ];
  it('finds the bar under the pointer', () => {
    expect(binAt(bars, 45)).toBe(0);
    expect(binAt(bars, 60)).toBe(1);
    expect(binAt(bars, 84)).toBe(2);
  });
  it('gives a gap between bars to the nearer bar, and nothing far outside', () => {
    expect(binAt(bars, 50.4)).toBe(0);
    expect(binAt(bars, 50.8)).toBe(1);
    expect(binAt(bars, 87)).toBe(2);
    expect(binAt(bars, 10)).toBeNull();
    expect(binAt(bars, 200)).toBeNull();
  });
});
