import { describe, expect, it } from 'vitest';
import { binAt, binDensities, proportionalBars } from './histogram';

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

describe('proportionalBars', () => {
  it('keeps true widths on one linear axis when every bin fits', () => {
    const { bars, mainEnd, tail } = proportionalBars([0, 0.1, 0.3], 0.3, 0, 300, 12);
    expect(tail).toBe(false);
    expect(mainEnd).toBe(300);
    expect(bars).toEqual([
      { x: 0, w: 100 },
      { x: 100, w: 200 },
    ]);
  });

  it('sets an open-ended last bin apart after a gap, at the same scale (area still equals count)', () => {
    // [0.5, 1] drawn up to 0.6: the axis 0..0.6 over 312 − 12 px = 500 px per unit.
    const { bars, mainEnd, tail } = proportionalBars([0, 0.2, 0.5, 1], 0.6, 0, 312, 12);
    expect(tail).toBe(true);
    expect(mainEnd).toBeCloseTo(250);
    expect(bars[1]!.x + bars[1]!.w).toBeCloseTo(250);
    expect(bars[2]!.x).toBeCloseTo(262);
    expect(bars[2]!.w).toBeCloseTo(50);
    expect(bars[2]!.x + bars[2]!.w).toBeCloseTo(312);
  });
});
