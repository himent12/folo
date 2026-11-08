import { describe, expect, it } from 'vitest';
import { binDensities } from './histogram';

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
