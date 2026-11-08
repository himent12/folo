import { describe, expect, it } from 'vitest';
import { downsample, nearest, smooth, ticks } from './downsample';

const pts = (ys: number[]) => ys.map((y, x) => ({ x, y }));

describe('chart helpers', () => {
  it('keeps small inputs as they are', () => {
    const p = pts([1, 2, 3]);
    expect(downsample(p, 10)).toBe(p);
  });

  it('keeps spikes and the last point when downsampling', () => {
    const ys = Array.from({ length: 1000 }, (_, i) => (i === 517 ? 50 : Math.sin(i / 20)));
    const out = downsample(pts(ys), 50);
    expect(out.length).toBeLessThanOrEqual(101);
    expect(out.some((p) => p.y === 50)).toBe(true);
    expect(out.at(-1)!.x).toBe(999);
    expect(out.map((p) => p.x)).toEqual([...out.map((p) => p.x)].sort((a, b) => a - b));
  });

  it('smooths with bias correction (first value unchanged)', () => {
    const out = smooth(pts([10, 0, 0, 0]), 0.6);
    expect(out[0]!.y).toBeCloseTo(10);
    expect(out[3]!.y).toBeLessThan(out[1]!.y);
  });

  it('finds the nearest x', () => {
    const p = pts([0, 0, 0, 0, 0]);
    expect(nearest(p, 2.4)).toBe(2);
    expect(nearest(p, 2.6)).toBe(3);
    expect(nearest(p, -5)).toBe(0);
    expect(nearest(p, 99)).toBe(4);
    expect(nearest([], 1)).toBe(-1);
  });

  it('makes nice ticks', () => {
    expect(ticks(0, 1, 4)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(ticks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
  });
});
