import { describe, expect, it } from 'vitest';
import { balancedColumns, fitColumns } from './balancedGrid';

describe('balancedColumns', () => {
  it('never leaves one card alone on the last row', () => {
    for (let n = 1; n <= 12; n++)
      for (let max = 1; max <= 5; max++) {
        const c = balancedColumns(n, max);
        expect(c).toBeLessThanOrEqual(Math.max(1, max));
        if (c > 1) expect(n % c).not.toBe(1);
      }
  });
  it('lays four out 2 × 2 where three fit, and keeps a full row where it can', () => {
    expect(balancedColumns(4, 3)).toBe(2);
    expect(balancedColumns(4, 4)).toBe(4);
    expect(balancedColumns(5, 3)).toBe(3);
    expect(balancedColumns(6, 3)).toBe(3);
    expect(balancedColumns(3, 2)).toBe(1);
    expect(balancedColumns(2, 5)).toBe(2);
  });
});

it('fitColumns counts the columns of at least `min` that fit with their gaps', () => {
  expect(fitColumns(1144, 280, 16)).toBe(3);
  expect(fitColumns(1200, 280, 16)).toBe(4);
  expect(fitColumns(200, 280, 16)).toBe(1);
});
