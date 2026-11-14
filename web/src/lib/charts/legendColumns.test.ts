import { describe, expect, it } from 'vitest';
import { legendColumns } from './ChartLegend';

describe('legendColumns', () => {
  it('wraps a legend into the fewest rows, its items spread over them evenly', () => {
    // Four loss terms where three fit: 2 + 2, never 3 + 1.
    expect(legendColumns(4, 3)).toBe(2);
    // Three where two fit: still two rows, never three.
    expect(legendColumns(3, 2)).toBe(2);
    expect(legendColumns(5, 4)).toBe(3);
    expect(legendColumns(6, 4)).toBe(3);
    expect(legendColumns(4, 4)).toBe(4);
    // Not even one fits: one item per row.
    expect(legendColumns(3, 0)).toBe(1);
  });
});
