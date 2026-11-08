import { describe, expect, it } from 'vitest';
import { cmIndex } from './confusion';

describe('cmIndex', () => {
  it('is the class itself in a full matrix', () => {
    expect(cmIndex(0, undefined)).toBe(0);
    expect(cmIndex(7, undefined)).toBe(7);
  });
  it('finds a kept class in a reduced matrix', () => {
    const ids = [0, 2, 5, 9, -1];
    expect(cmIndex(5, ids)).toBe(2);
    expect(cmIndex(9, ids)).toBe(3);
  });
  it('is null for a class folded into Other (and never matches the Other entry)', () => {
    expect(cmIndex(3, [0, 2, 5, -1])).toBeNull();
    expect(cmIndex(-1, [0, 2, 5, -1])).toBeNull();
  });
});
