import { describe, expect, it } from 'vitest';
import { compareHint, compareReason } from './compareHint';

describe('compareHint', () => {
  it('says what Compare needs at every count, in words', () => {
    expect(compareHint(0)).toBe('Select 2–4 runs to compare');
    expect(compareHint(1)).toBe('Select one more run to compare · 2–4 runs');
    expect(compareHint(2)).toBe('2 runs selected · up to 2 more');
    expect(compareHint(3)).toBe('3 runs selected · up to 1 more');
    expect(compareHint(4)).toBe('4 runs selected · the most Compare takes');
  });
  it('gives the disabled button the visible line as its reason, and none once it can run', () => {
    expect(compareReason(1)).toBe(compareHint(1));
    expect(compareReason(2)).toBeNull();
  });
});
