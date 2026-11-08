import { describe, expect, it } from 'vitest';
import { fitCount } from './fit';

const more = () => 20;

describe('fitCount', () => {
  it('shows every item when they all fit', () => {
    expect(fitCount([30, 30, 30], 10, more, 110)).toBe(3);
  });
  it('cuts by whole items and leaves room for the "+n"', () => {
    // 30 + 10 + 30 + 20 (more) = 90 fits; adding the third item would need 130.
    expect(fitCount([30, 30, 30, 30], 10, more, 100)).toBe(2);
  });
  it('drops the "+n" reservation for the last item', () => {
    expect(fitCount([30, 30], 10, more, 70)).toBe(2);
  });
  it('always keeps one item', () => {
    expect(fitCount([200, 30], 10, more, 50)).toBe(1);
    expect(fitCount([], 10, more, 50)).toBe(0);
  });
});
