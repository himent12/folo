import { describe, expect, it } from 'vitest';
import { oomScale } from './estimate';

const GiB = 2 ** 30;

describe('oomScale', () => {
  it('leaves the estimate alone when it already said the failed settings would not fit', () => {
    expect(oomScale(27 * GiB, 16 * GiB)).toBe(1);
  });
  it('scales a low estimate until the failed settings exceed the memory by 5%', () => {
    const k = oomScale(8 * GiB, 16 * GiB);
    expect(k).toBeCloseTo(2.1);
    expect(8 * GiB * k).toBeGreaterThan(16 * GiB);
  });
  it('does nothing without an estimate', () => {
    expect(oomScale(0, 16 * GiB)).toBe(1);
  });
});
