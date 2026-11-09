import { describe, expect, it } from 'vitest';
import { bestMark, bestWord } from './bestMark';

describe('bestMark', () => {
  it('marks the one highest value "best"', () => {
    const m = bestMark([0.599, 0.604, null]);
    expect([...m.best]).toEqual([1]);
    expect(m.joint).toBe(false);
    expect(bestWord(m)).toBe('best');
    expect(m.decimals).toBe(3);
  });

  it('marks every value of an exact tie "joint best" (fasteners-n-best and fasteners-n-f16)', () => {
    const m = bestMark([0.6041, 0.6041, 0.5]);
    expect([...m.best].sort()).toEqual([0, 1]);
    expect(m.joint).toBe(true);
    expect(bestWord(m)).toBe('joint best');
  });

  it('shows more decimals where a different value rounds to the same display', () => {
    const m = bestMark([0.60413, 0.60409]);
    expect([...m.best]).toEqual([0]);
    expect(m.decimals).toBe(5);
    expect((0.60413).toFixed(m.decimals)).not.toBe((0.60409).toFixed(m.decimals));
  });

  it('needs two values to compare: one score alone is not a best', () => {
    expect(bestMark([0.6]).best.size).toBe(0);
    expect(bestMark([0.6, null, undefined]).best.size).toBe(0);
  });
});
