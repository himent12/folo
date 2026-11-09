import { describe, expect, it } from 'vitest';
import { BEST_R, endMarkInk, endRingInk, MERGE_PX, mergedEnds } from './endMarks';

describe('end marks', () => {
  it('merges a best ring with the nearest end mark that has at most 16 px of clear space to it, and only then', () => {
    const ends = [
      { id: 'a', end: 'live' as const, x: 100, y: 50, r: 6.75 },
      { id: 'b', end: undefined, x: 300, y: 50, r: 4.25 },
    ];
    expect(MERGE_PX).toBe(16);
    // Round 24's Overview card: best and live 18 px apart, centre to centre, read as a lime "oo".
    expect(mergedEnds([{ x: 82, y: 50 }], ends)).toEqual(['a']);
    const edge = BEST_R + 6.75 + 16;
    expect(mergedEnds([{ x: 100 - edge, y: 50 }], ends)).toEqual(['a']);
    expect(mergedEnds([{ x: 100 - edge - 1, y: 50 }], ends)).toEqual([null]);
    expect(mergedEnds([{ x: 200, y: 50 }], ends)).toEqual([null]);
    expect(mergedEnds([{ x: 280, y: 50 }, { x: 10, y: 10 }], ends)).toEqual(['b', null]);
  });

  it('never puts lime on a failed or stopped end, and draws the failure in the danger ink', () => {
    expect(endRingInk('live', 'lime')).toBe('lime');
    expect(endRingInk(undefined, 'lime')).toBe('lime');
    expect(endRingInk('failed', 'lime')).toBe('var(--text-primary)');
    expect(endRingInk('stopped', 'lime')).toBe('var(--text-primary)');
    expect(endMarkInk('failed', 'red')).toBe('var(--danger-text)');
    expect(endMarkInk('stopped', 'red')).toBe('red');
  });
});
