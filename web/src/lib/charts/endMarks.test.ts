import { describe, expect, it } from 'vitest';
import { BEST_R, endMarkInk, LIVE_INK, MERGE_PX, mergedEnds } from './endMarks';

describe('end marks', () => {
  it('draws a best ring around an end mark only where the two would touch; apart, each keeps its own shape', () => {
    const ends = [
      { id: 'a', end: 'live' as const, x: 100, y: 50, r: 4.25 },
      { id: 'b', end: undefined, x: 300, y: 50, r: 3.75 },
    ];
    expect(MERGE_PX).toBe(2);
    // Round 25: best at epoch 36, live at 37, 18 px apart, drawn as two lime rings. They are a ring and a dot now.
    expect(mergedEnds([{ x: 82, y: 50 }], ends)).toEqual([null]);
    // The best is the latest: the ring goes around the dot.
    expect(mergedEnds([{ x: 100, y: 50 }], ends)).toEqual(['a']);
    // A phone's plot puts epochs 36 and 37 3 px apart: the marks would overlap, so they are drawn as one.
    expect(mergedEnds([{ x: 97, y: 50 }], ends)).toEqual(['a']);
    const edge = BEST_R + 4.25 + MERGE_PX;
    expect(mergedEnds([{ x: 100 - edge + 0.1, y: 50 }], ends)).toEqual(['a']);
    expect(mergedEnds([{ x: 100 - edge, y: 50 }], ends)).toEqual([null]);
    expect(mergedEnds([{ x: 299, y: 50 }, { x: 10, y: 10 }], ends)).toEqual(['b', null]);
  });

  it('draws the live end lime, a failure in the danger ink, and any other end in its line', () => {
    expect(endMarkInk('live', 'ink')).toBe(LIVE_INK);
    expect(endMarkInk(undefined, 'ink')).toBe('ink');
    expect(endMarkInk('failed', 'red')).toBe('var(--danger-text)');
    expect(endMarkInk('stopped', 'red')).toBe('red');
  });
});
