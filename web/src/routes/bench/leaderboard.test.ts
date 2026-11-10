import { describe, expect, it } from 'vitest';
import type { BenchEntry } from '$lib/api/types';
import { along, compareHint, compareIds, comparePath, compareReason, intervalDomain, keepSelection, overlaps, ranks, toggleSelection } from './leaderboard';

const e = (id: string, total: number | null, status: BenchEntry['status'] = total === null ? 'running' : 'done') => ({ id, status, score: total === null ? null : { total, partial: false, axes: [] } }) as unknown as BenchEntry;

describe('ranks', () => {
  it('ranks finished entries by score, ties sharing a rank (1, 2, 2, 4), and leaves the others unranked', () => {
    const r = ranks([e('a', 70), e('b', 60), e('c', 60), e('d', 50), e('x', null), e('f', null, 'failed')]);
    expect([...r.entries()]).toEqual([
      ['a', 1],
      ['b', 2],
      ['c', 2],
      ['d', 4],
    ]);
  });
});

describe('the Compare selection', () => {
  const board = [e('a', 70), e('b', 60), e('c', 55), e('d', 50), e('g', 40), e('x', null)];

  it('takes finished entries only, at most four, and toggles one off again', () => {
    let s: string[] = [];
    for (const id of ['a', 'x', 'b', 'c', 'd', 'g']) s = toggleSelection(s, id, board);
    expect(s).toEqual(['a', 'b', 'c', 'd']);
    expect(toggleSelection(s, 'b', board)).toEqual(['a', 'c', 'd']);
  });

  it('keeps what still holds once the board changes, and names the selection in board order', () => {
    expect(keepSelection(['a', 'gone', 'x'], board)).toEqual(['a']);
    expect(comparePath('bn_1', ['c', 'a'], board)).toBe('/benchmarks/bn_1/compare?entries=a,c');
    expect(compareIds(new URLSearchParams('entries=a,%20c,,d'))).toEqual(['a', 'c', 'd']);
  });

  it('says what Compare needs, and is enabled from two', () => {
    expect(compareHint(0)).toBe('Select 2–4 finished entries to compare');
    expect(compareHint(1)).toBe('Select one more entry to compare · 2–4 entries');
    expect(compareHint(3)).toBe('3 selected · up to 1 more');
    expect(compareHint(4)).toBe('4 selected (the maximum)');
    expect(compareReason(1)).toBe(compareHint(1));
    expect(compareReason(2)).toBeNull();
  });
});

describe('the intervals scale', () => {
  it('spans every interval with room, on a 0.05 grid, at least 0.1 wide, inside 0–1', () => {
    expect(intervalDomain([])).toBeNull();
    expect(intervalDomain([[0.56, 0.62], [0.6, 0.67]])).toEqual([0.55, 0.7]);
    expect(intervalDomain([[0.601, 0.62]])).toEqual([0.55, 0.65]);
    expect(intervalDomain([[0.0, 0.03]])).toEqual([0, 0.1]);
    expect(intervalDomain([[0.97, 0.995]])).toEqual([0.9, 1]);
  });

  it('places a value along it, clamped, and tells overlapping intervals', () => {
    expect(along([0.5, 0.7], 0.6)).toBeCloseTo(0.5, 9);
    expect(along([0.5, 0.7], 0.9)).toBe(1);
    expect(along([0.5, 0.7], 0.1)).toBe(0);
    expect(overlaps([0.56, 0.62], [0.6, 0.67])).toBe(true);
    expect(overlaps([0.3, 0.48], [0.6, 0.67])).toBe(false);
  });
});
