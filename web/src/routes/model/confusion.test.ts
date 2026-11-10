import { describe, expect, it } from 'vitest';
import { cmIndex, rankConfusions } from './confusion';

describe('cmIndex', () => {
  it('is the class itself in a full matrix, its place in a reduced one, and null for a class folded into Other', () => {
    expect(cmIndex(0, undefined)).toBe(0);
    expect(cmIndex(7, undefined)).toBe(7);
    const ids = [0, 2, 5, 9, -1];
    expect(cmIndex(5, ids)).toBe(2);
    expect(cmIndex(9, ids)).toBe(3);
    // Never the Other entry itself.
    expect(cmIndex(3, [0, 2, 5, -1])).toBeNull();
    expect(cmIndex(-1, [0, 2, 5, -1])).toBeNull();
  });
});

describe('rankConfusions', () => {
  // bolt, nut, spring + background. spring misses 30 %, is taken for nut 10 %; bolt for nut 5 %.
  const conf = [
    [90, 5, 0, 5],
    [0, 40, 0, 0],
    [0, 10, 60, 30],
    [4, 3, 2, 0],
  ];
  it('ranks confusions and misses by share of the true class, with counts', () => {
    expect(rankConfusions(conf)).toEqual([
      { kind: 'missed', r: 2, c: 3, count: 30, total: 100, v: 0.3 },
      { kind: 'confused', r: 2, c: 1, count: 10, total: 100, v: 0.1 },
      { kind: 'confused', r: 0, c: 1, count: 5, total: 100, v: 0.05 },
      { kind: 'missed', r: 0, c: 3, count: 5, total: 100, v: 0.05 },
    ]);
  });
  it('skips rare classes and the background row, and keeps to the limit', () => {
    expect(rankConfusions(conf, 2).map((e) => [e.r, e.c])).toEqual([
      [2, 3],
      [2, 1],
    ]);
    expect(rankConfusions(conf, 8, 101)).toEqual([]);
    expect(rankConfusions([[0]])).toEqual([]);
  });
});
