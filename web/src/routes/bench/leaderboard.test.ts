import { describe, expect, it } from 'vitest';
import type { BenchEntry } from '$lib/api/types';
import { along, bandEntry, boardOrder, stale, compareHint, compareIds, comparePath, compareReason, intervalDomain, intervalScale, keepSelection, overlaps, ranks, scaleTicks, TICK_GAP, toggleSelection } from './leaderboard';

const e = (id: string, total: number | null, status: BenchEntry['status'] = total === null ? 'running' : 'done', partial = false, created = '2026-11-01T00:00:00Z') =>
  ({ id, status, created, score: total === null ? null : { total, partial, axes: [] } }) as unknown as BenchEntry;

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

  it('ranks complete entries before partial ones, and the band is the leading complete one', () => {
    // A partial agent scores highest but ranks after every complete entry; a tie as printed shares a rank (an older
    // entry first), a partial tie with a complete one does not.
    const board = [e('agent', 80, 'done', true, '2026-11-02T00:00:00Z'), e('m1', 60, 'done', false, '2026-11-03T00:00:00Z'), e('m2', 60.02, 'done', false, '2026-11-01T00:00:00Z'), e('ext', 60, 'done', true), e('run', null)];
    expect([...ranks(board).entries()]).toEqual([
      ['m2', 1],
      ['m1', 1],
      ['agent', 3],
      ['ext', 4],
    ]);
    expect(bandEntry(board)?.id).toBe('m2');
    // No complete entry: the leading partial one.
    expect(bandEntry([e('agent', 80, 'done', true), e('ext', 60, 'done', true)])?.id).toBe('agent');
    expect(bandEntry([e('run', null)])).toBeUndefined();
  });
});

describe('entries graded on an earlier version of the benchmark', () => {
  it('rank below the current ones, the server’s order kept within each group, out of the band', () => {
    const fp = (x: BenchEntry, f: string) => ({ ...x, fingerprint: f }) as BenchEntry;
    const board = [fp(e('old-hi', 90, 'done', false, '2026-10-01T00:00:00Z'), 'old'), fp(e('m1', 70), 'now'), fp(e('agent', 80, 'done', true), 'now'), fp(e('old-lo', 40), 'old'), fp(e('run', null), 'now')];
    expect(boardOrder(board, 'now').map((x) => x.id)).toEqual(['m1', 'agent', 'old-hi', 'old-lo', 'run']);
    const current = board.filter((x) => !stale(x, 'now'));
    expect(bandEntry(current)?.id).toBe('m1');
    expect([...ranks(current).keys()]).toEqual(['m1', 'agent']);
    expect(stale(board[0]!, 'now')).toBe(true);
    expect(stale(board[1]!, 'now')).toBe(false);
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

describe('the focused scale', () => {
  it('keeps the leaders readable: an entry far behind is off the scale, marked below it', () => {
    const rows = [
      { id: 'a', ci: [0.62, 0.66] as [number, number] },
      { id: 'b', ci: [0.58, 0.63] as [number, number] },
      { id: 'c', ci: [0.53, 0.57] as [number, number] },
      { id: 'd', ci: [0.14, 0.17] as [number, number] },
      { id: 'e', ci: null },
    ];
    const s = intervalScale(rows)!;
    expect([...s.below]).toEqual(['d']);
    expect(s.domain).toEqual([0.5, 0.7]);
    // Without the far one the scale is as before.
    expect(intervalScale(rows.slice(0, 2))!.domain).toEqual([0.55, 0.7]);
    expect(intervalScale([{ id: 'x', ci: null }])).toBeNull();
  });

  it('ticks round steps inside it, at most five before its width is known, evenly spaced', () => {
    expect(scaleTicks([0.55, 0.7])).toEqual([0.55, 0.6, 0.65, 0.7]);
    expect(scaleTicks([0.35, 0.7])).toEqual([0.4, 0.5, 0.6, 0.7]);
    expect(scaleTicks([0, 0.1])).toEqual([0, 0.05, 0.1]);
    expect(scaleTicks([0.9, 1])).toEqual([0.9, 0.95, 1]);
  });

  // The real leaderboard of COCO photos (folo-n-coco-long, folo-n-coco, Claude Code on 24 images): its scale was ticked
  // 0.25 0.35 0.45 0.50, the last two labels touching at the right end.
  const real = [
    { id: 'long', ci: [0.337, 0.402] as [number, number] },
    { id: 'coco', ci: [0.341, 0.404] as [number, number] },
    { id: 'agent', ci: [0.264, 0.445] as [number, number] },
  ];
  /** The smallest distance between two neighbouring labels' centres on a plot `width` px wide. */
  const closest = (domain: readonly [number, number], ticks: number[], width: number) =>
    Math.min(...ticks.slice(1).map((t, i) => ((t - ticks[i]!) / (domain[1] - domain[0])) * width));

  it('never sets two labels closer than a readable gap, at any width: the real case', () => {
    const { domain, below } = intervalScale(real)!;
    expect(below.size).toBe(0);
    expect(domain).toEqual([0.25, 0.5]);
    // The plot is 148 px wide at 1440, 108 px on a laptop or tablet; and anything from a cramped 40 px up.
    expect(scaleTicks(domain, 148)).toEqual([0.3, 0.4, 0.5]);
    expect(scaleTicks(domain, 108)).toEqual([0.3, 0.4, 0.5]);
    for (let w = 40; w <= 600; w += 4) {
      const t = scaleTicks(domain, w);
      expect(t.length, `${w} px`).toBeGreaterThanOrEqual(1);
      if (t.length > 1) expect(closest(domain, t, w), `${w} px`).toBeGreaterThanOrEqual(TICK_GAP - 1e-9);
      // Evenly spaced: one step between every two.
      const steps = new Set(t.slice(1).map((x, i) => (x - t[i]!).toFixed(2)));
      expect(steps.size, `${w} px`).toBeLessThanOrEqual(1);
      for (const x of t) expect(x >= domain[0] && x <= domain[1]).toBe(true);
    }
    // Wide enough, the finest step: every 0.05.
    expect(scaleTicks(domain, 400)).toEqual([0.25, 0.3, 0.35, 0.4, 0.45, 0.5]);
    // Too narrow for two labels: its lower end alone.
    expect(scaleTicks(domain, 30)).toEqual([0.25]);
  });
});
