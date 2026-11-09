import { describe, expect, it } from 'vitest';
import { CLEAR, LABEL_GAP, LABEL_H, labelWidth, leaderOf, MAX_REACH, memoPlaceEndLabels, needsLeader, placeEndLabels, type Placed } from './endLabels';

const bounds = { x0: 0, y0: 0, x1: 600, y1: 300 };
const line = (pts: [number, number][]) => pts.map(([x, y]) => ({ x, y }));

describe('placeEndLabels', () => {
  it('puts a label right of its line when that spot is clear', () => {
    const c = line([[0, 200], [300, 100]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 100 }], [c], bounds);
    expect(p).toEqual({ id: 'a', label: 'run-a', x: 306, y: 100, ax: 300, ay: 100, w: labelWidth('run-a') });
    // Right beside its end marker: no leader.
    expect(needsLeader(p!)).toBe(false);
  });

  it("ties a label moved off its line's end to the end marker with a leader, from the marker to the label's edge", () => {
    // b runs flat through a's end, and c, d just above and below it to its right, so a's label steps further out,
    // up and to the left, where its leader crosses no other line.
    const a = line([[0, 200], [300, 100]]);
    const b = line([[0, 100], [600, 100]]);
    const c = line([[300, 85], [600, 85]]);
    const d = line([[300, 115], [600, 115]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'shapes-hard-m · failed', x: 300, y: 100 }], [a, b, c, d], bounds);
    expect(p).toBeDefined();
    const l = leaderOf(p!, 5)!;
    expect(l).not.toBeNull();
    // It starts just outside the marker and ends on the label's box edge.
    expect(Math.hypot(l.x1 - 300, l.y1 - 100)).toBeCloseTo(5, 5);
    expect(Math.abs(l.y2 - p!.y)).toBeLessThanOrEqual(LABEL_H / 2 + 2);
  });

  it('moves off another line that runs through the right-hand spot', () => {
    const a = line([[0, 200], [300, 100]]);
    // b continues flat through y = 100, right where a's label would go.
    const b = line([[0, 220], [300, 100], [600, 100]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 100 }], [a, b], bounds);
    expect(p).toBeDefined();
    const box = { x0: p!.x, x1: p!.x + labelWidth('run-a'), y0: p!.y - 6, y1: p!.y + 6 };
    expect(box.y1 <= 98 || box.y0 >= 102).toBe(true);
  });

  it('never overlaps two labels, and leaves one out when no spot is free', () => {
    const ends = [
      { id: 'a', label: 'shapes-hard-s', x: 590, y: 150 },
      { id: 'b', label: 'shapes-hard-m', x: 590, y: 151 },
    ];
    // Wall-to-wall curves left of the points leave only the right / above / below spots, and the right edge is close.
    const placed = placeEndLabels(ends, [], { x0: 0, y0: 0, x1: 600, y1: 300 });
    for (let i = 0; i < placed.length; i++)
      for (let j = i + 1; j < placed.length; j++) {
        const A = placed[i]!;
        const B = placed[j]!;
        const sep = Math.abs(A.y - B.y) >= 14 || A.x + labelWidth(A.label) <= B.x || B.x + labelWidth(B.label) <= A.x;
        expect(sep).toBe(true);
      }
    expect(placed.length).toBe(2);
    // A third label crowded into a box with no room is dropped rather than drawn on top.
    const tight = placeEndLabels([...ends, { id: 'c', label: 'x', x: 590, y: 150 }], [], { x0: 560, y0: 130, x1: 600, y1: 170 });
    expect(tight.length).toBeLessThan(3);
  });

  it('labels every line of a four-run comparison, a line with no free spot within reach included', () => {
    // Horizontal lines every 10 px across the plot (x 0..500) leave no free spot near the point; the clear gutter is
    // further than MAX_REACH from it, but a label is never left out while a clear box exists: it goes there on a leader.
    const hatch = Array.from({ length: 31 }, (_, i) => line([[0, i * 10], [500, i * 10]]));
    const [far] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 150 }], hatch, bounds);
    expect(far).toBeDefined();
    expect(far!.x).toBeGreaterThanOrEqual(500);
    expect(needsLeader(far!)).toBe(true);
    // Near the gutter, it goes there, with a leader.
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 600 - labelWidth('run-a') - MAX_REACH + 10, y: 150 }], hatch, bounds);
    expect(p).toBeDefined();
    expect(p!.x).toBeGreaterThanOrEqual(500);
    expect(p!.x + labelWidth('run-a')).toBeLessThanOrEqual(600);
    // Four runs ending close together all get a label, and no two overlap.
    const ends = ['fasteners-s', 'fasteners-n', 'shapes-hard-s', 'shapes-hard-m'].map((label, i) => ({ id: label, label, x: 200 + i * 4, y: 120 + i * 3 }));
    const placed = placeEndLabels(ends, [line([[0, 280], [200, 120]]), line([[0, 280], [600, 110]])], bounds);
    expect(placed.map((q) => q.id).sort()).toEqual(ends.map((e) => e.id).sort());
    for (let i = 0; i < placed.length; i++)
      for (let j = i + 1; j < placed.length; j++) {
        const A = placed[i]!;
        const B = placed[j]!;
        expect(Math.abs(A.y - B.y) >= 14 || A.x + labelWidth(A.label) <= B.x || B.x + labelWidth(B.label) <= A.x).toBe(true);
      }
  });

  it('prefers a spot over its own line to the far gutter', () => {
    // Its own line runs right through every near spot; another line blocks nothing near it.
    const own = Array.from({ length: 31 }, (_, i) => line([[0, i * 10], [500, i * 10]])).flat();
    const zigzag = own.map((p, i) => ({ x: (i % 2) * 500, y: p.y }));
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 150, curve: 0 }], [zigzag], bounds);
    expect(p!.x).toBeLessThan(500);
    expect(Math.abs(p!.y - 150)).toBeLessThanOrEqual(20);
  });
});

describe('placeEndLabels: the gutter column', () => {
  it('stacks labels of lines that end together next to each other, in order, near their ends', () => {
    // The loss chart near its end: total at 100, box and dfl meeting at 170, cls at 200; the plot ends at x = 500.
    const flat = (y: number) => line([[0, y], [500, y]]);
    const ends = [
      { id: 'total', label: 'total', x: 500, y: 100, curve: 0 },
      { id: 'box', label: 'box', x: 500, y: 170, curve: 1 },
      { id: 'cls', label: 'cls', x: 500, y: 200, curve: 2 },
      { id: 'dfl', label: 'dfl', x: 500, y: 171, curve: 3 },
    ];
    // Noisy curves fill the plot left of their ends, so only the gutter is free.
    const hatch = Array.from({ length: 31 }, (_, i) => flat(i * 10));
    const placed = placeEndLabels(ends, [flat(100), flat(170), flat(200), flat(171), ...hatch], bounds, 2, 500);
    const at = Object.fromEntries(placed.map((p) => [p.id, p]));
    expect(Object.keys(at).sort()).toEqual(['box', 'cls', 'dfl', 'total']);
    // Each within two label heights of its own end (none sent past another line's label), none overlapping.
    for (const e of ends) expect(Math.abs(at[e.id]!.y - e.y)).toBeLessThanOrEqual(28);
    for (const a of placed)
      for (const b of placed)
        if (a !== b) expect(Math.abs(a.y - b.y) >= 14 || a.x + labelWidth(a.label) <= b.x || b.x + labelWidth(b.label) <= a.x).toBe(true);
    // One ordered column: without it, dfl would take the nearest free height in the gutter, above total.
    for (const e of ends) expect(at[e.id]!.x).toBe(506);
    expect(at.total!.y).toBeLessThan(at.box!.y);
    expect(at.box!.y).toBeLessThan(at.dfl!.y);
    expect(at.dfl!.y).toBeLessThan(at.cls!.y);
  });

  it('keeps a crowded column inside the plot', () => {
    const ends = ['a', 'b', 'c', 'd'].map((id, i) => ({ id, label: id, x: 500, y: 296 + i * 0.5 }));
    const placed = placeEndLabels(ends, ends.map((e) => line([[0, e.y], [500, e.y]])), bounds, 2, 500);
    expect(placed).toHaveLength(4);
    for (const p of placed) expect(p.y + 7).toBeLessThanOrEqual(300);
  });

  it('keeps labels a clear gap apart, with leaders for those moved off their line', () => {
    // The loss chart's box and dfl ending 1 px apart at the plot's edge, and the comparison's two runs ending at 100 %.
    const flat = (y: number) => line([[0, y], [500, y]]);
    const ends = [
      { id: 'box', label: 'box', x: 500, y: 170, curve: 0 },
      { id: 'dfl', label: 'dfl', x: 500, y: 171, curve: 1 },
      { id: 'cls', label: 'cls', x: 500, y: 210, curve: 2 },
    ];
    const placed = placeEndLabels(ends, [flat(170), flat(171), flat(210)], bounds, 2, 500, { gap: 10, lead: 14 });
    expect(placed).toHaveLength(3);
    const ys = placed.map((p) => p.y).sort((a, b) => a - b);
    // A gap asked below the rule's 12 px is raised to it: one 16 px label box plus 12 px between centres.
    for (let i = 1; i < ys.length; i++) expect(ys[i]! - ys[i - 1]!).toBeGreaterThanOrEqual(LABEL_H + LABEL_GAP);
    // Moved apart, the column steps out for the leaders; the labels stay inside the bounds.
    for (const p of placed) {
      expect(p.x).toBe(514);
      expect(p.x + labelWidth(p.label)).toBeLessThanOrEqual(600);
    }
    expect(placed.some(needsLeader)).toBe(true);
  });

  it('labels a line ending at the plot edge in the column, beside its end when nothing crowds it', () => {
    const ends = [{ id: 'total', label: 'total', x: 500, y: 100, curve: 0 }];
    const [p] = placeEndLabels(ends, [line([[0, 100], [500, 100]])], bounds, 2, 500, { gap: 10, lead: 14 });
    expect(p).toMatchObject({ x: 506, y: 100 });
    expect(needsLeader(p!)).toBe(false);
  });
});

/** Clear space between two placed labels' drawn boxes (their measured width by the 16 px text line), px. */
function clearance(a: Placed, b: Placed): number {
  const dx = Math.max(b.x - (a.x + a.w), a.x - (b.x + b.w));
  const dy = Math.abs(a.y - b.y) - LABEL_H;
  return Math.max(dx, dy);
}

describe('placeEndLabels: the 12 px gap', () => {
  it('never draws two labels closer than 12 px, whatever the caller asks and however the lines end', () => {
    // Round 22: Run › Loss drew box and dfl with their 16 px text boxes 10 px apart (a 14 px box was assumed).
    const flat = (y: number) => line([[0, y], [500, y]]);
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const names = ['total', 'box', 'cls', 'dfl', 'fasteners-s', 'shapes-hard-m · failed'];
    for (let trial = 0; trial < 200; trial++) {
      const n = 2 + Math.floor(rnd() * 4);
      const atEdge = rnd() < 0.5;
      const ends = Array.from({ length: n }, (_, i) => ({
        id: `l${i}`,
        label: names[(trial + i) % names.length]!,
        x: atEdge ? 500 : 300 + Math.floor(rnd() * 200),
        y: 120 + Math.floor(rnd() * 60),
        curve: i,
      }));
      const curves = ends.map((e) => line([[0, e.y + 40], [e.x, e.y]]));
      const gap = [undefined, 2, 4, 10, 12, 16][trial % 6];
      const placed = placeEndLabels(ends, [...curves, flat(60)], bounds, 2, atEdge ? 500 : undefined, { gap, lead: 14 });
      for (let i = 0; i < placed.length; i++)
        for (let j = i + 1; j < placed.length; j++)
          expect(clearance(placed[i]!, placed[j]!), `trial ${trial}`).toBeGreaterThanOrEqual(Math.max(gap ?? 0, LABEL_GAP) - 1e-9);
    }
  });

  it("keeps the loss chart's box and dfl 12 px apart where both lines end together", () => {
    const flat = (y: number) => line([[0, y], [500, y]]);
    const ends = [
      { id: 'total', label: 'total', x: 500, y: 100, curve: 0 },
      { id: 'box', label: 'box', x: 500, y: 176, curve: 1 },
      { id: 'dfl', label: 'dfl', x: 500, y: 177, curve: 2 },
      { id: 'cls', label: 'cls', x: 500, y: 243, curve: 3 },
    ];
    const placed = placeEndLabels(ends, ends.map((e) => flat(e.y)), bounds, 2, 500, { lead: 14 });
    expect(placed).toHaveLength(4);
    const at = Object.fromEntries(placed.map((p) => [p.id, p]));
    expect(clearance(at.box!, at.dfl!)).toBeGreaterThanOrEqual(LABEL_GAP);
    expect(needsLeader(at.box!) || needsLeader(at.dfl!)).toBe(true);
  });
});

describe("placeEndLabels: other runs' lines are obstacles", () => {
  /** Least distance from a placed label's box to a polyline, px. */
  function distance(p: Placed, c: { x: number; y: number }[]): number {
    const box = { x0: p.x, x1: p.x + p.w, y0: p.y - LABEL_H / 2, y1: p.y + LABEL_H / 2 };
    let d = Infinity;
    for (let i = 1; i < c.length; i++) {
      const a = c[i - 1]!;
      const b = c[i]!;
      for (let t = 0; t <= 1; t += 1 / 200) {
        const x = a.x + (b.x - a.x) * t;
        const y = a.y + (b.y - a.y) * t;
        d = Math.min(d, Math.hypot(Math.max(box.x0 - x, 0, x - box.x1), Math.max(box.y0 - y, 0, y - box.y1)));
      }
    }
    return d;
  }

  it("keeps a label at least 6 px off another run's line", () => {
    // Round 24: "fasteners-s · live" sat 2 px over shapes-hard-s's line, which runs on past its end just below.
    const own = line([[0, 280], [200, 100]]);
    const other = line([[0, 290], [190, 120], [300, 112], [600, 112]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'fasteners-s · live', x: 200, y: 100, curve: 0 }], [own, other], bounds, 2);
    expect(CLEAR).toBe(6);
    expect(p).toBeDefined();
    expect(distance(p!, other)).toBeGreaterThanOrEqual(CLEAR - 1e-6);
  });

  it("never ties a label to its line with a leader that crosses another run's line where a clear spot allows", () => {
    // Round 24: "shapes-hard-m · failed" floated mid-plot, its leader running across two other runs' lines.
    const own = line([[0, 280], [200, 150]]);
    const other = line([[0, 290], [190, 120], [300, 112], [600, 112]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'shapes-hard-m · failed', x: 200, y: 150, curve: 0 }], [own, other], bounds, 2);
    expect(p).toBeDefined();
    const l = leaderOf(p!, 5);
    if (l) expect(crosses(l, other)).toBe(false);
  });

  it('labels a line that ends among the others: the clear spot whose leader crosses the fewest lines, nearest its end', () => {
    // Round 25: Compare's failed shapes-hard-m ended between two other runs with no free spot near it, and had no label.
    const own = line([[0, 280], [200, 150]]);
    const above = line([[100, 140], [600, 140]]);
    const below = [160, 175, 190].map((y) => line([[100, y], [600, y]]));
    const placed = placeEndLabels([{ id: 'a', label: 'shapes-hard-m · failed', x: 200, y: 150, curve: 0 }], [own, above, ...below], bounds, 2, 560);
    expect(placed).toHaveLength(1);
    const p = placed[0]!;
    // Above the one line, not below the three, and clear of every line it does not name.
    expect(p.y).toBeLessThan(140);
    const l = leaderOf(p, 5)!;
    expect([above, ...below].filter((c) => crosses(l, c))).toEqual([above]);
    for (const c of [above, ...below]) expect(distance(p, c)).toBeGreaterThanOrEqual(CLEAR - 1e-6);
  });
});

describe('placeEndLabels: leaders only from the gutter', () => {
  it('keeps a label inside the plot beside its marker or in the gutter while either has room', () => {
    // A line ending mid-plot with others crowding every spot beside its end: no spot beside it, none in reach of the
    // gutter, so it takes the last resort (a clear spot whose leader crosses the fewest lines) rather than no label.
    const own = line([[0, 280], [200, 150]]);
    const crowd = [line([[100, 150], [500, 150]]), line([[100, 132], [500, 132]]), line([[100, 168], [500, 168]])];
    const placed = placeEndLabels([{ id: 'a', label: 'shapes-hard-m · failed', x: 200, y: 150, curve: 0 }], [own, ...crowd], bounds, 2, 500);
    expect(placed).toHaveLength(1);
    // Without the crowd, beside its end.
    const [p] = placeEndLabels([{ id: 'a', label: 'shapes-hard-m · failed', x: 200, y: 150, curve: 0 }], [own], bounds, 2, 500);
    expect(needsLeader(p!)).toBe(false);
  });
});

describe('placeEndLabels: cost', () => {
  /** A crowded comparison: four noisy 300-point runs, two ending mid-plot among the others (seeded). */
  function crowded() {
    let seed = 7;
    const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
    const ends = [360, 820, 500, 820];
    const curves = ends.map((xe, k) => {
      let y = 60 + k * 50;
      return Array.from({ length: 300 }, (_, i) => {
        y = Math.min(270, Math.max(15, y + (rand() - 0.5) * 40));
        return { x: 40 + ((xe - 40) * i) / 299, y };
      });
    });
    const points = curves.map((c, k) => ({ id: `r${k}`, label: `shapes-hard-${k} · failed`, x: c[299]!.x, y: c[299]!.y, curve: k }));
    return { points, curves, bounds: { x0: 40, y0: 8, x1: 958, y1: 278 } };
  }

  it('places four labels among four 300-point lines in well under a frame, the last resort included', () => {
    // Round 26 review: the last resort tested every grid box of the plot against every segment, ~240 ms for four
    // 150-point lines. The fastest of five runs, so a busy CI machine does not fail it; it was ~140 ms, now ~2 ms.
    const { points, curves, bounds } = crowded();
    const placed = placeEndLabels(points, curves, bounds, 2, 820, { lead: 14 });
    expect(placed).toHaveLength(4);
    // Some label went to the last resort: inside the plot, off its line's end.
    expect(placed.some((p) => p.x < 820 && needsLeader(p))).toBe(true);
    let best = Infinity;
    for (let i = 0; i < 5; i++) {
      const t = performance.now();
      placeEndLabels(points, curves, bounds, 2, 820, { lead: 14 });
      best = Math.min(best, performance.now() - t);
    }
    expect(best).toBeLessThan(10);
  });

  it('places again only when what it places changes (memoPlaceEndLabels)', () => {
    const { points, curves, bounds } = crowded();
    const place = memoPlaceEndLabels();
    const first = place(points, curves, bounds, 2, 820, { lead: 14 });
    expect(first).toEqual(placeEndLabels(points, curves, bounds, 2, 820, { lead: 14 }));
    // Equal values in new arrays (a chart re-rendering): the same answer, not placed again.
    const copy = <T extends object>(v: T[]) => v.map((o) => ({ ...o }));
    expect(place(copy(points), curves.map(copy), { ...bounds }, 2, 820, { lead: 14 })).toBe(first);
    // A line that moved, a resized plot, a renamed label: placed again.
    const moved = curves.map((c, k) => (k === 1 ? c.map((p) => ({ x: p.x, y: p.y + 1 })) : c));
    expect(place(points, moved, bounds, 2, 820, { lead: 14 })).not.toBe(first);
    const again = place(points, curves, bounds, 2, 820, { lead: 14 });
    expect(again).not.toBe(first);
    expect(again).toEqual(first);
    expect(place(points, curves, { ...bounds, x1: 900 }, 2, 820, { lead: 14 })).not.toBe(again);
    const renamed = points.map((p, k) => (k === 0 ? { ...p, label: 'renamed' } : p));
    expect(place(renamed, curves, { ...bounds, x1: 900 }, 2, 820, { lead: 14 })[0]!.label).toBe('renamed');
  });
});

function crosses(l: { x1: number; y1: number; x2: number; y2: number }, c: { x: number; y: number }[]): boolean {
  const o = (p1: { x: number; y: number }, q: { x: number; y: number }, r: { x: number; y: number }) => Math.sign((q.x - p1.x) * (r.y - p1.y) - (q.y - p1.y) * (r.x - p1.x));
  const L1 = { x: l.x1, y: l.y1 };
  const L2 = { x: l.x2, y: l.y2 };
  for (let i = 1; i < c.length; i++) {
    const a = c[i - 1]!;
    const b = c[i]!;
    if (o(a, b, L1) !== o(a, b, L2) && o(L1, L2, a) !== o(L1, L2, b)) return true;
  }
  return false;
}
