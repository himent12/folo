import { describe, expect, it } from 'vitest';
import { labelWidth, needsLeader, placeEndLabels } from './endLabels';

const bounds = { x0: 0, y0: 0, x1: 600, y1: 300 };
const line = (pts: [number, number][]) => pts.map(([x, y]) => ({ x, y }));

describe('placeEndLabels', () => {
  it('puts a label right of its line when that spot is clear', () => {
    const c = line([[0, 200], [300, 100]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 100 }], [c], bounds);
    expect(p).toEqual({ id: 'a', label: 'run-a', x: 306, y: 100, ax: 300, ay: 100 });
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

  it('labels every line of a four-run comparison: further out, then in the clear gutter, when near spots are taken', () => {
    // Horizontal lines every 10 px across the plot (x 0..500) leave no free spot near the point; the gutter is clear.
    const hatch = Array.from({ length: 31 }, (_, i) => line([[0, i * 10], [500, i * 10]]));
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 150 }], hatch, bounds);
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
    for (let i = 1; i < ys.length; i++) expect(ys[i]! - ys[i - 1]!).toBeGreaterThanOrEqual(24);
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
