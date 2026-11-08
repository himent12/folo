import { describe, expect, it } from 'vitest';
import { labelWidth, placeEndLabels } from './endLabels';

const bounds = { x0: 0, y0: 0, x1: 600, y1: 300 };
const line = (pts: [number, number][]) => pts.map(([x, y]) => ({ x, y }));

describe('placeEndLabels', () => {
  it('puts a label right of its line when that spot is clear', () => {
    const c = line([[0, 200], [300, 100]]);
    const [p] = placeEndLabels([{ id: 'a', label: 'run-a', x: 300, y: 100 }], [c], bounds);
    expect(p).toEqual({ id: 'a', label: 'run-a', x: 306, y: 100 });
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
        const sep = Math.abs(A.y - B.y) >= 12 || A.x + labelWidth(A.label) <= B.x || B.x + labelWidth(B.label) <= A.x;
        expect(sep).toBe(true);
      }
    expect(placed.length).toBe(2);
    // A third label crowded into a box with no room is dropped rather than drawn on top.
    const tight = placeEndLabels([...ends, { id: 'c', label: 'x', x: 590, y: 150 }], [], { x0: 560, y0: 130, x1: 600, y1: 170 });
    expect(tight.length).toBeLessThan(3);
  });
});
