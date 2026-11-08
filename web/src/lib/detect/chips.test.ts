import { describe, expect, it } from 'vitest';
import { CHIP_H, placeChips } from './chips';

const bounds = { minX: 4, maxX: 196, minY: 2, maxY: 98 };

describe('placeChips', () => {
  it('puts a chip above its box and clamps it horizontally', () => {
    const [p] = placeChips([{ x: 180, y: 50, w: 60 }], bounds);
    expect(p!.inside).toBe(false);
    expect(p!.dy).toBe(-CHIP_H - 1);
    expect(180 + p!.dx + 60).toBeLessThanOrEqual(196);
  });

  it('goes inside at the top edge', () => {
    const [p] = placeChips([{ x: 20, y: 4, w: 50 }], bounds);
    expect(p!.inside).toBe(true);
    expect(p!.dy).toBe(0);
  });

  it('never runs past the bottom of the image', () => {
    const ps = placeChips(
      [
        { x: 20, y: 90, w: 50 },
        { x: 22, y: 90, w: 50 },
      ],
      bounds,
    );
    for (const p of ps) expect(90 + p!.dy + CHIP_H).toBeLessThanOrEqual(98);
    // The second chip found a free slot instead of overlapping the first.
    expect(ps[1]!.dy).not.toBe(ps[0]!.dy);
  });

  it('hides a label that has no free slot instead of covering another', () => {
    const tight = { minX: 4, maxX: 60, minY: 2, maxY: 40 };
    const ps = placeChips(
      [
        { x: 5, y: 20, w: 50 },
        { x: 6, y: 20, w: 50 },
        { x: 7, y: 20, w: 50 },
      ],
      tight,
    );
    expect(ps[0]!.hidden).toBe(false);
    expect(ps.some((p) => p!.hidden)).toBe(true);
  });

  it('moves a chip below its box rather than over a neighbouring box', () => {
    // A small box right below a tall neighbour: its chip would sit on the neighbour, so it goes under its own box.
    const ps = placeChips(
      [
        { x: 60, y: 10, w: 40, bw: 30, bh: 40 },
        { x: 30, y: 52, w: 60, bw: 20, bh: 20 },
      ],
      { minX: 4, maxX: 196, minY: 2, maxY: 198 },
    );
    expect(ps[1]!.dy).toBe(20 + 1);
    expect(ps[1]!.hidden).toBe(false);
  });

  it('tries every slot in the column before hiding a label', () => {
    const boxes = [
      [109, 21, 60, 32, 50],
      [10, 50, 60, 50, 108],
      [62, 40, 60, 66, 112],
      [69, 31, 60, 60, 77],
      [108, 7, 60, 59, 131],
    ].map(([x, y, w, bw, bh]) => ({ x: x!, y: y!, w: w!, bw, bh }));
    const ps = placeChips(boxes, { minX: 4, maxX: 196, minY: 2, maxY: 198 });
    expect(ps.every((p) => p && !p.hidden)).toBe(true);
  });

  it('clamps a non-reserving chip at the edges and keeps its slot free', () => {
    const b = { minX: 4, maxX: 796, minY: 2, maxY: 398 };
    const [top] = placeChips([{ x: 10, y: 0, w: 60, bw: 15, bh: 15, reserve: false }], b);
    expect(0 + top!.dy).toBeGreaterThanOrEqual(2);
    const [right] = placeChips([{ x: 790, y: 100, w: 60, bw: 6, bh: 6, reserve: false }], b);
    expect(790 + right!.dx + 60).toBeLessThanOrEqual(796);
    // A later reserving chip may take the same slot.
    const ps = placeChips(
      [
        { x: 50, y: 50, w: 60, reserve: false },
        { x: 50, y: 50, w: 60 },
      ],
      b,
    );
    expect(ps[1]!.dy).toBe(ps[0]!.dy);
  });

  it('finds an exact-fit slot between two placed chips', () => {
    const ps = placeChips([34, 49, 35].map((y) => ({ x: 20, y, w: 60 })), { minX: 4, maxX: 196, minY: 2, maxY: 65 });
    expect(ps[2]!.hidden).toBe(false);
    expect(35 + ps[2]!.dy).toBe(33);
  });
});
