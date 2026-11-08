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
});
