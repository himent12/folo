import { describe, expect, it } from 'vitest';
import { BELOW_GAP, CHIP_H, placeChips, type ChipBox, type ChipBounds, type Placement } from './chips';

const bounds = { minX: 4, maxX: 196, minY: 2, maxY: 98 };

/** Vertical gap between a placed chip and its own box (0 when they overlap or touch). */
function gap(box: ChipBox, p: Placement) {
  const y = box.y + p.dy;
  return Math.max(0, box.y - (y + CHIP_H), y - (box.y + (box.bh ?? 0)));
}

/** Overlap of a placed chip with another box's interior (2 px inset), in px². */
function coverArea(box: ChipBox, p: Placement, o: ChipBox) {
  const x = box.x + p.dx;
  const y = box.y + p.dy;
  const w = Math.min(x + box.w, o.x + o.bw! - 2) - Math.max(x, o.x + 2);
  const h = Math.min(y + CHIP_H, o.y + o.bh! - 2) - Math.max(y, o.y + 2);
  return w > 0 && h > 0 ? w * h : 0;
}

/** A small deterministic PRNG (mulberry32), so the property scenes are the same on every run. */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function scene(seed: number, n: number, b: ChipBounds): ChipBox[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => {
    const bw = 12 + Math.round(r() * 140);
    const bh = 12 + Math.round(r() * 140);
    const x = b.minX + Math.round(r() * (b.maxX - b.minX - bw));
    const y = b.minY + Math.round(r() * (b.maxY - b.minY - bh));
    return { x, y, w: 40 + Math.round(r() * 50), bw, bh };
  });
}

describe('placeChips', () => {
  it('puts a chip above its box and clamps it horizontally', () => {
    const [p] = placeChips([{ x: 180, y: 50, w: 60 }], bounds);
    expect(p!.side).toBe('above');
    expect(p!.dy).toBe(-CHIP_H - 1);
    expect(p!.leader).toBe(0);
    expect(180 + p!.dx + 60).toBeLessThanOrEqual(196);
  });

  it('goes inside at the top edge', () => {
    const [p] = placeChips([{ x: 20, y: 4, w: 50 }], bounds);
    expect(p!.inside).toBe(true);
    expect(p!.dy).toBe(0);
  });

  it('never runs past the bottom of the image, and hides a label rather than moving it away', () => {
    const ps = placeChips(
      [
        { x: 20, y: 90, w: 50 },
        { x: 22, y: 90, w: 50 },
      ],
      bounds,
    );
    for (const p of ps) expect(90 + p!.dy + CHIP_H).toBeLessThanOrEqual(98);
    expect(ps[0]!.hidden).toBe(false);
    // Above and inside are both taken by the first chip: the second waits for hover instead of drifting off.
    expect(ps[1]!.hidden).toBe(true);
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

  it('hangs a chip below a short box, with a leader, rather than over a neighbouring box', () => {
    // A small box right below a tall neighbour: above, its chip would sit on the neighbour; inside, it would cover the
    // small object. It hangs below its own box, a leader bridging the gap.
    const ps = placeChips(
      [
        { x: 60, y: 10, w: 40, bw: 30, bh: 40 },
        { x: 30, y: 52, w: 60, bw: 20, bh: 20 },
      ],
      { minX: 4, maxX: 196, minY: 2, maxY: 198 },
    );
    expect(ps[1]!.side).toBe('below');
    expect(ps[1]!.dy).toBe(20 + BELOW_GAP);
    expect(ps[1]!.leader).toBe(BELOW_GAP);
    expect(ps[1]!.hidden).toBe(false);
  });

  it('goes inside a tall box rather than over the box above it', () => {
    const b = { minX: 4, maxX: 196, minY: 2, maxY: 198 };
    const boxes = [
      { x: 30, y: 20, w: 50, bw: 60, bh: 38 },
      { x: 50, y: 60, w: 60, bw: 80, bh: 60 },
    ];
    const ps = placeChips(boxes, b);
    expect(ps[1]!.side).toBe('inside');
    expect(coverArea(boxes[1]!, ps[1]!, boxes[0]!)).toBe(0);
  });

  it('slides along its own top edge to clear a neighbouring box', () => {
    // The default slot (above, at the left edge) lies over the box to the upper left; a few px to the right is clear.
    const b = { minX: 4, maxX: 296, minY: 2, maxY: 198 };
    const boxes = [
      { x: 20, y: 20, w: 50, bw: 60, bh: 60 },
      { x: 70, y: 70, w: 50, bw: 120, bh: 60 },
    ];
    const ps = placeChips(boxes, b);
    expect(ps[1]!.side).toBe('above');
    expect(70 + ps[1]!.dx).toBeGreaterThanOrEqual(20 + 60);
    expect(coverArea(boxes[1]!, ps[1]!, boxes[0]!)).toBe(0);
  });

  it('nudges sideways past an earlier chip before going inside', () => {
    const ps = placeChips(
      [
        { x: 20, y: 50, w: 60, bw: 100, bh: 40 },
        { x: 60, y: 50, w: 60, bw: 100, bh: 40 },
      ],
      { minX: 4, maxX: 296, minY: 2, maxY: 198 },
    );
    expect(ps[1]!.side).toBe('above');
    expect(ps[1]!.hidden).toBe(false);
    expect(60 + ps[1]!.dx).toBe(18 + 60 + 2);
  });

  it('does not count the box that contains this one as covered', () => {
    const ps = placeChips(
      [
        { x: 10, y: 10, w: 50, bw: 180, bh: 180 },
        { x: 50, y: 80, w: 50, bw: 40, bh: 40 },
      ],
      { minX: 4, maxX: 196, minY: 2, maxY: 198 },
    );
    expect(ps[1]!.side).toBe('above');
    expect(ps[1]!.dx).toBe(-2);
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

  describe('ownership in crowded scenes', () => {
    const b = { minX: 4, maxX: 636, minY: 2, maxY: 478 };
    const seeds = Array.from({ length: 40 }, (_, i) => i + 1);

    it('keeps every chip next to its own box, overlapping it horizontally', () => {
      for (const seed of seeds) {
        const boxes = scene(seed, 14, b);
        placeChips(boxes, b).forEach((p, i) => {
          const box = boxes[i]!;
          const x = box.x + p!.dx;
          expect(gap(box, p!), `seed ${seed} box ${i}`).toBeLessThanOrEqual(BELOW_GAP);
          expect(Math.min(x + box.w, box.x + box.bw!) - Math.max(x, box.x), `seed ${seed} box ${i}`).toBeGreaterThan(0);
          // Only a chip hung below its box has a gap, and it always gets a leader.
          expect(p!.leader).toBe(p!.side === 'below' ? BELOW_GAP : 0);
        });
      }
    });

    it('never lets two shown chips overlap, and stays inside the image', () => {
      for (const seed of seeds) {
        const boxes = scene(seed, 14, b);
        const ps = placeChips(boxes, b);
        const rects = ps.flatMap((p, i) => (p!.hidden ? [] : [{ x: boxes[i]!.x + p!.dx, y: boxes[i]!.y + p!.dy, w: boxes[i]!.w }]));
        for (const r of rects) {
          expect(r.x).toBeGreaterThanOrEqual(b.minX);
          expect(r.x + r.w).toBeLessThanOrEqual(b.maxX);
          expect(r.y).toBeGreaterThanOrEqual(b.minY);
          expect(r.y + CHIP_H).toBeLessThanOrEqual(b.maxY);
        }
        for (let i = 0; i < rects.length; i++)
          for (let j = i + 1; j < rects.length; j++) {
            const [p, q] = [rects[i]!, rects[j]!];
            const overlap = p.x < q.x + q.w && p.x + p.w > q.x && p.y < q.y + CHIP_H && p.y + CHIP_H > q.y;
            expect(overlap, `seed ${seed}`).toBe(false);
          }
      }
    });

    it('covers another box only when every adjacent slot would', () => {
      for (const seed of seeds) {
        const boxes = scene(seed, 6, b);
        const ps = placeChips(boxes, b);
        ps.forEach((p, i) => {
          if (p!.hidden) return;
          const own = boxes[i]!;
          const contains = (o: ChipBox) => o.x <= own.x && o.y <= own.y && o.x + o.bw! >= own.x + own.bw! && o.y + o.bh! >= own.y + own.bh!;
          const covered = boxes.some((o, j) => j !== i && !contains(o) && coverArea(own, p!, o) > 0);
          if (!covered) return;
          // Placed with no other chips in the way: either every adjacent slot covers a box too, or the clear slot it
          // would take is held by an earlier chip.
          const solo = placeChips(boxes.map((o, j) => (j === i ? o : { ...o, reserve: false })), b)[i]!;
          if (boxes.some((o, j) => j !== i && !contains(o) && coverArea(own, solo, o) > 0)) return;
          const sx = own.x + solo.dx;
          const sy = own.y + solo.dy;
          const blocked = ps.some((q, j) => {
            if (j >= i || q!.hidden) return false;
            const [qx, qy] = [boxes[j]!.x + q!.dx, boxes[j]!.y + q!.dy];
            return sx < qx + boxes[j]!.w && sx + own.w > qx && sy < qy + CHIP_H && sy + CHIP_H > qy;
          });
          expect(blocked, `seed ${seed} box ${i}`).toBe(true);
        });
      }
    });

    it('shows most labels', () => {
      let shown = 0;
      let total = 0;
      for (const seed of seeds) {
        const boxes = scene(seed, 8, b);
        const ps = placeChips(boxes, b);
        total += ps.length;
        shown += ps.filter((p) => !p!.hidden).length;
      }
      expect(shown / total).toBeGreaterThan(0.8);
    });
  });
});
