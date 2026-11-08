// Label-chip placement for DetectionView, in screen pixels. Pure, so it can be tested without a DOM.

export const CHIP_H = 16;

export interface ChipBox {
  /** Box top-left on screen. */
  x: number;
  y: number;
  /** Estimated chip width. */
  w: number;
  /** The box's own size on screen, when known: other chips then avoid covering it. */
  bw?: number;
  bh?: number;
  /** false: place (and clamp) this chip but keep its slot free for others, e.g. a label shown only on hover. */
  reserve?: boolean;
}

export interface ChipBounds {
  minX: number;
  maxX: number;
  minY: number;
  /** Bottom edge of the visible image. */
  maxY: number;
}

export interface Placement {
  /** Offset of the chip from the box's top-left. */
  dx: number;
  dy: number;
  inside: boolean;
  /** No free slot: show this label only when its box is hovered or selected. */
  hidden: boolean;
}

/**
 * `chipH` is the chip height (16 px, 14 px on small views).
 * Above the box by default; inside it when that leaves the image or hits an earlier chip; nudged down on further
 * collisions while it fits; otherwise the nearest free slot above. A chip that would lie over another box (when box
 * sizes are given) moves just below its own box if that slot is free. Before a label is hidden, every slot in its
 * column is tried, nearest first. Always clamped to the visible image.
 */
export function placeChips(boxes: (ChipBox | null)[], b: ChipBounds, chipH = CHIP_H): (Placement | null)[] {
  const CHIP_H = chipH;
  const maxY = b.maxY - CHIP_H;
  const placed: { x: number; y: number; w: number }[] = [];
  const hits = (x: number, y: number, w: number) =>
    placed.some((p) => x < p.x + p.w && x + w > p.x && y < p.y + CHIP_H && y + CHIP_H > p.y);
  const covers = (x: number, y: number, w: number, self: number) =>
    boxes.some(
      (o, j) =>
        j !== self && !!o && o.bw !== undefined && o.bh !== undefined && x < o.x + o.bw && x + w > o.x && y < o.y + o.bh && y + CHIP_H > o.y,
    );
  return boxes.map((box, idx) => {
    if (!box) return null;
    const x = Math.max(b.minX, Math.min(b.maxX - box.w, box.x - 2));
    let y = box.y - CHIP_H - 1;
    let inside = false;
    if (y < b.minY || hits(x, y, box.w)) {
      y = box.y;
      inside = true;
      for (let k = 0; k < 3 && hits(x, y, box.w) && y + CHIP_H + 2 <= maxY; k++) y += CHIP_H + 2;
      if (hits(x, y, box.w) || y > maxY) {
        for (let up = box.y - CHIP_H - 1; up >= b.minY; up -= CHIP_H + 2) {
          if (!hits(x, up, box.w)) {
            y = up;
            inside = false;
            break;
          }
        }
      }
    }
    y = Math.max(b.minY, Math.min(maxY, y));
    if (box.bh !== undefined && covers(x, y, box.w, idx)) {
      const below = box.y + box.bh + 1;
      if (below <= maxY && !hits(x, below, box.w) && !covers(x, below, box.w, idx)) {
        y = below;
        inside = false;
      }
    }
    if (hits(x, y, box.w)) {
      // Last resort: the free slot in this column nearest the box's top edge.
      let best: number | null = null;
      for (let t = b.minY; t <= maxY; t += 2)
        if (!hits(x, t, box.w) && (best === null || Math.abs(t - box.y) < Math.abs(best - box.y))) best = t;
      if (best !== null) {
        y = best;
        inside = y >= box.y;
      }
    }
    // Earlier boxes win (callers pass them highest score first); a label that would cover one waits for hover.
    if (hits(x, y, box.w)) return { dx: x - box.x, dy: y - box.y, inside, hidden: true };
    if (box.reserve !== false) placed.push({ x, y, w: box.w });
    return { dx: x - box.x, dy: y - box.y, inside, hidden: false };
  });
}
