// Label-chip placement for DetectionView, in screen pixels. Pure, so it can be tested without a DOM.

export const CHIP_H = 16;

export interface ChipBox {
  /** Box top-left on screen. */
  x: number;
  y: number;
  /** Estimated chip width. */
  w: number;
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
}

/**
 * Above the box by default; inside it when that leaves the image or hits an earlier chip; nudged down on further
 * collisions while it fits; otherwise the nearest free slot above. Always clamped to the visible image.
 */
export function placeChips(boxes: (ChipBox | null)[], b: ChipBounds): (Placement | null)[] {
  const maxY = b.maxY - CHIP_H;
  const placed: { x: number; y: number; w: number }[] = [];
  const hits = (x: number, y: number, w: number) =>
    placed.some((p) => x < p.x + p.w && x + w > p.x && y < p.y + CHIP_H && y + CHIP_H > p.y);
  return boxes.map((box) => {
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
    placed.push({ x, y, w: box.w });
    return { dx: x - box.x, dy: y - box.y, inside };
  });
}
