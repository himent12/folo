// Label-chip placement for DetectionView, in screen pixels. Pure, so it can be tested without a DOM.

export const CHIP_H = 16;
/** A chip hung below its box keeps this gap (bridged by a leader line), so it never reads as the label of a box below. */
export const BELOW_GAP = 10;
/** How far a chip may slide sideways from its box's left edge; further only while it stays over its own box. */
export const MAX_NUDGE = 40;

export interface ChipBox {
  /** Box top-left on screen. */
  x: number;
  y: number;
  /** Estimated chip width. */
  w: number;
  /** The box's own size on screen, when known: chips then avoid covering other boxes and may sit below their own. */
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

export type ChipSide = 'above' | 'inside' | 'below' | 'beside';

export interface ChipOptions {
  /**
   * Small views (phones): a chip goes inside its box's top edge when it fits there (the box at least the chip's width
   * plus 4 px and two chips tall), and outside only otherwise; and a chip never covers another box: with no slot free
   * of boxes and chips it is hidden, and shows when its box is tapped (selected) or hovered.
   */
  small?: boolean;
}

export interface Placement {
  /** Offset of the chip from the box's top-left. */
  dx: number;
  dy: number;
  side: ChipSide;
  inside: boolean;
  /** Gap between the chip and its box's edge, bridged by a leader line (0: the chip touches its box). */
  leader: number;
  /** No free slot next to its own box: show this label only when its box is hovered or selected. */
  hidden: boolean;
}

/**
 * `chipH` is the chip height (16 px, 14 px on small views).
 * A chip only ever sits next to its own box: above it, inside its top edge, hung below it (with a leader), each
 * optionally slid sideways a little (or along the box's own top / bottom edge), or flipped to its left or right side.
 * The preferred slot is above at the left edge, then a sideways nudge, then inside (for boxes tall enough to carry a
 * chip), then below; on a small view (`opts.small`) inside comes first when the chip fits there, a chip may also flip to
 * the box's side, and no slot may cover another box. Slots that cover another box (other than one containing this box) lose to any slot that
 * does not. A chip never overlaps an earlier
 * chip: when no adjacent slot is free the label is hidden (shown on hover / selection) instead of moving far away.
 * Earlier boxes win, so callers pass them highest priority first. Always clamped to the visible image.
 */
export function placeChips(boxes: (ChipBox | null)[], b: ChipBounds, chipH = CHIP_H, opts: ChipOptions = {}): (Placement | null)[] {
  const H = chipH;
  const maxY = b.maxY - H;
  const placed: { x: number; y: number; w: number }[] = [];
  const hits = (x: number, y: number, w: number) =>
    placed.some((p) => x < p.x + p.w && x + w > p.x && y < p.y + H && y + H > p.y);
  const sized = boxes.map((o) => (o && o.bw !== undefined && o.bh !== undefined ? { x: o.x, y: o.y, w: o.bw, h: o.bh } : null));
  return boxes.map((box, idx) => {
    if (!box) return null;
    const own = sized[idx] ?? null;
    // Boxes this chip must not cover: every other sized box except one that contains this box (a chip inside a
    // container is unavoidable and reads fine). A 2 px inset ignores touching strokes.
    const others = sized.filter(
      (o, j): o is NonNullable<typeof o> =>
        j !== idx && !!o && !(own && o.x <= own.x && o.y <= own.y && o.x + o.w >= own.x + own.w && o.y + o.h >= own.y + own.h),
    );
    const covers = (x: number, y: number) =>
      others.filter((o) => x < o.x + o.w - 2 && x + box.w > o.x + 2 && y < o.y + o.h - 2 && y + H > o.y + 2).length;
    const clampX = (x: number) => Math.max(b.minX, Math.min(b.maxX - box.w, x));
    const x0 = clampX(box.x - 2);

    // Rows next to the box, with their rank (lower is better).
    const rows: { side: ChipSide; y: number; rank: number }[] = [];
    const above = box.y - H - 1;
    if (above >= b.minY) rows.push({ side: 'above', y: above, rank: 0 });
    // Inside the top edge covers the top of the object: fine for a tall box, a last resort for a short one. On a small
    // view, a box the chip fits inside takes it there first.
    const short = own !== null && own.h < 2 * H;
    const fitsIn = own !== null && own.w >= box.w + 4 && !short;
    rows.push({ side: 'inside', y: Math.max(b.minY, Math.min(maxY, box.y)), rank: opts.small && fitsIn ? -1 : short ? 3 : 1 });
    if (own) {
      const below = box.y + own.h + BELOW_GAP;
      if (below <= maxY) rows.push({ side: 'below', y: below, rank: 2 });
      // Small views: flipped to the box's side, level with its top (or its bottom), touching its edge, where nothing
      // above, inside or below is free (round 26: "hexagon 0.33" over the circle box beside it).
      if (opts.small) for (const y of new Set([own.y, own.y + own.h - H])) if (y >= b.minY && y <= maxY) rows.push({ side: 'beside', y, rank: 2.5 });
    }

    type Cand = { x: number; y: number; side: ChipSide; cost: number };
    let best: Cand | null = null;
    let fallback: Cand | null = null;
    for (const row of rows) {
      if (row.side === 'beside') {
        // Left or right of the box, never clamped back over it.
        for (const x of [own!.x - box.w - 3, own!.x + own!.w + 3]) {
          if (x < b.minX || x + box.w > b.maxX) continue;
          const cv = covers(x, row.y);
          const cost = cv * 100 + row.rank * 10;
          const cand: Cand = { x, y: row.y, side: 'beside', cost };
          if (!fallback || cost < fallback.cost) fallback = cand;
          if (!hits(x, row.y, box.w) && !cv && (!best || cost < best.cost)) best = cand;
        }
        continue;
      }
      const xs = new Set([x0]);
      if (own) {
        // Sideways candidates: aligned to the box's right edge, and just clear of each chip or box in the way.
        xs.add(clampX(own.x + own.w + 2 - box.w));
        for (const p of placed)
          if (row.y < p.y + H && row.y + H > p.y) xs.add(clampX(p.x + p.w + 2)).add(clampX(p.x - box.w - 2));
        for (const o of others)
          if (row.y < o.y + o.h && row.y + H > o.y) xs.add(clampX(o.x + o.w + 2)).add(clampX(o.x - box.w - 2));
      }
      for (const x of xs) {
        const nudge = Math.abs(x - x0);
        if (own && nudge > 0) {
          const along = x >= own.x - 2 && x + box.w <= own.x + own.w + 2;
          const overlap = Math.min(x + box.w, own.x + own.w) - Math.max(x, own.x);
          if ((nudge > MAX_NUDGE && !along) || overlap < Math.min(8, own.w)) continue;
        }
        // A cover outweighs any rank; a nudge costs half a rank step, growing slightly with distance.
        const cv = covers(x, row.y);
        const cost = cv * 100 + row.rank * 10 + (nudge > 0 ? 5 + nudge / 100 : 0);
        const cand: Cand = { x, y: row.y, side: row.side, cost };
        if (!fallback || cost < fallback.cost) fallback = cand;
        // A small view never lets a chip cover another box: it waits for a tap instead.
        if (!hits(x, row.y, box.w) && (!opts.small || !cv) && (!best || cost < best.cost)) best = cand;
      }
    }
    // `rows` always holds the inside slot, so a fallback exists.
    const pick = best ?? fallback!;
    if (best && box.reserve !== false) placed.push({ x: pick.x, y: pick.y, w: box.w });
    return {
      dx: pick.x - box.x,
      dy: pick.y - box.y,
      side: pick.side,
      inside: pick.side === 'inside',
      leader: pick.side === 'below' ? BELOW_GAP : 0,
      hidden: !best,
    };
  });
}
