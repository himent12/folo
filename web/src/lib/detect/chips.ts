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
  const n = boxes.length;
  const placed: { x: number; y: number; w: number }[] = [];
  type SizedBox = { idx: number; x: number; y: number; w: number; h: number };
  const sizedByIdx: (SizedBox | null)[] = new Array(n);
  const sizedList: SizedBox[] = [];
  for (let i = 0; i < n; i++) {
    const o = boxes[i];
    if (o && o.bw !== undefined && o.bh !== undefined) {
      const s: SizedBox = { idx: i, x: o.x, y: o.y, w: o.bw, h: o.bh };
      sizedByIdx[i] = s;
      sizedList.push(s);
    } else {
      sizedByIdx[i] = null;
    }
  }

  // Reusable scratch buffers: avoid O(N^2) / O(N^3) temporary arrays and Sets across crowded scenes.
  const others: SizedBox[] = [];
  const rowOthers: SizedBox[] = [];
  const rowPlaced: { x: number; y: number; w: number }[] = [];
  const rows: { side: ChipSide; y: number; rank: number }[] = [];
  const xs = new Set<number>();
  const out: (Placement | null)[] = new Array(n);

  const hitsRow = (x: number, w: number): boolean => {
    const x2 = x + w;
    for (let k = 0; k < rowPlaced.length; k++) {
      const p = rowPlaced[k]!;
      if (x < p.x + p.w && x2 > p.x) return true;
    }
    return false;
  };
  const coversRow = (x: number, w: number): number => {
    const x2 = x + w;
    let count = 0;
    for (let k = 0; k < rowOthers.length; k++) {
      const o = rowOthers[k]!;
      if (x < o.x + o.w - 2 && x2 > o.x + 2) count++;
    }
    return count;
  };

  for (let idx = 0; idx < n; idx++) {
    const box = boxes[idx];
    if (!box) {
      out[idx] = null;
      continue;
    }
    const own = sizedByIdx[idx]!;
    // Boxes this chip must not cover: every other sized box except one that contains this box (a chip inside a
    // container is unavoidable and reads fine). A 2 px inset ignores touching strokes.
    others.length = 0;
    for (let j = 0; j < sizedList.length; j++) {
      const o = sizedList[j]!;
      if (o.idx === idx) continue;
      if (own && o.x <= own.x && o.y <= own.y && o.x + o.w >= own.x + own.w && o.y + o.h >= own.y + own.h) continue;
      others.push(o);
    }

    const maxClampX = b.maxX - box.w;
    const clampX = (x: number) => Math.max(b.minX, Math.min(maxClampX, x));
    const x0 = clampX(box.x - 2);

    // Rows next to the box, with their rank (lower is better).
    rows.length = 0;
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
      if (opts.small) {
        const yTop = own.y;
        const yBot = own.y + own.h - H;
        if (yTop >= b.minY && yTop <= maxY) rows.push({ side: 'beside', y: yTop, rank: 2.5 });
        if (yBot !== yTop && yBot >= b.minY && yBot <= maxY) rows.push({ side: 'beside', y: yBot, rank: 2.5 });
      }
    }

    let hasBest = false;
    let bestX = 0;
    let bestY = 0;
    let bestSide: ChipSide = 'inside';
    let bestCost = 0;

    let hasFallback = false;
    let fbX = 0;
    let fbY = 0;
    let fbSide: ChipSide = 'inside';
    let fbCost = 0;

    for (let r = 0; r < rows.length; r++) {
      const row = rows[r]!;
      const ry = row.y;
      const ry2 = ry + H;

      rowPlaced.length = 0;
      for (let k = 0; k < placed.length; k++) {
        const p = placed[k]!;
        if (ry < p.y + H && ry2 > p.y) rowPlaced.push(p);
      }
      rowOthers.length = 0;
      for (let k = 0; k < others.length; k++) {
        const o = others[k]!;
        if (ry < o.y + o.h - 2 && ry2 > o.y + 2) rowOthers.push(o);
      }

      if (row.side === 'beside') {
        // Left or right of the box, never clamped back over it.
        const bx0 = own!.x - box.w - 3;
        const bx1 = own!.x + own!.w + 3;
        for (let sideIdx = 0; sideIdx < 2; sideIdx++) {
          const x = sideIdx === 0 ? bx0 : bx1;
          if (x < b.minX || x + box.w > b.maxX) continue;
          const cv = coversRow(x, box.w);
          const cost = cv * 100 + row.rank * 10;
          if (!hasFallback || cost < fbCost) {
            hasFallback = true;
            fbX = x;
            fbY = ry;
            fbSide = 'beside';
            fbCost = cost;
          }
          if (!hitsRow(x, box.w) && !cv && (!hasBest || cost < bestCost)) {
            hasBest = true;
            bestX = x;
            bestY = ry;
            bestSide = 'beside';
            bestCost = cost;
          }
        }
        continue;
      }

      xs.clear();
      xs.add(x0);
      if (own) {
        // Sideways candidates: aligned to the box's right edge, and just clear of each chip or box in the way.
        xs.add(clampX(own.x + own.w + 2 - box.w));
        for (let k = 0; k < rowPlaced.length; k++) {
          const p = rowPlaced[k]!;
          xs.add(clampX(p.x + p.w + 2));
          xs.add(clampX(p.x - box.w - 2));
        }
        for (let k = 0; k < others.length; k++) {
          const o = others[k]!;
          if (ry < o.y + o.h && ry2 > o.y) {
            xs.add(clampX(o.x + o.w + 2));
            xs.add(clampX(o.x - box.w - 2));
          }
        }
      }

      for (const x of xs) {
        const nudge = Math.abs(x - x0);
        if (own && nudge > 0) {
          const along = x >= own.x - 2 && x + box.w <= own.x + own.w + 2;
          const overlap = Math.min(x + box.w, own.x + own.w) - Math.max(x, own.x);
          if ((nudge > MAX_NUDGE && !along) || overlap < Math.min(8, own.w)) continue;
        }
        // A cover outweighs any rank; a nudge costs half a rank step, growing slightly with distance.
        const cv = coversRow(x, box.w);
        const cost = cv * 100 + row.rank * 10 + (nudge > 0 ? 5 + nudge / 100 : 0);
        if (!hasFallback || cost < fbCost) {
          hasFallback = true;
          fbX = x;
          fbY = ry;
          fbSide = row.side;
          fbCost = cost;
        }
        // A small view never lets a chip cover another box: it waits for a tap instead.
        if (!hitsRow(x, box.w) && (!opts.small || !cv) && (!hasBest || cost < bestCost)) {
          hasBest = true;
          bestX = x;
          bestY = ry;
          bestSide = row.side;
          bestCost = cost;
        }
      }
    }

    // `rows` always holds the inside slot, so a fallback exists.
    const pickX = hasBest ? bestX : fbX;
    const pickY = hasBest ? bestY : fbY;
    const pickSide = hasBest ? bestSide : fbSide;
    if (hasBest && box.reserve !== false) placed.push({ x: pickX, y: pickY, w: box.w });
    out[idx] = {
      dx: pickX - box.x,
      dy: pickY - box.y,
      side: pickSide,
      inside: pickSide === 'inside',
      leader: pickSide === 'below' ? BELOW_GAP : 0,
      hidden: !hasBest,
    };
  }
  return out;
}
