// Direct labels at the ends of lines (comparisons): each label takes the first spot beside its line's last point
// that covers no curve and no other label; one that finds none is left out (the legend still names the line).

export interface EndPoint {
  id: string;
  label: string;
  /** The line's last point, in pixels. */
  x: number;
  y: number;
}

export interface Placed {
  id: string;
  label: string;
  /** Text start (left edge) and vertical centre, in pixels. */
  x: number;
  y: number;
}

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

type Pt = { x: number; y: number };

/** Approximate width of `text` at the chart's 11 px label size. */
export const labelWidth = (text: string) => text.length * 6.2 + 4;

const H = 12;

/** Whether segment a–b crosses (or lies in) the box (Liang–Barsky clipping). */
function segmentHits(a: Pt, b: Pt, r: Box): boolean {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  let t0 = 0;
  let t1 = 1;
  for (const [p, q] of [
    [-dx, a.x - r.x0],
    [dx, r.x1 - a.x],
    [-dy, a.y - r.y0],
    [dy, r.y1 - a.y],
  ] as const) {
    if (p === 0) {
      if (q < 0) return false;
    } else {
      const t = q / p;
      if (p < 0) t0 = Math.max(t0, t);
      else t1 = Math.min(t1, t);
      if (t0 > t1) return false;
    }
  }
  return true;
}

const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/**
 * Places labels in order (the caller's order is the priority). `curves` are every drawn line in pixels; `bounds`
 * is where labels may go (the plot plus the right gutter).
 */
export function placeEndLabels(ends: readonly EndPoint[], curves: readonly (readonly Pt[])[], bounds: Box, pad = 2): Placed[] {
  const taken: Box[] = [];
  const out: Placed[] = [];
  for (const e of ends) {
    const w = labelWidth(e.label);
    // Right of the point first (where the eye goes after a line ends), then above / below it, then to the left.
    const spots: Box[] = [
      { x0: e.x + 6, y0: e.y - H / 2, x1: e.x + 6 + w, y1: e.y + H / 2 },
      { x0: e.x + 4, y0: e.y - 6 - H, x1: e.x + 4 + w, y1: e.y - 6 },
      { x0: e.x + 4, y0: e.y + 6, x1: e.x + 4 + w, y1: e.y + 6 + H },
      { x0: e.x - 4 - w, y0: e.y - 6 - H, x1: e.x - 4, y1: e.y - 6 },
      { x0: e.x - 4 - w, y0: e.y + 6, x1: e.x - 4, y1: e.y + 6 + H },
    ];
    const fits = (b: Box) => {
      if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) return false;
      const g = { x0: b.x0 - pad, y0: b.y0 - pad, x1: b.x1 + pad, y1: b.y1 + pad };
      if (taken.some((t) => overlaps(t, g))) return false;
      for (const c of curves) for (let i = 1; i < c.length; i++) if (segmentHits(c[i - 1]!, c[i]!, g)) return false;
      return true;
    };
    const spot = spots.find(fits);
    if (!spot) continue;
    taken.push(spot);
    out.push({ id: e.id, label: e.label, x: spot.x0, y: (spot.y0 + spot.y1) / 2 });
  }
  return out;
}
