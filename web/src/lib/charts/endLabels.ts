// Direct labels at the ends of lines (comparisons): each label takes the first spot beside its line's last point
// that covers no curve and no other label, stepping further up or down when the near spots are taken; then the same
// spots allowing the label over its own line (never another's); finally the clear gutter right of the plot at the
// nearest free height. Only a label with no free spot anywhere is left out (the
// legend still names the line).

export interface EndPoint {
  id: string;
  label: string;
  /** The line's last point, in pixels. */
  x: number;
  y: number;
  /** Index of this label's own line in `curves`, which a label may cover before it is sent to the gutter. */
  curve?: number;
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

/** Approximate width of `text` at the chart's 12 px label size. */
export const labelWidth = (text: string) => text.length * 6.8 + 4;

const H = 14;

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
 * Places labels in rounds, nearest spots first: every label tries its near spots before any label takes a spot
 * further out, so one crowded line cannot take the spot another line's label needed. Within a round the caller's
 * order is the priority. `curves` are every drawn line in pixels; `bounds` is where labels may go (the plot plus the
 * right gutter). Output follows the input order.
 */
export function placeEndLabels(ends: readonly EndPoint[], curves: readonly (readonly Pt[])[], bounds: Box, pad = 2): Placed[] {
  const taken: Box[] = [];
  const placed = new Map<number, Placed>();
  const fits = (b: Box, skip?: number) => {
    if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) return false;
    const g = { x0: b.x0 - pad, y0: b.y0 - pad, x1: b.x1 + pad, y1: b.y1 + pad };
    if (taken.some((t) => overlaps(t, g))) return false;
    for (let k = 0; k < curves.length; k++) {
      if (k === skip) continue;
      const c = curves[k]!;
      for (let i = 1; i < c.length; i++) if (segmentHits(c[i - 1]!, c[i]!, g)) return false;
    }
    return true;
  };
  /** Spots `level` label heights out: above / below to the right, to the left and centred (level 0 also beside it). */
  const ring = (e: EndPoint, w: number, level: number): Box[] => {
    const d = level * (H + 2);
    const out: Box[] = level ? [] : [{ x0: e.x + 6, y0: e.y - H / 2, x1: e.x + 6 + w, y1: e.y + H / 2 }];
    for (const x0 of [e.x + 4, e.x - 4 - w, e.x - w / 2])
      out.push({ x0, y0: e.y - 6 - H - d, x1: x0 + w, y1: e.y - 6 - d }, { x0, y0: e.y + 6 + d, x1: x0 + w, y1: e.y + 6 + H + d });
    return out;
  };
  const LEVELS = 5;
  // Rounds: each level clear of every curve; then the near spots allowing a label over its own line (never
  // another's); last the right-hand gutter (no curve runs there), at the free height nearest the line's end.
  const rounds: ((e: EndPoint, w: number) => Box | undefined)[] = [];
  for (let k = 0; k < LEVELS; k++) rounds.push((e, w) => ring(e, w, k).find((b) => fits(b)));
  rounds.push((e, w) => (e.curve === undefined ? undefined : ring(e, w, 0).find((b) => fits(b, e.curve))));
  rounds.push((e, w) => {
    for (let d = 0; d <= bounds.y1 - bounds.y0; d += 2)
      for (const y of d ? [e.y - d, e.y + d] : [e.y]) {
        const b = { x0: bounds.x1 - w, y0: y - H / 2, x1: bounds.x1, y1: y + H / 2 };
        if (fits(b)) return b;
      }
    return undefined;
  });
  for (const round of rounds)
    ends.forEach((e, i) => {
      if (placed.has(i)) return;
      const spot = round(e, labelWidth(e.label));
      if (!spot) return;
      taken.push(spot);
      placed.set(i, { id: e.id, label: e.label, x: spot.x0, y: (spot.y0 + spot.y1) / 2 });
    });
  return ends.flatMap((_, i) => placed.get(i) ?? []);
}
