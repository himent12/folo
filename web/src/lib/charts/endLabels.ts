// Direct labels at the ends of lines (comparisons): each label takes the first spot beside its line's last point
// that covers no curve and no other label, stepping further up or down when the near spots are taken; then the same
// spots allowing the label over its own line (never another's); finally the clear gutter right of the plot: as one
// column in the order of the lines' ends when the caller says where the plot ends (so two lines ending together get
// neighbouring labels), else at the nearest free height. A line that ends at the plot's right edge goes straight to that
// column. Labels keep `gap` px between them; a label moved off its line's end gets a short leader (`Placed.ax/ay`).
// Only a label with no free spot anywhere is left out (the legend still names the line).

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
  /** The line's end the label names, in pixels: a caller draws a leader from it when the label sits away from it. */
  ax: number;
  ay: number;
}

export interface PlaceOptions {
  /** Clear space between two labels' boxes, px (default 2 × pad). */
  gap?: number;
  /** How far right of the plot the gutter column starts when a label there sits off its line's end (room for a leader). */
  lead?: number;
}

/** Whether a placed label sits away from its line's end, so it needs a leader to show which line it names. */
export const needsLeader = (p: Placed) => p.x - p.ax >= 12 && Math.abs(p.y - p.ay) > 3;

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
 * right gutter); `gutter`, the x where the plot ends and the gutter starts. Output follows the input order.
 */
export function placeEndLabels(
  ends: readonly EndPoint[],
  curves: readonly (readonly Pt[])[],
  bounds: Box,
  pad = 2,
  gutter?: number,
  opts: PlaceOptions = {},
): Placed[] {
  const gap = opts.gap ?? 2 * pad;
  const taken: Box[] = [];
  const spots = new Map<number, Box>();
  const placed = new Map<number, Placed>();
  const fits = (b: Box, skip?: number) => {
    if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) return false;
    const g = { x0: b.x0 - pad, y0: b.y0 - pad, x1: b.x1 + pad, y1: b.y1 + pad };
    const room = { x0: b.x0 - gap, y0: b.y0 - gap, x1: b.x1 + gap, y1: b.y1 + gap };
    if (taken.some((t) => overlaps(t, room))) return false;
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
  // With a gutter, a line that ends at the plot's right edge is labelled beside its end or in the ordered column below,
  // never stepped further out (up into the plot or the gutter, across other lines and out of order); so is any spot
  // further out that lies in the gutter.
  const atEdge = (e: EndPoint) => gutter !== undefined && e.x >= gutter - 1;
  const away = (e: EndPoint, b: Box, k: number) => gutter !== undefined && k > 0 && (b.x0 >= gutter || atEdge(e));
  // A line ending at the plot's edge is labelled in the gutter column (below), never above or below its end inside the
  // plot, where it would sit beside another line's label.
  for (let k = 0; k < LEVELS; k++) rounds.push((e, w) => (atEdge(e) ? undefined : ring(e, w, k).find((b) => !away(e, b, k) && fits(b))));
  rounds.push((e, w) => (e.curve === undefined || atEdge(e) ? undefined : ring(e, w, 0).find((b) => fits(b, e.curve))));
  const take = (i: number, spot: Box) => {
    const e = ends[i]!;
    taken.push(spot);
    spots.set(i, spot);
    placed.set(i, { id: e.id, label: e.label, x: spot.x0, y: (spot.y0 + spot.y1) / 2, ax: e.x, ay: e.y });
  };
  const run = (round: (e: EndPoint, w: number) => Box | undefined) =>
    ends.forEach((e, i) => {
      if (placed.has(i)) return;
      const spot = round(e, labelWidth(e.label));
      if (spot) take(i, spot);
    });
  for (const round of rounds) run(round);
  if (gutter !== undefined && placed.size < ends.length) column(gutter);
  // Whatever is still without a spot: the gutter at the free height nearest its line's end.
  run((e, w) => {
    for (let d = 0; d <= bounds.y1 - bounds.y0; d += 2)
      for (const y of d ? [e.y - d, e.y + d] : [e.y]) {
        const b = { x0: bounds.x1 - w, y0: y - H / 2, x1: bounds.x1, y1: y + H / 2 };
        if (fits(b)) return b;
      }
    return undefined;
  });
  return ends.flatMap((_, i) => placed.get(i) ?? []);

  /**
   * The labels left without a spot and those already beside their line's end in the gutter, re-laid as one column in
   * the order of their ends: each group of labels that would overlap is centred on its lines' ends and stacked one label
   * height apart, inside the bounds. Kept only if every label then fits; otherwise nothing changes.
   */
  function column(x: number) {
    const members = ends.map((_, i) => i).filter((i) => !placed.has(i) || spots.get(i)!.x0 >= x);
    const old = members.flatMap((i) => spots.get(i) ?? []);
    for (const b of old) taken.splice(taken.indexOf(b), 1);
    const step = H + Math.max(gap, pad + 1);
    const lo = bounds.y0 + H / 2;
    const order = [...members].sort((a, b) => ends[a]!.y - ends[b]!.y || a - b);
    type Group = { items: number[]; sum: number };
    const top = (g: Group) => {
      const n = g.items.length;
      return Math.min(Math.max(g.sum / n - ((n - 1) * step) / 2, lo), bounds.y1 - H / 2 - (n - 1) * step);
    };
    const groups: Group[] = [];
    for (const i of order) {
      groups.push({ items: [i], sum: ends[i]!.y });
      // Merge backwards while the newest group runs into the one above it.
      while (groups.length > 1) {
        const b = groups[groups.length - 1]!;
        const a = groups[groups.length - 2]!;
        if (top(a) + a.items.length * step <= top(b) + 1e-9) break;
        groups.splice(groups.length - 2, 2, { items: [...a.items, ...b.items], sum: a.sum + b.sum });
      }
    }
    const ys = new Map<number, number>();
    for (const g of groups) g.items.forEach((i, k) => ys.set(i, top(g) + k * step));
    // Beside the line's end when every label in the column sits level with its own; set further out when any was
    // moved, so its leader has room.
    const moved = [...ys].some(([i, y]) => Math.abs(y - ends[i]!.y) > 3);
    const x0 = x + (moved ? (opts.lead ?? 6) : 6);
    const next = new Map<number, Box>();
    for (const [i, y] of ys) next.set(i, { x0, y0: y - H / 2, x1: x0 + labelWidth(ends[i]!.label), y1: y + H / 2 });
    const ok = [...next.values()].every((b) => fits(b));
    if (!ok) {
      taken.push(...old);
      return;
    }
    for (const i of members) {
      spots.delete(i);
      placed.delete(i);
    }
    for (const [i, b] of next) take(i, b);
  }
}
