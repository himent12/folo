// Direct labels at the ends of lines (comparisons): each label takes the first spot beside its line's last point
// that covers no curve and no other label, stepping further up or down when the near spots are taken; then the same
// spots allowing the label over its own line (never another's); finally the clear gutter right of the plot: as one
// column in the order of the lines' ends when the caller says where the plot ends (so two lines ending together get
// neighbouring labels), else at the free height nearest its end, at most MAX_REACH px away. A line that ends at the
// plot's right edge goes straight to that column. Labels keep at least LABEL_GAP (12 px) between their drawn boxes, a
// box being the label's measured width by its 16 px line (`LABEL_H`, the 12 px text's own line box, not the glyphs'
// height), so the clear space a reader sees is the space the rule names. Every label is anchored to its
// line's end marker: one that does not sit right beside it gets a leader from the marker to the label's nearest edge
// (`leaderOf`), so a label never floats free over another line. Other runs' lines are obstacles with a margin: a
// label keeps CLEAR (6 px) off every line but its own, and its leader crosses none of them. Every line is labelled
// (round 25 left a failed run's line, ending among the others, with only its cross): a label with no such spot takes
// the clear spot, in the plot or the gutter, whose leader crosses the fewest other lines, nearest its line's end
// (`lastResort`). Only a label with no clear box anywhere in the bounds is left out.

export interface EndPoint {
  id: string;
  label: string;
  /** The line's last point, in pixels. */
  x: number;
  y: number;
  /** Index of this label's own line in `curves`, which a label may cover before it is sent to the gutter. */
  curve?: number;
  /** The label's measured width, px; `labelWidth` estimates it when absent. */
  w?: number;
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
  /** The label's width, px (`labelWidth`). */
  w: number;
}

export interface PlaceOptions {
  /** Clear space between two labels' boxes, px: LABEL_GAP, or more (a smaller value is raised to it). */
  gap?: number;
  /** How far right of the plot the gutter column starts when a label there sits off its line's end (room for a leader). */
  lead?: number;
  /** Clear space between a label's box and any other line, px: CLEAR, or more. */
  clear?: number;
}

/** The least clear space between two end labels' boxes, px (DESIGN_DIRECTION §2.4). */
export const LABEL_GAP = 12;

/** A label's box height, px: the 16 px line of the 12 px label text, as the browser draws it. */
export const LABEL_H = 16;
/** The narrowest plot that draws direct labels: below it (every phone) the legend carries names, values and status. */
export const END_LABELS_MIN = 560;

/**
 * The least clear space between a label's box and another line, px: a label 2 px off another run's line reads as
 * sitting on it (round 24: "fasteners-s · live" on shapes-hard-s's line). Its own line may come as close as `pad`.
 */
export const CLEAR = 6;

/** A label whose box is further than this (px) from its line's end marker gets a leader. */
const BESIDE = 8;
/** The furthest a label's box may sit from its line's end (px); beyond it the label is left out. */
export const MAX_REACH = 64;

/**
 * The leader tying a label to its line's end: from just outside the end marker (`r` px) to just outside the label's
 * nearest edge, or null when the label sits right beside the marker.
 */
export function leaderOf(p: Placed, r = 5): { x1: number; y1: number; x2: number; y2: number } | null {
  const nx = Math.min(Math.max(p.ax, p.x - 3), p.x + p.w + 3);
  const ny = Math.min(Math.max(p.ay, p.y - H / 2 - 2), p.y + H / 2 + 2);
  const d = Math.hypot(nx - p.ax, ny - p.ay);
  if (d <= BESIDE) return null;
  return { x1: p.ax + ((nx - p.ax) / d) * r, y1: p.ay + ((ny - p.ay) / d) * r, x2: nx, y2: ny };
}

/** Whether a placed label sits away from its line's end, so it needs a leader to show which line it names. */
export const needsLeader = (p: Placed) => leaderOf(p) !== null;

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

type Pt = { x: number; y: number };

/** Approximate width of `text` at the chart's 12 px label size. */
export const labelWidth = (text: string) => text.length * 6.8 + 4;

const H = LABEL_H;

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

/** Whether segments a–b and c–d cross (or touch). */
function segmentsCross(a: Pt, b: Pt, c: Pt, d: Pt): boolean {
  const o = (p: Pt, q: Pt, r: Pt) => Math.sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x));
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b);
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
  const gap = Math.max(opts.gap ?? LABEL_GAP, LABEL_GAP);
  const taken: Box[] = [];
  const spots = new Map<number, Box>();
  const placed = new Map<number, Placed>();
  const clear = Math.max(opts.clear ?? CLEAR, pad);
  /** The label's own line: `curve`, else the curve that ends at its point. */
  const ownOf = (e: EndPoint) => {
    if (e.curve !== undefined) return e.curve;
    const k = curves.findIndex((c) => {
      const p = c[c.length - 1];
      return !!p && Math.abs(p.x - e.x) < 0.5 && Math.abs(p.y - e.y) < 0.5;
    });
    return k < 0 ? undefined : k;
  };
  /**
   * Whether box `b` can hold `e`'s label: inside the bounds, LABEL_GAP from every other label, `pad` px off its own
   * line (or over it, with `overOwn`) and `clear` px off every other line, so a label never sits on another run's line;
   * and the leader from its line's end, where it needs one, crosses no other line.
   */
  const fits = (b: Box, e: EndPoint, overOwn = false) => {
    if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) return false;
    const room = { x0: b.x0 - gap, y0: b.y0 - gap, x1: b.x1 + gap, y1: b.y1 + gap };
    if (taken.some((t) => overlaps(t, room))) return false;
    const own = ownOf(e);
    const lead = leaderOf({ id: e.id, label: e.label, x: b.x0, y: (b.y0 + b.y1) / 2, ax: e.x, ay: e.y, w: b.x1 - b.x0 });
    // Where the plot's edge is known, a label inside the plot sits beside its marker: only the gutter column, right of
    // every line, ties a label to its line with a leader (round 24: a label floating mid-plot on a leader).
    if (lead && gutter !== undefined && b.x0 < gutter) return false;
    for (let k = 0; k < curves.length; k++) {
      if (k === own && overOwn) continue;
      // In the gutter, right of where every line ends, the lines' ends are beside the column, not under it: `pad`.
      const m = k === own || (gutter !== undefined && b.x0 >= gutter) ? pad : clear;
      const g = { x0: b.x0 - m, y0: b.y0 - m, x1: b.x1 + m, y1: b.y1 + m };
      const c = curves[k]!;
      for (let i = 1; i < c.length; i++) {
        if (segmentHits(c[i - 1]!, c[i]!, g)) return false;
        if (lead && k !== own && segmentsCross(c[i - 1]!, c[i]!, { x: lead.x1, y: lead.y1 }, { x: lead.x2, y: lead.y2 })) return false;
      }
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
  for (let k = 0; k < LEVELS; k++) rounds.push((e, w) => (atEdge(e) ? undefined : ring(e, w, k).find((b) => !away(e, b, k) && fits(b, e))));
  rounds.push((e, w) => (ownOf(e) === undefined || atEdge(e) ? undefined : ring(e, w, 0).find((b) => fits(b, e, true))));
  const take = (i: number, spot: Box) => {
    const e = ends[i]!;
    taken.push(spot);
    spots.set(i, spot);
    placed.set(i, { id: e.id, label: e.label, x: spot.x0, y: (spot.y0 + spot.y1) / 2, ax: e.x, ay: e.y, w: spot.x1 - spot.x0 });
  };
  const run = (round: (e: EndPoint, w: number) => Box | undefined) =>
    ends.forEach((e, i) => {
      if (placed.has(i)) return;
      const spot = round(e, e.w ?? labelWidth(e.label));
      if (spot) take(i, spot);
    });
  for (const round of rounds) run(round);
  if (gutter !== undefined && placed.size < ends.length) column(gutter);
  // Whatever is still without a spot: the gutter at the free height nearest its line's end, within reach.
  run((e, w) => {
    // As near the line's end as the gutter allows: just right of it, or right-aligned in the bounds.
    const x0 = Math.max(bounds.x0, Math.min(bounds.x1 - w, Math.max(e.x + 6, gutter ?? e.x + 6)));
    if (x0 - e.x > MAX_REACH) return undefined;
    for (let d = 0; d <= Math.min(MAX_REACH, bounds.y1 - bounds.y0); d += 2)
      for (const y of d ? [e.y - d, e.y + d] : [e.y]) {
        const b = { x0, y0: y - H / 2, x1: x0 + w, y1: y + H / 2 };
        if (fits(b, e)) return b;
      }
    return undefined;
  });
  // Still without a spot: any clear box in the bounds, tied to its line's end by a leader that crosses the fewest other
  // lines (a line ending among the others, as Compare's failed shapes-hard-m), nearest the end. A label is never left
  // out while a clear box exists.
  run(lastResort);
  return ends.flatMap((_, i) => placed.get(i) ?? []);

  /** The clear box for `e`'s label whose leader crosses the fewest other lines, then the nearest (a 4 px grid). */
  function lastResort(e: EndPoint, w: number): Box | undefined {
    const own = ownOf(e);
    const spots: { b: Box; dist: number }[] = [];
    for (let cy = bounds.y0 + H / 2; cy <= bounds.y1 - H / 2; cy += 4)
      for (let x0 = bounds.x0; x0 + w <= bounds.x1; x0 += 4) {
        const b = { x0, y0: cy - H / 2, x1: x0 + w, y1: cy + H / 2 };
        const dx = Math.max(b.x0 - e.x, 0, e.x - b.x1);
        const dy = Math.max(b.y0 - e.y, 0, e.y - b.y1);
        spots.push({ b, dist: Math.hypot(dx, dy) });
      }
    spots.sort((a, b) => a.dist - b.dist);
    let best: { b: Box; crossed: number } | undefined;
    for (const { b } of spots) {
      if (best?.crossed === 0) break;
      if (!clearOf(b, e)) continue;
      const lead = leaderOf({ id: e.id, label: e.label, x: b.x0, y: (b.y0 + b.y1) / 2, ax: e.x, ay: e.y, w });
      let crossed = 0;
      if (lead) {
        const a = { x: lead.x1, y: lead.y1 };
        const z = { x: lead.x2, y: lead.y2 };
        for (let k = 0; k < curves.length; k++) {
          if (k === own) continue;
          const c = curves[k]!;
          for (let i = from(c, Math.min(a.x, z.x)); i < c.length && c[i - 1]!.x <= Math.max(a.x, z.x); i++)
            if (segmentsCross(c[i - 1]!, c[i]!, a, z)) {
              crossed++;
              break;
            }
        }
      }
      if (!best || crossed < best.crossed) best = { b, crossed };
    }
    return best?.b;
  }

  /** The first segment (its end index, at least 1) of an x-sorted curve that can reach `x0`. */
  function from(c: readonly Pt[], x0: number): number {
    let lo = 1;
    let hi = c.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (c[mid]!.x < x0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Box `b` in the bounds, LABEL_GAP from every label, `pad` off its own line and `clear` off every other. */
  function clearOf(b: Box, e: EndPoint): boolean {
    if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) return false;
    const room = { x0: b.x0 - gap, y0: b.y0 - gap, x1: b.x1 + gap, y1: b.y1 + gap };
    if (taken.some((t) => overlaps(t, room))) return false;
    const own = ownOf(e);
    for (let k = 0; k < curves.length; k++) {
      const m = k === own ? pad : clear;
      const g = { x0: b.x0 - m, y0: b.y0 - m, x1: b.x1 + m, y1: b.y1 + m };
      const c = curves[k]!;
      for (let i = from(c, g.x0); i < c.length && c[i - 1]!.x <= g.x1; i++) if (segmentHits(c[i - 1]!, c[i]!, g)) return false;
    }
    return true;
  }

  /**
   * The labels left without a spot and those already beside their line's end in the gutter, re-laid as one column in
   * the order of their ends: each group of labels that would overlap is centred on its lines' ends and stacked one label
   * height apart, inside the bounds. Kept only if every label then fits; otherwise nothing changes.
   */
  function column(x: number) {
    // Only lines ending within reach of the gutter: a label far from its line's end is left out instead.
    const members = ends.map((_, i) => i).filter((i) => (!placed.has(i) && x - ends[i]!.x <= MAX_REACH) || (placed.has(i) && spots.get(i)!.x0 >= x));
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
    for (const [i, y] of ys) next.set(i, { x0, y0: y - H / 2, x1: x0 + (ends[i]!.w ?? labelWidth(ends[i]!.label)), y1: y + H / 2 });
    const ok = [...next].every(([i, b]) => fits(b, ends[i]!));
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
