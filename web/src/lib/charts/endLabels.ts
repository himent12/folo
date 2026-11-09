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
// (`lastResort`, a bounded search: a few milliseconds on a crowded comparison). Only a label with no clear box in that
// search is left out. Curves are tested through an index by x (`Obstacles`), and `memoPlaceEndLabels` places again
// only when what it places changes, so a chart does not redo the work on every render or animation frame.

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
  // The four edges as (p, q) pairs, unrolled: this runs for every candidate box against every segment under it.
  for (let s = 0; s < 4; s++) {
    const p = s === 0 ? -dx : s === 1 ? dx : s === 2 ? -dy : dy;
    const q = s === 0 ? a.x - r.x0 : s === 1 ? r.x1 - a.x : s === 2 ? a.y - r.y0 : r.y1 - a.y;
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

/** Width of a column of the obstacle index, px. */
const COL = 16;
/** Side of a cell of the index's summed-area table, px. */
const CELL = 2;

/**
 * Every curve's segments bucketed into COL px columns by x, with each column's y extent per curve, so a box or a leader
 * is tested only against the segments under it, and not at all where a curve's column misses its height (round 26
 * review: the last resort tested every segment of every curve for every candidate, ~240 ms for four 150-point lines).
 * Exact: a box or leader that touches a segment always shares one of its columns.
 */
class Obstacles {
  private readonly n: number;
  /** Per curve, per column: the segments (end index i: c[i-1]–c[i]) that reach into it. */
  private readonly segs: (number[] | undefined)[][];
  /** Per curve, per column: the y extent of those segments within the column. */
  private readonly lo: Float64Array[];
  private readonly hi: Float64Array[];

  constructor(
    private readonly curves: readonly (readonly Pt[])[],
    private readonly x0: number,
    x1: number,
  ) {
    this.n = Math.max(1, Math.ceil((x1 - x0) / COL) + 1);
    this.segs = curves.map(() => []);
    this.lo = curves.map(() => new Float64Array(this.n).fill(Infinity));
    this.hi = curves.map(() => new Float64Array(this.n).fill(-Infinity));
    curves.forEach((c, k) => {
      const segs = this.segs[k]!;
      const lo = this.lo[k]!;
      const hi = this.hi[k]!;
      for (let i = 1; i < c.length; i++) {
        const a = c[i - 1]!;
        const b = c[i]!;
        const ax = Math.min(a.x, b.x);
        const bx = Math.max(a.x, b.x);
        const yAt = (x: number) => (b.x === a.x ? NaN : a.y + ((x - a.x) / (b.x - a.x)) * (b.y - a.y));
        for (let j = this.col(ax); j <= this.col(bx); j++) {
          (segs[j] ??= []).push(i);
          // The segment's height over the part of the column it spans (all of it for a vertical one, or at the edges,
          // where the column clamps).
          const sx = Math.max(ax, this.x0 + j * COL);
          const ex = Math.min(bx, this.x0 + (j + 1) * COL);
          const edge = j === 0 || j === this.n - 1;
          const y0 = edge ? NaN : yAt(sx);
          const y1 = edge ? NaN : yAt(ex);
          const ylo = Number.isNaN(y0 + y1) ? Math.min(a.y, b.y) : Math.min(y0, y1);
          const yhi = Number.isNaN(y0 + y1) ? Math.max(a.y, b.y) : Math.max(y0, y1);
          lo[j] = Math.min(lo[j]!, ylo - 1e-6);
          hi[j] = Math.max(hi[j]!, yhi + 1e-6);
        }
      }
    });
  }

  /** Counts of points sampled along every curve, summed over cells (lazily, for `surelyHit`). */
  private sat?: { x0: number; y0: number; w: number; h: number; sum: Uint32Array };

  /**
   * Whether some curve surely passes through box `g`: a point sampled along it (1 px apart) lies in a CELL px cell
   * inside the box. O(1) from a summed-area table, so the last resort's thousands of boxes over a crowded plot are
   * turned down without testing segments; false says nothing (test the segments).
   */
  surelyHit(g: Box, area: Box): boolean {
    const t = (this.sat ??= this.table(area));
    const cx0 = Math.max(0, Math.ceil((g.x0 + 1e-6 - t.x0) / CELL));
    const cy0 = Math.max(0, Math.ceil((g.y0 + 1e-6 - t.y0) / CELL));
    const cx1 = Math.min(t.w, Math.floor((g.x1 - 1e-6 - t.x0) / CELL));
    const cy1 = Math.min(t.h, Math.floor((g.y1 - 1e-6 - t.y0) / CELL));
    if (cx1 <= cx0 || cy1 <= cy0) return false;
    const W = t.w + 1;
    return t.sum[cy1 * W + cx1]! - t.sum[cy0 * W + cx1]! - t.sum[cy1 * W + cx0]! + t.sum[cy0 * W + cx0]! > 0;
  }

  private table(area: Box) {
    const w = Math.max(1, Math.ceil((area.x1 - area.x0) / CELL));
    const h = Math.max(1, Math.ceil((area.y1 - area.y0) / CELL));
    const W = w + 1;
    const sum = new Uint32Array(W * (h + 1));
    for (const c of this.curves)
      for (let i = 1; i < c.length; i++) {
        const a = c[i - 1]!;
        const b = c[i]!;
        const n = Math.ceil(Math.hypot(b.x - a.x, b.y - a.y));
        for (let s = 0; s <= n; s++) {
          const f = n ? s / n : 0;
          const cx = Math.floor((a.x + (b.x - a.x) * f - area.x0) / CELL);
          const cy = Math.floor((a.y + (b.y - a.y) * f - area.y0) / CELL);
          if (cx >= 0 && cx < w && cy >= 0 && cy < h) sum[(cy + 1) * W + cx + 1]!++;
        }
      }
    for (let y = 1; y <= h; y++)
      for (let x = 1; x <= w; x++) sum[y * W + x]! += sum[(y - 1) * W + x]! + sum[y * W + x - 1]! - sum[(y - 1) * W + x - 1]!;
    return { x0: area.x0, y0: area.y0, w, h, sum };
  }

  private col(x: number): number {
    return Math.min(this.n - 1, Math.max(0, Math.floor((x - this.x0) / COL)));
  }

  /** Whether curve `k` crosses (or touches) box `g`. */
  hitsBox(k: number, g: Box): boolean {
    const c = this.curves[k]!;
    const segs = this.segs[k]!;
    const lo = this.lo[k]!;
    const hi = this.hi[k]!;
    for (let j = this.col(g.x0), z = this.col(g.x1); j <= z; j++) {
      if (hi[j]! < g.y0 || lo[j]! > g.y1) continue;
      for (const i of segs[j]!) if (segmentHits(c[i - 1]!, c[i]!, g)) return true;
    }
    return false;
  }

  /** Whether curve `k` crosses (or touches) segment a–b. */
  crosses(k: number, a: Pt, b: Pt): boolean {
    const c = this.curves[k]!;
    const segs = this.segs[k]!;
    const lo = this.lo[k]!;
    const hi = this.hi[k]!;
    const y0 = Math.min(a.y, b.y);
    const y1 = Math.max(a.y, b.y);
    for (let j = this.col(Math.min(a.x, b.x)), z = this.col(Math.max(a.x, b.x)); j <= z; j++) {
      if (hi[j]! < y0 || lo[j]! > y1) continue;
      for (const i of segs[j]!) if (segmentsCross(c[i - 1]!, c[i]!, a, b)) return true;
    }
    return false;
  }
}

/** The last resort's search (`lastResort`): a 4 px grid within NEAR px of the line's end, nearest first, ... */
const NEAR = 96;
/** ... then an 8 px grid over the rest of the bounds; at most SPOTS boxes are tried, and TRIES clear ones' leaders. */
const SPOTS = 6000;
const TRIES = 400;

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
  // Labels stay inside the bounds and the furthest margin is `clear`: the index covers that much either side.
  const obstacles = new Obstacles(curves, bounds.x0 - clear - COL, bounds.x1 + clear + COL);
  const area = { x0: bounds.x0 - pad, y0: bounds.y0 - pad, x1: bounds.x1 + pad, y1: bounds.y1 + pad };
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
      if (obstacles.hitsBox(k, { x0: b.x0 - m, y0: b.y0 - m, x1: b.x1 + m, y1: b.y1 + m })) return false;
      if (lead && k !== own && obstacles.crosses(k, { x: lead.x1, y: lead.y1 }, { x: lead.x2, y: lead.y2 })) return false;
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

  /**
   * The clear box for `e`'s label whose leader crosses the fewest other lines, then the nearest: a 4 px grid within NEAR
   * px of the line's end first, nearest first, then an 8 px grid over the rest of the bounds. The first clear box whose
   * leader crosses nothing ends the search; after TRIES clear boxes, or SPOTS boxes in all, the best so far is kept, so
   * a crowded chart's last resort stays a few milliseconds (round 26 review).
   */
  function lastResort(e: EndPoint, w: number): Box | undefined {
    const own = ownOf(e);
    let best: { b: Box; crossed: number } | undefined;
    let tries = 0;
    let left = SPOTS;
    for (const [step, near] of [
      [4, true],
      [8, false],
    ] as const) {
      const cols = Math.max(0, Math.floor((bounds.x1 - w - bounds.x0) / step) + 1);
      const rows = Math.max(0, Math.floor((bounds.y1 - bounds.y0 - H) / step) + 1);
      // The near pass only looks at the columns within NEAR of the line's end.
      const c0 = near ? Math.max(0, Math.ceil((e.x - w - NEAR - bounds.x0) / step)) : 0;
      const c1 = near ? Math.min(cols, Math.floor((e.x + NEAR - bounds.x0) / step) + 1) : cols;
      // Nearest first, by a counting sort on the distance in 1/16 px (the grid's own order breaks ties).
      const at = new Uint32Array(Math.max(0, rows * (c1 - c0)));
      const dist = new Uint32Array(at.length);
      let n = 0;
      let far = 0;
      for (let r = 0; r < rows; r++) {
        const cy = bounds.y0 + H / 2 + r * step;
        const dy = Math.max(cy - H / 2 - e.y, 0, e.y - cy - H / 2);
        if (near && dy > NEAR) continue;
        for (let c = c0; c < c1; c++) {
          const x0 = bounds.x0 + c * step;
          const d = Math.hypot(Math.max(x0 - e.x, 0, e.x - x0 - w), dy);
          if (d <= NEAR !== near) continue;
          at[n] = r * cols + c;
          dist[n] = Math.round(d * 16);
          far = Math.max(far, dist[n++]!);
        }
      }
      const starts = new Uint32Array(far + 2);
      for (let i = 0; i < n; i++) starts[dist[i]! + 1]!++;
      for (let d = 1; d < starts.length; d++) starts[d]! += starts[d - 1]!;
      const order = new Uint32Array(n);
      for (let i = 0; i < n; i++) order[starts[dist[i]!]!++] = at[i]!;
      for (const k of order) {
        if (left-- <= 0) return best?.b;
        const x0 = bounds.x0 + (k % cols) * step;
        const cy = bounds.y0 + H / 2 + Math.floor(k / cols) * step;
        const b = { x0, y0: cy - H / 2, x1: x0 + w, y1: cy + H / 2 };
        if (!clearOf(b, e)) continue;
        const lead = leaderOf({ id: e.id, label: e.label, x: b.x0, y: cy, ax: e.x, ay: e.y, w });
        let crossed = 0;
        if (lead) {
          const a = { x: lead.x1, y: lead.y1 };
          const z = { x: lead.x2, y: lead.y2 };
          for (let k = 0; k < curves.length; k++) if (k !== own && obstacles.crosses(k, a, z)) crossed++;
        }
        if (!best || crossed < best.crossed) best = { b, crossed };
        if (best.crossed === 0 || ++tries >= TRIES) return best.b;
      }
    }
    return best?.b;
  }

  /** Box `b` in the bounds, LABEL_GAP from every label, `pad` off its own line and `clear` off every other. */
  function clearOf(b: Box, e: EndPoint): boolean {
    if (b.x0 < bounds.x0 || b.x1 > bounds.x1 || b.y0 < bounds.y0 || b.y1 > bounds.y1) return false;
    const room = { x0: b.x0 - gap, y0: b.y0 - gap, x1: b.x1 + gap, y1: b.y1 + gap };
    if (taken.some((t) => overlaps(t, room))) return false;
    // Every line keeps at least `pad` off the box: one surely inside it rules the box out without testing segments.
    if (obstacles.surelyHit({ x0: b.x0 - pad, y0: b.y0 - pad, x1: b.x1 + pad, y1: b.y1 + pad }, area)) return false;
    const own = ownOf(e);
    for (let k = 0; k < curves.length; k++) {
      const m = k === own ? pad : clear;
      if (obstacles.hitsBox(k, { x0: b.x0 - m, y0: b.y0 - m, x1: b.x1 + m, y1: b.y1 + m })) return false;
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

/**
 * `placeEndLabels` that remembers its last call: given the same ends, curves, bounds and options (equal values, not
 * the same arrays), it returns the last placement without placing again. One per chart; the caller passes new arrays
 * rather than changing the old ones in place.
 */
export function memoPlaceEndLabels(): typeof placeEndLabels {
  let last: { args: Parameters<typeof placeEndLabels>; out: Placed[] } | undefined;
  const samePt = (a: Pt, b: Pt) => a.x === b.x && a.y === b.y;
  const sameEnd = (a: EndPoint, b: EndPoint) =>
    a.id === b.id && a.label === b.label && a.x === b.x && a.y === b.y && a.curve === b.curve && a.w === b.w;
  const sameList = <T>(a: readonly T[], b: readonly T[], eq: (p: T, q: T) => boolean) =>
    a === b || (a.length === b.length && a.every((v, i) => eq(v, b[i]!)));
  const same = (a: Parameters<typeof placeEndLabels>, b: Parameters<typeof placeEndLabels>) =>
    sameList(a[0], b[0], sameEnd) &&
    sameList(a[1], b[1], (p, q) => sameList(p, q, samePt)) &&
    a[2].x0 === b[2].x0 &&
    a[2].y0 === b[2].y0 &&
    a[2].x1 === b[2].x1 &&
    a[2].y1 === b[2].y1 &&
    a[3] === b[3] &&
    a[4] === b[4] &&
    a[5]?.gap === b[5]?.gap &&
    a[5]?.lead === b[5]?.lead &&
    a[5]?.clear === b[5]?.clear;
  return (...args) => {
    if (last && same(last.args, args)) return last.out;
    last = { args, out: placeEndLabels(...args) };
    return last.out;
  };
}
