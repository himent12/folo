/** A line chart's readings under the pointer (LineChart): the cursor's rows, the line a pointer picks, the x pill. */
import { nearest, type Pt } from './downsample';
import type { Scale } from './lineAxes';
import type { Series } from './lineSeries';

/** What the cursor reads: each keyed series' nearest point (only where it has data), under one x title. */
export function hoverOf(
  cursor: number | null | undefined,
  hasData: boolean,
  xExtent: [number, number],
  keyed: Series[],
  xTitle: ((x: number) => string) | undefined,
  xFormat: (x: number) => string,
  x: Scale,
) {
  if (cursor === null || cursor === undefined || !hasData) return null;
  if (cursor < xExtent[0] || cursor > xExtent[1]) return null;
  const c = cursor as number;
  const rows = keyed
    .map((s) => {
      const pts = s.points;
      const first = pts[0];
      const last = pts[pts.length - 1];
      if (!first || !last) return null;
      // A series only reads out where it has data: never its last value under a cursor past its end.
      const tol = pts.length > 1 ? (last.x - first.x) / (pts.length - 1) / 2 : 0.5;
      if (c < first.x - tol || c > last.x + tol) return null;
      const p = pts[nearest(pts, c)];
      return p ? { s, p } : null;
    })
    .filter((r): r is { s: Series; p: Pt } => r !== null);
  if (!rows.length) return null;
  // Title at the series with the densest sampling (closest to the cursor); rows sampled elsewhere say where.
  const ref = rows.reduce((a, b) => (Math.abs(b.p.x - c) < Math.abs(a.p.x - c) ? b : a));
  const title = xTitle ? xTitle(ref.p.x) : xFormat(ref.p.x);
  const labelled = rows.map((r) => {
    const t = xTitle ? xTitle(r.p.x) : xFormat(r.p.x);
    return { ...r, at: t === title ? null : t };
  });
  return { rows: labelled, px: x(ref.p.x), xv: ref.p.x, title };
}

/** The pickable series whose drawn line passes nearest (px, py), within `r` px; null for none. */
export function pickNearest(
  pick: ((s: Series) => boolean) | undefined,
  inPlot: (px: number) => boolean,
  visible: Series[],
  x: Scale,
  y: Scale,
  px: number,
  py: number,
  r: number,
): string | null {
  if (!pick || !inPlot(px)) return null;
  let best: string | null = null;
  let bd = r;
  for (const s of visible) {
    if (!pick(s)) continue;
    const pts = s.points;
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1]!;
      const b = pts[i]!;
      if (!Number.isFinite(a.y) || !Number.isFinite(b.y)) continue;
      const ax = x(a.x);
      const bx = x(b.x);
      if (Math.max(ax, bx) < px - bd || Math.min(ax, bx) > px + bd) continue;
      const ay = y(a.y);
      const by = y(b.y);
      // Distance from the point to the segment.
      const dx = bx - ax;
      const dy = by - ay;
      const t = dx || dy ? Math.min(1, Math.max(0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy))) : 0;
      const d = Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
      if (d < bd) {
        bd = d;
        best = s.id;
      }
    }
  }
  return best;
}

/** Narrow plots: the cursor's x title as a pill over the x ticks it covers. */
export function pillOf(compact: boolean, hover: { title: string; px: number } | null, width: number) {
  if (!compact || !hover) return null;
  const w = Math.ceil(hover.title.length * 6.8) + 12;
  const cx = Math.min(width - w / 2 - 1, Math.max(w / 2 + 1, hover.px));
  return { w, cx, text: hover.title };
}
