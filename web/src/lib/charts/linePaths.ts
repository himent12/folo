/** A line chart's drawn shapes (LineChart): each series as SVG paths, and which best markers sit on a line's end. */
import { area as d3area, curveLinear, curveMonotoneX, line as d3line } from 'd3-shape';
import { downsample, type Pt } from './downsample';
import { LIVE_GROW, mergedEnds } from './endMarks';
import type { Margin, Scale } from './lineAxes';
import type { Marker, Series } from './lineSeries';

/** Each visible series as paths: its line (downsampled to the plot's width) and area, or its band. */
export function pathsOf(visible: Series[], curve: 'linear' | 'monotone', width: number, margin: Margin, height: number, x: Scale, y: Scale, yLog: boolean) {
  const c = curve === 'monotone' ? curveMonotoneX : curveLinear;
  const buckets = Math.max(32, Math.floor((width - margin.left - margin.right) / 1.5));
  const ok = (p: Pt) => Number.isFinite(p.x) && Number.isFinite(p.y) && (!yLog || p.y > 0);
  const ln = d3line<Pt>()
    .defined(ok)
    .x((p) => x(p.x))
    .y((p) => y(p.y))
    .curve(c);
  const ar = d3area<Pt>()
    .defined(ok)
    .x((p) => x(p.x))
    .y0(height - margin.bottom)
    .y1((p) => y(p.y))
    .curve(c);
  return visible.map((s) => {
    if (s.lower) {
      const lo = s.lower;
      const zipped = s.points.map((p, i) => ({ x: p.x, y: p.y, y0: lo[i]?.y ?? p.y }));
      const band = d3area<{ x: number; y: number; y0: number }>()
        .defined((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.y0) && (!yLog || (p.y > 0 && p.y0 > 0)))
        .x((p) => x(p.x))
        .y0((p) => y(p.y0))
        .y1((p) => y(p.y))
        .curve(c);
      return { s, d: '', a: '', b: band(zipped) ?? '' };
    }
    const pts = downsample(s.points, buckets);
    return { s, d: ln(pts) ?? '', a: s.area ? (ar(pts) ?? '') : '', b: '' };
  });
}

// A best ring on its line's end (or so close that the two would overlap) is drawn around the end mark, as one mark
// (./endMarks); apart, each keeps its own shape. Only markers in the x domain, and only where end marks are drawn.
export function mergesOf(endDots: boolean, hasData: boolean, markers: Marker[], keyed: Series[], x: Scale, y: Scale, xExtent: [number, number]): (string | null)[] {
  if (!endDots || !hasData) return markers.map(() => null);
  const ends = keyed.flatMap((s) => {
    const p = s.points[s.points.length - 1];
    // Outer radii as drawn below: the mark (3 px, the live dot LIVE_GROW more) with its 0.75 px halo.
    const py = p && Number.isFinite(p.y) ? y(p.y) : NaN;
    return p && Number.isFinite(py) ? [{ id: s.id, end: s.end, x: x(p.x), y: py, r: (s.end === 'live' ? 3 + LIVE_GROW : 3) + 0.75 }] : [];
  });
  const inDomain = (m: Marker) => m.x >= xExtent[0] && m.x <= xExtent[1];
  return mergedEnds(
    markers.map((m) => {
      const my = inDomain(m) && Number.isFinite(m.y) ? y(m.y) : NaN;
      return Number.isFinite(my) ? { x: x(m.x), y: my } : { x: -1e9, y: -1e9 };
    }),
    ends,
  );
}
