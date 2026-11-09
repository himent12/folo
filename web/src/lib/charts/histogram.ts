// Histogram geometry shared by Histogram.svelte, pure so it can be tested.

/**
 * Bar heights for a histogram drawn with true bin widths: each bar's height is its count per `per` units of the
 * *displayed* span, so its area is its count. A last bin that reaches past `xMax` is drawn only up to `xMax`, and its
 * density uses that clipped span (otherwise a clipped tail would show a fraction of its count).
 * `per` defaults to the narrowest displayed bin.
 */
export function binDensities(edges: number[], counts: number[], xMax = Infinity, per?: number): number[] {
  const spans = counts.map((_, i) => {
    const lo = edges[i] ?? 0;
    const hi = Math.min(edges[i + 1] ?? lo, xMax);
    return Math.max(1e-12, hi - lo);
  });
  const unit = per ?? Math.min(...spans);
  return counts.map((c, i) => c / (spans[i]! / unit));
}

/**
 * The bin whose bar spans pixel `px` (bars as `{ x, w }` in px, left to right); outside every bar, the nearest one
 * within half a bar's width, else null. Gaps between bars belong to the nearer bar, so a scrub never flickers off.
 */
export function binAt(bars: readonly { x: number; w: number }[], px: number): number | null {
  let best: number | null = null;
  let dist = Infinity;
  bars.forEach((b, i) => {
    const d = px < b.x ? b.x - px : px > b.x + b.w ? px - (b.x + b.w) : 0;
    if (d < dist && d <= Math.max(2, b.w / 2)) {
      dist = d;
      best = i;
    }
  });
  return best;
}

/**
 * Bars on a linear value axis from px `x0` to `x1` covering `[edges[0], xMax]`, each at its true width. When the last
 * bin is open-ended (its upper edge past `xMax`), it is set apart: the axis breaks for `gap` px before it, and it is
 * drawn `xMax − its lower edge` wide at the same scale (so its area is still its count). `mainEnd` is where the
 * continuous axis stops; `tail` says whether the last bar is set apart.
 */
export function proportionalBars(
  edges: readonly number[],
  xMax: number,
  x0: number,
  x1: number,
  gap: number,
): { bars: { x: number; w: number }[]; mainEnd: number; tail: boolean } {
  const n = Math.max(0, edges.length - 1);
  const lo0 = edges[0] ?? 0;
  const hi = Math.min(edges[n] ?? 1, xMax);
  const lastLo = edges[n - 1] ?? lo0;
  const tail = n > 0 && (edges[n] ?? 0) > hi && lastLo < hi;
  const k = (x1 - x0 - (tail ? gap : 0)) / Math.max(1e-12, hi - lo0);
  const at = (v: number) => x0 + (Math.min(Math.max(v, lo0), hi) - lo0) * k;
  const bars = Array.from({ length: n }, (_, i) => {
    const a = at(edges[i] ?? 0);
    const w = Math.max(1, at(edges[i + 1] ?? 0) - a);
    return tail && i === n - 1 ? { x: a + gap, w } : { x: a, w };
  });
  return { bars, mainEnd: tail ? at(lastLo) : x1, tail };
}
