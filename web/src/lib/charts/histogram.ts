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
