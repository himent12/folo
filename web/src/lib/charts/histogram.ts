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
