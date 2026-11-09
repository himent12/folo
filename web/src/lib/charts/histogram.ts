// Histogram geometry shared by Histogram.svelte, pure so it can be tested.

/**
 * `n` evenly spaced category bars from px `x0` to `x1`, as `{ x, w }` (each bin one category, whatever its value span:
 * the bar's height carries the count, its label the range).
 */
export function categoryBars(n: number, x0: number, x1: number): { x: number; w: number }[] {
  const w = n > 0 ? Math.max(0, x1 - x0) / n : 0;
  return Array.from({ length: n }, (_, i) => ({ x: x0 + i * w, w }));
}

/**
 * Pixel x of value `v` on a category axis: a bin edge sits on the boundary between its two bars; a value inside a bin
 * lies proportionally inside that bin's bar. Clamped to the first and last bar.
 */
export function edgeX(edges: readonly number[], bars: readonly { x: number; w: number }[], v: number): number {
  const n = bars.length;
  if (!n) return 0;
  const first = bars[0]!;
  const last = bars[n - 1]!;
  if (v <= (edges[0] ?? 0)) return first.x;
  for (let i = 0; i < n; i++) {
    const lo = edges[i] ?? 0;
    const hi = edges[i + 1] ?? lo;
    if (v < hi) {
      const b = bars[i]!;
      return b.x + ((v - lo) / Math.max(1e-12, hi - lo)) * b.w;
    }
  }
  return last.x + last.w;
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
