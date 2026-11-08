// Precision-recall curve helpers for the Model view.

/**
 * Drops the tail of a PR curve past its maximum recall: evaluators pad the curve with precision 0 up to recall 1,
 * which would draw a line along the x-axis that no detector reached. Keeps everything up to the last point with
 * precision above zero (an all-zero curve stays empty).
 */
export function trimPrTail<P extends { y: number }>(points: readonly P[]): P[] {
  let end = points.length;
  while (end > 0 && !(points[end - 1]!.y > 0)) end--;
  return points.slice(0, end);
}

/** Case-insensitive substring match of a class name against a filter query (an empty query matches all). */
export function matchesClass(name: string, query: string): boolean {
  const q = query.trim().toLowerCase();
  return !q || name.toLowerCase().includes(q);
}

/** Interpolated precision of a PR curve at recall `r`: the best precision at any recall >= r (0 past its end). */
export function precisionAt(curve: readonly { x: number; y: number }[], r: number): number {
  let best = 0;
  for (const p of curve) if (p.x >= r - 1e-9 && p.y > best) best = p.y;
  return best;
}

/**
 * The spread of many PR curves: at each recall step, the `lo` and `hi` quantiles of the classes' interpolated
 * precision. Drawn as a band behind the mean, it shows how classes vary without 80 overlapping lines.
 */
export function prEnvelope(
  curves: readonly (readonly { x: number; y: number }[])[],
  lo = 0.1,
  hi = 0.9,
  steps = 101,
): { lower: { x: number; y: number }[]; upper: { x: number; y: number }[] } {
  const lower: { x: number; y: number }[] = [];
  const upper: { x: number; y: number }[] = [];
  const q = (sorted: number[], f: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(f * (sorted.length - 1))))] ?? 0;
  for (let i = 0; i < steps; i++) {
    const r = i / (steps - 1);
    const v = curves.map((c) => precisionAt(c, r)).sort((a, b) => a - b);
    lower.push({ x: r, y: q(v, lo) });
    upper.push({ x: r, y: q(v, hi) });
  }
  return { lower, upper };
}
