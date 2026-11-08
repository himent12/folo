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
