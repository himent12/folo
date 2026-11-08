// Fits a comma-separated list on one line by whole items, so a cut list never ends on a dangling separator.

/**
 * How many leading items fit in `avail` px. `widths` are the item widths, `sep` the separator's width and `more(n)`
 * the width of the "+n" that follows a cut list. At least one item is always shown (it truncates on its own).
 */
export function fitCount(widths: readonly number[], sep: number, more: (rest: number) => number, avail: number): number {
  const n = widths.length;
  let used = 0;
  for (let k = 0; k < n; k++) {
    used += (k ? sep : 0) + widths[k]!;
    const tail = k + 1 < n ? more(n - k - 1) : 0;
    if (used + tail > avail) return Math.max(1, k);
  }
  return n;
}
