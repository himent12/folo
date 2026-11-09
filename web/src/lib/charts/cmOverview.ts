// The whole-matrix overview shown beside a confusion matrix that scrolls sideways (phones): every cell as a small
// square, so the pattern stays visible while the full matrix shows a few columns at a time. A tap brings those columns
// into view.

export interface OverviewGeometry {
  /** Cell side and gap between cells, in px. */
  cell: number;
  gap: number;
  /** Side of the whole square, in px. */
  size: number;
}

/** Cells as large as fit `maxSize` (at least `minCell` px, 6 by default; at most 20 px) for an n × n matrix. */
export function overviewGeometry(n: number, maxSize = 168, minCell = 6): OverviewGeometry {
  const gap = 1;
  const k = Math.max(1, n);
  const cell = Math.max(minCell, Math.min(20, Math.floor((maxSize + gap) / k) - gap));
  return { cell, gap, size: k * cell + (k - 1) * gap };
}

/** The cell under a point of the overview (px from its top-left corner), or null outside it. A gap counts as the cell before it. */
export function overviewHit(px: number, py: number, g: OverviewGeometry, n: number): { r: number; c: number } | null {
  if (n < 1 || px < 0 || py < 0 || px >= g.size || py >= g.size) return null;
  const step = g.cell + g.gap;
  return { r: Math.min(n - 1, Math.floor(py / step)), c: Math.min(n - 1, Math.floor(px / step)) };
}

/** Left edge and width (px) of the band covering columns `first`..`last` (1-based, inclusive): what the full matrix shows. */
export function overviewWindow(first: number, last: number, g: OverviewGeometry): { x: number; w: number } {
  const step = g.cell + g.gap;
  return { x: (first - 1) * step, w: (last - first + 1) * step - g.gap };
}

/**
 * Whole columns per page of a matrix that scrolls sideways: how many columns of at least `minCell` px fit beside the
 * pinned row labels (`head` px, then a `gap`), and the cell size that makes exactly that many fill the width, so no
 * column is ever half in view (round 24: "Columns 1–6 of 9" with column 7 cut through its "71"). Null when every
 * column fits without scrolling.
 */
export function pageFit(width: number, head: number, minCell: number, gap: number, n: number): { cols: number; cell: number } | null {
  const room = width - head;
  const cols = Math.max(1, Math.floor(room / (minCell + gap)));
  if (cols >= n) return null;
  return { cols, cell: Math.floor((room / cols - gap) * 100) / 100 };
}

/** The columns in view (1-based, inclusive) at a scroll offset, when pages hold `cols` whole columns of `step` px. */
export function pageRange(scrollLeft: number, step: number, cols: number, n: number): { first: number; last: number } {
  const first = Math.min(Math.max(1, n - cols + 1), Math.max(1, Math.round(scrollLeft / step) + 1));
  return { first, last: Math.min(n, first + cols - 1) };
}
