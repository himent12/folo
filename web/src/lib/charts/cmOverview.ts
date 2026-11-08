// The whole-matrix overview shown beside a confusion matrix that scrolls sideways (phones): every cell as a small
// square, so the pattern stays visible while the full matrix shows a few columns at a time. A tap on it jumps there.

export interface OverviewGeometry {
  /** Cell side and gap between cells, in px. */
  cell: number;
  gap: number;
  /** Side of the whole square, in px. */
  size: number;
}

/** Cells as large as fit `maxSize` (at least 6 px, at most 20 px) for an n × n matrix. */
export function overviewGeometry(n: number, maxSize = 168): OverviewGeometry {
  const gap = 1;
  const k = Math.max(1, n);
  const cell = Math.max(6, Math.min(20, Math.floor((maxSize + gap) / k) - gap));
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
