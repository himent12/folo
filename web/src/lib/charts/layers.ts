// Paint order of a line chart's series: context and ghost lines at the back, emphasised lines in front. Among the
// ordinary lines, solid ones go first and patterned (dashed, dotted) ones over them, so a patterned line that runs
// along a solid one keeps its pattern instead of disappearing under it; within each, the shorter ones last, so a run
// that stopped early is drawn over the longer runs it would otherwise disappear under. Ties keep the caller's order.

export interface Layered {
  points: readonly { x: number }[];
  ghost?: boolean;
  context?: boolean;
  emphasis?: boolean;
  dashed?: boolean;
  pattern?: 'dash' | 'dot' | 'dashdot';
}

const layer = (s: Layered) => (s.ghost || s.context ? 0 : s.emphasis ? 3 : s.pattern || s.dashed ? 2 : 1);
const end = (s: Layered) => s.points[s.points.length - 1]?.x ?? -Infinity;

/** `items` in paint order (back to front); `of` picks each item's series. */
export function paintOrder<T>(items: readonly T[], of: (t: T) => Layered): T[] {
  const ordinary = (s: Layered) => layer(s) === 1 || layer(s) === 2;
  return items
    .map((t, i) => ({ t, i, s: of(t) }))
    .sort((a, b) => layer(a.s) - layer(b.s) || (ordinary(a.s) ? end(b.s) - end(a.s) : 0) || a.i - b.i)
    .map((o) => o.t);
}
