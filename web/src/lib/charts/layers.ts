// Paint order of a line chart's series: context and ghost lines at the back, emphasised lines in front, and among the
// ordinary lines the shorter ones last, so a run that stopped early is drawn over the longer runs it would otherwise
// disappear under. Ties keep the caller's order.

export interface Layered {
  points: readonly { x: number }[];
  ghost?: boolean;
  context?: boolean;
  emphasis?: boolean;
}

const layer = (s: Layered) => (s.ghost || s.context ? 0 : s.emphasis ? 2 : 1);
const end = (s: Layered) => s.points[s.points.length - 1]?.x ?? -Infinity;

/** `items` in paint order (back to front); `of` picks each item's series. */
export function paintOrder<T>(items: readonly T[], of: (t: T) => Layered): T[] {
  return items
    .map((t, i) => ({ t, i, s: of(t) }))
    .sort((a, b) => layer(a.s) - layer(b.s) || (layer(a.s) === 1 ? end(b.s) - end(a.s) : 0) || a.i - b.i)
    .map((o) => o.t);
}
