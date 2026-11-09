// Paint order of a line chart's series: context and ghost lines at the back, emphasised lines in front. Among the
// ordinary lines, **no series is hidden under the others**: a line that has ended (failed, stopped) goes over the
// lines that ran on, since it is the shorter one and its end is what people look for (round 27: Compare's failed
// dotted run, ~3 % of the plot, lay in the bundle of the others' climb); then, within each, solid ones first and
// patterned (dashed, dotted) ones over them, so a patterned line that runs along a solid one keeps its pattern instead
// of disappearing under it; then the shorter ones last, so a run that stopped early is drawn over the longer runs it
// would otherwise disappear under. A `front` line goes over every ordinary one (a solid line a dotted one would
// otherwise cover, e.g. box under dfl in the loss chart), still under the emphasised ones. Ties keep the caller's order.

export interface Layered {
  points: readonly { x: number }[];
  ghost?: boolean;
  context?: boolean;
  emphasis?: boolean;
  front?: boolean;
  dashed?: boolean;
  pattern?: 'dash' | 'dot' | 'dashdot';
  end?: 'live' | 'stopped' | 'failed';
}

/** 0 back (context, ghost); 1–4 ordinary (solid, patterned, ended solid, ended patterned); 5 front; 6 emphasised. */
const layer = (s: Layered) => {
  if (s.ghost || s.context) return 0;
  if (s.emphasis) return 6;
  if (s.front) return 5;
  return 1 + (s.pattern || s.dashed ? 1 : 0) + (s.end === 'failed' || s.end === 'stopped' ? 2 : 0);
};
const end = (s: Layered) => s.points[s.points.length - 1]?.x ?? -Infinity;

/** `items` in paint order (back to front); `of` picks each item's series. */
export function paintOrder<T>(items: readonly T[], of: (t: T) => Layered): T[] {
  const ordinary = (s: Layered) => layer(s) >= 1 && layer(s) <= 4;
  return items
    .map((t, i) => ({ t, i, s: of(t) }))
    .sort((a, b) => layer(a.s) - layer(b.s) || (ordinary(a.s) ? end(b.s) - end(a.s) : 0) || a.i - b.i)
    .map((o) => o.t);
}
