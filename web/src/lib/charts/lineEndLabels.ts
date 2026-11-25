/** A line chart's direct labels (LineChart): where each sits beside its line's end, its ink, and its leader. */
import { KEY_W, LABEL_GAP, leaderOf, type memoPlaceEndLabels } from './endLabels';
import { LIVE_GROW } from './endMarks';
import { seriesText } from './ink';
import type { Margin, Scale } from './lineAxes';
import { endShown, endText, type Series } from './lineSeries';

/** How far right of the plot the label column moves when it needs leaders: short ones, fanning from each line's end. */
export const LEAD = 14;
/** The widest the gutter column grows for its labels, px ("shapes-hard-m · failed" at 12 px semibold is ~150, keyed ~168). */
export const GUTTER_MAX = 196;
/** Where a leader starts: just outside the line's end marker (the live ring is wider), px from its centre. */
export function endRadius(id: string, endDots: boolean, receding: boolean, highlight: string | null, visible: Series[], merges: (string | null)[]): number {
  if (!endDots) return 3;
  const r = receding && id === highlight ? 4 : 3;
  const mark = visible.find((s) => s.id === id)?.end === 'live' ? r + LIVE_GROW : r;
  // A best ring around the end mark (./endMarks) is part of it: the leader starts outside the ring.
  return (merges.includes(id) ? mark + 3.75 : mark) + 2;
}

/** The labels placed (endLabels.ts), each in a spot that covers no curve and no other label. */
// Placed against the y domain the axis eases to, not each frame of the 240 ms ease, and again only when the lines,
// the plot's size or the labels change (endLabels.ts memoPlaceEndLabels): placing is the costly part.
export function placedEndsOf(
  showEnds: boolean,
  visible: Series[],
  x: Scale,
  yTo: Scale,
  textWidth: (t: string) => number,
  highlight: string | null,
  margin: Margin,
  width: number,
  height: number,
  placeEnds: ReturnType<typeof memoPlaceEndLabels>,
) {
  if (!showEnds) return [];
  const drawn = visible.filter((s) => endText(s) !== null);
  const points = drawn
    .map((s) => {
      const p = s.points[s.points.length - 1];
      const curve = visible.indexOf(s);
      const label = endShown(s)!;
      // A line that has ended short of the plot's edge (failed, stopped) is labelled beside its end like any other.
      // Non-positive values on a log scale map to non-finite coordinates and are skipped so they never poison placement.
      const py = p && Number.isFinite(p.y) ? yTo(p.y) : NaN;
      return p && Number.isFinite(py) ? { id: s.id, label, x: x(p.x), y: py, curve, w: textWidth(label) } : null;
    })
    .filter((e): e is NonNullable<typeof e> => e !== null)
    // The highlighted line is labelled first, so it never loses its spot; then the shortest lines, which are the
    // easiest to lose among the others.
    .sort((a, b) => Number(b.id === highlight) - Number(a.id === highlight) || a.x - b.x);
  const curves = visible.map((s) =>
    s.points.flatMap((p) => {
      if (!Number.isFinite(p.y)) return [];
      const py = yTo(p.y);
      return Number.isFinite(py) ? [{ x: x(p.x), y: py }] : [];
    }),
  );
  const bounds = { x0: margin.left, y0: margin.top, x1: width - 2, y1: height - margin.bottom };
  return placeEnds(points, curves, bounds, 2, width - margin.right, { gap: LABEL_GAP, lead: LEAD });
}

/** The placed labels as drawn: riding with their lines' ends while the axis eases, keyed, inked and led. */
export function endsOf(visible: Series[], placedEnds: ReturnType<typeof placedEndsOf>, y: Scale, receding: boolean, highlight: string | null, endR: (id: string) => number) {
  const color = new Map(visible.map((s) => [s.id, s.color]));
  const last = new Map(visible.map((s) => [s.id, s.points[s.points.length - 1]?.y ?? NaN]));
  const byId = new Map(visible.map((s) => [s.id, s]));
  return placedEnds.map((p) => {
    // While the axis eases, each label rides with its line's end: moved as far as the end is from where it will be.
    // The column's "+N more" entry has no line, and stays where it was placed.
    const rawAy = last.has(p.id) ? y(last.get(p.id)!) : p.ay;
    const ay = Number.isFinite(rawAy) ? rawAy : p.ay;
    const e = { ...p, y: p.y + ay - p.ay, ay };
    const s = byId.get(e.id);
    return {
      ...e,
      // A keyed label (in the column, away from its line's end) starts with the line's swatch: its ink and dash.
      key: e.detached && s ? s : null,
      tx: e.detached && s ? e.x + KEY_W : e.x,
      // A receded run is a context curve: its label in ink, so the one hue left on the plot is the run in front.
      color: receding && e.id !== highlight ? 'var(--text-tertiary)' : (color.get(e.id) ?? 'var(--text-secondary)'),
      // The label is text: the series' text ink (./ink), 4.5:1 where the line's own ink is only a 3:1 mark.
      text: receding && e.id !== highlight ? 'var(--text-tertiary)' : seriesText(color.get(e.id) ?? 'var(--text-secondary)'),
      // Every label is anchored to its end marker: one not right beside it is tied to it by a leader.
      lead: leaderOf(e, endR(e.id)),
    };
  });
}
