// The marks of ./XYChart.svelte: its series, references, labelled points and count strip.

/** One point of an `XYSeries`: `lo` / `hi` draw an interval on y around it (whiskers with caps). */
export interface XYPoint {
  x: number;
  y: number | null;
  lo?: number;
  hi?: number;
  /** The dot's radius in px (default 3.5): a point that stands for more weighs more. */
  r?: number;
  /** A hollow dot: a point that stands for too little to trust (a bin of a few detections). */
  hollow?: boolean;
}

export interface XYSeries {
  id: string;
  label: string;
  /** A token: ink (`var(--text-primary)`) for a single result's own lines, `var(--chart-N)` for an entity's. */
  color: string;
  points: XYPoint[];
  /** Join the points with a line (default true). */
  line?: boolean;
  /** Draw a dot at every point (default false; always with intervals). */
  dots?: boolean;
  pattern?: 'dash' | 'dot';
  /** A 1 px line: a secondary quantity beside the focal one. */
  thin?: boolean;
}

/** A reference on the plot: a level (`axis: 'y'`), a threshold on x, or the diagonal y = x. */
export type XYRef =
  /** A level: its label over the line at the plot's right end, or under it (`below`) where a curve runs above. */
  | { axis: 'y'; value: number; label?: string; below?: boolean }
  /** A threshold on x: its label beside the line at the plot's foot, clear of curves that leave the top. */
  | { axis: 'x'; value: number; label?: string }
  | { axis: 'diagonal'; label?: string };

/** A labelled point (a threshold found on a curve): a ring and its words. */
export interface XYMark {
  x: number;
  y: number;
  label: string;
  /** Where the words go from the ring. */
  side?: 'left' | 'right' | 'above' | 'below';
}

/** A strip of bars under the plot, on the plot's x scale: how much each span stands on (detections per bin). */
export interface XYStrip {
  items: { x0: number; x1: number; v: number }[];
  /** The unit the readout names, e.g. "detections". */
  unit: string;
}

/**
 * What the chart says aloud at a cursor stop (its live region): `speak` when the caller gives one; else the table row
 * whose first cell is the stop as the axis prints it; else, since a table's rows need not print x as the axis ticks do
 * ("0.50" against "0.5") or may hold only some stops, each series' value there, named, so every stop says something.
 */
export function spokenAt(
  c: number,
  o: {
    series: readonly XYSeries[];
    table: { head: readonly string[]; rows: readonly (readonly string[])[] };
    xFormat: (v: number) => string;
    yFormat: (v: number) => string;
    xLabel?: string;
    speak?: (x: number) => string;
  },
): string {
  if (o.speak) return o.speak(c);
  const row = o.table.rows.find((r) => r[0] === o.xFormat(c));
  if (row) return row.map((cell, k) => (k ? `${o.table.head[k]} ${cell}` : cell)).join(', ');
  const parts = o.series.flatMap((s) => {
    const p = s.points.find((q) => Math.abs(q.x - c) < 1e-9 && q.y !== null);
    return p ? [`${s.label} ${o.yFormat(p.y!)}`] : [];
  });
  return `${o.xLabel ?? o.table.head[0] ?? 'At'} ${+c.toFixed(4)}${parts.length ? `: ${parts.join(', ')}` : ''}`;
}
