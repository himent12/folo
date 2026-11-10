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
