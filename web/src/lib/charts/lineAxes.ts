/**
 * A line chart's axes (LineChart): the data's extents, the ticks and their labels, the margins around the plot, and
 * the x mark's place. Pure functions of the chart's props and size; the component derives from them.
 */
import { ticks as niceTicks } from './downsample';
import { capTicks, logTicks, MAX_TICKS, stepDecimals } from './domain';
import { KEY_W, labelWidth } from './endLabels';
import { GUTTER_MAX, LEAD } from './lineEndLabels';
import { endShown, type Series, type XMark } from './lineSeries';

/** The space around the plot, px: tick labels, captions and the end labels' column. */
export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** A scale from data to pixels (d3's linear or log scale). */
export type Scale = (v: number) => number;

/** The x extent: the caller's domain, else the lines' first to last x (one x widened by one either side). */
export function xExtentOf(xDomain: [number, number] | undefined, visible: Series[]): [number, number] {
  if (xDomain) return xDomain;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of visible) {
    const a = s.points[0];
    const b = s.points[s.points.length - 1];
    if (a && a.x < lo) lo = a.x;
    if (b && b.x > hi) hi = b.x;
  }
  if (!Number.isFinite(lo)) return [0, 1];
  return lo === hi ? [lo - 1, hi + 1] : [lo, hi];
}

/** The y extent: the caller's domain, else the values with 8 % padding (from 0 when that is close), a log axis wider. */
export function yExtentOf(yDomain: [number, number] | undefined, visible: Series[], yLog: boolean): [number, number] {
  if (yDomain) return yDomain;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of visible)
    for (const p of s.points) {
      if (!Number.isFinite(p.y) || (yLog && p.y <= 0)) continue;
      if (p.y < lo) lo = p.y;
      if (p.y > hi) hi = p.y;
    }
  if (!Number.isFinite(lo)) return yLog ? [0.1, 1] : [0, 1];
  if (yLog) return [lo * 0.9, hi * 1.1];
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
  return [lo >= 0 && lo - pad < 0 ? 0 : lo - pad, hi + pad];
}

/** The y ticks: the caller's within the extent, else 3–5 nice ones (a log axis's decades). */
export function yTicksOf(yTickProp: number[] | undefined, yExtent: [number, number], yLog: boolean, yTicks: number): number[] {
  if (yTickProp) return yTickProp.filter((t) => t >= yExtent[0] - 1e-9 && t <= yExtent[1] + 1e-9);
  // 3–5 ticks in nice steps (./domain): more gridlines read as a texture, not a scale.
  if (yLog) return logTicks(yExtent[0], yExtent[1]);
  return capTicks(yExtent[0], yExtent[1], Math.min(yTicks, MAX_TICKS));
}

// Decimals follow the tick step, so a 0..6 axis reads "0 2 4 6", a 0..1 axis "0 0.5 1" and quarter steps
// "0.25 0.50"; a fitted domain's own decimals win (`yDecimals`), as its ticks may skip steps; log ticks use 3 digits.
export function yFormatOf(yFormatProp: ((y: number) => string) | undefined, yLog: boolean, yTickValues: number[], yDecimals: number | undefined): (v: number) => string {
  if (yFormatProp) return yFormatProp;
  if (yLog) return (v: number) => String(+v.toPrecision(3));
  const step = yTickValues.length > 1 ? Math.abs(yTickValues[1]! - yTickValues[0]!) : 1;
  const d = Math.min(6, yDecimals ?? stepDecimals(step));
  return (v: number) => v.toFixed(d);
}

/** The margins: room for the y tick labels, the captions and an x mark's label, and the end labels' column. */
export function marginOf(
  yTickValues: number[],
  yFormat: (v: number) => string,
  xExtent: [number, number],
  series: Series[],
  textWidth: (t: string) => number,
  showEnds: boolean,
  yLabel: string | undefined,
  capInPlot: boolean,
  xMark: XMark | undefined,
  xLabel: string | undefined,
): Margin {
  const longest = Math.max(...yTickValues.map((v) => yFormat(v).length), 1);
  // ~7 px per character at 11 px Geist (tabular digits); end labels are 12 px, measured (textWidth).
  // Plus room for a leader, when labels are moved apart (LEAD).
  // A line that ends inside the plot may be labelled in the column, keyed by its swatch (KEY_W) before its name.
  const key = (s: Series) => ((s.points.at(-1)?.x ?? Infinity) < xExtent[1] ? KEY_W : 0);
  const widest = Math.max(0, ...series.map((s) => [endShown(s), s] as const).map(([t, s]) => (t === null ? 0 : textWidth(t) + key(s))));
  // The gutter column holds the widest label whole.
  const labelW = showEnds ? Math.min(GUTTER_MAX, widest + 6) + LEAD - 6 : 0;
  const left = Math.ceil(longest * 7) + 14;
  // An x-axis title gets its own line under the tick labels; a y-axis name or an x mark's label one above the plot,
  // unless the y caption fits over the top tick (capInPlot).
  return { top: (yLabel && !capInPlot) || xMark ? 24 : 8, right: 12 + labelW, bottom: xLabel ? 36 : 22, left };
}

/**
 * A y caption ("y from 0.40", the zoom note) sits right over its axis's top tick label, never a row higher with the
 * plot's headroom between them: where the domain leaves at least a line (16 px) above the top tick, the caption goes
 * there and the plot keeps the row. Linear axes only; an x mark's label keeps the row above the plot.
 */
export function capInPlotOf(
  yLabel: string | undefined,
  xMark: XMark | undefined,
  yLog: boolean,
  yTickValues: number[],
  yExtent: [number, number],
  height: number,
  xLabel: string | undefined,
): boolean {
  if (!yLabel || xMark || yLog || !yTickValues.length) return false;
  const span = yExtent[1] - yExtent[0];
  const room = span > 0 ? ((yExtent[1] - Math.max(...yTickValues)) / span) * (height - 8 - (xLabel ? 36 : 22)) : 0;
  return room >= 16;
}

/** The x ticks: whole numbers where they fit (`xInteger`), else nice steps, ending on a fixed domain's end. */
// When the x domain is fixed (e.g. 0..epochs), prefer a tick step that lands exactly on the end.
export function xTicksOf(width: number, xTicks: number, xExtent: [number, number], xInteger: boolean, xDomain: [number, number] | undefined): number[] {
  const want = Math.max(2, Math.min(xTicks, Math.floor(width / 70)));
  const [lo, hi] = xExtent;
  if (xInteger && hi - lo <= want) {
    const out: number[] = [];
    for (let v = Math.ceil(lo); v <= Math.floor(hi); v++) out.push(v);
    if (out.length >= 2) return out;
  }
  if (!xDomain) return niceTicks(lo, hi, want);
  for (let c = want; c >= 2; c--) {
    const t = niceTicks(lo, hi, c);
    if (t[t.length - 1] === hi) return t;
  }
  return niceTicks(lo, hi, want);
}

// The x mark: its line, its span hatched along the axis to the end of the domain, and its label above the plot beside
// the line (on its left when the right has no room).
export function markOf(xMark: XMark | undefined, xExtent: [number, number], x: Scale, width: number, margin: Margin) {
  if (!xMark || !(xMark.x > xExtent[0] && xMark.x < xExtent[1])) return null;
  const px = x(xMark.x);
  const end = width - margin.right;
  const before = px + 4 + labelWidth(xMark.label) > end;
  return { px, end, label: xMark.label, tx: before ? px - 4 : px + 4, anchor: before ? 'end' : 'start' };
}
