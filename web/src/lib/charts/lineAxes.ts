/**
 * A line chart's axes (LineChart): the data's extents, the ticks and their labels, the margins around the plot, and
 * the x mark's place. Pure functions of the chart's props and size; the component derives from them.
 */
import { ticks as niceTicks } from './downsample';
import { capTicks, logTicks, MAX_TICKS, stepDecimals } from './domain';
import { KEY_W, labelWidth } from './endLabels';
import { GUTTER_MAX, LEAD } from './lineEndLabels';
import { endShown, type Future, type Series, type XMark } from './lineSeries';

/** The space around the plot, px: tick labels, captions and the end labels' column. */
export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** A scale from data to pixels (d3's linear or log scale). */
export type Scale = (v: number) => number;

/** The x extent: the caller's domain, else the lines' finite min to max x (one x widened by one either side). */
export function xExtentOf(xDomain: [number, number] | undefined, visible: Series[]): [number, number] {
  if (xDomain && Number.isFinite(xDomain[0]) && Number.isFinite(xDomain[1])) return xDomain;
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of visible)
    for (const p of s.points) {
      if (!Number.isFinite(p.x)) continue;
      if (p.x < lo) lo = p.x;
      if (p.x > hi) hi = p.x;
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
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || (yLog && p.y <= 0)) continue;
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
  band: boolean,
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
  // An x-axis title gets its own line under the tick labels; a y-axis name or the band of an x mark's or the future's
  // label one above the plot, unless the y caption fits over the top tick (capInPlot).
  return { top: (yLabel && !capInPlot) || band ? 24 : 8, right: 12 + labelW, bottom: xLabel ? 36 : 22, left };
}

/**
 * A y caption ("y from 0.40", the zoom note) sits right over its axis's top tick label, never a row higher with the
 * plot's headroom between them: where the domain leaves at least a line (16 px) above the top tick, the caption goes
 * there and the plot keeps the row. Linear axes only; the band of an x mark's (or the future's) label keeps the row
 * above the plot.
 */
export function capInPlotOf(
  yLabel: string | undefined,
  band: boolean,
  yLog: boolean,
  yTickValues: number[],
  yExtent: [number, number],
  height: number,
  xLabel: string | undefined,
): boolean {
  if (!yLabel || band || yLog || !yTickValues.length) return false;
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

/**
 * An event on x (a failed run's "failed during epoch 3"): a hairline at `x`, which may be the domain's end, where the
 * line stopped, and its label in the band above the plot beside it (on its left when the right has no room). Where it
 * would touch the x mark's label it goes inside the plot's top, as the future's does (both 12 px captions; 6 px apart).
 */
export function eventOf(
  xEvent: XMark | undefined,
  mark: { tx: number; anchor: string; label: string } | null,
  xExtent: [number, number],
  x: Scale,
  width: number,
  margin: Margin,
) {
  if (!xEvent || !(xEvent.x > xExtent[0] && xEvent.x <= xExtent[1])) return null;
  const px = x(xEvent.x);
  const end = width - margin.right;
  const w = labelWidth(xEvent.label);
  const before = px + 4 + w > end;
  const tx = before ? px - 4 : px + 4;
  let inBand = true;
  if (mark) {
    const mw = labelWidth(mark.label);
    const m0 = mark.anchor === 'end' ? mark.tx - mw : mark.tx;
    const e0 = before ? tx - w : tx;
    inBand = e0 + w + 6 <= m0 || m0 + mw + 6 <= e0;
  }
  return { px, label: xEvent.label, tx, anchor: before ? 'end' : 'start', ty: inBand ? margin.top - 8 : margin.top + 14 };
}

/**
 * The future's shade (a live run's epochs still to train): from the latest point to the end of the plot, its label
 * after the rule in the band above the plot, or inside the plot's top where the x mark's label already holds that
 * spot (both are 12 px captions; 6 px between them).
 */
export function futureOf(
  future: Future | undefined,
  mark: { tx: number; anchor: string; label: string } | null,
  xExtent: [number, number],
  x: Scale,
  width: number,
  margin: Margin,
  height: number,
) {
  if (!future || !(future.x < xExtent[1])) return null;
  const end = width - margin.right;
  const px = Math.max(margin.left, x(Math.max(xExtent[0], future.x)));
  if (px >= end) return null;
  const label = future.label ?? '';
  const w = label ? labelWidth(label) : 0;
  const tx = Math.max(margin.left, Math.min(px + 6, end - w - 4));
  let inBand = true;
  if (mark && label) {
    const mw = labelWidth(mark.label);
    const m0 = mark.anchor === 'end' ? mark.tx - mw : mark.tx;
    inBand = tx + w + 6 <= m0 || m0 + mw + 6 <= tx;
  }
  return { px, end, top: margin.top, bottom: height - margin.bottom, label, tx, ty: inBand ? margin.top - 8 : margin.top + 14 };
}
