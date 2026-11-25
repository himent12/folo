/**
 * A line chart's series (LineChart's `Series`, `Marker`, `XMark` and `Future`) and how each one is drawn and named: its line
 * pattern and ink, its legend swatch, its direct label, its latest value.
 */
import type { Pt } from './downsample';
import { seriesText } from './ink';

export interface Series {
  id: string;
  label: string;
  /** Any CSS colour, usually a token: `var(--chart-1)`. */
  color: string;
  points: Pt[];
  dashed?: boolean;
  /** Line pattern for secondary series of the same hue: dashed, dotted or dash-dot. */
  pattern?: 'dash' | 'dot' | 'dashdot';
  /** Fill the area under the line. */
  area?: boolean;
  /** A 1 px line: a secondary series in ink beside the focal one. */
  thin?: boolean;
  /** Thin, recessive line (e.g. raw values under a smoothed line); hidden from legend and tooltip. */
  ghost?: boolean;
  /**
   * Context line (e.g. each class under their mean): 1 px in `--chart-context`, which holds >= 3:1 on the plot
   * without opacity, so it stays traceable while the emphasised series carry weight and ink. Like a ghost it is
   * hidden from the legend, the readout and end labels; `color` is ignored.
   */
  context?: boolean;
  /** Thicker line drawn above the others (e.g. a mean over many thin per-class lines). */
  emphasis?: boolean;
  /** Painted over the other ordinary lines, patterned ones included (./layers), but under emphasised ones. */
  front?: boolean;
  /** The direct label at the line's end (with `endLabels`): a short name instead of `label`, or false for none. */
  endLabel?: string | false;
  /**
   * How the line ends, drawn at its last point with `endDots` (./endMarks): a filled lime dot with the heartbeat while
   * live, a square when stopped, a cross when failed (a plain dot otherwise). The end label then names it too.
   */
  end?: 'live' | 'stopped' | 'failed';
  /**
   * A band instead of a line: `points` is the upper edge, `lower` the lower edge at the same x positions (e.g. the
   * spread of many curves). Drawn as a soft fill under the lines; not in the legend or the readout.
   */
  lower?: Pt[];
  /** Optional revision bumped when `points` grows in place, so live callers do not need to `.slice()` every tick. */
  rev?: number;
  /** Optional precomputed running min/max (`lo`, `hi`) of finite `y` values, letting `seriesSummary` run in O(1). */
  bounds?: { lo: number; hi: number };
}

/**
 * The best epoch: a hollow ring in `text.primary`, 1.5 px, the "○ best" glyph of the chart's header (./endMarks). It
 * has one ink on every run, so it never reads as the live mark (a filled lime dot).
 */
export interface Marker {
  x: number;
  y: number;
  label: string;
  /** Ring only: the label is stated elsewhere (e.g. the panel header), not floated over the data. */
  quiet?: boolean;
  /**
   * Bump to pulse the ring once (e.g. a live run just set a new best): a ring grows out of it and fades over the
   * chart duration. Nothing moves under reduced motion. 0 or unset: no pulse.
   */
  pulse?: number;
  /**
   * Bump to focus the marker once (the run just completed): lime viewfinder brackets close in on it over 200 ms, hold
   * and fade, the one moment the brand's mark is drawn on a chart. Skipped under reduced motion. 0 or unset: none.
   */
  focus?: number;
}

/**
 * A phase change on the x axis (e.g. "mosaic off" for a run's last epochs): a thin vertical line at `x` with its 12 px
 * label above the plot, and the span from `x` to the end of the domain hatched along the axis, in the hatch the run's
 * progress bar marks that span with.
 */
export interface XMark {
  x: number;
  label: string;
}

/**
 * The planned part of a live run's x axis, not trained yet: shaded from `x` (the latest point) to the end of the
 * domain, and `label` (the ETA, "ETA 12m 02s") over its start in the band above the plot.
 */
export interface Future {
  x: number;
  label?: string;
}

/** A line's direct label: its short `endLabel`, else its legend label; null for none. */
export const endText = (s: Series) => (s.ghost || s.context || s.endLabel === false ? null : (s.endLabel ?? s.label));
/** The end label as drawn: the name cut at 18 characters, then how the line ends ("· failed"), never cut. */
export const endShown = (s: Series) => {
  const t = endText(s);
  if (t === null) return null;
  const name = t.length > 18 ? `${t.slice(0, 17)}…` : t;
  return s.end ? `${name} · ${s.end}` : name;
};

// Dots are 0-length dashes drawn with round caps (stroke width), so they read as dots, not a faint hairline.
// Dots sit 4.5 px apart, so a line they run along (box under dfl in the loss chart) still shows between them.
const PATTERNS = { dash: [6, 3], dot: [0.01, 4.5], dashdot: [7, 3, 0.01, 3] } as const;
// The legend swatch (12 px, 2 px stroke, round caps) shows the same rhythm as its line: two dashes, three dots.
const SWATCH = { dash: [3, 4], dot: [0.01, 4.5], dashdot: [4, 3.5, 0.01, 3.5] } as const;
export const patternOf = (s: Series) => s.pattern ?? (s.dashed ? 'dash' : null);
/**
 * A line's ink. A dash or a dot carries less ink than a line, so a patterned series takes its hue's text ink
 * (./ink: 4.5:1 per dash, the same hue in dark, darker in light) and, in light, a 2 px dash (LineChart's widthOf), so a dotted
 * fourth run clears 3:1 in practice, not only per pixel (DESIGN_DIRECTION §2.5).
 */
export const inkOf = (s: Series) => (patternOf(s) ? seriesText(s.color) : s.color);
/** stroke-dasharray for a series; `k` scales it with a heavier stroke (a highlighted line keeps its rhythm). */
export function dashOf(s: Series, k = 1): string | undefined {
  const p = patternOf(s);
  return p ? PATTERNS[p].map((v) => +(v * k).toFixed(3)).join(' ') : undefined;
}
export const swatchDash = (s: Series) => {
  const p = patternOf(s);
  return p ? SWATCH[p].join(' ') : undefined;
};

export const latest = (s: Series) => {
  for (let i = s.points.length - 1; i >= 0; i--) {
    const p = s.points[i]!;
    if (Number.isFinite(p.y)) return p.y;
  }
  return null;
};

/** Per-series summary for assistive tech: latest, minimum and maximum. Uses precomputed `s.bounds` in O(1) when provided. */
export function seriesSummary(keyed: Series[]) {
  return keyed.map((s) => {
    if (s.bounds) {
      return {
        s,
        last: latest(s),
        lo: Number.isFinite(s.bounds.lo) ? s.bounds.lo : null,
        hi: Number.isFinite(s.bounds.hi) ? s.bounds.hi : null,
      };
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of s.points) {
      if (!Number.isFinite(p.y)) continue;
      if (p.y < lo) lo = p.y;
      if (p.y > hi) hi = p.y;
    }
    return { s, last: latest(s), lo: Number.isFinite(lo) ? lo : null, hi: Number.isFinite(hi) ? hi : null };
  });
}

/** Stroke width of a series' line in px, as LineChart's `.line` classes draw it (`light`: the light theme). */
export function strokeWidth(s: Series, receding: boolean, highlight: string | null, light: boolean): number {
  if (s.ghost || s.context || s.thin) return 1;
  const p = patternOf(s);
  if (receding && s.id === highlight) return p === 'dot' ? 3.5 : 3;
  // Receded: a 1 px context curve, dots a little heavier so a dotted run keeps its rhythm.
  if (receding) return p === 'dot' ? 1.75 : 1;
  if (s.emphasis || (highlight !== null && s.id === highlight)) return p === 'dot' ? 3.25 : 2.75;
  return p === 'dot' ? 2.25 : p ? (light ? 2 : 1.5) : 1.75;
}
