// A focal chart's marks at a line's end (DESIGN_DIRECTION §2.4): each mark has one shape for one meaning, whatever
// sits next to it. The best epoch is a hollow ink ring (the "○ best" glyph of the chart's header); the latest value of
// a live run is a filled lime dot with the heartbeat (StatusBadge's running mark); a stopped run ends in a square and
// a failed one in a cross. **The glyph says the status; the hue says whose line it is**: on a line in a series hue (a
// run in Compare, where hue identifies the run) the square and the cross take the line's hue, never the danger red,
// which on a plot of hues would read as one more run (round 27); on a line in ink (a single run's focal chart) the
// cross takes the danger ink, the status hue on its mark. The best ring and the end mark are always two different shapes, so best
// at epoch 36 and live at 37 read as a ring and a dot, never "oo". Only when the best is the line's end (or so close
// that the two marks would overlap) is the ring drawn around the end mark, as one mark: the status mark is never
// hidden, and a ring never sits half over a dot.

export type EndKind = 'live' | 'stopped' | 'failed' | undefined;

/**
 * A best ring with less than this much clear space (px, edge to edge) to its line's end mark is drawn around it: the
 * two would otherwise overlap or touch. Marks further apart are drawn where they are, each in its own shape.
 */
export const MERGE_PX = 2;

/** The best ring's radius, px: a 4.5 px ring, 1.5 px wide (the "○" glyph's weight). */
export const BEST_R = 5.25;

export interface EndAt {
  id: string;
  end: EndKind;
  /** The line's last point, px. */
  x: number;
  y: number;
  /** The end mark's outer radius, px. */
  r: number;
}

/**
 * For each marker (px), the id of the line end it is drawn around: the nearest end mark with less than `MERGE_PX` of
 * clear space between the two marks' edges, or null when they are apart and each is drawn as it is.
 */
export function mergedEnds(markers: readonly { x: number; y: number }[], ends: readonly EndAt[], within = MERGE_PX): (string | null)[] {
  return markers.map((m) => {
    let best: string | null = null;
    let d = within;
    for (const e of ends) {
      const k = Math.hypot(e.x - m.x, e.y - m.y) - BEST_R - e.r;
      if (k < d) {
        d = k;
        best = e.id;
      }
    }
    return best;
  });
}

/** How much larger (px of radius) a live run's filled lime dot is than the 3 px end dot: the heartbeat's mark. */
export const LIVE_GROW = 1;

/** The latest value of a live run: the running mark's lime, whatever the line's own ink. */
export const LIVE_INK = 'var(--accent-line)';

/** A line in a series hue (`--chart-N`, its fill or text ink) or receded behind one (`--chart-context`): an entity's line. */
const SERIES_HUE = /^var\(--chart-(?:(?:fill-|text-)?[1-6]|context)\)$/;

/**
 * The status mark's own ink: a live end lime (the running mark, whatever the line); a failed run's cross in the danger
 * ink on an ink line, in the line's own hue on a series-hued one (the glyph alone says it failed); the line's ink
 * otherwise.
 */
export function endMarkInk(end: EndKind, line: string): string {
  if (end === 'live') return LIVE_INK;
  return end === 'failed' && !SERIES_HUE.test(line.trim()) ? 'var(--danger-text)' : line;
}
