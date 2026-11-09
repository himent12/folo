// A focal chart's marks at a line's end (DESIGN_DIRECTION §2.4): the best-epoch ring and the status end mark (a
// ringed dot while live, a square stopped, a cross failed) each say one thing. Two marks that touch read as one shape
// ("oo"), so a best ring with at most MERGE_PX of clear space between it and its line's end mark is drawn as one
// combined mark at the end: the end mark inside the best ring. The status mark is never hidden: a ring that would
// cover it holds it instead. Lime is for "now" and the best of a run that is still worth reading as good; a run that
// failed or was stopped draws a best ring that sits on its end in ink, so lime never sits on a failed line's end.

export type EndKind = 'live' | 'stopped' | 'failed' | undefined;

/** A best ring with at most this much clear space (px, edge to edge) to its line's end mark joins it as one mark. */
export const MERGE_PX = 16;

/** The best ring's outer radius, px: a 4.5 px ring drawn 2 px wide. */
export const BEST_R = 5.5;

export interface EndAt {
  id: string;
  end: EndKind;
  /** The line's last point, px. */
  x: number;
  y: number;
  /** The end mark's outer radius, px (the live ring is the widest). */
  r: number;
}

/**
 * For each marker (px), the id of the line end it merges with: the nearest end mark with at most `MERGE_PX` of clear
 * space between the two marks' edges, or null when they are far enough apart to be drawn as they are.
 */
export function mergedEnds(markers: readonly { x: number; y: number }[], ends: readonly EndAt[], within = MERGE_PX): (string | null)[] {
  return markers.map((m) => {
    let best: string | null = null;
    let d = within + 1e-9;
    for (const e of ends) {
      const k = Math.hypot(e.x - m.x, e.y - m.y) - BEST_R - e.r;
      if (k <= d) {
        d = k;
        best = e.id;
      }
    }
    return best;
  });
}

/** The colour of a best ring on a line's end: lime while live or completed, ink once the run failed or was stopped. */
export function endRingInk(end: EndKind, lime: string): string {
  return end === 'failed' || end === 'stopped' ? 'var(--text-primary)' : lime;
}

/** The status mark's own ink: a failed run's cross in the danger ink, the line's ink otherwise. */
export function endMarkInk(end: EndKind, line: string): string {
  return end === 'failed' ? 'var(--danger-text)' : line;
}
