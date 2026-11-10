// The robustness heat table's ink: a cell's tone is the accuracy the condition took away (1 - kept), as ink mixed into
// the panel (in oklab, the confusion matrix's way, ./lib/charts/confusionInk), so where a model breaks is where the
// table is darkest in light and brightest in dark: one sequential ramp, one hue, never a status colour. A cell that
// kept everything is the panel itself. Labels are pure white or black and switch where the far extreme wins, so every
// number keeps >= 4.5:1 (./heat.test.ts).

/** Lost from this share on, a cell is at the ramp's end: half of the accuracy gone is as bad as the table shows. */
export const LOST_FULL = 0.6;

const RAMP = { dark: { from: 0.1, end: 0.9 }, light: { from: 0.07, end: 0.9 } } as const;
const FLIP = { dark: 0.5, light: 0.52 } as const;

/** The ink share of a cell that kept `kept` of its accuracy (0..1): 0 when nothing was lost. */
export function heatShare(kept: number, theme: 'dark' | 'light'): number {
  const lost = 1 - Math.min(1, Math.max(0, kept));
  if (lost < 0.005) return 0;
  const r = RAMP[theme];
  return r.from + (Math.min(1, lost / LOST_FULL) * (r.end - r.from));
}

/** Whether a cell's number takes the high-ramp ink. */
export const heatLabelHigh = (kept: number, theme: 'dark' | 'light') => heatShare(kept, theme) > FLIP[theme];
