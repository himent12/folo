// The confusion-matrix cell ramp and its label ink. A cell with no objects is the panel itself, edged by the hairline
// grid; any share above zero is ink mixed into the panel (in oklab). Small shares are where a matrix's story is, so the
// ramp's floor is stepped: under 1 %, 1–5 % and 5–10 % each get a step of their own (RAMP[theme].steps), clearly apart,
// and from 10 % the ramp runs evenly to RAMP[theme].end. Light starts lower: on white a heavier first step turns a
// matrix of 1-4 % confusions into a grey chequerboard, while on dark the step must clear a near-black panel. Labels are
// pure white or black and switch where the far extreme starts to win (CONFUSION_FLIP, as an ink share), so every label
// keeps >= 4.5:1 (checked over the whole ramp in confusionInk.test.ts). ConfusionMatrix.svelte draws from rampShare.

/** Where the floor steps start, as shares of the row: under 1 %, then 1 %, 5 % and the even ramp from 10 %. */
export const FLOOR = [0.01, 0.05, 0.1] as const;

/**
 * Ink share mixed into the panel: one per floor step (a share above zero under 1 %, from 1 %, from 5 %), then from
 * `from` at 10 % evenly to `end` at 100 %.
 */
export const RAMP = {
  dark: { steps: [0.2, 0.3, 0.4], from: 0.48, end: 0.94 },
  light: { steps: [0.12, 0.2, 0.28], from: 0.36, end: 0.94 },
} as const;

/** Ink share above which a cell's label switches to the high-ramp ink. */
export const CONFUSION_FLIP = { dark: 0.5, light: 0.52 } as const;

/** Where a cell of value `v` sits on the ramp, as the ink share mixed into the panel; 0 for an empty cell. */
export function rampShare(v: number, theme: 'dark' | 'light'): number {
  if (!(v > 0)) return 0;
  const r = RAMP[theme];
  if (v < FLOOR[0]) return r.steps[0];
  if (v < FLOOR[1]) return r.steps[1];
  if (v < FLOOR[2]) return r.steps[2];
  return r.from + ((Math.min(1, v) - FLOOR[2]) / (1 - FLOOR[2])) * (r.end - r.from);
}

/** Whether a cell of value `v` takes the high-ramp label ink. */
export const labelHigh = (v: number, theme: 'dark' | 'light') => rampShare(v, theme) > CONFUSION_FLIP[theme];

/**
 * A share of the true class as the matrix writes it, one rule everywhere (cells, readout, summaries): whole
 * percentages, "<1" for a share above zero that rounds below 1, and "" for no objects at all. No "%" sign: callers
 * add it where the text needs it.
 */
export function sharePct(v: number): string {
  if (!(v > 0)) return '';
  return v < 0.005 ? '<1' : String(Math.max(1, Math.round(v * 100)));
}
