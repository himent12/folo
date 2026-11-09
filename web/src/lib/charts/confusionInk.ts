// The confusion-matrix cell ramp and its label ink. A cell with no objects is the panel itself, edged by the hairline
// grid; any share above zero starts at a clearly visible first step (RAMP[theme].start of the ink mixed into the panel,
// in oklab) and runs to RAMP[theme].end. Light starts lower: on white a heavier first step turns a matrix of 1-4 %
// confusions into a grey chequerboard, while on dark the step must clear a near-black panel. Labels are pure white or
// black and switch just below the point where the far extreme starts to win, so every label keeps >= 4.5:1 (all
// checked over the whole ramp in confusionInk.test.ts). ConfusionMatrix.svelte mirrors RAMP in --cm-start / --cm-span.

/** Ink share mixed into the panel at the lowest non-zero cell and at a full (100 %) cell. */
export const RAMP = { dark: { start: 0.2, end: 0.94 }, light: { start: 0.12, end: 0.94 } } as const;

/** Cell value (0..1 share of the row) above which a label switches to the high-ramp ink. */
export const CONFUSION_FLIP = { dark: 0.41, light: 0.49 } as const;

/** Where a cell of value `v` sits on the ramp, as the ink share mixed into the panel; 0 for an empty cell. */
export const rampShare = (v: number, theme: 'dark' | 'light') => (v > 0 ? RAMP[theme].start + v * (RAMP[theme].end - RAMP[theme].start) : 0);
