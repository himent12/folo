// Label ink for confusion-matrix cells. The cell ramp runs from the plot well (`bg.inset`) to full ink (`text.primary`)
// in oklab; labels are pure white or black and switch just below the point where the far extreme starts to win, so
// every label keeps >= 4.5:1 (checked over the whole ramp in confusionInk.test.ts).

/** Cell value (0..1 share of the row) above which a label switches to the high-ramp ink. */
export const CONFUSION_FLIP = { dark: 0.545, light: 0.505 } as const;

/** Where a cell of value `v` sits on the ramp, as the ink share mixed into the well. */
export const rampShare = (v: number) => v * 0.9 + 0.04;
