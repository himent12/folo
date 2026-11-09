// What Runs' Compare needs, said where the selection is made (DESIGN_DIRECTION §2.8, prerequisites are visible): a
// disabled Compare is never explained only by a tooltip, which touch and a keyboard on a disabled button never show.

/** The fewest and most runs Compare takes. */
export const COMPARE_MIN = 2;
export const COMPARE_MAX = 4;

/** The line beside (desktop) or under (phones) the selection controls, for `n` runs selected. */
export function compareHint(n: number): string {
  if (n === 0) return `Select ${COMPARE_MIN}–${COMPARE_MAX} runs to compare`;
  if (n < COMPARE_MIN) return `Select one more run to compare · ${COMPARE_MIN}–${COMPARE_MAX} runs`;
  if (n < COMPARE_MAX) return `${n} selected · up to ${COMPARE_MAX - n} more`;
  return `${n} selected (the maximum)`;
}

/** Why Compare is disabled, or null once it can run: the same words as the visible line. */
export function compareReason(n: number): string | null {
  return n < COMPARE_MIN ? compareHint(n) : null;
}
