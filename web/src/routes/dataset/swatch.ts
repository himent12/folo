// Class swatches for classes with no boxes ("absent"): the hue at 40 % as the fill, inside a 1 px outline in the
// class's stroke ink (≥ 3:1 on the surface, also on light panels), so an absent class still reads as a swatch.

import { classInk, classStroke } from '$lib/design/classInk.svelte';

/** Inline styles for class `i`'s swatch: the plain fill, or the absent look (40 % fill, full-contrast outline). */
export function swatchStyle(i: number, absent: boolean): string {
  if (!absent) return `background: ${classInk(i)}`;
  return `background: color-mix(in srgb, ${classInk(i)} 40%, transparent); box-shadow: inset 0 0 0 1px ${classStroke(i)}`;
}
