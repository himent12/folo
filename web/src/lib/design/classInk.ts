// Class colours for UI marks on page surfaces. Fills (swatches, chips, legend dots, bars, strips) use the exact
// palette hue in both themes so a class looks the same in a legend as on its box; they carry a 1 px
// `--chart-swatch-edge` inset (global `.sw`, or set it yourself) for contrast on light surfaces. Thin strokes
// (1-2 px curves) cannot carry an edge, so on light surfaces they use a darker ink of the same hue.
// Detection boxes on images always use the raw palette (`classColor`).

import { prefs } from '$lib/prefs';
import { classColor } from './tokens';

/** Fill colour for class `i` on a page surface: the palette hue itself. */
export function classInk(i: number): string {
  return classColor(i);
}

/** Stroke colour for class `i` (thin lines on page surfaces): darkened on light surfaces to keep ≥ 3:1. */
export function classStroke(i: number): string {
  return prefs.resolvedTheme === 'light' ? `color-mix(in oklab, ${classColor(i)} 66%, #000)` : classColor(i);
}
