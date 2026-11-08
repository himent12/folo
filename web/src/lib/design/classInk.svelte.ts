// Class colours for UI marks (swatches, bars, strips, curves) on page surfaces. The palette is tuned for boxes on
// photos; on light surfaces some hues are too pale, so the light theme uses a darker ink of the same hue (≥ 3:1).
// Detection boxes on images keep the raw palette (`classColor`).

import { prefs } from '$lib/prefs.svelte';
import { classColor } from './tokens';

export function classInk(i: number): string {
  return prefs.resolvedTheme === 'light' ? `color-mix(in oklab, ${classColor(i)} 66%, #000)` : classColor(i);
}
