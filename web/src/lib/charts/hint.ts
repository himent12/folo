// The one-line hints a chart shows in its readout row, so its reading paths are discoverable: a keyboard focus says
// "← → to inspect" while it lasts; a touch screen says "Tap to pin a value" once per browser session.

export const KEY_HINT = '← → to inspect';
export const TAP_HINT = 'Tap to pin a value';
export const TAP_HINT_KEY = 'folo.chart-tap-hint';

/**
 * True the first time it is asked in this session (and remembers that it was), false after that or when storage is
 * unavailable (a private window, blocked site data): a hint that cannot be remembered is not shown at all.
 */
export function claimOnce(key: string, store: () => Storage = () => sessionStorage): boolean {
  try {
    const s = store();
    if (s.getItem(key)) return false;
    s.setItem(key, '1');
    return true;
  } catch {
    return false;
  }
}

/** The hint to show, if any: none while the cursor reads out a value (the row shows it then). */
export function chartHint(o: { reading: boolean; keyFocus: boolean; tap: boolean }): string | null {
  if (o.reading) return null;
  if (o.keyFocus) return KEY_HINT;
  return o.tap ? TAP_HINT : null;
}
