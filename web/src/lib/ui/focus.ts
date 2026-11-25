// Focus return for transient UI (dialogs, the palette, sheets): focus goes back to whatever opened it.

function isFocusableTarget(el: HTMLElement | null): el is HTMLElement {
  if (!el || el === document.body || !el.isConnected) return false;
  if ('disabled' in el && Boolean((el as HTMLButtonElement).disabled)) return false;
  if (el.closest('[inert], dialog:not([open])')) return false;
  return true;
}

/**
 * Remembers what has focus now (a dialog's opener). The returned function puts focus back there, unless focus has
 * already gone somewhere real (the browser restored it, or the action moved it on purpose). If the opener was removed,
 * disabled, or hidden before close, an optional fallback supplies the next sensible control.
 */
export function rememberFocus(fallback?: () => HTMLElement | null): () => void {
  const opener = typeof document === 'undefined' ? null : (document.activeElement as HTMLElement | null);
  return () => {
    const now = document.activeElement;
    const lost = !now || now === document.body || !now.isConnected || !!now.closest('dialog:not([open])');
    if (!lost) return;
    if (isFocusableTarget(opener)) opener.focus({ preventScroll: true });
    else {
      const fb = fallback?.() ?? null;
      if (isFocusableTarget(fb)) fb.focus({ preventScroll: true });
    }
  };
}
