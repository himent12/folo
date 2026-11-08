// Focus return for transient UI (dialogs, the palette, sheets): focus goes back to whatever opened it.

/**
 * Remembers what has focus now (a dialog's opener). The returned function puts focus back there, unless focus has
 * already gone somewhere real (the browser restored it, or the action moved it on purpose).
 */
export function rememberFocus(): () => void {
  const opener = typeof document === 'undefined' ? null : (document.activeElement as HTMLElement | null);
  return () => {
    const now = document.activeElement;
    const lost = !now || now === document.body || !now.isConnected || !!now.closest('dialog:not([open])');
    if (lost && opener && opener !== document.body && opener.isConnected) opener.focus({ preventScroll: true });
  };
}
