// Focus for the Playground's notice strip (a failed prediction, a missing weights file). Its heading takes focus as it
// appears, so it is announced and its actions are next in tab order, without the page jumping: the move never scrolls
// on its own (`preventScroll`), and the heading is brought into view only when none of it is on screen. While the
// notice stays, focus that falls back to the document (the focused element was re-rendered away) returns to the
// heading instead of the page start; focus the user moves elsewhere ends the hold.

/** The notice's heading: our own `.n-title`, or ErrorState's `.title`. */
export function noticeHeading(strip: HTMLElement): HTMLElement | null {
  return strip.querySelector<HTMLElement>('.n-title, .title');
}

/** Focuses the heading without scrolling; scrolls it into view (nearest edge) only when it is entirely off screen. */
export function focusNotice(strip: HTMLElement): HTMLElement | null {
  const h = noticeHeading(strip);
  if (!h) return null;
  h.tabIndex = -1;
  h.focus({ preventScroll: true });
  const r = h.getBoundingClientRect();
  if (r.height > 0 && (r.top >= window.innerHeight || r.bottom <= 0)) h.scrollIntoView?.({ block: 'nearest' });
  return h;
}

/** `focusNotice`, then holds focus on the heading while the strip is shown. Returns the cleanup. */
export function holdNoticeFocus(strip: HTMLElement): () => void {
  let target = focusNotice(strip);
  const lost = () => {
    const a = document.activeElement;
    return a === null || a === document.body || !a.isConnected;
  };
  const observer = new MutationObserver(() => {
    if (!strip.isConnected || !lost()) return;
    const h = noticeHeading(strip);
    if (!h) return;
    target = h;
    h.tabIndex = -1;
    h.focus({ preventScroll: true });
  });
  observer.observe(strip, { childList: true, subtree: true });
  const onFocus = (e: FocusEvent) => {
    if (e.target !== target) stop();
  };
  document.addEventListener('focusin', onFocus);
  function stop() {
    observer.disconnect();
    document.removeEventListener('focusin', onFocus);
  }
  return stop;
}
