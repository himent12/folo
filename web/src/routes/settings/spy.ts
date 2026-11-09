// Settings' scroll-spy: which section the rail and the phone strip mark as current.

/**
 * The section under the sticky chrome: the last one whose top has reached `line` (the bottom of the top bar, and below
 * 1024 px of the jump strip, plus a little), else the first. Reaching the page's end changes nothing: a short last
 * section (About) is current only once its own top is under the chrome, never because the page cannot scroll further
 * (a full-page shot, a tall window).
 */
export function sectionAt(tops: readonly { id: string; top: number }[], line: number): string | undefined {
  let id = tops[0]?.id;
  for (const t of tops) if (t.top <= line) id = t.id;
  return id;
}

/** Keys that scroll the page: the person's own scroll, which releases a section chosen by a click or the URL. */
export const SCROLL_KEYS = new Set(['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' ']);
