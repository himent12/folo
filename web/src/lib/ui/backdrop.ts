// Closing a modal <dialog> with a press on its backdrop (outside its box), as Dialog and the command palette do. Only a
// click that started on the backdrop counts (a text selection or a slider drag released outside is not a request to
// close), and not the second click of a double-click on the button that opened it: that press lands on the backdrop
// right after opening, and would close the dialog before the person saw it (runs F5). Only that press is let through:
// the first one after a dialog a pointer opened, soon after and where the opening press was. A dialog opened from the
// keyboard or by code closes on the first press outside. That same press selects no text either: landing on the
// dialog's title, a double-click's second press would select a word of it (v-data-runs N5).

/** How long after opening a press where the opening one was is taken for the rest of its double-click. */
export const OPEN_GRACE_MS = 500;
/** How far from the opening press the second one of a double-click lands, at most (a finger's double tap included). */
export const DOUBLE_CLICK_SLOP_PX = 16;
/** A dialog shown within this long of a press's release was opened by it (its click handler, or an effect after it). */
const OPENER_MS = 250;

/** The last press anywhere: where it was, and when it was released (still down: Infinity). */
let last: { x: number; y: number; up: number } | null = null;
let listening = false;
function listen() {
  if (listening || typeof document === 'undefined') return;
  listening = true;
  const opts = { capture: true, passive: true };
  document.addEventListener('pointerdown', (e) => (last = { x: e.clientX, y: e.clientY, up: Number.POSITIVE_INFINITY }), opts);
  document.addEventListener('pointerup', () => last && (last.up = performance.now()), opts);
}

/**
 * The backdrop handlers for `dialog()`: put `onpointerdown`, `onmousedown` and `onclick` on the <dialog>, and call
 * `opened()` when it is shown. `close` runs for a press that started and ended on the backdrop, unless it continues the
 * double-click that opened the dialog; that press does nothing else either (no text selected, no focus moved).
 */
export function backdropClose(dialog: () => HTMLDialogElement, close: () => void) {
  listen();
  /** The press that opened the dialog (when a pointer did), until the next press. */
  let opener: { x: number; y: number; at: number } | null = null;
  let down = false;
  /** The press under way continues the opening double-click. */
  let continuing = false;
  return {
    opened() {
      const now = performance.now();
      opener = last && now - Math.min(last.up, now) < OPENER_MS ? { x: last.x, y: last.y, at: now } : null;
    },
    onpointerdown(e: PointerEvent) {
      const o = opener;
      opener = null;
      const continues = !!o && performance.now() - o.at < OPEN_GRACE_MS && Math.hypot(e.clientX - o.x, e.clientY - o.y) <= DOUBLE_CLICK_SLOP_PX;
      continuing = continues;
      down = e.target === dialog() && !continues;
    },
    // After `onpointerdown` for the same press: a mouse's selection (a word on a double-click) starts here.
    onmousedown(e: MouseEvent) {
      if (continuing) e.preventDefault();
      continuing = false;
    },
    onclick(e: MouseEvent) {
      if (e.target === dialog() && down) close();
      down = false;
    },
  };
}
