// Closing a modal <dialog> with a press on its backdrop (outside its box), as Dialog and the command palette do. Only a
// click that started on the backdrop counts (a text selection or a slider drag released outside is not a request to
// close), and not one that started right after the dialog opened: the second click of a double-click on the button
// that opened it lands on the backdrop, and would close the dialog before the person saw it (runs F5).

/** How long after opening a press on the backdrop is taken for the rest of the click that opened the dialog. */
export const OPEN_GRACE_MS = 500;

/**
 * The backdrop handlers for `dialog()`: put `onpointerdown` and `onclick` on the <dialog>, and call `opened()` when it is
 * shown. `close` runs for a press that started and ended on the backdrop, outside the grace period.
 */
export function backdropClose(dialog: () => HTMLDialogElement, close: () => void) {
  let openedAt = Number.NEGATIVE_INFINITY;
  let down = false;
  return {
    opened() {
      openedAt = performance.now();
    },
    onpointerdown(e: PointerEvent) {
      down = e.target === dialog() && performance.now() - openedAt >= OPEN_GRACE_MS;
    },
    onclick(e: MouseEvent) {
      if (e.target === dialog() && down) close();
      down = false;
    },
  };
}
