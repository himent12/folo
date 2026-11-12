// @vitest-environment happy-dom
// A double-click on the button that opens a dialog must not close it again (runs F5): the second press lands on the
// backdrop right after opening. A press that starts on the backdrop later closes it; one that starts inside never does.
// Only that second press is let through: a dialog opened from the keyboard, or a press elsewhere, closes at once
// (review shell R5).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { backdropClose, OPEN_GRACE_MS } from './backdrop';

describe('backdropClose', () => {
  let dialog: HTMLDialogElement;
  let inside: HTMLElement;
  let closed: number;
  let b: ReturnType<typeof backdropClose>;

  /** A press anywhere on the page, as the document hears it (the button that opens the dialog, say). */
  const pagePress = (x: number, y: number) => {
    document.dispatchEvent(new PointerEvent('pointerdown', { clientX: x, clientY: y }));
    document.dispatchEvent(new PointerEvent('pointerup', { clientX: x, clientY: y }));
  };
  /** A press on the open dialog: it starts on `down` and is released on `up`. */
  const press = (down: Element, up: Element, x = 100, y = 100) => {
    b.onpointerdown({ target: down, clientX: x, clientY: y } as unknown as PointerEvent);
    b.onclick({ target: up } as unknown as MouseEvent);
  };

  beforeEach(() => {
    vi.useFakeTimers();
    dialog = document.createElement('dialog');
    inside = dialog.appendChild(document.createElement('button'));
    closed = 0;
    b = backdropClose(() => dialog, () => closed++);
  });
  afterEach(() => vi.useRealTimers());

  it('ignores the rest of the double-click that opened it, then closes on a press outside', () => {
    pagePress(100, 100);
    b.opened();
    vi.advanceTimersByTime(120);
    press(dialog, dialog, 101, 99); // the double-click's second click
    expect(closed).toBe(0);

    vi.advanceTimersByTime(OPEN_GRACE_MS);
    press(inside, dialog); // a drag from inside, released outside
    expect(closed).toBe(0);
    press(dialog, dialog);
    expect(closed).toBe(1);
  });

  it('closes on the first press outside when the keyboard or code opened it (Ctrl+K)', () => {
    // A click long before does not make this opening a pointer's.
    pagePress(100, 100);
    vi.advanceTimersByTime(2000);
    b.opened();
    vi.advanceTimersByTime(50);
    press(dialog, dialog);
    expect(closed).toBe(1);
  });

  it('lets through only the press that continues the opening one: elsewhere, or a second one, closes', () => {
    pagePress(100, 100);
    b.opened();
    vi.advanceTimersByTime(450);
    press(dialog, dialog, 600, 400); // a quick dismissal, away from the button
    expect(closed).toBe(1);

    pagePress(100, 100);
    b.opened();
    vi.advanceTimersByTime(100);
    press(inside, inside, 100, 100); // a press inside the box first: the double-click is over
    vi.advanceTimersByTime(100);
    press(dialog, dialog, 100, 100);
    expect(closed).toBe(2);
  });
});
