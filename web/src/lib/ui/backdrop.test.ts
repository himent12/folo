// @vitest-environment happy-dom
// A double-click on the button that opens a dialog must not close it again (runs F5): the second press lands on the
// backdrop right after opening. A press that starts on the backdrop later closes it; one that starts inside never does.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { backdropClose, OPEN_GRACE_MS } from './backdrop';

describe('backdropClose', () => {
  afterEach(() => vi.useRealTimers());

  it('ignores the rest of the double-click that opened it, then closes on a press outside', () => {
    vi.useFakeTimers();
    const dialog = document.createElement('dialog');
    const inside = dialog.appendChild(document.createElement('button'));
    let closed = 0;
    const b = backdropClose(() => dialog, () => closed++);
    const press = (down: Element, up: Element) => {
      b.onpointerdown({ target: down } as unknown as PointerEvent);
      b.onclick({ target: up } as unknown as MouseEvent);
    };

    b.opened();
    vi.advanceTimersByTime(120);
    press(dialog, dialog); // the double-click's second click
    expect(closed).toBe(0);

    vi.advanceTimersByTime(OPEN_GRACE_MS);
    press(inside, dialog); // a drag from inside, released outside
    expect(closed).toBe(0);
    press(dialog, dialog);
    expect(closed).toBe(1);
  });
});
