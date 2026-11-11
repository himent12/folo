// @vitest-environment happy-dom
// fw twin of rowNav.test.ts, on rowNav.fw.ts.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { rowNav, settleTabStop, stepRow } from './rowNav.fw';

afterEach(() => {
  document.body.innerHTML = '';
});

function table(n: number) {
  const t = document.createElement('table');
  t.innerHTML = `<tbody>${Array.from({ length: n }, (_, i) => `<tr data-row data-id="r${i}"><td><a href="#/r${i}">r${i}</a></td></tr>`).join('')}</tbody>`;
  document.body.appendChild(t);
  return { t, rows: [...t.querySelectorAll<HTMLTableRowElement>('tr')] };
}
const key = (el: Element, k: string, init: KeyboardEventInit = {}) =>
  el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init }));

describe('stepRow / settleTabStop', () => {
  const rows = [1, 2, 3].map(() => document.createElement('tr'));

  it('steps and clamps', () => {
    expect(stepRow(rows, rows[0]!, 1)).toBe(rows[1]);
    expect(stepRow(rows, rows[2]!, 1)).toBe(rows[2]);
    expect(stepRow(rows, rows[0]!, -1)).toBe(rows[0]);
    expect(stepRow(rows, null, 1)).toBe(rows[0]);
    expect(stepRow(rows, rows[1]!, 'last')).toBe(rows[2]);
    expect(stepRow([], null, 1)).toBeUndefined();
  });

  it('gives exactly one row the tab stop', () => {
    settleTabStop(rows, rows[1]);
    expect(rows.map((r) => r.tabIndex)).toEqual([-1, 0, -1]);
    settleTabStop(rows, document.createElement('tr'));
    expect(rows.map((r) => r.tabIndex)).toEqual([0, -1, -1]);
  });
});

describe('rowNav', () => {
  it('moves a roving focus with J / K and the arrows, and opens with Enter', () => {
    const { t, rows } = table(3);
    const nav = rowNav(t);
    expect(rows.map((r) => r.tabIndex)).toEqual([0, -1, -1]);
    rows[0]!.focus();
    key(rows[0]!, 'j');
    expect(document.activeElement).toBe(rows[1]);
    expect(rows.map((r) => r.tabIndex)).toEqual([-1, 0, -1]);
    key(rows[1]!, 'ArrowDown');
    expect(document.activeElement).toBe(rows[2]);
    key(rows[2]!, 'k');
    key(rows[1]!, 'ArrowUp');
    expect(document.activeElement).toBe(rows[0]);
    const open = vi.fn();
    rows[0]!.addEventListener('click', open);
    key(rows[0]!, 'Enter');
    expect(open).toHaveBeenCalledOnce();
    nav.destroy();
  });

  it('toggles with X only when asked to, and leaves keys inside a row alone', () => {
    const { t, rows } = table(2);
    const toggle = vi.fn();
    const nav = rowNav(t, { toggle });
    rows[1]!.focus();
    key(rows[1]!, 'x');
    expect(toggle).toHaveBeenCalledWith(rows[1]);
    // A key on the link inside the row is the link's.
    const link = rows[1]!.querySelector('a')!;
    link.focus();
    expect(key(link, 'k')).toBe(true);
    expect(document.activeElement).toBe(link);
    nav.destroy();
  });

  it('enters the list with J when nothing has focus, but not while typing or with a dialog open', () => {
    const { t, rows } = table(2);
    const nav = rowNav(t);
    (document.activeElement as HTMLElement | null)?.blur();
    key(document.body, 'j');
    expect(document.activeElement).toBe(rows[0]);

    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    key(input, 'j');
    expect(document.activeElement).toBe(input);

    input.blur();
    const dialog = document.createElement('dialog');
    dialog.setAttribute('open', '');
    document.body.appendChild(dialog);
    key(document.body, 'j');
    expect(document.activeElement).not.toBe(rows[0]);
    nav.destroy();
  });
});
