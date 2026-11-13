// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { forgetRows, rowNav, settleTabStop, stepRow } from './rowNav';

afterEach(() => {
  document.body.innerHTML = '';
  forgetRows();
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

  it('starts again on the row it was last on when the list comes back, once that row is there (shell F6)', () => {
    location.hash = '#/runs';
    const first = table(4);
    let nav = rowNav(first.t);
    first.rows[0]!.focus();
    key(first.rows[0]!, 'j');
    key(first.rows[1]!, 'j');
    key(first.rows[2]!, 'Enter');
    nav.destroy();
    first.t.remove();
    // Esc or Back: the page mounts a new table, its rows arrive after the data.
    const again = document.createElement('table');
    document.body.appendChild(again);
    nav = rowNav(again);
    again.innerHTML = `<tbody>${[0, 1, 2, 3].map((i) => `<tr data-row data-id="r${i}"></tr>`).join('')}</tbody>`;
    return new Promise<void>((done) =>
      setTimeout(() => {
        const rows = [...again.querySelectorAll<HTMLElement>('tr')];
        expect(rows.map((r) => r.tabIndex)).toEqual([-1, -1, 0, -1]);
        (document.activeElement as HTMLElement | null)?.blur();
        key(window as unknown as Element, 'j');
        expect(document.activeElement).toBe(rows[2]);
        key(rows[2]!, 'j');
        expect(document.activeElement).toBe(rows[3]);
        nav.destroy();
        // Another list, or the same list on another page, starts at its top.
        location.hash = '#/models';
        const other = table(3);
        const n2 = rowNav(other.t);
        expect(other.rows.map((r) => r.tabIndex)).toEqual([0, -1, -1]);
        n2.destroy();
        done();
      }),
    );
  });

  it('restores a row by its id, not its name: two rows can share a name (review of #223)', () => {
    location.hash = '#/datasets';
    const make = () => {
      const t = document.createElement('table');
      t.setAttribute('aria-label', 'Datasets');
      t.innerHTML = `<tbody>${['ds_a', 'ds_b', 'ds_c'].map((id) => `<tr data-row data-id="${id}" aria-label="shapes"></tr>`).join('')}</tbody>`;
      document.body.appendChild(t);
      return { t, rows: [...t.querySelectorAll<HTMLElement>('tr')] };
    };
    const first = make();
    let nav = rowNav(first.t);
    first.rows[1]!.focus();
    nav.destroy();
    first.t.remove();
    const again = make();
    nav = rowNav(again.t);
    expect(again.rows.map((r) => r.tabIndex)).toEqual([-1, 0, -1]);
    nav.destroy();
  });
});

