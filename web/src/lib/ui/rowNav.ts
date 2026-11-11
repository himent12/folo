// Keyboard rows for list tables (Runs, Models, Datasets): one roving tab stop over the rows, J / K and ↓ / ↑ move it,
// Home / End jump, Enter opens the row (its click), and an optional `toggle` answers X (select for compare). J or K
// pressed while nothing on the page has focus enters the list at its first row, as in Linear. With single-key shortcuts
// off (Settings, ui/keyPrefs) the letters do nothing; the arrows, Home, End and Enter keep working on a focused row.

import { isTyping } from '$lib/keys';
import { keyPrefs } from './keyPrefs';

export interface RowNavOptions {
  /** X on a focused row (e.g. toggle its Compare selection). */
  toggle?: (row: HTMLElement) => void;
}

/** Rows that take part: any `[data-row]` that is rendered (phones hide the desktop table). */
export function navRows(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('[data-row]')].filter((r) => r.checkVisibility?.() ?? true);
}

/** Gives exactly one row the tab stop: `keep` when it is still a row, else the first. */
export function settleTabStop(rows: HTMLElement[], keep?: HTMLElement | null): HTMLElement | undefined {
  const stop = keep && rows.includes(keep) ? keep : rows[0];
  for (const r of rows) r.tabIndex = r === stop ? 0 : -1;
  return stop;
}

/** The row `delta` steps from `from` (clamped), or the first / last row for `'first'` / `'last'`. */
export function stepRow(rows: HTMLElement[], from: HTMLElement | null, delta: number | 'first' | 'last'): HTMLElement | undefined {
  if (!rows.length) return undefined;
  if (delta === 'first') return rows[0];
  if (delta === 'last') return rows[rows.length - 1];
  const i = from ? rows.indexOf(from) : -1;
  if (i < 0) return rows[0];
  return rows[Math.min(rows.length - 1, Math.max(0, i + delta))];
}

const MOVES: Record<string, number | 'first' | 'last'> = { j: 1, ArrowDown: 1, k: -1, ArrowUp: -1, Home: 'first', End: 'last' };
/** The letters among the keys above (and X), which the single-key preference turns off. */
const LETTERS = new Set(['j', 'k', 'x']);

/** Action (`use:rowNav`) for the element holding the rows (a table, or a list). */
export function rowNav(root: HTMLElement, opts: RowNavOptions = {}) {
  let options = opts;
  let current: HTMLElement | null = null;

  const settle = () => {
    current = settleTabStop(navRows(root), current) ?? null;
  };
  const focusRow = (row: HTMLElement | undefined) => {
    if (!row) return;
    current = row;
    settleTabStop(navRows(root), row);
    row.focus();
    row.scrollIntoView?.({ block: 'nearest' });
  };

  const onkeydown = (e: KeyboardEvent) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    const row = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-row]');
    // Only the row itself: keys inside its checkbox or link keep their own meaning.
    if (!row || row !== e.target || !root.contains(row)) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (LETTERS.has(key) && !keyPrefs.singleKeys) return;
    const move = MOVES[key];
    if (move !== undefined) {
      e.preventDefault();
      focusRow(stepRow(navRows(root), row, move));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      row.click();
    } else if (key === 'x' && options.toggle) {
      e.preventDefault();
      options.toggle(row);
    }
  };

  const onfocusin = (e: FocusEvent) => {
    const row = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-row]');
    if (row && root.contains(row) && row !== current) {
      current = row;
      settleTabStop(navRows(root), row);
    }
  };

  // J / K with nothing focused enters the list (the page's other single keys live in shell/shortcuts.ts).
  const onwindowkey = (e: KeyboardEvent) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
    if (!keyPrefs.singleKeys || document.querySelector('dialog[open]')) return;
    const k = e.key.toLowerCase();
    if (k !== 'j' && k !== 'k') return;
    const active = document.activeElement;
    if (active && active !== document.body && !(active instanceof HTMLElement && active.matches('main, h1, [data-route-focus]'))) return;
    const rows = navRows(root);
    if (!rows.length) return;
    e.preventDefault();
    focusRow(current && rows.includes(current) ? current : rows[0]);
  };

  const observer = typeof MutationObserver === 'undefined' ? null : new MutationObserver(settle);
  observer?.observe(root, { childList: true, subtree: true });
  root.addEventListener('keydown', onkeydown);
  root.addEventListener('focusin', onfocusin);
  window.addEventListener('keydown', onwindowkey);
  settle();

  return {
    update(next: RowNavOptions = {}) {
      options = next;
    },
    destroy() {
      observer?.disconnect();
      root.removeEventListener('keydown', onkeydown);
      root.removeEventListener('focusin', onfocusin);
      window.removeEventListener('keydown', onwindowkey);
    },
  };
}
