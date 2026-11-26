// Keyboard rows for list tables (Runs, Models, Datasets): one roving tab stop over the rows, J / K and ↓ / ↑ move it,
// Home / End jump, Enter opens the row (its click), and an optional `toggle` answers X (select for compare). J or K
// pressed while nothing on the page has focus enters the list at its first row, or at the row it was last on when the
// page is shown again (Esc or Back from that row's page), as in Linear. With single-key shortcuts
// off (Settings, ui/keyPrefs) the letters do nothing; the arrows, Home, End and Enter keep working on a focused row.

import { isTyping } from '$lib/keys';
import { keyPrefs } from './keyPrefs';

export interface RowNavOptions {
  /** X on a focused row (e.g. toggle its Compare selection). */
  toggle?: (row: HTMLElement) => void;
}

function isRowVisible(r: HTMLElement): boolean {
  if (r.hidden || r.style.display === 'none') return false;
  return !r.checkVisibility || r.checkVisibility();
}

/** Rows that take part: any visible `[data-row]` inside a visible container (phones hide the desktop table).
 *  Checks container visibility once up front so hidden tables skip querying and inspecting every row. */
export function navRows(root: HTMLElement): HTMLElement[] {
  if (root.checkVisibility && !root.checkVisibility()) return [];
  const list = root.querySelectorAll<HTMLElement>('[data-row]');
  const out: HTMLElement[] = [];
  for (let i = 0; i < list.length; i++) {
    const r = list[i]!;
    if (isRowVisible(r)) out.push(r);
  }
  return out;
}

/** Steps from `from` to the next visible `[data-row]` without calling `checkVisibility()` on every row in the table. */
function stepVisibleRow(root: HTMLElement, from: HTMLElement, move: number | 'first' | 'last'): HTMLElement | undefined {
  if (root.checkVisibility && !root.checkVisibility()) return undefined;
  const list = root.querySelectorAll<HTMLElement>('[data-row]');
  const n = list.length;
  if (!n) return undefined;
  if (move === 'first') {
    for (let i = 0; i < n; i++) if (isRowVisible(list[i]!)) return list[i];
    return undefined;
  }
  if (move === 'last') {
    for (let i = n - 1; i >= 0; i--) if (isRowVisible(list[i]!)) return list[i];
    return undefined;
  }
  let idx = -1;
  for (let i = 0; i < n; i++) {
    if (list[i] === from) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    for (let i = 0; i < n; i++) if (isRowVisible(list[i]!)) return list[i];
    return undefined;
  }
  const dir = move > 0 ? 1 : -1;
  for (let i = idx + dir; i >= 0 && i < n; i += dir) {
    if (isRowVisible(list[i]!)) return list[i];
  }
  return isRowVisible(from) ? from : undefined;
}

function hasRowNode(nodes: NodeList): boolean {
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n && n.nodeType === 1) {
      const el = n as Element;
      if (el.hasAttribute('data-row') || el.querySelector('[data-row]')) return true;
    }
  }
  return false;
}

function hasRowMutation(records: MutationRecord[]): boolean {
  for (let i = 0; i < records.length; i++) {
    const rec = records[i]!;
    if (hasRowNode(rec.addedNodes) || hasRowNode(rec.removedNodes)) return true;
  }
  return false;
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

/**
 * The row each list was last on (focused or opened), by the list's page and label: a list mounted again (Esc or Back
 * from the row's page) puts its tab stop there, so J, K and Tab go on from the row just read instead of the first one
 * (shell F6). For this tab's session only.
 */
const lastRow = new Map<string, string>();
const listKey = (root: HTMLElement) => `${location.hash.split('?')[0]}|${root.getAttribute('aria-label') ?? ''}`;
/** A row's identity: its entity's `data-id` (names are not unique); a row without one is not remembered. */
const rowKey = (row: HTMLElement) => row.dataset.id;

/** Forgets the rows lists were on (tests). */
export function forgetRows() {
  lastRow.clear();
}

/** Action (`use:rowNav`) for the element holding the rows (a table, or a list). */
export function rowNav(root: HTMLElement, opts: RowNavOptions = {}) {
  let options = opts;
  let current: HTMLElement | null = null;
  // Until the user moves in this list, the row it was last on is where it starts, as soon as that row is rendered.
  let resumed = false;
  const list = listKey(root);
  const remember = (row: HTMLElement) => {
    resumed = true;
    const id = rowKey(row);
    if (id === undefined) lastRow.delete(list);
    else lastRow.set(list, id);
  };

  const settle = () => {
    const rows = navRows(root);
    const was = resumed ? undefined : lastRow.get(list);
    const back = was === undefined ? undefined : rows.find((r) => rowKey(r) === was);
    current = settleTabStop(rows, back ?? current) ?? null;
  };
  const focusRow = (row: HTMLElement | undefined, rows?: HTMLElement[]) => {
    if (!row) return;
    remember(row);
    const prev = current;
    current = row;
    if (rows) settleTabStop(rows, row);
    else if (prev && prev !== row && root.contains(prev) && prev.tabIndex === 0) {
      prev.tabIndex = -1;
      row.tabIndex = 0;
    } else settleTabStop(navRows(root), row);
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
      focusRow(stepVisibleRow(root, row, move));
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
    if (row && root.contains(row)) {
      remember(row);
      if (row === current && row.tabIndex === 0) return;
      const prev = current;
      current = row;
      if (prev && root.contains(prev) && prev.tabIndex === 0) {
        prev.tabIndex = -1;
        row.tabIndex = 0;
      } else {
        settleTabStop(navRows(root), row);
      }
    }
  };
  // A row opened by a click counts as the one the list was on.
  const onclick = (e: MouseEvent) => {
    const row = (e.target as HTMLElement | null)?.closest<HTMLElement>('[data-row]');
    if (row && root.contains(row)) remember(row);
  };

  const hasOpenPopover = (): boolean => {
    if (document.querySelector('.info-pop.open')) return true;
    try {
      return !!document.querySelector('[popover]:popover-open');
    } catch {
      return false;
    }
  };

  // J / K with nothing focused enters the list (the page's other single keys live in shell/shortcuts.ts).
  const onwindowkey = (e: KeyboardEvent) => {
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey || isTyping(e)) return;
    if (!keyPrefs.singleKeys || document.querySelector('dialog[open], [role="menu"], .g-chord') || hasOpenPopover()) return;
    const k = e.key.toLowerCase();
    if (k !== 'j' && k !== 'k') return;
    const active = document.activeElement;
    if (active && active !== document.body && !(active instanceof HTMLElement && active.matches('main, h1, [data-route-focus]'))) return;
    const rows = navRows(root);
    if (!rows.length) return;
    e.preventDefault();
    focusRow(current && rows.includes(current) ? current : rows[0], rows);
  };

  const observer = typeof MutationObserver === 'undefined' ? null : new MutationObserver((records) => {
    if (hasRowMutation(records)) settle();
  });
  observer?.observe(root, { childList: true, subtree: true });
  root.addEventListener('keydown', onkeydown);
  root.addEventListener('focusin', onfocusin);
  root.addEventListener('click', onclick);
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
      root.removeEventListener('click', onclick);
      window.removeEventListener('keydown', onwindowkey);
    },
  };
}
