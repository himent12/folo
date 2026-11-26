// @vitest-environment happy-dom
// Single-key shortcuts can be turned off (WCAG 2.1.4): the global keys and the list letters stop, Mod+K, the arrows
// and the Settings button that opens the shortcuts dialog keep working. The choice persists in localStorage.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { router } from '$lib/router';
import { keyPrefs } from '$lib/ui/keyPrefs';
import { rowNav } from '$lib/ui/rowNav';
import { backPath, G_CHORD_MS, gChord, installShortcuts, shortcutList, showShortcuts } from './shortcuts';

const key = (k: string, init: KeyboardEventInit = {}, el: EventTarget = window) =>
  el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init }));

let uninstall: () => void;
const palette = vi.fn();
const shortcuts = vi.fn();
const onGChord = vi.fn();

beforeEach(() => {
  palette.mockReset();
  shortcuts.mockReset();
  onGChord.mockReset();
  uninstall = installShortcuts({ palette, shortcuts, gChord: onGChord });
});
afterEach(() => {
  uninstall();
  keyPrefs.singleKeys = true;
  document.body.innerHTML = '';
  vi.useRealTimers();
});

describe('single-key shortcuts preference', () => {
  it('is on by default and persists when turned off', () => {
    expect(keyPrefs.singleKeys).toBe(true);
    keyPrefs.singleKeys = false;
    expect(localStorage.getItem('folo.singleKeys')).toBe('off');
    keyPrefs.singleKeys = true;
    expect(localStorage.getItem('folo.singleKeys')).toBe('on');
  });

  it('on: ? opens the shortcuts and N starts a run', () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    key('?');
    expect(shortcuts).toHaveBeenCalledOnce();
    key('n');
    expect(go).toHaveBeenCalledWith('/runs/new');
    go.mockRestore();
  });

  it('touch-only devices get no shortcuts reference: ? does nothing', () => {
    const mm = vi.spyOn(window, 'matchMedia').mockImplementation(
      (q: string) => ({ matches: q.includes('hover: none'), media: q, addEventListener() {}, removeEventListener() {} }) as unknown as MediaQueryList,
    );
    key('?');
    showShortcuts();
    expect(shortcuts).not.toHaveBeenCalled();
    mm.mockRestore();
  });

  it('off: ?, N and G sequences do nothing; Mod+K still opens the palette', () => {
    keyPrefs.singleKeys = false;
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    key('?');
    key('n');
    key('g');
    key('r');
    expect(shortcuts).not.toHaveBeenCalled();
    expect(go).not.toHaveBeenCalled();
    key('k', { ctrlKey: true });
    expect(palette).toHaveBeenCalledOnce();
    // The Settings button still opens the shortcuts dialog.
    showShortcuts();
    expect(shortcuts).toHaveBeenCalledOnce();
    go.mockRestore();
  });

  it('off: list rows ignore J / K / X but keep the arrow keys', () => {
    const t = document.createElement('table');
    t.innerHTML = `<tbody>${[0, 1, 2].map((i) => `<tr data-row data-id="r${i}"><td>r${i}</td></tr>`).join('')}</tbody>`;
    document.body.appendChild(t);
    const rows = [...t.querySelectorAll<HTMLTableRowElement>('tr')];
    const toggle = vi.fn();
    const nav = rowNav(t, { toggle });
    keyPrefs.singleKeys = false;
    // J with nothing focused does not enter the list.
    key('j');
    expect(document.activeElement).not.toBe(rows[0]);
    rows[0]!.focus();
    key('j', {}, rows[0]!);
    expect(document.activeElement).toBe(rows[0]);
    key('x', {}, rows[0]!);
    expect(toggle).not.toHaveBeenCalled();
    key('ArrowDown', {}, rows[0]!);
    expect(document.activeElement).toBe(rows[1]);
    nav.destroy();
  });

});

describe('Escape on a detail page', () => {
  it('goes back to the list when nothing on the page is using it', () => {
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const route = vi.spyOn(router, 'route', 'get').mockReturnValue({ name: 'run', params: { id: 'r' }, query: new URLSearchParams(), path: '/runs/r' });
    key('Escape');
    expect(go).toHaveBeenCalledWith('/runs');
    go.mockClear();
    // A focused control keeps its own Escape.
    const b = document.createElement('button');
    document.body.appendChild(b);
    b.focus();
    key('Escape', {}, b);
    expect(go).not.toHaveBeenCalled();
    b.remove();

    // An open fallback info popover keeps Escape from navigating back even when focus is at rest.
    const pop = document.createElement('div');
    pop.className = 'info-pop open';
    document.body.appendChild(pop);
    key('Escape');
    expect(go).not.toHaveBeenCalled();
    pop.remove();

    // A native Popover API popover (:popover-open) also keeps Escape from navigating back.
    const nativePop = document.createElement('div');
    nativePop.setAttribute('popover', 'auto');
    document.body.appendChild(nativePop);
    const origQs = document.querySelector.bind(document);
    const qsSpy = vi.spyOn(document, 'querySelector').mockImplementation((sel: string) =>
      sel === '[popover]:popover-open' ? nativePop : origQs(sel),
    );
    try {
      key('Escape');
      expect(go).not.toHaveBeenCalled();
    } finally {
      qsSpy.mockRestore();
      nativePop.remove();
    }

    route.mockRestore();
    go.mockRestore();
  });

  it('knows where each detail page goes back to', () => {
    expect(['run', 'run-compare', 'model', 'dataset', 'label-jobs', 'label-job', 'runs', 'settings'].map((name) => backPath({ name }))).toEqual([
      '/runs',
      '/runs',
      '/models',
      '/datasets',
      '/datasets',
      '/datasets/label',
      null,
      null,
    ]);
  });
});

describe('G-chord navigation state and guards', () => {
  it('arms gChord and calls onGChord on bare g, ignores Shift, navigates on target, and clears', () => {
    vi.useFakeTimers();
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    expect(gChord.get()).toBe(false);

    key('g');
    expect(gChord.get()).toBe(true);
    expect(onGChord).toHaveBeenCalledTimes(1);
    // Holding g (repeat) does not re-fire onGChord.
    key('g', { repeat: true });
    expect(onGChord).toHaveBeenCalledTimes(1);
    // Modifier press before second key does not cancel the chord.
    key('Shift', { shiftKey: true });
    expect(gChord.get()).toBe(true);

    key('R', { shiftKey: true });
    expect(gChord.get()).toBe(false);
    expect(go).toHaveBeenCalledWith('/runs');
    go.mockRestore();
  });

  it('cancels gChord on timeout, Escape (without navigating back), blur, and uninstall', () => {
    vi.useFakeTimers();
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    const route = vi.spyOn(router, 'route', 'get').mockReturnValue({ name: 'run', params: { id: 'r' }, query: new URLSearchParams(), path: '/runs/r' });

    // Timeout
    key('g');
    expect(gChord.get()).toBe(true);
    vi.advanceTimersByTime(G_CHORD_MS);
    expect(gChord.get()).toBe(false);

    // Escape cancels chord without triggering detail-page back navigation.
    key('g');
    expect(gChord.get()).toBe(true);
    key('Escape');
    expect(gChord.get()).toBe(false);
    expect(go).not.toHaveBeenCalled();

    // Window blur cancels chord.
    key('g');
    expect(gChord.get()).toBe(true);
    window.dispatchEvent(new Event('blur'));
    expect(gChord.get()).toBe(false);

    // Uninstall while armed clears chord cleanly.
    key('g');
    expect(gChord.get()).toBe(true);
    uninstall();
    expect(gChord.get()).toBe(false);
    // Re-install so afterEach uninstall stays a clean teardown.
    uninstall = installShortcuts({ palette, shortcuts, gChord: onGChord });

    route.mockRestore();
    go.mockRestore();
  });

  it('never arms gChord or invokes onGChord while typing, modified (including Shift+G), in an open dialog, or with singleKeys off', () => {
    const input = document.body.appendChild(document.createElement('input'));
    input.focus();
    key('g', {}, input);
    expect(gChord.get()).toBe(false);
    expect(onGChord).not.toHaveBeenCalled();
    input.remove();

    key('g', { ctrlKey: true });
    expect(gChord.get()).toBe(false);
    expect(onGChord).not.toHaveBeenCalled();

    key('G', { shiftKey: true });
    expect(gChord.get()).toBe(false);
    expect(onGChord).not.toHaveBeenCalled();

    const dlg = document.body.appendChild(document.createElement('dialog'));
    dlg.setAttribute('open', '');
    key('g');
    expect(gChord.get()).toBe(false);
    expect(onGChord).not.toHaveBeenCalled();
    // Ctrl+K is also suppressed over unrelated open modal dialogs, but works when Shortcuts or Palette is open.
    key('k', { ctrlKey: true });
    expect(palette).not.toHaveBeenCalled();
    dlg.className = 'shortcuts';
    key('k', { ctrlKey: true });
    expect(palette).toHaveBeenCalledTimes(1);
    dlg.className = 'palette';
    key('k', { ctrlKey: true });
    expect(palette).toHaveBeenCalledTimes(2);
    dlg.remove();

    keyPrefs.singleKeys = false;
    key('g');
    expect(gChord.get()).toBe(false);
    expect(onGChord).not.toHaveBeenCalled();
  });
});

describe('shortcutList reference', () => {
  it('documents Compare 1–4 pin, Escape unpin, and updated B baseline wording', () => {
    expect(shortcutList).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ keys: 'B', group: 'Charts', label: 'Make the next run or entry the baseline (Compare)', single: true }),
        expect.objectContaining({ keys: '1–4', group: 'Charts', label: 'Pin or unpin a run or entry (Compare)', single: true }),
        expect.objectContaining({ keys: 'Escape', group: 'Charts', label: 'Unpin the highlighted run or entry (Compare)', single: true }),
      ]),
    );
  });
});
