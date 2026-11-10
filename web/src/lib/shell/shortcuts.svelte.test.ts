// @vitest-environment happy-dom
// Single-key shortcuts can be turned off (WCAG 2.1.4): the global keys and the list letters stop, Mod+K, the arrows
// and the Settings button that opens the shortcuts dialog keep working. The choice persists in localStorage.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { router } from '$lib/router.svelte';
import { keyPrefs } from '$lib/ui/keyPrefs.svelte';
import { rowNav } from '$lib/ui/rowNav';
import { backPath, installShortcuts, showShortcuts } from './shortcuts';

const key = (k: string, init: KeyboardEventInit = {}, el: EventTarget = window) =>
  el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init }));

let uninstall: () => void;
const palette = vi.fn();
const shortcuts = vi.fn();

beforeEach(() => {
  palette.mockReset();
  shortcuts.mockReset();
  uninstall = installShortcuts({ palette, shortcuts });
});
afterEach(() => {
  uninstall();
  keyPrefs.singleKeys = true;
  document.body.innerHTML = '';
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
