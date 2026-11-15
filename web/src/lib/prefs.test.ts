// @vitest-environment happy-dom
// Preferences in two tabs of one browser (shell F2): a tab never puts back what another tab changed, and it shows the
// other tab's choices without a reload (the server address excepted: it stays on the server it is connected to).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PrefsStore } from './prefs';
import { keyPrefs } from './ui/keyPrefs';

const KEY = 'folo.prefs';
const read = () => JSON.parse(localStorage.getItem(KEY)!) as Record<string, unknown>;
/** What the browser does in every other tab when one tab writes a key. */
const fromOtherTab = (key: string, value: string) => {
  localStorage.setItem(key, value);
  window.dispatchEvent(new StorageEvent('storage', { key, newValue: value }));
};

describe('preferences across tabs', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("saving one preference keeps what another tab stored for the others", () => {
    const a = new PrefsStore();
    const b = new PrefsStore();
    a.theme = 'light';
    a.units = 'metric';
    a.serverUrl = 'http://gpu-box:7878/';
    // Tab B, opened before, collapses its sidebar.
    b.sidebarCollapsed = true;
    expect(read()).toMatchObject({ theme: 'light', units: 'metric', serverUrl: 'http://gpu-box:7878', sidebarCollapsed: true });
    expect(localStorage.getItem('folo.theme')).toBe('light');
    // And a later version's keys stay.
    fromOtherTab(KEY, JSON.stringify({ ...read(), later: 1 }));
    b.units = 'binary';
    expect(read()).toMatchObject({ later: 1, theme: 'light', units: 'binary' });
  });

  it('keeps a change whose write failed for the next save (review of #220)', () => {
    const a = new PrefsStore();
    const full = vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('full', 'QuotaExceededError');
    });
    a.serverUrl = 'http://gpu-box:7878';
    full.mockRestore();
    expect(localStorage.getItem(KEY)).toBeNull();
    a.sidebarCollapsed = true;
    expect(read()).toMatchObject({ serverUrl: 'http://gpu-box:7878', sidebarCollapsed: true });
  });

  it("keeps a change whose write failed when another tab saves meanwhile (review of #220, round 2)", () => {
    const a = new PrefsStore();
    const full = vi.spyOn(localStorage, 'setItem').mockImplementation(() => {
      throw new DOMException('full', 'QuotaExceededError');
    });
    a.units = 'metric';
    full.mockRestore();
    // Space is freed; tab B saves its sidebar (still binary in storage).
    fromOtherTab(KEY, JSON.stringify({ theme: 'dark', serverUrl: '', units: 'binary', sidebarCollapsed: true, labelAgent: null }));
    expect([a.units, a.sidebarCollapsed]).toEqual(['metric', true]);
    a.theme = 'light';
    expect(read()).toMatchObject({ units: 'metric', sidebarCollapsed: true, theme: 'light' });
  });

  it("shows another tab's theme, units and sidebar at once, but keeps its own server address", () => {
    const b = new PrefsStore();
    expect([b.theme, b.units, b.sidebarCollapsed, b.serverUrl]).toEqual(['dark', 'binary', false, '']);
    fromOtherTab(KEY, JSON.stringify({ theme: 'light', serverUrl: 'http://gpu-box:7878', units: 'metric', sidebarCollapsed: true, labelAgent: null }));
    fromOtherTab('folo.theme', 'light');
    expect([b.theme, b.resolvedTheme, b.units, b.sidebarCollapsed, b.serverUrl]).toEqual(['light', 'light', 'metric', true, '']);
    // A storage cleared elsewhere: back to the defaults.
    localStorage.clear();
    window.dispatchEvent(new StorageEvent('storage', { key: null }));
    expect([b.theme, b.units, b.sidebarCollapsed]).toEqual(['dark', 'binary', false]);
  });

  it('turns single-key shortcuts off and on as another tab does', () => {
    fromOtherTab('folo.singleKeys', 'off');
    expect(keyPrefs.singleKeys).toBe(false);
    fromOtherTab('folo.singleKeys', 'on');
    expect(keyPrefs.singleKeys).toBe(true);
  });
});

// Stored preferences of the wrong type (another writer, an older version, a hand edit; s6-shell-1): a number or null for
// the server address made every request go to "42" and Settings throw while rendering, so nothing in the app could fix it.
describe('stored preferences of the wrong type', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  for (const [what, stored] of [
    ['wrong types', { serverUrl: 42, theme: 'neon', units: 7, sidebarCollapsed: 'yes', labelAgent: 'gpt' }],
    ['nulls', { serverUrl: null, theme: null, units: null, sidebarCollapsed: null, labelAgent: null }],
    ['an array', [1, 2, 3]],
    ['a string', 'http://x'],
  ] as const) {
    it(`fall back to the defaults: ${what}`, () => {
      localStorage.setItem(KEY, JSON.stringify(stored));
      const p = new PrefsStore();
      expect(p.serverUrl).toBe('');
      expect(p.theme).toBe('dark');
      expect(p.units).toBe('binary');
      expect(p.sidebarCollapsed).toBe(false);
      expect(p.labelAgent).toBeNull();
    });
  }

  it('keep the values of the right type and the keys of a later version', () => {
    localStorage.setItem(KEY, JSON.stringify({ serverUrl: 'http://gpu-box:7878', theme: 'light', units: 'metric', sidebarCollapsed: 1, later: { a: 1 } }));
    const p = new PrefsStore();
    expect([p.serverUrl, p.theme, p.units, p.sidebarCollapsed]).toEqual(['http://gpu-box:7878', 'light', 'metric', false]);
    p.units = 'binary';
    expect(read()).toMatchObject({ later: { a: 1 }, serverUrl: 'http://gpu-box:7878', units: 'binary' });
  });

  it('a value of the wrong type from another tab is not taken', () => {
    const p = new PrefsStore();
    fromOtherTab(KEY, JSON.stringify({ theme: 5, sidebarCollapsed: 'yes' }));
    expect(p.theme).toBe('dark');
    expect(p.sidebarCollapsed).toBe(false);
  });
});
