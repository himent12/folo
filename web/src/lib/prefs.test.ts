// @vitest-environment happy-dom
// Preferences in two tabs of one browser (shell F2): a tab never puts back what another tab changed, and it shows the
// other tab's choices without a reload (the server address excepted: it stays on the server it is connected to).
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
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
