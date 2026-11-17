// @vitest-environment happy-dom
// Themes in the preferences (design/THEMES.md): each mode's slot, the mode/slot rule of Choosing a theme, the custom
// themes store (`folo.themes`) across tabs, and what is stored.
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { CustomThemeFile } from './design/themes';
import { PrefsStore, sanitize, THEMES_KEY } from './prefs';

const prefsOf = () => JSON.parse(localStorage.getItem('folo.prefs')!) as Record<string, unknown>;
const night: CustomThemeFile = { folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', base: 'midnight', color: { bg: { app: '#0B0D12' } } };
const fromOtherTab = (key: string, value: string) => {
  localStorage.setItem(key, value);
  window.dispatchEvent(new StorageEvent('storage', { key, newValue: value }));
};

describe('theme preferences', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it('defaults each mode to its own theme', () => {
    const p = new PrefsStore();
    expect([p.darkTheme, p.lightTheme, p.activeTheme.id]).toEqual(['dark', 'light', 'dark']);
    p.theme = 'light';
    expect(p.activeTheme).toMatchObject({ id: 'light', base: 'light', mode: 'light', vars: [] });
  });

  it('keeps a stored id that names no theme, and shows the default for it', () => {
    expect(sanitize({ darkTheme: 'gone', lightTheme: 7 })).toMatchObject({ darkTheme: 'gone', lightTheme: 'light' });
    localStorage.setItem('folo.prefs', JSON.stringify({ theme: 'dark', darkTheme: 'gone' }));
    const p = new PrefsStore();
    expect([p.darkTheme, p.activeTheme.id, p.slotTheme('dark').id]).toEqual(['gone', 'dark', 'dark']);
    // Installing a theme with that id brings it back.
    p.installThemes([{ name: 'gone.json', text: JSON.stringify({ ...night, id: 'gone' }) }]);
    expect(p.activeTheme).toMatchObject({ id: 'gone', custom: true });
  });

  it("picks a theme of the other mode: switches the mode under Dark or Light, keeps System", () => {
    const p = new PrefsStore();
    expect(p.useTheme('paper')).toBe(true);
    expect([p.theme, p.lightTheme, p.darkTheme, p.activeTheme.id]).toEqual(['light', 'paper', 'dark', 'paper']);
    p.useTheme('midnight');
    expect([p.theme, p.darkTheme, p.lightTheme]).toEqual(['dark', 'midnight', 'paper']);
    p.theme = 'system';
    p.useTheme('contrast-light');
    expect([p.theme, p.lightTheme]).toEqual(['system', 'contrast-light']);
    expect(p.useTheme('nothing')).toBe(false);
    expect(prefsOf()).toMatchObject({ theme: 'system', darkTheme: 'midnight', lightTheme: 'contrast-light' });
    // A new tab reads them back.
    const q = new PrefsStore();
    expect([q.theme, q.darkTheme, q.lightTheme]).toEqual(['system', 'midnight', 'contrast-light']);
  });

  it('installs, stores, removes custom themes; a mode that used a removed one shows its default', () => {
    const p = new PrefsStore();
    const r = p.installThemes([
      { name: 'night.json', text: JSON.stringify(night) },
      { name: 'bad.json', text: '{' },
    ]);
    expect(r.map((x) => x.ok)).toEqual([true, false]);
    expect(JSON.parse(localStorage.getItem(THEMES_KEY)!)).toEqual([night]);
    expect(p.themesFor('dark').map((t) => t.id)).toEqual(['dark', 'midnight', 'dim', 'contrast-dark', 'night']);
    p.useTheme('night');
    expect(p.activeTheme).toMatchObject({ id: 'night', base: 'midnight', vars: [['--bg-app', '#0b0d12']] });
    p.removeTheme('night');
    expect([p.darkTheme, p.activeTheme.id, p.customThemes.length]).toEqual(['night', 'dark', 0]);
    expect(JSON.parse(localStorage.getItem(THEMES_KEY)!)).toEqual([]);
  });

  it("follows another tab's custom themes and theme choices", () => {
    const p = new PrefsStore();
    fromOtherTab(THEMES_KEY, JSON.stringify([night]));
    fromOtherTab('folo.prefs', JSON.stringify({ theme: 'dark', darkTheme: 'night' }));
    expect(p.activeTheme.id).toBe('night');
    // Installing here keeps what the other tab installed.
    const q = new PrefsStore();
    q.installThemes([{ name: 'b.json', text: JSON.stringify({ ...night, id: 'b', name: 'B' }) }]);
    fromOtherTab(THEMES_KEY, localStorage.getItem(THEMES_KEY)!);
    expect(p.customThemes.map((t) => t.id)).toEqual(['night', 'b']);
    localStorage.clear();
    window.dispatchEvent(new StorageEvent('storage', { key: null }));
    expect([p.customThemes.length, p.activeTheme.id]).toEqual([0, 'dark']);
  });

  it('ignores a stored list that is not JSON or holds broken entries', () => {
    localStorage.setItem(THEMES_KEY, '[{"folo_theme":1');
    expect(new PrefsStore().customThemes).toEqual([]);
    localStorage.setItem(THEMES_KEY, JSON.stringify([night, { ...night, id: 'x', mode: 'grey' }]));
    expect(new PrefsStore().customThemes.map((t) => t.id)).toEqual(['night']);
  });
});
