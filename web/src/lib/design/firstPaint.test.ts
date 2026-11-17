// @vitest-environment happy-dom
// index.html's first-paint script (design/THEMES.md: no flash of the default theme) gives <html> what the app then
// gives it (applyTheme with prefs' resolution), for every kind of stored state, and knows the built-in themes tokens.json
// lists.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { PrefsStore, THEMES_KEY } from '../prefs';
import { applyTheme } from './applyTheme';
import { themeIds, themes } from './tokens';

const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8');
const source = /<script id="first-paint">([\s\S]*?)<\/script>/.exec(html)![1]!;
const root = document.documentElement;

/** <html>'s theme state: the two attributes and every inline custom property. */
function state() {
  const vars = [...Array.from({ length: root.style.length }, (_, i) => root.style.item(i))].filter((n) => n.startsWith('--')).sort();
  return { theme: root.dataset.theme, themeId: root.dataset.themeId, vars: vars.map((n) => [n, root.style.getPropertyValue(n)]) };
}
function reset() {
  root.removeAttribute('style');
  delete root.dataset.theme;
  delete root.dataset.themeId;
}
const firstPaint = () => {
  reset();
  new Function(source)();
  return state();
};
const app = () => {
  reset();
  applyTheme(root, new PrefsStore().activeTheme);
  return state();
};

const night = { folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', base: 'midnight', color: { $c: 1, bg: { app: '#0B0D12' }, chart: { series: ['#000001', '#000002', '#000003', '#000004', '#000005', '#000006'], area_opacity: 0.3 } } };
const pale = { folo_theme: 1, id: 'pale', name: 'Pale', mode: 'light', color: { text: { primary: '#111111' } } };

describe('the first-paint script', () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => {
    localStorage.clear();
    reset();
  });

  it('lists the built-in themes of tokens.json by mode', () => {
    const list = /const BUILTIN = (\{[^;]*\});/.exec(source)![1]!;
    const builtin = new Function(`return ${list}`)() as Record<string, string[]>;
    expect(builtin).toEqual({ dark: themeIds.filter((id) => themes[id].mode === 'dark'), light: themeIds.filter((id) => themes[id].mode === 'light') });
  });

  const states: [string, Record<string, string>][] = [
    ['nothing stored', {}],
    ['light mode', { 'folo.theme': 'light' }],
    ['a built-in dark theme', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ darkTheme: 'midnight' }) }],
    ['a built-in light theme', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ lightTheme: 'contrast-light' }) }],
    ['a theme of the other mode in a slot', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ darkTheme: 'paper' }) }],
    ['an id that names nothing', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ lightTheme: 'gone' }) }],
    ['a custom dark theme on its base', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night, pale]) }],
    ['a custom light theme on the default', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ lightTheme: 'pale' }), [THEMES_KEY]: JSON.stringify([night, pale]) }],
    ['the other mode in use', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night]) }],
    ['broken JSON everywhere', { 'folo.theme': 'dark', 'folo.prefs': '{', [THEMES_KEY]: '[' }],
  ];
  for (const [what, stored] of states) {
    it(`shows what the app shows: ${what}`, () => {
      for (const [k, v] of Object.entries(stored)) localStorage.setItem(k, v);
      const first = firstPaint();
      expect(first).toEqual(app());
      expect(first.theme).toBe(stored['folo.theme'] === 'light' ? 'light' : 'dark');
    });
  }

  it('sets a custom theme as its base and its own colours', () => {
    localStorage.setItem('folo.prefs', JSON.stringify({ darkTheme: 'night' }));
    localStorage.setItem(THEMES_KEY, JSON.stringify([night]));
    const s = firstPaint();
    expect(s.themeId).toBe('midnight');
    expect(s.vars).toContainEqual(['--bg-app', '#0b0d12']);
    expect(s.vars).toContainEqual(['--chart-6', '#000006']);
    expect(s.vars).toContainEqual(['--chart-area-opacity', '0.3']);
  });

  it('leaves the app other inline properties of <html> (the dock height) when the theme changes', () => {
    reset();
    root.style.setProperty('--dock-h', '64px');
    localStorage.setItem('folo.prefs', JSON.stringify({ darkTheme: 'night' }));
    localStorage.setItem(THEMES_KEY, JSON.stringify([night]));
    const p = new PrefsStore();
    applyTheme(root, p.activeTheme);
    p.useTheme('dim');
    applyTheme(root, p.activeTheme);
    expect(state().vars).toEqual([['--dock-h', '64px']]);
    expect(root.dataset.themeId).toBe('dim');
  });
});
