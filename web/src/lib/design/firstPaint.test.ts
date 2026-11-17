// @vitest-environment happy-dom
// index.html's first-paint script (design/THEMES.md: no flash of the default theme) gives <html> what the app then
// gives it (applyTheme with prefs' resolution), for every kind of stored state: every case of design/theme-cases.json
// stored as the active theme, malformed entries, the mode in folo.prefs only, System under both OS appearances. Its
// generated data (the built-in ids, the colour token table) matches tokens.ts, and handing over from the script to the
// app on one document leaves nothing of the script's behind.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PrefsStore, THEMES_KEY } from '../prefs';
import { applyTheme } from './applyTheme';
import { colorVars, themeIds, themes } from './tokens';

const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8');
const source = /<script id="first-paint">([\s\S]*?)<\/script>/.exec(html)![1]!;
const root = document.documentElement;
const cases = JSON.parse(readFileSync(resolve(process.cwd(), '../design/theme-cases.json'), 'utf8')) as {
  invalid: { what: string; theme?: unknown; text?: string }[];
  valid: { what: string; theme?: unknown; text?: string }[];
};

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
const runScript = () => new Function(source)();
const firstPaint = () => {
  reset();
  runScript();
  return state();
};
const app = () => {
  reset();
  applyTheme(root, new PrefsStore().activeTheme);
  return state();
};
/** The OS appearance matchMedia reports (System mode), for the script and for PrefsStore. */
function osDark(dark: boolean) {
  vi.stubGlobal('matchMedia', (q: string) => ({ matches: dark && q.includes('dark'), media: q, addEventListener() {}, removeEventListener() {} }));
}
const store = (stored: Record<string, string>) => {
  for (const [k, v] of Object.entries(stored)) localStorage.setItem(k, v);
};

const night = { folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', base: 'midnight', color: { $c: 1, bg: { app: '#0B0D12' }, chart: { series: ['#000001', '#000002', '#000003', '#000004', '#000005', '#000006'], area_opacity: 0.3 } } };
const pale = { folo_theme: 1, id: 'pale', name: 'Pale', mode: 'light', color: { text: { primary: '#111111' } } };
/** Sol's entry: flattened and non-colour keys, which must not reach <html>'s style. */
const malformed = { ...night, color: { 'bg.app': '#123456', space: { '4': 999 }, 'size-sidebar': 123 } };

describe('the first-paint script', () => {
  beforeEach(() => {
    localStorage.clear();
    osDark(true);
  });
  afterEach(() => {
    localStorage.clear();
    reset();
    vi.unstubAllGlobals();
  });

  it("has tokens.ts's built-in themes and colour tokens (scripts/tokens.mjs writes them)", () => {
    const line = /const T = (\{[^\n]*\});\n/.exec(source)![1]!;
    const T = JSON.parse(line) as { dark: string[]; light: string[]; color: Record<string, unknown>; number: string[] };
    expect({ dark: T.dark, light: T.light }).toEqual({ dark: themeIds.filter((id) => themes[id].mode === 'dark'), light: themeIds.filter((id) => themes[id].mode === 'light') });
    const names = (v: unknown): string[] => (typeof v === 'string' ? [v] : Array.isArray(v) ? (v as string[]) : Object.values(v as object).flatMap(names));
    expect(names(T.color)).toEqual(colorVars.map(([, name]) => name));
    expect(T.number).toEqual(['--chart-area-opacity']);
  });

  const states: [string, Record<string, string>, 'dark' | 'light'][] = [
    ['nothing stored', {}, 'dark'],
    ['light mode', { 'folo.theme': 'light' }, 'light'],
    ['a built-in dark theme', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ darkTheme: 'midnight' }) }, 'dark'],
    ['a built-in light theme', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ lightTheme: 'contrast-light' }) }, 'light'],
    ['a theme of the other mode in a slot', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ darkTheme: 'paper' }) }, 'dark'],
    ['an id that names nothing', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ lightTheme: 'gone' }) }, 'light'],
    ['a custom dark theme on its base', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night, pale]) }, 'dark'],
    ['a custom light theme on the default', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ lightTheme: 'pale' }), [THEMES_KEY]: JSON.stringify([night, pale]) }, 'light'],
    ['the other mode in use', { 'folo.theme': 'light', 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night]) }, 'light'],
    ['broken JSON everywhere', { 'folo.theme': 'dark', 'folo.prefs': '{', [THEMES_KEY]: '[' }, 'dark'],
    // The mode as prefs.ts reads it: a valid legacy key, else folo.prefs' theme, else dark.
    ['the mode in folo.prefs only', { 'folo.prefs': JSON.stringify({ theme: 'light', lightTheme: 'paper' }) }, 'light'],
    ['an invalid legacy key over folo.prefs', { 'folo.theme': 'blue', 'folo.prefs': JSON.stringify({ theme: 'light', lightTheme: 'paper' }) }, 'light'],
    ['a valid legacy key wins over folo.prefs', { 'folo.theme': 'dark', 'folo.prefs': JSON.stringify({ theme: 'light', darkTheme: 'dim' }) }, 'dark'],
    ['an invalid mode everywhere', { 'folo.theme': 'blue', 'folo.prefs': JSON.stringify({ theme: 7 }) }, 'dark'],
    ['folo.prefs not an object', { 'folo.theme': 'light', 'folo.prefs': '["paper"]' }, 'light'],
    ['an active entry with flattened and non-colour keys', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([malformed]) }, 'dark'],
    ['a broken entry before a good one with its id', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([{ ...night, name: '' }, night]) }, 'dark'],
    ['a good entry before a broken one with its id', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night, malformed]) }, 'dark'],
    ['an entry with an unknown key', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([{ ...night, extra: 1 }]) }, 'dark'],
    ['an entry deeper than the palette', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([{ ...night, color: { bg: { app: { deeper: { still: '#000000' } } } } }]) }, 'dark'],
    ['a stored list that is not a list', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify({ night }) }, 'dark'],
  ];
  for (const [what, stored, mode] of states) {
    it(`shows what the app shows: ${what}`, () => {
      store(stored);
      const first = firstPaint();
      expect(first).toEqual(app());
      expect(first.theme).toBe(mode);
    });
  }

  for (const dark of [true, false]) {
    it(`follows the OS under System, ${dark ? 'dark' : 'light'}`, () => {
      osDark(dark);
      store({ 'folo.theme': 'system', 'folo.prefs': JSON.stringify({ theme: 'system', darkTheme: 'night', lightTheme: 'pale' }), [THEMES_KEY]: JSON.stringify([night, pale]) });
      const first = firstPaint();
      expect(first).toEqual(app());
      expect([first.theme, first.themeId]).toEqual(dark ? ['dark', 'midnight'] : ['light', 'light']);
      expect(first.vars).toContainEqual(dark ? ['--bg-app', '#0b0d12'] : ['--text-primary', '#111111']);
    });
  }

  // Every shared format case, stored as the active theme: the script applies exactly the entries the app accepts. The
  // byte cases (`hex`) test the file reader (encoding), which a stored entry never meets: they are left out.
  const entries = [...cases.invalid.map((c) => ({ ...c, valid: false })), ...cases.valid.map((c) => ({ ...c, valid: true }))]
    .filter((c) => (c as { hex?: string }).hex === undefined)
    .map((c) => ({
      what: c.what,
      valid: c.valid,
      entry: (c.text !== undefined ? JSON.parse(c.text.replace(/^\uFEFF/, '')) : c.theme) as Record<string, unknown>,
    }));
  it('has the shared format cases to replay', () => expect(entries.length).toBeGreaterThan(50));
  for (const { what, valid, entry } of entries) {
    it(`applies a stored entry only when the app would (${valid ? 'valid' : 'invalid'}: ${what})`, () => {
      const mode = entry && (entry as { mode?: unknown }).mode === 'light' ? 'light' : 'dark';
      const id = entry && typeof (entry as { id?: unknown }).id === 'string' ? (entry as { id: string }).id : 'x';
      store({ 'folo.theme': mode, 'folo.prefs': JSON.stringify({ [`${mode}Theme`]: id }), [THEMES_KEY]: JSON.stringify([entry]) });
      const first = firstPaint();
      const shown = app();
      expect(first).toEqual(shown);
      // A valid case shows itself (its base's block); an invalid one, the mode's default with nothing inline (or the
      // built-in its id names: "id of a built-in theme").
      if (valid) expect(first.themeId).toBe((entry.base as string | undefined) ?? mode);
      else expect([first.themeId, first.vars]).toEqual([(themes as Record<string, { mode: string }>)[id]?.mode === mode ? id : mode, []]);
    });
  }

  it('sets a custom theme as its base and its own colours', () => {
    store({ 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night]) });
    const s = firstPaint();
    expect(s.themeId).toBe('midnight');
    expect(s.vars).toContainEqual(['--bg-app', '#0b0d12']);
    expect(s.vars).toContainEqual(['--chart-6', '#000006']);
    expect(s.vars).toContainEqual(['--chart-area-opacity', '0.3']);
  });

  // The script runs, then the app applies its theme to the same <html>: the result is the app's alone.
  const handovers: [string, Record<string, string>][] = [
    ['a custom theme', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night]) }],
    ['a malformed entry', { 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([malformed]) }],
    ['the mode in folo.prefs only', { 'folo.prefs': JSON.stringify({ theme: 'light', lightTheme: 'pale' }), [THEMES_KEY]: JSON.stringify([pale]) }],
  ];
  for (const [what, stored] of handovers) {
    it(`hands over to applyTheme on one document, leaving nothing behind: ${what}`, () => {
      store(stored);
      const shown = app();
      reset();
      runScript();
      applyTheme(root, new PrefsStore().activeTheme);
      expect(state()).toEqual(shown);
    });
  }

  it('leaves the app other inline properties of <html> (the dock height) when the theme changes', () => {
    reset();
    root.style.setProperty('--dock-h', '64px');
    store({ 'folo.prefs': JSON.stringify({ darkTheme: 'night' }), [THEMES_KEY]: JSON.stringify([night]) });
    const p = new PrefsStore();
    applyTheme(root, p.activeTheme);
    p.useTheme('dim');
    applyTheme(root, p.activeTheme);
    expect(state().vars).toEqual([['--dock-h', '64px']]);
    expect(root.dataset.themeId).toBe('dim');
  });
});
