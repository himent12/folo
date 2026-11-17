// The custom theme format and theme resolution (design/THEMES.md). design/theme-cases.json is shared with
// design/contrast.mjs and Desktop, so the three validators refuse the same files with the same words.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { checkCustomTheme, cssVarsOf, customTheme, findTheme, importThemes, parseThemeFile, readStoredThemes, resolveTheme, themesFor, type CustomThemeFile } from './themes';
import { colorVars, themeIds, themes } from './tokens';

const design = new URL('../../../../design/', import.meta.url);
const cases = JSON.parse(readFileSync(new URL('theme-cases.json', design), 'utf8')) as {
  invalid: { what: string; theme: unknown; error: string }[];
  valid: { what: string; theme: unknown; expect: { base: string; name: string; color: Record<string, unknown> } }[];
};
const at = (o: unknown, path: string) => path.split('.').reduce<unknown>((v, k) => (v as Record<string, unknown> | undefined)?.[k], o);
const file = (over: Partial<CustomThemeFile> = {}): CustomThemeFile => ({ folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', ...over });

describe('custom theme files (design/theme-cases.json)', () => {
  it('refuses every invalid case with its exact path and message', () => {
    expect(cases.invalid.length).toBeGreaterThan(30);
    for (const c of cases.invalid) expect(checkCustomTheme(c.theme), c.what).toEqual({ ok: false, error: c.error });
  });

  it('accepts every valid case and resolves it on its base', () => {
    for (const c of cases.valid) {
      const r = checkCustomTheme(c.theme);
      if (!r.ok) throw new Error(`${c.what}: ${r.error}`);
      const t = customTheme(r.theme);
      expect(t.base, c.what).toBe(c.expect.base);
      expect(t.name, c.what).toBe(c.expect.name);
      for (const [path, v] of Object.entries(c.expect.color)) expect(at(t.color, path), `${c.what}: ${path}`).toBe(v);
    }
  });

  it('says a file that is not JSON is not, and drops comments from what it stores', () => {
    const bad = parseThemeFile('{ "folo_theme": 1,');
    expect(bad.ok).toBe(false);
    expect(!bad.ok && bad.error).toMatch(/^\(file\): not valid JSON \(/);
    const r = parseThemeFile(JSON.stringify({ $comment: 'x', folo_theme: 1, id: 'a', name: ' A ', mode: 'dark', color: { $n: 1, bg: { $w: 2, app: '#000000' } } }));
    expect(r).toEqual({ ok: true, theme: { folo_theme: 1, id: 'a', name: 'A', mode: 'dark', color: { bg: { app: '#000000' } } } });
  });
});

describe('resolution', () => {
  it('lists each mode: built-ins in their order, then custom themes by name', () => {
    const customs = [file({ id: 'zz', name: 'Zebra' }), file({ id: 'aa', name: 'Apple' }), file({ id: 'pl', name: 'Pale', mode: 'light' })];
    expect(themesFor('dark', customs).map((t) => t.id)).toEqual([...themeIds.filter((id) => themes[id].mode === 'dark'), 'aa', 'zz']);
    expect(themesFor('light', customs).map((t) => t.id)).toEqual(['light', 'paper', 'contrast-light', 'pl']);
    expect(themeIds[0]).toBe('dark');
  });

  it("shows the mode's default for an id that names no theme of that mode", () => {
    expect(resolveTheme('midnight', 'dark', []).id).toBe('midnight');
    expect(resolveTheme('gone', 'dark', []).id).toBe('dark');
    expect(resolveTheme('paper', 'dark', []).id).toBe('dark');
    expect(resolveTheme('night', 'light', [file()]).id).toBe('light');
    expect(resolveTheme('night', 'dark', [file()])).toMatchObject({ id: 'night', custom: true, base: 'dark' });
    expect(findTheme('night', [])).toBeUndefined();
    expect(findTheme('paper', [])?.mode).toBe('light');
  });

  it('merges a custom theme over its base and sets only its own colours inline', () => {
    const t = customTheme(file({ base: 'midnight', color: { bg: { app: '#0B0D12' }, chart: { series: ['#000001', '#000002', '#000003', '#000004', '#000005', '#000006'], area_opacity: 0.2 } } }));
    expect(t.base).toBe('midnight');
    expect(t.color.bg.app).toBe('#0B0D12');
    expect(t.color.bg.panel).toBe(themes.midnight.color.bg.panel);
    expect(t.color.chart.fill).toEqual(themes.midnight.color.chart.fill);
    expect(t.vars).toEqual([
      ['--bg-app', '#0b0d12'],
      ...[1, 2, 3, 4, 5, 6].map((i) => [`--chart-${i}`, `#00000${i}`]),
      ['--chart-area-opacity', '0.2'],
    ]);
    expect(resolveTheme('midnight', 'dark', []).vars).toEqual([]);
  });

  it('names every colour token as tokens.css does', () => {
    const cssText = readFileSync(new URL('./tokens.css', import.meta.url), 'utf8');
    for (const [, name] of colorVars) expect(cssText).toContain(`  ${name}: `);
    expect(cssVarsOf({ focus: '#ABCDEF', chart: { text: ['#000000'], swatch_edge: '#FFFFFF1F' }, accent: { solid_hover: '#111111' } })).toEqual([
      ['--focus', '#abcdef'],
      ['--chart-text-1', '#000000'],
      ['--chart-swatch-edge', '#ffffff1f'],
      ['--accent-solid-hover', '#111111'],
    ]);
  });

  it('has a CSS block for every built-in theme but the defaults', () => {
    const cssText = readFileSync(new URL('./tokens.css', import.meta.url), 'utf8');
    for (const id of themeIds) expect(cssText.includes(`:root[data-theme-id="${id}"]`), id).toBe(id !== 'dark' && id !== 'light');
  });
});

describe('importing', () => {
  it('installs, replaces by id in place, and says why a file was refused', () => {
    const { themes: out, results } = importThemes([file({ id: 'a', name: 'A' }), file({ id: 'b', name: 'B' })], [
      { name: 'a2.json', text: JSON.stringify(file({ id: 'a', name: 'A two' })) },
      { name: 'c.json', text: JSON.stringify(file({ id: 'c', name: 'C' })) },
      { name: 'typo.json', text: JSON.stringify({ ...file({ id: 'd' }), color: { bg: { ap: '#000000' } } }) },
      { name: 'c-again.json', text: JSON.stringify(file({ id: 'c', name: 'C again' })) },
    ]);
    expect(out.map((t) => `${t.id}:${t.name}`)).toEqual(['a:A two', 'b:B', 'c:C again']);
    expect(results).toEqual([
      { file: 'a2.json', ok: true, id: 'a', name: 'A two', replaced: true },
      { file: 'c.json', ok: true, id: 'c', name: 'C', replaced: false },
      { file: 'typo.json', ok: false, error: 'color.bg.ap: not a colour token' },
      { file: 'c-again.json', ok: true, id: 'c', name: 'C again', replaced: true },
    ]);
  });

  it('reads the stored list, leaving out entries a hand edit broke and repeated ids', () => {
    expect(readStoredThemes({})).toEqual([]);
    expect(readStoredThemes([file(), { ...file(), mode: 'dusk' }, file({ name: 'Twice' }), 7]).map((t) => t.name)).toEqual(['Night']);
  });

  it('accepts the example files in design/themes', () => {
    for (const name of ['night-shift.json', 'sepia.json']) {
      const r = parseThemeFile(readFileSync(new URL(`themes/${name}`, design), 'utf8'));
      expect(r.ok, name).toBe(true);
    }
  });
});
