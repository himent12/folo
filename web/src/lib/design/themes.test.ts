// The custom theme format and theme resolution (design/THEMES.md). design/theme-cases.json is shared with
// design/contrast.mjs and Desktop, so the three validators refuse the same files with the same words.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { checkCustomTheme, compareCodePoints, cssVarsOf, customTheme, findTheme, importThemes, parseThemeFile, readStoredThemes, readThemeFile, resolveTheme, themesFor, type CustomThemeFile, type ParsedFile } from './themes';
import { colorVars, themeIds, themes } from './tokens';

const design = new URL('../../../../design/', import.meta.url);
type Case = { what: string; theme?: unknown; text?: string; hex?: string };
const cases = JSON.parse(readFileSync(new URL('theme-cases.json', design), 'utf8')) as {
  invalid: (Case & { error: string })[];
  valid: (Case & { expect: { base: string; name: string; description?: string; color: Record<string, unknown> } })[];
};
/** A case's file bytes: its `hex`, or its raw `text` as UTF-8; none for a parsed `theme`. */
const bytesOf = (c: Case) => (c.hex !== undefined ? Uint8Array.from(c.hex.match(/../g)!.map((h) => parseInt(h, 16))) : c.text !== undefined ? new TextEncoder().encode(c.text) : undefined);
/**
 * A case checked as Studio reads it: a file's bytes through the import's own reader (a picked File, `readThemeFile`),
 * or its parsed `theme` as a stored entry.
 */
async function check(c: Case): Promise<{ ok: true; theme: CustomThemeFile } | { ok: false; error: string }> {
  const bytes = bytesOf(c);
  if (!bytes) return checkCustomTheme(c.theme);
  const { file: _, ...r } = await readThemeFile(new File([bytes], 'case.json', { type: 'application/json' }));
  return r;
}
/** The case's error: exact, but `(file): not valid JSON` takes any parser's detail after it (theme-cases.json). */
const sameError = (got: string, want: string) => got === want || (want === '(file): not valid JSON' && got.startsWith(`${want} (`));
const at = (o: unknown, path: string) => path.split('.').reduce<unknown>((v, k) => (v as Record<string, unknown> | undefined)?.[k], o);
const file = (over: Partial<CustomThemeFile> = {}): CustomThemeFile => ({ folo_theme: 1, id: 'night', name: 'Night', mode: 'dark', ...over });

describe('custom theme files (design/theme-cases.json)', () => {
  it('refuses every invalid case with its exact path and message', async () => {
    expect(cases.invalid.length).toBeGreaterThan(30);
    expect([...cases.invalid, ...cases.valid].filter((c) => c.text !== undefined).length, 'the raw cases').toBeGreaterThanOrEqual(8);
    expect([...cases.invalid, ...cases.valid].filter((c) => c.hex !== undefined).length, 'the byte cases').toBeGreaterThanOrEqual(10);
    for (const c of cases.invalid) {
      const r = await check(c);
      expect(r.ok, c.what).toBe(false);
      if (!r.ok && !sameError(r.error, c.error)) expect(r.error, c.what).toBe(c.error);
    }
  });

  it('accepts every valid case and resolves it on its base', async () => {
    for (const c of cases.valid) {
      const r = await check(c);
      if (!r.ok) throw new Error(`${c.what}: ${r.error}`);
      const t = customTheme(r.theme);
      expect(t.base, c.what).toBe(c.expect.base);
      expect(t.name, c.what).toBe(c.expect.name);
      if (c.expect.description !== undefined) expect(t.description, c.what).toBe(c.expect.description);
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

  it('refuses a file over 1 MiB by its size, unread, and takes one of exactly 1 MiB', async () => {
    const big = new File(['{}'.padEnd((1 << 20) + 1, ' ')], 'big.json');
    big.arrayBuffer = () => Promise.reject(new Error('a refused file was read'));
    expect(await readThemeFile(big)).toEqual({ file: 'big.json', ok: false, error: '(file): larger than 1 MB; not a theme' });
    const text = JSON.stringify(file({ id: 'edge' }));
    expect(await readThemeFile(new File([text.padEnd(1 << 20, ' ')], 'edge.json'))).toMatchObject({ file: 'edge.json', ok: true, theme: { id: 'edge' } });
  });

  it('refuses a file nested about 1 MiB deep as not JSON, without overflowing the stack (the scan is iterative)', async () => {
    const n = (1 << 19) - 100;
    const deep = `{"folo_theme": 1, "id": "deep", "name": "Deep", "mode": "dark", "$c": ${'['.repeat(n)}${']'.repeat(n)}}`;
    expect(deep.length).toBeLessThanOrEqual(1 << 20);
    expect(await readThemeFile(new File([deep], 'deep.json'))).toEqual({ file: 'deep.json', ok: false, error: '(file): not valid JSON (nested too deep)' });
  });

  it("turns anything thrown while one file is read into that file's error", async () => {
    const throwing = (what: string) => ({ name: `${what}.json`, size: 10, arrayBuffer: () => Promise.reject(new Error(what)) }) as unknown as File;
    expect(await readThemeFile(throwing('denied'))).toEqual({ file: 'denied.json', ok: false, error: "(file): couldn't read (denied)" });
  });
});

describe('resolution', () => {
  it('lists each mode: built-ins in their order, then custom themes by name', () => {
    const customs = [file({ id: 'zz', name: 'Zebra' }), file({ id: 'aa', name: 'Apple' }), file({ id: 'pl', name: 'Pale', mode: 'light' })];
    expect(themesFor('dark', customs).map((t) => t.id)).toEqual([...themeIds.filter((id) => themes[id].mode === 'dark'), 'aa', 'zz']);
    expect(themesFor('light', customs).map((t) => t.id)).toEqual(['light', 'paper', 'contrast-light', 'pl']);
    expect(themeIds[0]).toBe('dark');
  });

  it('orders custom themes by lower-cased name, then id, by code point (as Desktop does; no locale collation)', () => {
    // The same list as Desktop's catalog test (apps/desktop/src/theme/catalog.rs).
    const names = ['😀 Smile', 'Ａpple', 'éclair', 'Ember', 'apple', 'Same', 'same'];
    const customs = names.map((name, i) => file({ id: `t${names.length - i}`, name }));
    // a (U+61) < e < s (two "same": by id, t1 then t2) < é (U+E9) < fullwidth a (U+FF41) < 😀 (U+1F600).
    expect(themesFor('dark', customs).slice(4).map((t) => t.name)).toEqual(['apple', 'Ember', 'same', 'Same', 'éclair', 'Ａpple', '😀 Smile']);
    // By code point, not by UTF-16 unit: U+FFFF is before U+1F600, whose first unit (U+D83D) is lower.
    expect(compareCodePoints('￿', '😀')).toBeLessThan(0);
    expect('￿' < '😀').toBe(false);
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
  // #382 (after #342): a zero-width name showed as an empty one and an empty "Remove " label; a control character or a
  // bidi override showed its own way in each list. Studio's import only, for now (Desktop's twin and the shared cases
  // follow together); a theme stored before the rule stays installed.
  it('refuses an imported name with control or invisible characters, or with nothing visible', () => {
    const named = (name: string) => parseThemeFile(JSON.stringify(file({ name })));
    const hidden = 'name: must not hold control or invisible characters';
    for (const name of ['​', 'A​B', 'A\u0007B', 'A\nB', 'A\tB', 'A‮B', 'A B', '﻿A'])
      expect(named(name), JSON.stringify(name)).toEqual({ ok: false, error: hidden });
    for (const name of ['‍', '́́', '⠀', 'ㅤ', 'ᅟᅠ', 'ﾠ ⠀']) expect(named(name), JSON.stringify(name)).toEqual({ ok: false, error: 'name: must have a visible character' });
    // Joiners inside emoji sequences and scripts, variation selectors, accents, CJK and emoji stay names.
    for (const name of ['👩‍💻 night', 'می‌خواهم', '☀️ day', 'Café', '夜班', '🌈'.repeat(40)]) expect(named(name).ok, JSON.stringify(name)).toBe(true);
    expect(readStoredThemes([file({ name: 'A​B' })]).map((t) => t.name)).toEqual(['A​B']);
    expect(readStoredThemes([file({ name: 'ㅤ' })]).map((t) => t.name)).toEqual(['ㅤ']);
  });

  it('installs, replaces by id in place, and says why a file was refused', () => {
    const parsed = (name: string, v: unknown): ParsedFile => ({ file: name, ...parseThemeFile(JSON.stringify(v)) });
    const { themes: out, results } = importThemes([file({ id: 'a', name: 'A' }), file({ id: 'b', name: 'B' })], [
      parsed('a2.json', file({ id: 'a', name: 'A two' })),
      parsed('c.json', file({ id: 'c', name: 'C' })),
      parsed('typo.json', { ...file({ id: 'd' }), color: { bg: { ap: '#000000' } } }),
      parsed('c-again.json', file({ id: 'c', name: 'C again' })),
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
