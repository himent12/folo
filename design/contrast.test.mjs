// Tests for contrast.mjs's theme half: the custom theme format (the shared cases in theme-cases.json), resolution,
// and the checks. Run: node --test design/contrast.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { builtIns, builtInPalette, validateTheme, checkTheme } from './contrast.mjs';

const read = (name) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const tokens = read('./tokens.json');
const cases = read('./theme-cases.json');
const at = (palette, path) => path.split('.').reduce((v, k) => v?.[k], palette);

test('every invalid case is refused with its path and message', () => {
  for (const { what, theme, error } of cases.invalid) {
    assert.equal(validateTheme(theme).error, error, what);
  }
});

test('every valid case resolves on its base', () => {
  for (const { what, theme, expect } of cases.valid) {
    const r = validateTheme(theme);
    assert.equal(r.error, undefined, `${what}: ${r.error}`);
    assert.equal(r.theme.base, expect.base, what);
    assert.equal(r.theme.name, expect.name, what);
    for (const [path, value] of Object.entries(expect.color)) assert.equal(at(r.theme.palette, path), value, `${what}: ${path}`);
  }
});

test('built-ins: listing order, the defaults have no overrides, every palette is complete', () => {
  assert.deepEqual(builtIns.map((t) => t.id), ['dark', 'midnight', 'dim', 'contrast-dark', 'light', 'paper', 'contrast-light']);
  assert.deepEqual(builtInPalette('dark'), tokens.color.dark);
  assert.deepEqual(builtInPalette('light'), tokens.color.light);
  const paths = (o, p = '') => Object.entries(o).flatMap(([k, v]) => (v && typeof v === 'object' && !Array.isArray(v) ? paths(v, `${p}${k}.`) : [`${p}${k}`]));
  for (const t of builtIns) {
    const palette = builtInPalette(t.id);
    assert.deepEqual(paths(palette), paths(tokens.color[t.mode]), t.id);
    for (const key of ['series', 'fill', 'text']) assert.equal(palette.chart[key].length, 6, `${t.id} chart.${key}`);
    // The accent stays the signal lime in every built-in theme.
    assert.equal(palette.accent.solid, tokens.color[t.mode].accent.solid, `${t.id} accent.solid`);
  }
});

test('every built-in theme and example file passes', () => {
  for (const t of builtIns) {
    const r = checkTheme({ mode: t.mode, palette: builtInPalette(t.id), highContrast: t.id.startsWith('contrast-') });
    assert.deepEqual(r.failures, [], t.id);
  }
  for (const f of ['night-shift', 'sepia']) {
    const { theme, error } = validateTheme(read(`./themes/${f}.json`));
    assert.equal(error, undefined, f);
    assert.deepEqual(checkTheme(theme).failures, [], f);
  }
});

test('the checks catch a low-contrast pair, a broken elevation and the high-contrast floors', () => {
  const dim = (color, base = 'dark') => validateTheme({ folo_theme: 1, id: 'z', name: 'Z', mode: 'dark', base, color }).theme;
  const low = checkTheme(dim({ text: { tertiary: '#4A4F57' } }));
  assert.ok(low.failures.some((f) => f.startsWith('text.tertiary') && f.includes('(min 4.5)')), low.failures.join('\n'));
  const flat = checkTheme(dim({ bg: { inset: '#0E1013' } }));
  assert.ok(flat.failures.some((f) => f.startsWith('elevation: bg.inset')), flat.failures.join('\n'));
  // Dark's own palette passes at AA but not at the high-contrast floors.
  const hc = checkTheme({ mode: 'dark', palette: builtInPalette('dark'), highContrast: true });
  assert.ok(hc.failures.some((f) => f.startsWith('text.tertiary') && f.includes('(min 7)')));
  assert.ok(hc.failures.some((f) => f.startsWith('line.default') && f.includes('on bg.app')));
});

test('--theme: exit 0 on a passing file, 1 with the path on an invalid or failing one', () => {
  const script = fileURLToPath(new URL('./contrast.mjs', import.meta.url));
  const dir = mkdtempSync(join(tmpdir(), 'folo-theme-'));
  try {
    const run = (theme) => {
      const file = join(dir, 't.json');
      writeFileSync(file, typeof theme === 'string' ? theme : JSON.stringify(theme));
      return spawnSync(process.execPath, [script, '--theme', file], { encoding: 'utf8' });
    };
    const ok = run({ folo_theme: 1, id: 'ok', name: 'Ok', mode: 'light', base: 'paper' });
    assert.equal(ok.status, 0, ok.stderr);
    assert.match(ok.stdout, /^ok {3}ok /m);
    const typo = run({ folo_theme: 1, id: 'typo', name: 'Typo', mode: 'dark', color: { bg: { ap: '#000000' } } });
    assert.equal(typo.status, 1);
    assert.match(typo.stderr, /t\.json: color\.bg\.ap: not a colour token/);
    const broken = run('{ "folo_theme": 1, ');
    assert.equal(broken.status, 1);
    assert.match(broken.stderr, /\(file\): not valid JSON/);
    const faint = run({ folo_theme: 1, id: 'faint', name: 'Faint', mode: 'dark', color: { text: { secondary: '#3A3D42' } } });
    assert.equal(faint.status, 1);
    assert.match(faint.stderr, /FAIL faint .*\n.*text\.secondary/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
