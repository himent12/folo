// Tests for contrast.mjs's theme half: the custom theme format (the shared cases in theme-cases.json), resolution,
// and the checks. Run: node --test design/contrast.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { existsSync, openSync, closeSync } from 'node:fs';
import { builtIns, builtInPalette, validateTheme, validateCase, checkTheme, caseBytes, readThemeFile, readAtMost, sameError, THEME_FILE_MAX } from './contrast.mjs';
import { jsonProblem, JSON_MAX_DEPTH } from './json-scan.mjs';

const read = (name) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const tokens = read('./tokens.json');
const cases = read('./theme-cases.json');
const at = (palette, path) => path.split('.').reduce((v, k) => v?.[k], palette);

/**
 * Every case through the CLI's own file reader (`readThemeFile`, as `--theme` reads a file): its bytes written to a
 * file, or its parsed `theme` validated as is.
 */
function readCases(list, fn) {
  const dir = mkdtempSync(join(tmpdir(), 'folo-cases-'));
  try {
    for (const c of list) {
      const bytes = caseBytes(c);
      let r;
      if (bytes) {
        const file = join(dir, 'case.json');
        writeFileSync(file, bytes);
        r = readThemeFile(file);
      } else r = validateCase(c);
      fn(c, r);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test('every invalid case is refused with its path and message', () => {
  assert.ok(cases.invalid.filter((c) => c.text !== undefined).length >= 5, 'the raw cases (file text) are there');
  assert.ok([...cases.invalid, ...cases.valid].filter((c) => c.hex !== undefined).length >= 10, 'the byte cases are there');
  readCases(cases.invalid, (c, r) => {
    if (!sameError(r.error, c.error)) assert.equal(r.error, c.error, c.what);
  });
});

test('every valid case resolves on its base', () => {
  readCases(cases.valid, (c, r) => {
    const { what, expect } = c;
    assert.equal(r.error, undefined, `${what}: ${r.error}`);
    assert.equal(r.theme.base, expect.base, what);
    assert.equal(r.theme.name, expect.name, what);
    if (expect.description !== undefined) assert.equal(r.theme.description, expect.description, what);
    for (const [path, value] of Object.entries(expect.color)) assert.equal(at(r.theme.palette, path), value, `${what}: ${path}`);
  });
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
      writeFileSync(file, typeof theme === 'string' || Buffer.isBuffer(theme) ? theme : JSON.stringify(theme));
      return spawnSync(process.execPath, [script, '--theme', file], { encoding: 'utf8' });
    };
    const ok = run({ folo_theme: 1, id: 'ok', name: 'Ok', mode: 'light', base: 'paper' });
    assert.equal(ok.status, 0, ok.stderr);
    assert.match(ok.stdout, /^ok {3}ok /m);
    const typo = run({ folo_theme: 1, id: 'typo', name: 'Typo', mode: 'dark', color: { bg: { ap: '#000000' } } });
    assert.equal(typo.status, 1);
    assert.match(typo.stderr, /t\.json: color\.bg\.ap: not a colour token/);
    // A byte order mark (PowerShell 5.1's Set-Content -Encoding utf8 writes one) is skipped.
    const bom = run(`\uFEFF${JSON.stringify({ folo_theme: 1, id: 'bom', name: 'Bom', mode: 'dark' })}`);
    assert.equal(bom.status, 0, bom.stderr);
    // Exactly 1 MiB is read; one byte more is refused by its size, as in both apps.
    const edge = JSON.stringify({ folo_theme: 1, id: 'edge', name: 'Edge', mode: 'dark' });
    assert.equal(run(edge.padEnd(THEME_FILE_MAX, ' ')).status, 0);
    const big = run(edge.padEnd(THEME_FILE_MAX + 1, ' '));
    assert.equal(big.status, 1);
    assert.match(big.stderr, /t\.json: \(file\): larger than 1 MB; not a theme/);
    // Two byte order marks: only one is stripped. Windows-1252 (PowerShell 5.1's Set-Content default) is not UTF-8.
    assert.match(run(`\uFEFF\uFEFF${edge}`).stderr, /\(file\): not valid JSON/);
    assert.match(run(Buffer.from(edge.replace('Edge', 'Caf\u00e9'), 'latin1')).stderr, /\(file\): not UTF-8 text/);
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

test('json-scan: the JSON serde_json refuses and JSON.parse accepts, without recursion', () => {
  const header = '{"folo_theme": 1, "id": "x", "name": "X", "mode": "dark", "$c": ';
  const nest = (n, open = '[', close = ']') => `${header}${open.repeat(n)}${close.repeat(n)}}`;
  // The top-level object counts: JSON_MAX_DEPTH - 1 arrays inside it is the limit, one more is past it.
  assert.equal(JSON_MAX_DEPTH, 127);
  assert.equal(jsonProblem(nest(JSON_MAX_DEPTH - 1)), undefined);
  assert.equal(jsonProblem(nest(JSON_MAX_DEPTH)), 'nested too deep');
  assert.equal(jsonProblem(`${header}${'{"a":'.repeat(JSON_MAX_DEPTH - 2)}[]${'}'.repeat(JSON_MAX_DEPTH - 2)}}`), undefined);
  assert.equal(jsonProblem(`${header}${'{"a":'.repeat(JSON_MAX_DEPTH - 1)}[]${'}'.repeat(JSON_MAX_DEPTH - 1)}}`), 'nested too deep');
  // Brackets inside strings are not nesting; siblings don't add up.
  assert.equal(jsonProblem(`${header}"${'['.repeat(500)}"}`), undefined);
  assert.equal(jsonProblem(`${header}[${'[],'.repeat(500)}[]]}`), undefined);
  // About 1 MiB of nesting: refused, no stack overflow.
  assert.equal(jsonProblem(nest(500_000)), 'nested too deep');
  // Surrogate escapes, checked on the raw text (a duplicate member can't hide one); keys and values alike.
  const lone = 'a lone surrogate in a string';
  // The escapes as the file holds them (String.raw: a backslash, then u and four hex digits).
  for (const s of [String.raw`\uD800`, String.raw`\uDC00`, String.raw`\uDE00\uD83D`, String.raw`\uD83D\uD83D`, String.raw`\uD83Dx`, String.raw`\uD83D\n`, String.raw`A\uD800`]) {
    assert.equal(jsonProblem(`{"name": "${s}", "name": "ok"}`), lone, s);
    assert.equal(jsonProblem(`{"${s}": 1}`), lone, `key ${s}`);
  }
  assert.equal(jsonProblem(String.raw`{"name": "😀 😀 \"x\" A"}`), undefined, 'escaped pairs, either case');
  assert.equal(jsonProblem(String.raw`{"name": "\\uD800"}`), undefined, 'an escaped backslash, then text');
  assert.equal(jsonProblem('{"name": "\uD800"}'), lone, 'a raw lone surrogate (text that never was UTF-8)');
  assert.equal(jsonProblem('{"name": "😀"}'), undefined, 'a raw pair');
  // Numbers out of a double's range (JSON.parse reads Infinity); tiny ones read as 0 in both.
  for (const n of ['1e400', '-1e400', '1E309', `1${'0'.repeat(400)}`, '-2.5e+500']) assert.equal(jsonProblem(`{"a": ${n}}`), 'a number out of range', n);
  for (const n of ['1e-400', '1.7976931348623157e308', '-0', '0e999', '123456789012345678901234567890']) assert.equal(jsonProblem(`{"a": [${n}]}`), undefined, n);
  // Not JSON at all: left to JSON.parse.
  assert.equal(jsonProblem(String.raw`{"a": "\uZZZZ"}`), undefined);
  assert.equal(jsonProblem('{"a": "open'), undefined);
});

test('readThemeFile: a deeply nested file is an error naming the file, not a crash', () => {
  const dir = mkdtempSync(join(tmpdir(), 'folo-deep-'));
  try {
    const file = join(dir, 'deep.json');
    const n = 500_000;
    writeFileSync(file, `{"folo_theme": 1, "id": "deep", "name": "Deep", "mode": "dark", "$c": ${'['.repeat(n)}${']'.repeat(n)}}`);
    assert.deepEqual(readThemeFile(file), { error: '(file): not valid JSON (nested too deep)' });
    const script = fileURLToPath(new URL('./contrast.mjs', import.meta.url));
    const r = spawnSync(process.execPath, [script, '--theme', file], { encoding: 'utf8' });
    assert.equal(r.status, 1);
    assert.match(r.stderr, /deep\.json: \(file\): not valid JSON \(nested too deep\)\n$/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('readThemeFile: the read itself stops one byte past the limit', () => {
  const dir = mkdtempSync(join(tmpdir(), 'folo-cap-'));
  try {
    const file = join(dir, 'f.bin');
    writeFileSync(file, Buffer.alloc(100, 0x20));
    const fd = openSync(file, 'r');
    try {
      assert.equal(readAtMost(fd, 10).length, 10);
      assert.equal(readAtMost(fd, 1000).length, 90);
    } finally {
      closeSync(fd);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// A file whose size says 0 but whose read never ends (Linux's /dev/zero): the size check passes, the capped read refuses.
test('readThemeFile: a file that reads past its stated size is refused at the limit', { skip: !existsSync('/dev/zero') && 'no /dev/zero here' }, () => {
  assert.deepEqual(readThemeFile('/dev/zero'), { error: '(file): larger than 1 MB; not a theme' });
});
