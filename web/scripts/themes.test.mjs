// scripts/themes.mjs: the built-in themes tokens.mjs writes, resolved, and refused loudly when malformed.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { builtinThemes, cssVar, leaves } from './themes.mjs';

const tokens = JSON.parse(readFileSync(new URL('../../design/tokens.json', import.meta.url), 'utf8'));
/** tokens.json with `themes` replaced. */
const withThemes = (/** @type {Record<string, unknown>} */ themes) => ({ ...tokens, themes });
const dflt = { dark: { name: 'Dark', mode: 'dark', color: {} }, light: { name: 'Light', mode: 'light', color: {} } };

describe('builtinThemes', () => {
  it("resolves tokens.json's themes in order, each over its mode's base", () => {
    const out = builtinThemes(tokens);
    expect(out.map((t) => t.id)).toEqual(['dark', 'midnight', 'dim', 'contrast-dark', 'light', 'paper', 'contrast-light']);
    expect(out[0]?.color).toEqual(Object.fromEntries(Object.entries(tokens.color.dark).filter(([k]) => !k.startsWith('$'))));
    for (const t of out) expect(leaves(t.color).map(([p]) => p)).toEqual(leaves(tokens.color[t.mode]).map(([p]) => p));
  });

  it('merges key by key and replaces arrays whole', () => {
    const series = ['#000001', '#000002', '#000003', '#000004', '#000005', '#000006'];
    const t = builtinThemes(withThemes({ ...dflt, x: { name: 'X', mode: 'dark', color: { bg: { app: '#010203' }, chart: { series } } } })).find((th) => th.id === 'x');
    expect(t?.color.bg).toMatchObject({ app: '#010203', panel: tokens.color.dark.bg.panel });
    expect(t?.color.chart).toMatchObject({ series, fill: tokens.color.dark.chart.fill });
  });

  it('fails loudly, naming the path, on a malformed theme', () => {
    const bad = (/** @type {Record<string, unknown>} */ extra) => () => builtinThemes(withThemes({ ...dflt, ...extra }));
    expect(bad({ x: { name: 'X', mode: 'dusk', color: {} } })).toThrow('themes.x.mode');
    expect(bad({ x: { name: '', mode: 'dark', color: {} } })).toThrow('themes.x.name');
    expect(bad({ x: { name: 'X', mode: 'dark', colour: {} } })).toThrow('themes.x.colour: unknown key');
    expect(bad({ x: { name: 'X', mode: 'dark', color: { bg: { ap: '#000000' } } } })).toThrow('themes.x.color.bg.ap: not a colour token');
    expect(bad({ x: { name: 'X', mode: 'dark', color: { bg: { app: 'black' } } } })).toThrow('themes.x.color.bg.app');
    expect(bad({ x: { name: 'X', mode: 'dark', color: { chart: { series: ['#000000'] } } } })).toThrow('themes.x.color.chart.series');
    expect(bad({ x: { name: 'X', mode: 'dark', color: { chart: { area_opacity: 2 } } } })).toThrow('themes.x.color.chart.area_opacity');
    expect(() => builtinThemes(withThemes({ midnight: { name: 'M', mode: 'dark', color: {} }, light: dflt.light }))).toThrow('the first dark theme must be "dark"');
    expect(() => builtinThemes(withThemes({ dark: { ...dflt.dark, color: { bg: { app: '#000000' } } }, light: dflt.light }))).toThrow('has no overrides');
    expect(() => builtinThemes(withThemes({ dark: dflt.dark }))).toThrow('no light theme');
  });
});

describe('cssVar', () => {
  it("names a token's custom property as tokens.css does", () => {
    expect(['bg.app', 'accent.solid_hover', 'focus', 'chart.series.0', 'chart.fill.5', 'chart.text.2', 'chart.area_opacity'].map(cssVar)).toEqual([
      '--bg-app',
      '--accent-solid-hover',
      '--focus',
      '--chart-1',
      '--chart-fill-6',
      '--chart-text-3',
      '--chart-area-opacity',
    ]);
  });
});
