// A theme's chart.area_opacity (design/THEMES.md) reaches the charts' area fills: LineChart's gradient top and
// Sparkline's fill scale with it relative to the mode's default, so the default Dark and Light draw exactly what they
// drew before (0.08 and 0.12) and a custom theme's 0 removes the fill, as on Desktop.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { customTheme } from '../design/themes';
import { themeIds, themes } from '../design/tokens';

const read = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');
const cssText = read('../design/tokens.css');

/** The `calc(...)` a rule of `file`'s scoped CSS gives `property` under `selector`. */
function calcOf(file: string, selector: string, property: string): string {
  const rule = new RegExp(`\\${selector} \\{[^}]*?${property}: (calc\\([^;]*\\));`).exec(read(file));
  if (!rule) throw new Error(`${file}: no ${property} on ${selector}`);
  return rule[1]!;
}

/** `calc` evaluated with the custom properties `vars` (only * and / of numbers and var()s, as the rules use). */
function evaluate(calc: string, vars: Record<string, number>): number {
  const expr = calc.replace(/^calc\(|\)$/g, '').replace(/var\((--[\w-]+)\)/g, (_, name: string) => {
    if (!(name in vars)) throw new Error(`no ${name}`);
    return String(vars[name]);
  });
  if (!/^[\d.\s*/]+$/.test(expr)) throw new Error(`not a product: ${expr}`);
  return Function(`return ${expr}`)() as number;
}

/** A mode block's value of `name` in tokens.css. */
function modeValue(mode: 'dark' | 'light', name: string): number {
  const sel = mode === 'dark' ? ':root,\n:root[data-theme="dark"] {' : ':root[data-theme="light"] {';
  const block = cssText.slice(cssText.indexOf(sel), cssText.indexOf('}', cssText.indexOf(sel)));
  const m = new RegExp(`  ${name}: ([\\d.]+);`).exec(block);
  if (!m) throw new Error(`tokens.css: no ${name} in the ${mode} block`);
  return Number(m[1]);
}

const charts: [string, string, string, number][] = [
  ['./LineChart.tsx', '.area-top', 'stop-opacity', 0.08],
  ['./Sparkline.tsx', '.a', 'opacity', 0.12],
];

describe('chart.area_opacity', () => {
  it("each mode block holds its default theme's value", () => {
    for (const mode of ['dark', 'light'] as const) {
      expect(modeValue(mode, '--chart-area-opacity-default')).toBe(themes[mode].color.chart.area_opacity);
      expect(modeValue(mode, '--chart-area-opacity')).toBe(themes[mode].color.chart.area_opacity);
    }
  });

  for (const [file, selector, property, before] of charts) {
    it(`${file} draws ${before} in every built-in theme of the defaults' value and scales with a custom one`, () => {
      const calc = calcOf(file, selector, property);
      for (const id of themeIds) {
        const t = themes[id];
        const vars = { '--chart-area-opacity': t.color.chart.area_opacity, '--chart-area-opacity-default': themes[t.mode].color.chart.area_opacity };
        // Exactly the old constant when the theme keeps its mode's value (every built-in today).
        if (vars['--chart-area-opacity'] === vars['--chart-area-opacity-default']) expect(evaluate(calc, vars), id).toBe(before);
      }
      const custom = (area: number) => customTheme({ folo_theme: 1, id: 'x', name: 'X', mode: 'dark', color: { chart: { area_opacity: area } } });
      expect(custom(0).vars).toEqual([['--chart-area-opacity', '0']]);
      const at = (area: number) => evaluate(calc, { '--chart-area-opacity': area, '--chart-area-opacity-default': themes.dark.color.chart.area_opacity });
      expect(at(0)).toBe(0);
      expect(at(0.28)).toBeCloseTo(before * 2, 10);
    });
  }
});
