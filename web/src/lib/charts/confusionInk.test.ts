import { describe, expect, it } from 'vitest';
import raw from '../../../../design/tokens.json?raw';
import { CONFUSION_FLIP, rampShare, sharePct } from './confusionInk';

const tokens = JSON.parse(raw);

const rgb = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const gam = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
function toOklab(c: number[]) {
  const [r, g, b] = c.map(lin) as [number, number, number];
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function fromOklab([L, a, b]: number[]) {
  const l = (L! + 0.3963377774 * a! + 0.2158037573 * b!) ** 3;
  const m = (L! - 0.1055613458 * a! - 0.0638541728 * b!) ** 3;
  const s = (L! - 0.0894841775 * a! - 1.291485548 * b!) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s].map((x) =>
    Math.min(1, Math.max(0, gam(x))),
  );
}
const lum = (c: number[]) => {
  const [r, g, b] = c.map(lin) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a: number[], b: number[]) => (Math.max(lum(a), lum(b)) + 0.05) / (Math.min(lum(a), lum(b)) + 0.05);

describe('confusion-matrix ramp', () => {
  for (const theme of ['dark', 'light'] as const) {
    const ink = toOklab(rgb(tokens.color[theme].text.primary));
    const panel = rgb(tokens.color[theme].bg.panel);
    const base = toOklab(panel);
    const cell = (v: number) => {
      const t = rampShare(v, theme);
      return fromOklab(ink.map((x, k) => x * t + base[k]! * (1 - t)));
    };

    it(`keeps every label >= 4.5:1 over the whole ramp (${theme})`, () => {
      // Dark: low cells are dark (white labels); light: low cells are light (black labels).
      const [lo, hi] = theme === 'dark' ? [[1, 1, 1], [0, 0, 0]] : [[0, 0, 0], [1, 1, 1]];
      let worst = Infinity;
      for (let i = 1; i <= 1000; i++) {
        const v = i / 1000;
        worst = Math.min(worst, contrast(v > CONFUSION_FLIP[theme] ? hi! : lo!, cell(v)));
      }
      expect(worst).toBeGreaterThanOrEqual(4.5);
    });

    it(`tells the smallest share apart from an empty cell (${theme})`, () => {
      expect(rampShare(0, theme)).toBe(0);
      // 0.1 % of a row already sits on the first step, clearly off the panel an empty cell shows.
      expect(contrast(cell(0.001), panel)).toBeGreaterThanOrEqual(theme === 'dark' ? 1.5 : 1.3);
    });
  }
});

describe('sharePct', () => {
  it('writes whole percentages, "<1" for small shares and nothing for none', () => {
    expect(sharePct(0)).toBe('');
    expect(sharePct(0.001)).toBe('<1');
    expect(sharePct(0.0049)).toBe('<1');
    expect(sharePct(0.006)).toBe('1');
    expect(sharePct(0.083)).toBe('8');
    expect(sharePct(0.085)).toBe('9');
    expect(sharePct(0.79)).toBe('79');
    expect(sharePct(1)).toBe('100');
  });
});
