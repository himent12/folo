// WCAG 2 contrast check for tokens.json: every pairing design/README.md documents, in both themes.
//   node design/contrast.mjs            exit 1 if any pairing is below its minimum
//   node design/contrast.mjs --all      also print every measured pair
// Text needs 4.5:1; non-text marks (focus rings, dots, chart lines, bars) and large text need 3:1. Translucent
// colours (tints, overlays) are measured composited over each surface they sit on. Run by `npm run check` in web/.

import { readFileSync } from 'node:fs';

const tokens = JSON.parse(readFileSync(new URL('./tokens.json', import.meta.url), 'utf8'));
const TEXT = 4.5;
const MARK = 3;
const SURFACES = ['bg.app', 'bg.panel', 'bg.raised', 'bg.overlay', 'bg.inset'];
const STATUS = ['success', 'warning', 'danger', 'info'];

/** Every documented pairing: `fg` on each of `on` (token paths; `[tint, surface]` is a tint over a surface). */
function rules(theme) {
  const light = theme === 'light';
  const tinted = (tint) => SURFACES.map((s) => [tint, s]);
  return [
    // Colour: text tones meet AA on every surface.
    ...['text.primary', 'text.secondary', 'text.tertiary'].map((fg) => ({ fg, on: SURFACES, min: TEXT })),
    // Accent: text in accent.text; accent.on is the label on a lime fill (and on accent.line).
    { fg: 'accent.text', on: SURFACES, min: TEXT },
    { fg: 'accent.on', on: ['accent.solid', 'accent.solid_hover', 'accent.line'], min: TEXT },
    // Non-text marks that must be seen against the surface: accent.line and focus rings.
    { fg: 'accent.line', on: SURFACES, min: MARK },
    { fg: 'focus', on: SURFACES, min: MARK },
    // Status: text on every surface and on its own tint (badges, banners); solid dots and icons are marks.
    ...STATUS.map((s) => ({ fg: `${s}.text`, on: [...SURFACES, ...tinted(`${s}.muted`)], min: TEXT })),
    ...STATUS.map((s) => ({ fg: `${s}.solid`, on: SURFACES, min: MARK })),
    // Waiting states: text.secondary on bg.active.
    { fg: 'text.secondary', on: tinted('bg.active'), min: TEXT },
    // Charts: tick labels are text on both plot surfaces; light series and context curves are lines.
    { fg: 'chart.label', on: ['bg.panel', 'bg.inset'], min: TEXT },
    ...(light ? tokens.color.light.chart.series.map((_, i) => ({ fg: `chart.series.${i}`, on: ['bg.panel', 'bg.inset'], min: MARK })) : []),
    { fg: 'chart.context', on: ['bg.panel'], min: MARK },
  ];
}

const hex = (h) => {
  const m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(h);
  if (!m) throw new Error(`not a #RRGGBB(AA) colour: ${h}`);
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: m[2] ? parseInt(m[2], 16) / 255 : 1 };
};
const over = (top, under) => {
  const t = hex(top);
  const u = hex(under);
  const mix = (k) => Math.round(t[k] * t.a + u[k] * (1 - t.a));
  return `#${[mix('r'), mix('g'), mix('b')].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
};
const luminance = (h) => {
  const lin = (v) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  const { r, g, b } = hex(h);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
};
export const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

function get(theme, path) {
  let v = tokens.color[theme];
  for (const k of path.split('.')) v = v?.[k];
  if (typeof v !== 'string') throw new Error(`${theme}: no colour at color.${theme}.${path}`);
  return v;
}

const all = process.argv.includes('--all');
const failures = [];
let checked = 0;
for (const theme of ['dark', 'light']) {
  for (const { fg, on, min } of rules(theme)) {
    for (const bg of on) {
      const [tint, surface] = Array.isArray(bg) ? bg : [null, bg];
      const ground = tint ? over(get(theme, tint), get(theme, surface)) : get(theme, surface);
      const ink = get(theme, fg);
      // A translucent foreground (none today) would be composited too.
      const ratio = contrast(hex(ink).a < 1 ? over(ink, ground) : ink, ground);
      const where = tint ? `${tint} over ${surface}` : surface;
      const line = `${theme.padEnd(5)} ${fg.padEnd(18)} on ${where.padEnd(30)} ${ratio.toFixed(2).padStart(5)} (min ${min})`;
      checked++;
      if (ratio < min) failures.push(line);
      else if (all) console.log(line);
    }
  }
}

if (failures.length) {
  console.error(`contrast: ${failures.length} of ${checked} pairings below the minimum (design/README.md, Colour):`);
  for (const f of failures) console.error(`  ${f}`);
  process.exit(1);
}
console.log(`contrast: all ${checked} documented colour pairings meet WCAG (text 4.5:1, marks 3:1) in both themes`);
