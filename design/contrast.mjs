// Colour checks for tokens.json, in both themes: WCAG 2 contrast for every pairing design/README.md documents, and
// colour-vision separation for colours it documents as told apart.
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

// Colours documented as told apart stay apart under colour-vision deficiency: OKLab distance (×100) under normal
// vision and Machado et al. (2009) full-severity protan / deutan / tritan simulation, in linear RGB.
const SEPARATION = 7;
const CVD = {
  normal: null,
  protan: [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
  deutan: [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]],
  tritan: [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]],
};
const toLinear = (h) => {
  const { r, g, b } = hex(h);
  return [r, g, b].map((v) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
};
const oklab = ([r, g, b]) => {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
};
const seen = (h, mat) => {
  const c = toLinear(h);
  return mat ? mat.map((row) => Math.min(1, Math.max(0, row[0] * c[0] + row[1] * c[1] + row[2] * c[2]))) : c;
};
const distance = (a, b, mat) => 100 * Math.hypot(...oklab(seen(a, mat)).map((v, i) => v - oklab(seen(b, mat))[i]));

/** Sets whose members must be told apart: neighbouring chart series, and the loss components (box, cls, dfl). */
const SETS = [
  ...[0, 1, 2, 3, 4].map((i) => ({ what: `chart.series ${i} / ${i + 1}`, paths: [`chart.series.${i}`, `chart.series.${i + 1}`] })),
  { what: 'loss box / cls / dfl', paths: ['chart.series.1', 'chart.series.4', 'chart.series.3'] },
];
for (const theme of ['dark', 'light']) {
  for (const { what, paths } of SETS) {
    for (let i = 0; i < paths.length; i++) {
      for (let j = i + 1; j < paths.length; j++) {
        for (const [vision, mat] of Object.entries(CVD)) {
          const d = distance(get(theme, paths[i]), get(theme, paths[j]), mat);
          const line = `${theme.padEnd(5)} ${what}: ${paths[i]} vs ${paths[j]} (${vision})`.padEnd(78) + `ΔE ${d.toFixed(1).padStart(5)} (min ${SEPARATION})`;
          checked++;
          if (d < SEPARATION) failures.push(line);
          else if (all) console.log(line);
        }
      }
    }
  }
}

if (failures.length) {
  console.error(`contrast: ${failures.length} of ${checked} checks below the minimum (design/README.md, Colour):`);
  for (const f of failures) console.error(`  ${f}`);
  process.exit(1);
}
console.log(`contrast: all ${checked} documented colour checks pass in both themes (text 4.5:1, marks 3:1, CVD ΔE ${SEPARATION})`);
