// Shared pieces of the chart generator (tools/charts/render.mjs): the theme read from design/tokens.json, SVG
// primitives and a text-width estimate. The charts are static SVG files shown through <img>, so there is no script,
// no external font and no CSS that depends on the page: every colour is written out per theme.

import { readFileSync } from 'node:fs';

export const tokens = JSON.parse(readFileSync(new URL('../../design/tokens.json', import.meta.url), 'utf8'));

const rgba = (h) => {
  const m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(h);
  if (!m) throw new Error(`not a #RRGGBB(AA) colour: ${h}`);
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255, a: m[2] ? parseInt(m[2], 16) / 255 : 1 };
};

/** `top` (may be translucent) over the opaque `under`, as #RRGGBB: an <img> SVG then needs no opacity maths. */
export const over = (top, under) => {
  const t = rgba(top);
  const u = rgba(under);
  const mix = (k) => Math.round(t[k] * t.a + u[k] * (1 - t.a));
  return `#${[mix('r'), mix('g'), mix('b')].map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase()}`;
};

/**
 * The colours of one theme, all opaque. `series` is `chart.series` in its fixed order (design/README.md, Colour).
 * `reference` is the grey of whatever FOLO is compared with: `chart.context` in dark, `text.tertiary` in light,
 * where `chart.context` is too close to the sky series for a filled bar (OKLab ΔE 13 against 16; both ≥ 3:1 on the
 * panel).
 */
export function theme(mode) {
  const c = tokens.color[mode];
  const surface = c.bg.panel;
  return {
    mode,
    surface,
    border: over(c.line.default, surface),
    ink: c.text.primary,
    ink2: c.text.secondary,
    ink3: c.text.tertiary,
    grid: over(c.chart.grid, surface),
    axis: over(c.chart.axis, surface),
    tick: c.chart.label,
    series: c.chart.series,
    reference: mode === 'dark' ? c.chart.context : c.text.tertiary,
  };
}

export const FONT = [tokens.font.sans.family, ...tokens.font.sans.fallback]
  .map((f) => (/\s/.test(f) ? `'${f}'` : f))
  .join(', ');

export const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Coordinates with at most two decimals, so regenerated files do not differ in float noise. */
export const n = (v) => String(Math.round(v * 100) / 100);

/**
 * A rough advance width for `str` in the UI sans at `size` px. The viewer's font is not known (Geist where it is
 * installed, the system's sans otherwise), so layouts only use this to reserve room and always leave slack.
 */
export function textWidth(str, size, weight = 400) {
  let em = 0;
  for (const ch of String(str)) {
    if (ch === ' ') em += 0.28;
    else if (/[0-9]/.test(ch)) em += 0.6;
    else if (/[.,:;·'|!]/.test(ch)) em += 0.27;
    else if (/[iljtfr()\-–−/]/.test(ch)) em += 0.36;
    else if (/[mwMW]/.test(ch)) em += 0.86;
    else if (/[A-Z]/.test(ch)) em += 0.68;
    else em += 0.56;
  }
  return em * size * (weight >= 600 ? 1.05 : 1);
}

/** A <text> element. `anchor` is start | middle | end. */
export function text(x, y, str, { size = 13, weight = 400, fill, anchor = 'start', tracking } = {}) {
  const attrs = [
    `x="${n(x)}"`,
    `y="${n(y)}"`,
    `font-size="${size}"`,
    weight !== 400 ? `font-weight="${weight}"` : '',
    anchor !== 'start' ? `text-anchor="${anchor}"` : '',
    tracking ? `letter-spacing="${tracking}"` : '',
    `fill="${fill}"`,
  ].filter(Boolean);
  return `<text ${attrs.join(' ')}>${esc(str)}</text>`;
}

/** Text made of runs with their own weight and colour: `[[str, {weight, fill, size}], ...]`. */
export function runs(x, y, parts, { size = 13, anchor = 'start' } = {}) {
  const spans = parts
    .map(([str, o = {}]) => {
      const a = [o.size ? `font-size="${o.size}"` : '', o.weight ? `font-weight="${o.weight}"` : '', `fill="${o.fill}"`].filter(Boolean);
      return `<tspan ${a.join(' ')}>${esc(str)}</tspan>`;
    })
    .join('');
  return `<text x="${n(x)}" y="${n(y)}" font-size="${size}"${anchor !== 'start' ? ` text-anchor="${anchor}"` : ''} xml:space="preserve">${spans}</text>`;
}

/** A horizontal bar from `x` growing right: square at the baseline, a rounded data end. */
export function bar(x, y, w, h, fill) {
  const width = Math.max(w, 1);
  const r = Math.min(4, h / 2, width);
  return `<path d="M${n(x)} ${n(y)}h${n(width - r)}a${n(r)} ${n(r)} 0 0 1 ${n(r)} ${n(r)}v${n(h - 2 * r)}a${n(r)} ${n(r)} 0 0 1 ${n(-r)} ${n(r)}h${n(-(width - r))}z" fill="${fill}"/>`;
}

export const line = (x1, y1, x2, y2, stroke, width = 1) =>
  `<path d="M${n(x1)} ${n(y1)}L${n(x2)} ${n(y2)}" stroke="${stroke}" stroke-width="${width}" fill="none"/>`;

/** The document around a chart: its accessible name, the font, and the panel it sits on. */
export function frame(width, height, t, { title, desc }, body) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${n(height)}" width="${width}" height="${n(height)}" role="img" aria-labelledby="t d" font-family="${esc(FONT)}">`,
    `<title id="t">${esc(title)}</title>`,
    `<desc id="d">${esc(desc)}</desc>`,
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${n(height - 1)}" rx="${tokens.radius.lg}" fill="${t.surface}" stroke="${t.border}"/>`,
    ...body,
    '</svg>',
    '',
  ].join('\n');
}
