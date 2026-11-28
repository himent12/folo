// Shared pieces of the chart generator (tools/charts/render.mjs): the theme read from design/tokens.json, SVG
// primitives and a text-width estimate. The charts are static SVG files shown through <img>, so there is no script,
// no external font and no CSS that depends on the page: every colour is written out per theme.

import { readFileSync } from 'node:fs';

export const tokens = JSON.parse(readFileSync(new URL('../../design/tokens.json', import.meta.url), 'utf8'));

const rgba = (h) => {
  const m = /^#([0-9a-f]{6})([0-9a-f]{2})?$/i.exec(h);
  if (!m) throw new Error(`not a #RRGGBB(AA) colour: ${h}`);
  const v = parseInt(m[1], 16);
  return { r: (v >> 16) & 255, g: (v >> 8) & 255, b: v & 255, a: m[2] ? parseInt(m[2], 16) / 255 : 1 };
};

/** `top` (may be translucent) over the opaque `under`, as #RRGGBB: an <img> SVG then needs no opacity maths. */
export const over = (top, under) => {
  const t = rgba(top);
  const u = rgba(under);
  const mix = (k) => Math.round(t[k] * t.a + u[k] * (1 - t.a));
  return `#${[mix('r'), mix('g'), mix('b')].map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase()}`;
};

/** Mix `hex` toward `other` by `amount` (0..1) for subtle bar-gradient stops without leaving the design palette. */
export const blend = (hex, other, amount) => {
  const a = rgba(hex);
  const b = rgba(other);
  const mix = (k) => Math.round(a[k] * (1 - amount) + b[k] * amount);
  return `#${[mix('r'), mix('g'), mix('b')].map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase()}`;
};

/**
 * The colours of one theme, all opaque. `series` is `chart.series` in its fixed order (design/README.md, Colour).
 * Every bar gradient endpoint stays >= 3:1 against the bar track, and every pill ink stays >= 4.5:1 on its pill fill.
 */
export function theme(mode) {
  const c = tokens.color[mode];
  const surface = c.bg.panel;
  const card = c.bg.raised;
  const track = mode === 'dark' ? over('#FFFFFF09', card) : over('#0A0E140B', card);
  const seriesGrad = c.chart.series.map((col) => ({
    from: mode === 'dark' ? blend(col, '#000000', 0.14) : blend(col, '#000000', 0.18),
    to: mode === 'dark' ? blend(col, '#FFFFFF', 0.1) : blend(col, '#000000', 0.04),
  }));
  return {
    mode,
    surface,
    card,
    track,
    border: over(c.line.default, surface),
    cardBorder: over(c.line.subtle, surface),
    divider: over(c.line.subtle, card),
    ink: c.text.primary,
    ink2: c.text.secondary,
    ink3: c.text.tertiary,
    grid: over(c.chart.grid, card),
    axis: over(c.chart.axis, card),
    tick: c.chart.label,
    series: c.chart.series,
    seriesGrad,
    reference: mode === 'dark' ? c.chart.context : c.text.tertiary,
    accentBg: over(c.accent.muted, card),
    accentBorder: mode === 'dark' ? over('#C5F82A38', card) : over('#67940040', card),
    accentText: c.accent.text,
    neutralBg: over(c.bg.active, card),
    neutralBorder: over(c.line.default, card),
    neutralText: c.text.secondary,
    mutedBg: over(c.bg.hover, card),
    mutedBorder: over(c.line.subtle, card),
    mutedText: c.text.secondary,
    warnBg: over(c.warning.muted, card),
    warnBorder: mode === 'dark' ? over('#FFB54738', card) : over('#BD780038', card),
    warnText: c.warning.text,
  };
}

const fontStack = (f) =>
  [f.family, ...f.fallback].map((name) => (/\s/.test(name) ? `'${name}'` : name)).join(', ');

export const FONT = fontStack(tokens.font.sans);
export const FONT_MONO = fontStack(tokens.font.mono);

export const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Coordinates with at most two decimals, so regenerated files do not differ in float noise. */
export const n = (v) => String(Math.round(v * 100) / 100);

/**
 * A rough advance width for `str` in the UI sans (or mono) at `size` px. The viewer's font is not known (Geist where
 * it is installed, the system's sans otherwise), so layouts only use this to reserve room and always leave slack.
 */
export function textWidth(str, size, weight = 400, mono = false) {
  if (mono) return String(str).length * size * (weight >= 600 ? 0.63 : 0.61);
  let em = 0;
  for (const ch of String(str)) {
    if (ch === ' ') em += 0.28;
    else if (/[0-9]/.test(ch)) em += 0.6;
    else if (/[.,:;·'|!]/.test(ch)) em += 0.28;
    else if (/[iljtfr()\-–−/]/.test(ch)) em += 0.36;
    else if (/[mwMW]/.test(ch)) em += 0.86;
    else if (/[A-Z]/.test(ch)) em += 0.68;
    else em += 0.56;
  }
  return em * size * (weight >= 600 ? 1.05 : 1);
}

/** `str` broken into lines of at most `width` px at `size` px, splitting overlong tokens when needed. */
export function wrap(str, size, width) {
  const lines = [];
  let cur = '';
  const pushToken = (word) => {
    if (textWidth(word, size) <= width) {
      cur = word;
      return;
    }
    let chunk = '';
    for (const ch of word) {
      if (chunk && textWidth(chunk + ch, size) > width) {
        lines.push(chunk);
        chunk = ch;
      } else {
        chunk += ch;
      }
    }
    cur = chunk;
  };
  for (const word of String(str).split(' ')) {
    const next = cur ? `${cur} ${word}` : word;
    if (cur && textWidth(next, size) > width) {
      lines.push(cur);
      pushToken(word);
    } else if (!cur && textWidth(word, size) > width) {
      pushToken(word);
    } else {
      cur = next;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

/** A <text> element. `anchor` is start | middle | end. */
export function text(x, y, str, { size = 13, weight = 400, fill, anchor = 'start', tracking, mono = false } = {}) {
  const attrs = [
    `x="${n(x)}"`,
    `y="${n(y)}"`,
    `font-size="${size}"`,
    mono ? `font-family="${esc(FONT_MONO)}"` : '',
    weight !== 400 ? `font-weight="${weight}"` : '',
    anchor !== 'start' ? `text-anchor="${anchor}"` : '',
    tracking ? `letter-spacing="${tracking}"` : '',
    `fill="${fill}"`,
  ].filter(Boolean);
  return `<text ${attrs.join(' ')}>${esc(str)}</text>`;
}

/** Text made of runs with their own weight, font and colour: `[[str, {weight, fill, size, mono}], ...]`. */
export function runs(x, y, parts, { size = 13, anchor = 'start', mono = false } = {}) {
  const spans = parts
    .map(([str, o = {}]) => {
      const a = [
        o.size ? `font-size="${o.size}"` : '',
        o.mono ? `font-family="${esc(FONT_MONO)}"` : '',
        o.weight ? `font-weight="${o.weight}"` : '',
        `fill="${o.fill}"`,
      ].filter(Boolean);
      return `<tspan ${a.join(' ')}>${esc(str)}</tspan>`;
    })
    .join('');
  return `<text x="${n(x)}" y="${n(y)}" font-size="${size}"${mono ? ` font-family="${esc(FONT_MONO)}"` : ''}${anchor !== 'start' ? ` text-anchor="${anchor}"` : ''} xml:space="preserve">${spans}</text>`;
}

/** A horizontal bar from `x` growing right: square at the baseline, gently rounded at the data end. Zero draws nothing. */
export function bar(x, y, w, h, fill) {
  if (!(w > 0)) return '';
  const width = Math.max(w, 2);
  const r = Math.min(3.5, h / 2, width);
  return `<path d="M${n(x)} ${n(y)}h${n(width - r)}a${n(r)} ${n(r)} 0 0 1 ${n(r)} ${n(r)}v${n(h - 2 * r)}a${n(r)} ${n(r)} 0 0 1 ${n(-r)} ${n(r)}h${n(-(width - r))}z" fill="${fill}"/>`;
}

/** Pixel width a pill needs for `str`. */
export const pillWidth = (str, { size = 11.5, weight = 600, mono = true, padX = 8, minW = 0 } = {}) =>
  Math.max(minW, Math.ceil(textWidth(str, size, weight, mono) + padX * 2));

/** A rounded badge pill, aligned on its right edge (`anchor: 'end'`) or left edge (`anchor: 'start'`). */
export function pill(x, y, str, { bg, border, fill, size = 11.5, weight = 600, mono = true, anchor = 'end', padX = 8, height = 22, minW = 0 } = {}) {
  const w = pillWidth(str, { size, weight, mono, padX, minW });
  const rx = anchor === 'end' ? x - w : anchor === 'middle' ? x - w / 2 : x;
  const cx = rx + w / 2;
  const ty = y + height / 2 + size * 0.34;
  return [
    `<rect x="${n(rx)}" y="${n(y)}" width="${n(w)}" height="${n(height)}" rx="6" fill="${bg}"${border ? ` stroke="${border}"` : ''}/>`,
    text(cx, ty, str, { size, weight, fill, anchor: 'middle', mono }),
  ].join('\n');
}

export const line = (x1, y1, x2, y2, stroke, width = 1, dash = '') =>
  `<path d="M${n(x1)} ${n(y1)}L${n(x2)} ${n(y2)}" stroke="${stroke}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ''} fill="none"/>`;

export const hGrad = (id, from, to) =>
  `<linearGradient id="${id}" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="${from}"/><stop offset="100%" stop-color="${to}"/></linearGradient>`;

export const vAreaGrad = (id, colour, topOpacity) =>
  `<linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${colour}" stop-opacity="${topOpacity}"/><stop offset="100%" stop-color="${colour}" stop-opacity="0"/></linearGradient>`;

/** The document around a chart: its accessible name, the font, optional defs, and the panel it sits on. */
export function frame(width, height, t, { title, desc, defs = [] }, body) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${n(height)}" width="${width}" height="${n(height)}" role="img" aria-labelledby="t d" font-family="${esc(FONT)}">`,
    `<title id="t">${esc(title)}</title>`,
    `<desc id="d">${esc(desc)}</desc>`,
    ...(defs.length ? ['<defs>', ...defs, '</defs>'] : []),
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${n(height - 1)}" rx="10" fill="${t.surface}" stroke="${t.border}"/>`,
    ...body,
    '</svg>',
    '',
  ].join('\n');
}
