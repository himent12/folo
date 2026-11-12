// A line chart of a few series over a shared x axis (epochs), each line ending in a marker and a direct label.

import { frame, line, n, runs, text, textWidth, wrap } from './lib.mjs';

const W = 720;
const PAD = 28;
const PLOT_H = 260;
const LEFT = 40;
const RIGHT = 112;

/**
 * `spec`: { title, subtitle, desc, xTitle, xTicks, xMax, yTicks, yMax, yFormat, series: [{ label, short, values,
 * labelAt }], footer }. `footer` is a list of paragraphs, wrapped here. `xMax` / `yMax` end the axes past the last tick. `values[i]` is the value at x = i + 1. `labelAt` puts the end label `right` of the line's end or
 * `below` it (for a line that ends under another one).
 */
export function linesFigure(spec, t) {
  const out = [];
  let y = PAD;
  for (const [what, str, size] of [['title', spec.title, 17], ['subtitle', spec.subtitle, 13]]) {
    if (textWidth(str, size, size === 17 ? 600 : 400) > W - 2 * PAD) throw new Error(`${spec.title}: the ${what} does not fit`);
  }
  out.push(text(PAD, y + 15, spec.title, { size: 17, weight: 600, fill: t.ink, tracking: '-0.01em' }));
  y += 26;
  out.push(text(PAD, y + 12, spec.subtitle, { size: 13, fill: t.ink2 }));
  y += 22;

  y += 8;
  let lx = PAD;
  spec.series.forEach((s, i) => {
    out.push(line(lx, y + 5, lx + 14, y + 5, t.series[i], 2));
    out.push(text(lx + 20, y + 9.5, s.label, { size: 13, fill: t.ink2 }));
    lx += 20 + textWidth(s.label, 13) + 22;
  });
  if (lx - 22 > W - PAD) throw new Error(`${spec.title}: the legend does not fit`);
  y += 10 + 22;

  const x0 = PAD + LEFT;
  const x1 = W - PAD - RIGHT;
  const xMax = spec.xMax ?? spec.xTicks.at(-1);
  const yMax = spec.yMax ?? spec.yTicks.at(-1);
  const px = (v) => x0 + (v / xMax) * (x1 - x0);
  const py = (v) => y + PLOT_H - (v / yMax) * PLOT_H;

  for (const tick of spec.yTicks) {
    const gy = Math.round(py(tick)) + 0.5;
    out.push(line(x0, gy, x1, gy, tick === 0 ? t.axis : t.grid));
    out.push(text(x0 - 10, gy + 4, spec.yFormat(tick), { size: 12.5, fill: t.tick, anchor: 'end' }));
  }
  for (const tick of spec.xTicks) {
    out.push(text(px(tick), y + PLOT_H + 18, String(tick), { size: 12.5, fill: t.tick, anchor: 'middle' }));
  }
  out.push(text((x0 + x1) / 2, y + PLOT_H + 38, spec.xTitle, { size: 12.5, fill: t.ink3, anchor: 'middle' }));

  // Lines first, then the end markers with their surface ring, so a marker is never under another line.
  spec.series.forEach((s, i) => {
    const d = s.values.map((v, k) => `${k ? 'L' : 'M'}${n(px(k + 1))} ${n(py(v))}`).join('');
    out.push(`<path d="${d}" stroke="${t.series[i]}" stroke-width="2" fill="none" stroke-linejoin="round" stroke-linecap="round"/>`);
  });
  spec.series.forEach((s, i) => {
    const ex = px(s.values.length);
    const ey = py(s.values.at(-1));
    out.push(`<circle cx="${n(ex)}" cy="${n(ey)}" r="5" fill="${t.series[i]}" stroke="${t.surface}" stroke-width="2"/>`);
    const parts = [
      [spec.yFormat(s.values.at(-1), true), { weight: 600, fill: t.ink }],
      [`  ${s.short}`, { fill: t.ink2 }],
    ];
    const width = textWidth(parts.map((part) => part[0]).join(''), 13, 600);
    const at = s.labelAt === 'below' ? ex - 6 : ex + 12;
    if (at + width > W - PAD) throw new Error(`${spec.title}: the end label of "${s.label}" does not fit`);
    out.push(runs(at, s.labelAt === 'below' ? ey + 24 : ey + 4.4, parts, { size: 13 }));
  });
  y += PLOT_H + 52;

  if (spec.footer?.length) {
    out.push(line(PAD, y + 0.5, W - PAD, y + 0.5, t.grid));
    y += 12;
    for (const l of spec.footer.flatMap((paragraph) => wrap(paragraph, 12.5, W - 2 * PAD))) {
      out.push(text(PAD, y + 12, l, { size: 12.5, fill: t.ink3 }));
      y += 18;
    }
    y += 2;
  }
  return frame(W, y + PAD - 12, t, { title: spec.title, desc: spec.desc }, out);
}
