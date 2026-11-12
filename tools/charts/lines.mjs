// A line chart of a few series over a shared x axis (epochs), each line ending in a marker and a direct label.

import { frame, line, n, runs, text, textWidth } from './lib.mjs';

const W = 720;
const PAD = 28;
const PLOT_H = 260;
const LEFT = 40;
const RIGHT = 112;

/**
 * `spec`: { title, subtitle, desc, xTitle, xTicks, xMax, yTicks, yMax, yFormat, series: [{ label, short, values,
 * labelAt }], footer }. `xMax` / `yMax` end the axes past the last tick. `values[i]` is the value at x = i + 1. `labelAt` puts the end label `right` of the line's end or
 * `below` it (for a line that ends under another one).
 */
export function linesFigure(spec, t) {
  const out = [];
  let y = PAD;
  out.push(text(PAD, y + 15, spec.title, { size: 17, weight: 600, fill: t.ink, tracking: '-0.01em' }));
  y += 26;
  out.push(text(PAD, y + 12, spec.subtitle, { size: 13, fill: t.ink2 }));
  y += 22;

  y += 8;
  let lx = PAD;
  spec.series.forEach((s, i) => {
    out.push(line(lx, y + 5, lx + 14, y + 5, t.series[i], 2));
    out.push(text(lx + 20, y + 9.5, s.label, { size: 12.5, fill: t.ink2 }));
    lx += 20 + textWidth(s.label, 12.5) + 22;
  });
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
    out.push(text(x0 - 10, gy + 4, spec.yFormat(tick), { size: 11, fill: t.tick, anchor: 'end' }));
  }
  for (const tick of spec.xTicks) {
    out.push(text(px(tick), y + PLOT_H + 18, String(tick), { size: 11, fill: t.tick, anchor: 'middle' }));
  }
  out.push(text((x0 + x1) / 2, y + PLOT_H + 38, spec.xTitle, { size: 12, fill: t.ink3, anchor: 'middle' }));

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
    if (s.labelAt === 'below') out.push(runs(ex - 6, ey + 24, parts, { size: 12.5 }));
    else out.push(runs(ex + 12, ey + 4.4, parts, { size: 12.5 }));
  });
  y += PLOT_H + 52;

  if (spec.footer?.length) {
    out.push(line(PAD, y + 0.5, W - PAD, y + 0.5, t.grid));
    y += 12;
    for (const l of spec.footer) {
      out.push(text(PAD, y + 12, l, { size: 12, fill: t.ink3 }));
      y += 17;
    }
    y += 2;
  }
  return frame(W, y + PAD - 12, t, { title: spec.title, desc: spec.desc }, out);
}
