// A line chart of a few series over a shared x axis (epochs), inside a raised card with a soft gradient fill under
// the lead curve, dashed gridlines, ringed end markers, and Geist Mono direct labels.

import { frame, line, n, runs, text, textWidth, vAreaGrad, wrap } from './lib.mjs';

const W = 840;
const PAD = 24;
const CARD_PAD_X = 20;
const CARD_PAD_Y = 18;
const PLOT_H = 240;
const LEFT = 38;
const RIGHT = 118;

/**
 * `spec`: { title, subtitle, desc, xTitle, xTicks, xMax, yTicks, yMax, yFormat, series: [{ label, short, values,
 * labelAt }], footer }. `values[i]` is the value at x = i + 1.
 */
export function linesFigure(spec, t) {
  const out = [];
  const defs = [];
  let y = PAD;

  for (const [what, str, size] of [['title', spec.title, 17], ['subtitle', spec.subtitle, 12]]) {
    if (textWidth(str, size, size === 17 ? 600 : 400) > W - 2 * PAD) throw new Error(`${spec.title}: the ${what} does not fit`);
  }
  out.push(text(PAD, y + 16, spec.title, { size: 17, weight: 600, fill: t.ink, tracking: '-0.015em' }));
  out.push(text(PAD, y + 35, spec.subtitle, { size: 12, fill: t.ink3 }));

  y += 46;
  let lx = PAD;
  spec.series.forEach((s, i) => {
    out.push(`<rect x="${n(lx)}" y="${n(y + 4)}" width="14" height="3" rx="1.5" fill="${t.series[i]}"/>`);
    out.push(text(lx + 20, y + 9.5, s.label, { size: 12, fill: t.ink2 }));
    lx += 20 + textWidth(s.label, 12) + 20;
  });
  if (lx - 20 > W - PAD) throw new Error(`${spec.title}: the legend does not fit`);
  y += 22;

  const cardX = PAD;
  const cardW = W - 2 * PAD;
  const cardH = CARD_PAD_Y + PLOT_H + 42 + CARD_PAD_Y - 6;
  out.push(`<rect x="${n(cardX)}" y="${n(y)}" width="${n(cardW)}" height="${n(cardH)}" rx="8" fill="${t.card}" stroke="${t.cardBorder}"/>`);

  const plotTop = y + CARD_PAD_Y;
  const x0 = cardX + CARD_PAD_X + LEFT;
  const x1 = cardX + cardW - CARD_PAD_X - RIGHT;
  const xMax = spec.xMax ?? spec.xTicks.at(-1);
  const yMax = spec.yMax ?? spec.yTicks.at(-1);
  const px = (v) => x0 + (v / xMax) * (x1 - x0);
  const py = (v) => plotTop + PLOT_H - (v / yMax) * PLOT_H;

  for (const tick of spec.yTicks) {
    const gy = Math.round(py(tick)) + 0.5;
    out.push(line(x0, gy, x1, gy, tick === 0 ? t.axis : t.grid, 1, tick === 0 ? '' : '3 3'));
    out.push(text(x0 - 10, gy + 4, spec.yFormat(tick), { size: 11.5, fill: t.tick, anchor: 'end', mono: true }));
  }
  for (const tick of spec.xTicks) {
    const gx = Math.round(px(tick)) + 0.5;
    if (tick > 0) out.push(line(gx, plotTop + PLOT_H, gx, plotTop + PLOT_H + 4, t.axis));
    out.push(text(px(tick), plotTop + PLOT_H + 18, String(tick), { size: 11.5, fill: t.tick, anchor: 'middle', mono: true }));
  }
  out.push(text((x0 + x1) / 2, plotTop + PLOT_H + 36, spec.xTitle, { size: 11.5, fill: t.ink3, anchor: 'middle' }));

  // Soft vertical gradient fill under the highest-ending series so the lead curve reads effortlessly.
  const leadIndex = spec.series.reduce((best, s, i) => (s.values.at(-1) > spec.series[best].values.at(-1) ? i : best), 0);
  const lead = spec.series[leadIndex];
  defs.push(vAreaGrad('area-lead', t.series[leadIndex], t.mode === 'dark' ? 0.16 : 0.12));
  const leadPath = lead.values.map((v, k) => `${k ? 'L' : 'M'}${n(px(k + 1))} ${n(py(v))}`).join('');
  out.push(`<path d="${leadPath}L${n(px(lead.values.length))} ${n(py(0))}L${n(px(1))} ${n(py(0))}Z" fill="url(#area-lead)"/>`);

  // Lines first, then the end markers with their card-surface ring.
  spec.series.forEach((s, i) => {
    const d = s.values.map((v, k) => `${k ? 'L' : 'M'}${n(px(k + 1))} ${n(py(v))}`).join('');
    out.push(`<path d="${d}" stroke="${t.series[i]}" stroke-width="2.25" fill="none" stroke-linejoin="round" stroke-linecap="round"/>`);
  });
  spec.series.forEach((s, i) => {
    const ex = px(s.values.length);
    const ey = py(s.values.at(-1));
    out.push(`<circle cx="${n(ex)}" cy="${n(ey)}" r="4.5" fill="${t.series[i]}" stroke="${t.card}" stroke-width="2"/>`);
    const parts = [
      [spec.yFormat(s.values.at(-1), true), { weight: 600, fill: t.ink, mono: true }],
      [` ${s.short}`, { fill: t.ink2 }],
    ];
    const width = textWidth(spec.yFormat(s.values.at(-1), true), 12, 600, true) + textWidth(` ${s.short}`, 12);
    const at = s.labelAt === 'below' ? ex - 6 : ex + 10;
    if (at + width > cardX + cardW - CARD_PAD_X + 4) throw new Error(`${spec.title}: the end label of "${s.label}" does not fit`);
    out.push(runs(at, s.labelAt === 'below' ? ey + 21 : ey + 4, parts, { size: 12 }));
  });
  y += cardH + 14;

  if (spec.footer?.length) {
    y -= 2;
    for (const l of spec.footer.flatMap((paragraph) => wrap(paragraph, 11.5, W - 2 * PAD))) {
      out.push(text(PAD, y + 12, l, { size: 11.5, fill: t.ink3 }));
      y += 16;
    }
  }
  return frame(W, y + PAD - 10, t, { title: spec.title, desc: spec.desc, defs }, out);
}
