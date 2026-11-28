// Horizontal grouped bars (used for multi-size speedups in docs/BENCHMARKS.md): raised card plot area, full-width
// bar tracks, right-aligned Geist Mono values outside the track, vertical reference line, and crisp gradients.

import { bar, frame, hGrad, line, n, text, textWidth, wrap } from './lib.mjs';

const W = 840;
const PAD = 24;
const CARD_PAD_X = 18;
const CARD_PAD_Y = 16;
const LABEL_W = 224;
const ROW_H = 44;
const BAR_H = 11;
const BAR_GAP = 4;
const VALUE_SIZE = 12;
const SMALL = 11.5;

const colourOf = (key, t) => (key === 'reference' ? t.reference : `url(#g-${key})`);
const swatchOf = (key, t) => (key === 'reference' ? t.reference : t.series[Number(key.replace('series', ''))]);

/**
 * `spec`: { title, subtitle, desc, sharedScale, series: [{ label, colour, strong }], panels: [{ title, hint, unit,
 * refLine: { value, label }, rows: [{ label, detail, values, text }] }], footer }.
 */
export function barsFigure(spec, t) {
  const out = [];
  const defs = [];
  const seen = new Set();
  for (const s of spec.series) {
    if (s.colour === 'reference' || seen.has(s.colour)) continue;
    seen.add(s.colour);
    const g = t.seriesGrad[Number(s.colour.replace('series', ''))];
    defs.push(hGrad(`g-${s.colour}`, g.from, g.to));
  }

  let y = PAD;
  const fits = (what, width, room) => {
    if (width > room) throw new Error(`${spec.title}: ${what} needs about ${Math.round(width)} px and has ${Math.round(room)}`);
  };
  fits('the title', textWidth(spec.title, 17, 600), W - 2 * PAD);
  fits('the subtitle', textWidth(spec.subtitle ?? '', 12.5), W - 2 * PAD);

  out.push(text(PAD, y + 16, spec.title, { size: 17, weight: 600, fill: t.ink, tracking: '-0.015em' }));
  if (spec.subtitle) out.push(text(PAD, y + 35, spec.subtitle, { size: 12, fill: t.ink3 }));

  const legendItems = spec.series.map((s) => ({ s, w: 15 + textWidth(s.label, 12, s.strong ? 500 : 400) }));
  const legendTotal = legendItems.reduce((a, b) => a + b.w, 0) + Math.max(0, legendItems.length - 1) * 18;
  let lx = W - PAD - legendTotal;
  if (PAD + textWidth(spec.title, 17, 600) + 20 > lx) {
    lx = PAD;
    y += 44;
    for (const item of legendItems) {
      out.push(`<rect x="${n(lx)}" y="${n(y + 1)}" width="9" height="9" rx="2.5" fill="${swatchOf(item.s.colour, t)}"/>`);
      out.push(text(lx + 14, y + 9.5, item.s.label, { size: 12, weight: item.s.strong ? 500 : 400, fill: item.s.strong ? t.ink : t.ink2 }));
      lx += item.w + 18;
    }
    y += 22;
  } else {
    const ly = y + 6;
    for (const item of legendItems) {
      out.push(`<rect x="${n(lx)}" y="${n(ly + 1)}" width="9" height="9" rx="2.5" fill="${swatchOf(item.s.colour, t)}"/>`);
      out.push(text(lx + 14, ly + 9.5, item.s.label, { size: 12, weight: item.s.strong ? 500 : 400, fill: item.s.strong ? t.ink : t.ink2 }));
      lx += item.w + 18;
    }
    y += 48;
  }

  const cardX = PAD;
  const cardW = W - 2 * PAD;
  const barX = cardX + CARD_PAD_X + LABEL_W;
  const valRightX = cardX + cardW - CARD_PAD_X;
  const valueText = (panel, row, i) => `${row.text[i]}${panel.unit ? ` ${panel.unit}` : ''}`;

  const extent = (panel) => {
    const labelRoom = Math.ceil(Math.max(...panel.rows.flatMap((row) => row.values.map((_, i) => textWidth(valueText(panel, row, i), VALUE_SIZE, 600, true))))) + 14;
    return {
      width: valRightX - barX - labelRoom,
      top: Math.max(panel.refLine?.value ?? 0, ...panel.rows.flatMap((row) => row.values)),
    };
  };
  const extents = spec.panels.map(extent);
  const shared = spec.sharedScale ? Math.min(...extents.map((e) => e.width)) / Math.max(...extents.map((e) => e.top)) : null;

  for (const [index, panel] of spec.panels.entries()) {
    const panelH = CARD_PAD_Y + 28 + panel.rows.length * ROW_H + (panel.refLine ? 22 : 6) + CARD_PAD_Y - 6;
    out.push(`<rect x="${n(cardX)}" y="${n(y)}" width="${n(cardW)}" height="${n(panelH)}" rx="8" fill="${t.card}" stroke="${t.cardBorder}"/>`);

    const hx = cardX + CARD_PAD_X;
    let py = y + CARD_PAD_Y;
    out.push(text(hx, py + 12, panel.title, { size: 13, weight: 600, fill: t.ink }));
    if (panel.hint) out.push(text(valRightX, py + 12, panel.hint, { size: 11.5, fill: t.ink3, anchor: 'end' }));
    py += 26;

    const scale = shared ?? extents[index].width / extents[index].top;
    const barMaxW = extents[index].width;
    const x = (v) => barX + v * scale;
    const rowsTop = py;
    const rowsBottom = py + panel.rows.length * ROW_H;

    for (const row of panel.rows) {
      fits(`the label "${row.label}"`, Math.max(textWidth(row.label, 12.5, 500), textWidth(row.detail ?? '', SMALL)), LABEL_W - 10);
      if (row.detail) {
        out.push(text(hx, py + 18, row.label, { size: 12.5, weight: 500, fill: t.ink }));
        out.push(text(hx, py + 33, row.detail, { size: 11, fill: t.ink3 }));
      } else {
        out.push(text(hx, py + ROW_H / 2 + 4, row.label, { size: 12.5, weight: 500, fill: t.ink }));
      }
      const count = row.values.length;
      let by = py + (ROW_H - (count * BAR_H + (count - 1) * BAR_GAP)) / 2;
      for (let i = 0; i < count; i++) {
        const s = spec.series[i];
        const tip = x(row.values[i]);
        out.push(`<rect x="${n(barX)}" y="${n(by)}" width="${n(barMaxW)}" height="${BAR_H}" rx="3" fill="${t.track}"/>`);
        out.push(bar(barX, by, tip - barX, BAR_H, colourOf(s.colour, t)));
        out.push(
          text(valRightX, by + BAR_H / 2 + 4, valueText(panel, row, i), {
            size: VALUE_SIZE,
            weight: s.strong ? 600 : 400,
            fill: s.strong ? t.ink : t.ink2,
            anchor: 'end',
            mono: true,
          }),
        );
        by += BAR_H + BAR_GAP;
      }
      py += ROW_H;
    }

    if (panel.refLine) {
      const rx = Math.round(x(panel.refLine.value)) + 0.5;
      out.push(line(rx, rowsTop + 2, rx, rowsBottom - 2, t.axis, 1.2, '3 3'));
      out.push(text(rx, rowsBottom + 14, panel.refLine.label, { size: 11, fill: t.ink3, anchor: 'middle', mono: true }));
    }
    y += panelH + 14;
  }

  if (spec.footer?.length) {
    y -= 2;
    for (const l of spec.footer.flatMap((paragraph) => wrap(paragraph, SMALL, W - 2 * PAD))) {
      out.push(text(PAD, y + 12, l, { size: SMALL, fill: t.ink3 }));
      y += 16;
    }
  }
  return frame(W, y + PAD - 10, t, { title: spec.title, desc: spec.desc, defs }, out);
}
