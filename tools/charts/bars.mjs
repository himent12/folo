// Horizontal grouped bars: one figure, one or more panels, each panel its own value axis (its unit and direction are
// in its heading). Every bar carries its value at its tip (and the range of its repeats, when it has one), so there
// are no axis ticks; a row can also carry a note in a column at the right ("4.3×", "level").

import { bar, frame, line, n, runs, text, textWidth } from './lib.mjs';

const W = 720;
const PAD = 28;
const LABEL_W = 196;
const ROW_H = 52;
const BAR_H = 12;
const BAR_GAP = 4;
const NOTE_W = 84;
const VALUE_SIZE = 12.5;

const rangeText = ([lo, hi]) => `${lo.toFixed(3)} to ${hi.toFixed(3)}`;
const colourOf = (key, t) => (key === 'reference' ? t.reference : t.series[Number(key.replace('series', ''))]);

/**
 * `spec`: { title, subtitle, desc, sharedScale, series: [{ label, colour, strong }], panels: [{ title, hint, unit,
 * notes, refLine: { value, label }, rows: [{ label, detail, values, text, ranges, note, noteSub }] }], footer }.
 * A series' `colour` is `series0`..`series5` or `reference`; `strong` series get the heavier value label. With
 * `sharedScale` every panel uses one value axis (panels of one unit, so a bar's length means the same in each).
 */
export function barsFigure(spec, t) {
  const out = [];
  let y = PAD;
  // Text is not measured (lib.mjs, textWidth), so anything that could run into its neighbour fails here instead.
  const fits = (what, width, room) => {
    if (width > room) throw new Error(`${spec.title}: ${what} needs about ${Math.round(width)} px and has ${Math.round(room)}`);
  };
  const barX = PAD + LABEL_W;
  const valueText = (panel, row, i) => `${row.text[i]}${panel.unit ? ` ${panel.unit}` : ''}${row.ranges ? `  ${rangeText(row.ranges[i])}` : ''}`;
  /** Where a panel's bars may end, and its largest value. */
  const extent = (panel) => {
    const labelRoom = Math.max(...panel.rows.flatMap((row) => row.values.map((_, i) => textWidth(valueText(panel, row, i), VALUE_SIZE, 600)))) + 12;
    return {
      width: W - PAD - (panel.notes ? NOTE_W : 0) - barX - labelRoom,
      top: Math.max(panel.refLine?.value ?? 0, ...panel.rows.flatMap((row) => row.values)),
    };
  };
  const extents = spec.panels.map(extent);
  const shared = spec.sharedScale ? Math.min(...extents.map((e) => e.width)) / Math.max(...extents.map((e) => e.top)) : null;

  out.push(text(PAD, y + 15, spec.title, { size: 17, weight: 600, fill: t.ink, tracking: '-0.01em' }));
  y += 26;
  if (spec.subtitle) {
    out.push(text(PAD, y + 12, spec.subtitle, { size: 13, fill: t.ink2 }));
    y += 22;
  }

  // Legend: always there for two or more series, in the series' order.
  y += 8;
  let lx = PAD;
  for (const s of spec.series) {
    out.push(`<rect x="${n(lx)}" y="${n(y)}" width="10" height="10" rx="2" fill="${colourOf(s.colour, t)}"/>`);
    out.push(text(lx + 16, y + 9.5, s.label, { size: VALUE_SIZE, fill: t.ink2 }));
    lx += 16 + textWidth(s.label, VALUE_SIZE) + 22;
  }
  fits('the legend', lx - 22, W - PAD);
  y += 10 + 18;

  for (const [index, panel] of spec.panels.entries()) {
    out.push(line(PAD, y + 0.5, W - PAD, y + 0.5, t.grid));
    y += 22;
    out.push(
      runs(PAD, y + 4, [
        [panel.title, { weight: 600, fill: t.ink }],
        [`   ${panel.hint}`, { fill: t.ink3, size: 12 }],
      ]),
    );
    y += 16;

    fits(`the heading of "${panel.title}"`, textWidth(panel.title, 13, 600) + textWidth(`   ${panel.hint}`, 12), W - 2 * PAD);
    const scale = shared ?? extents[index].width / extents[index].top;
    const x = (v) => barX + v * scale;

    const rowsTop = y;
    const rowsBottom = y + panel.rows.length * ROW_H;
    if (panel.refLine) {
      out.push(line(x(panel.refLine.value) + 0.5, rowsTop + 2, x(panel.refLine.value) + 0.5, rowsBottom - 2, t.axis));
    }
    for (const row of panel.rows) {
      fits(`the label "${row.label}"`, Math.max(textWidth(row.label, 13, 600), textWidth(row.detail ?? '', 12)), LABEL_W - 8);
      if (row.detail) {
        out.push(text(PAD, y + 22, row.label, { weight: 500, fill: t.ink }));
        out.push(text(PAD, y + 39, row.detail, { size: 12, fill: t.ink3 }));
      } else {
        out.push(text(PAD, y + ROW_H / 2 + 4.5, row.label, { weight: 500, fill: t.ink }));
      }
      const count = row.values.length;
      let by = y + (ROW_H - (count * BAR_H + (count - 1) * BAR_GAP)) / 2;
      for (let i = 0; i < count; i++) {
        const s = spec.series[i];
        const tip = x(row.values[i]);
        out.push(bar(barX, by, tip - barX, BAR_H, colourOf(s.colour, t)));
        const parts = [[row.text[i], { weight: s.strong ? 600 : 400, fill: s.strong ? t.ink : t.ink2 }]];
        if (panel.unit) parts.push([` ${panel.unit}`, { fill: t.ink3, size: 12 }]);
        // Min to max over repeated runs, as text: on an axis from zero the spread is narrower than a whisker's caps.
        if (row.ranges) parts.push([`  ${rangeText(row.ranges[i])}`, { fill: t.ink3, size: 12 }]);
        out.push(runs(tip + 8, by + BAR_H / 2 + 4.4, parts, { size: VALUE_SIZE }));
        by += BAR_H + BAR_GAP;
      }
      if (panel.notes && row.note) {
        const single = !row.noteSub;
        out.push(text(W - PAD, y + (single ? ROW_H / 2 + 5.5 : 24), row.note, { size: 16, weight: 600, fill: t.ink, anchor: 'end' }));
        if (row.noteSub) out.push(text(W - PAD, y + 40, row.noteSub, { size: 12, fill: t.ink3, anchor: 'end' }));
      }
      y += ROW_H;
    }
    // The baseline every bar of the panel grows from.
    out.push(line(barX - 0.5, rowsTop + 4, barX - 0.5, rowsBottom - 4, t.axis));
    if (panel.refLine) {
      out.push(text(x(panel.refLine.value), rowsBottom + 12, panel.refLine.label, { size: 12, fill: t.ink3, anchor: 'middle' }));
      y += 14;
    }
    y += 12;
  }

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
