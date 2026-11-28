// Multi-card dashboard layout for the README's headline benchmark charts: two cards per row inside a crisp panel,
// each card with its own value axis (and optional explicit subgroup scale labels when a card holds two scale bands),
// full-width bar tracks, right-aligned Geist Mono value columns, high-contrast range whiskers, and semantic pills.

import { bar, frame, hGrad, line, n, pill, pillWidth, text, textWidth, wrap } from './lib.mjs';

const W = 840;
const PAD = 22;
const GAP = 14;
const CARD_PAD_X = 16;
const CARD_PAD_Y = 14;
const CARD_HEAD_H = 26;
const GROUP_HEAD_H = 18;
const ROW_H = 44;
const BAR_H = 10;
const BAR_GAP = 4;
const GROUP_GAP = 10;

const badgePalette = (tone, t) => {
  if (tone === 'accent') return { bg: t.accentBg, border: t.accentBorder, fill: t.accentText };
  if (tone === 'warn') return { bg: t.warnBg, border: t.warnBorder, fill: t.warnText };
  if (tone === 'muted') return { bg: t.mutedBg, border: t.mutedBorder, fill: t.mutedText };
  return { bg: t.neutralBg, border: t.neutralBorder, fill: t.neutralText };
};

const barFill = (s, t) => (s.colour === 'reference' ? t.reference : `url(#g-${s.colour})`);
const swatchFill = (s, t) => (s.colour === 'reference' ? t.reference : t.series[Number(s.colour.replace('series', ''))]);

/** Height a card needs to draw its groups (including optional subgroup headings) and wrapped note. */
function cardHeight(card, cw) {
  const rows = card.groups.reduce((acc, g) => acc + g.rows.length, 0);
  const subHeads = card.groups.filter((g) => g.title || g.scale).length * GROUP_HEAD_H;
  const gaps = Math.max(0, card.groups.length - 1) * GROUP_GAP;
  const noteLines = card.note ? wrap(card.note, 11, cw - 2 * CARD_PAD_X).length : 0;
  const noteH = noteLines ? noteLines * 15 + 10 : 0;
  return CARD_PAD_Y + CARD_HEAD_H + subHeads + rows * ROW_H + gaps + noteH + CARD_PAD_Y - 4;
}

/**
 * `spec`: {
 *   title, subtitle, desc,
 *   series: [{ label, colour, strong }],
 *   grid: [[cardA, cardB], ...],
 *   footer: string[]
 * }
 */
export function dashboardFigure(spec, t) {
  const out = [];
  const defs = [];
  const seenGrad = new Set();
  for (const s of spec.series) {
    if (s.colour === 'reference' || seenGrad.has(s.colour)) continue;
    seenGrad.add(s.colour);
    const idx = Number(s.colour.replace('series', ''));
    const g = t.seriesGrad[idx];
    defs.push(hGrad(`g-${s.colour}`, g.from, g.to));
  }

  const fits = (what, width, room) => {
    if (width > room) throw new Error(`${spec.title}: ${what} needs about ${Math.round(width)} px and has ${Math.round(room)}`);
  };
  fits('the title', textWidth(spec.title, 17, 600), W - 2 * PAD);
  fits('the subtitle', textWidth(spec.subtitle ?? '', 12), W - 2 * PAD);

  // Header: title + subtitle on the left, compact series legend aligned right.
  let y = PAD;
  out.push(text(PAD, y + 16, spec.title, { size: 17, weight: 600, fill: t.ink, tracking: '-0.015em' }));
  if (spec.subtitle) out.push(text(PAD, y + 35, spec.subtitle, { size: 12, fill: t.ink3 }));

  const legendItems = spec.series.map((s) => ({
    s,
    w: 14 + textWidth(s.label, 12, s.strong ? 500 : 400),
  }));
  const legendTotal = legendItems.reduce((a, b) => a + b.w, 0) + Math.max(0, legendItems.length - 1) * 16;
  let lx = W - PAD - legendTotal;
  const titleW = textWidth(spec.title, 17, 600);
  if (PAD + titleW + 20 > lx) {
    lx = PAD;
    y += 44;
    for (const item of legendItems) {
      out.push(`<rect x="${n(lx)}" y="${n(y + 1)}" width="9" height="9" rx="2.5" fill="${swatchFill(item.s, t)}"/>`);
      out.push(text(lx + 14, y + 9.5, item.s.label, { size: 12, weight: item.s.strong ? 500 : 400, fill: item.s.strong ? t.ink : t.ink2 }));
      lx += item.w + 16;
    }
    y += 22;
  } else {
    const ly = y + 6;
    for (const item of legendItems) {
      out.push(`<rect x="${n(lx)}" y="${n(ly + 1)}" width="9" height="9" rx="2.5" fill="${swatchFill(item.s, t)}"/>`);
      out.push(text(lx + 14, ly + 9.5, item.s.label, { size: 12, weight: item.s.strong ? 500 : 400, fill: item.s.strong ? t.ink : t.ink2 }));
      lx += item.w + 16;
    }
    y += 48;
  }

  const innerW = W - 2 * PAD;
  for (const row of spec.grid) {
    const totalFlex = row.reduce((a, c) => a + (c.flex ?? 1), 0);
    const availW = innerW - GAP * (row.length - 1);
    const widths = row.map((card) => (availW * (card.flex ?? 1)) / totalFlex);
    const rowH = Math.max(...row.map((card, i) => cardHeight(card, widths[i])));
    let cx = PAD;

    for (const [ci, card] of row.entries()) {
      const cw = widths[ci];
      out.push(`<rect x="${n(cx)}" y="${n(y)}" width="${n(cw)}" height="${n(rowH)}" rx="8" fill="${t.card}" stroke="${t.cardBorder}"/>`);

      const hx = cx + CARD_PAD_X;
      const hy = y + CARD_PAD_Y + 12;
      const cardInnerW = cw - 2 * CARD_PAD_X;
      fits(
        `header of "${card.title}"`,
        textWidth(card.title, 13, 600) + (card.hint ? textWidth(card.hint, 11) + 12 : 0),
        cardInnerW,
      );
      out.push(text(hx, hy, card.title, { size: 13, weight: 600, fill: t.ink }));
      if (card.hint) {
        out.push(text(cx + cw - CARD_PAD_X, hy, card.hint, { size: 11, fill: t.ink3, anchor: 'end' }));
      }

      const labelW = card.labelW ?? 116;
      const allBadges = card.groups.flatMap((g) => g.rows.map((r) => r.badge).filter(Boolean));
      const basePillW = card.pillW ?? 52;
      const effPillW =
        basePillW > 0 && allBadges.length
          ? Math.max(basePillW, ...allBadges.map((b) => pillWidth(b, { size: 11, weight: 600, mono: true, padX: 7, minW: basePillW })))
          : 0;
      const barX = cx + CARD_PAD_X + labelW;
      const valStr = (rowItem, i) => `${rowItem.text[i]}${card.unit ? ` ${card.unit}` : ''}`;
      const maxValW = Math.ceil(
        Math.max(
          ...card.groups.flatMap((g) => g.rows.flatMap((r) => r.values.map((_, i) => textWidth(valStr(r, i), 11.5, 600, true)))),
        ),
      );
      const valGap = 10;
      const pillGap = 10;
      const valRightX = cx + cw - CARD_PAD_X - (effPillW > 0 ? effPillW + pillGap : 0);
      const barMaxW = valRightX - maxValW - valGap - barX;
      if (barMaxW < 42) throw new Error(`${spec.title} / ${card.title}: bars have only ${Math.round(barMaxW)} px`);

      let ry = y + CARD_PAD_Y + CARD_HEAD_H;
      const sharedTop = card.sharedMax ?? null;

      for (const [gi, group] of card.groups.entries()) {
        if (gi > 0) {
          out.push(line(cx + CARD_PAD_X, ry + GROUP_GAP / 2, cx + cw - CARD_PAD_X, ry + GROUP_GAP / 2, t.divider));
          ry += GROUP_GAP;
        }
        if (group.title || group.scale) {
          if (group.title && group.scale) {
            fits(
              `subgroup header "${group.title}"`,
              textWidth(group.title, 10.5, 600) + textWidth(group.scale, 10.5, 400, true) + 12,
              barX + barMaxW - hx,
            );
          }
          if (group.title) out.push(text(hx, ry + 11, group.title, { size: 10.5, weight: 600, fill: t.ink2, tracking: '0.03em' }));
          if (group.scale) out.push(text(barX + barMaxW, ry + 11, group.scale, { size: 10.5, fill: t.ink3, anchor: 'end', mono: true }));
          ry += GROUP_HEAD_H;
        }
        const dataTop = Math.max(
          ...group.rows.flatMap((r) => [
            ...r.values,
            ...(r.ranges ? r.ranges.flatMap((rg) => rg) : []),
          ]),
        );
        const groupTop = sharedTop ?? group.max ?? dataTop;
        if (!(groupTop > 0)) throw new Error(`${spec.title} / ${card.title}: group scale must be positive (got ${groupTop})`);
        if (dataTop > groupTop) {
          throw new Error(`${spec.title} / ${card.title}: value ${dataTop} exceeds fixed scale ceiling ${groupTop}`);
        }
        const scale = barMaxW / groupTop;

        for (const item of group.rows) {
          const maxLabelW = Math.max(textWidth(item.label, 12.5, 500), textWidth(item.detail ?? '', 10.5));
          if (maxLabelW > labelW - 6) {
            throw new Error(`${spec.title} / ${card.title}: "${item.label}" needs ${Math.round(maxLabelW)} px, has ${labelW - 6}`);
          }
          if (item.detail) {
            out.push(text(hx, ry + 18, item.label, { size: 12.5, weight: 500, fill: t.ink }));
            out.push(text(hx, ry + 32, item.detail, { size: 10.5, fill: t.ink3 }));
          } else {
            out.push(text(hx, ry + ROW_H / 2 + 4, item.label, { size: 12.5, weight: 500, fill: t.ink }));
          }

          const count = item.values.length;
          let by = ry + (ROW_H - (count * BAR_H + (count - 1) * BAR_GAP)) / 2;
          for (let i = 0; i < count; i++) {
            const s = spec.series[i];
            const bw = item.values[i] * scale;
            out.push(`<rect x="${n(barX)}" y="${n(by)}" width="${n(barMaxW)}" height="${BAR_H}" rx="3" fill="${t.track}"/>`);
            const barEl = bar(barX, by, bw, BAR_H, barFill(s, t));
            if (barEl) out.push(barEl);
            if (item.ranges) {
              const [lo, hi] = item.ranges[i];
              const xLo = barX + lo * scale;
              const xHi = barX + hi * scale;
              const wy = by + BAR_H / 2;
              // Contrasting halo behind the whisker keeps min-max readable over both the bar fill and empty track.
              const halo = t.mode === 'dark' ? '#08090B' : '#FFFFFF';
              const wStroke = t.ink;
              out.push(line(xLo, wy, xHi, wy, halo, 3.2));
              out.push(line(xLo, wy - 3.5, xLo, wy + 3.5, halo, 3.2));
              out.push(line(xHi, wy - 3.5, xHi, wy + 3.5, halo, 3.2));
              out.push(line(xLo, wy, xHi, wy, wStroke, 1.5));
              out.push(line(xLo, wy - 3, xLo, wy + 3, wStroke, 1.5));
              out.push(line(xHi, wy - 3, xHi, wy + 3, wStroke, 1.5));
            }
            out.push(
              text(valRightX, by + BAR_H / 2 + 3.8, valStr(item, i), {
                size: 11.5,
                weight: s.strong ? 600 : 400,
                fill: s.strong ? t.ink : t.ink2,
                anchor: 'end',
                mono: true,
              }),
            );
            by += BAR_H + BAR_GAP;
          }

          if (item.badge && effPillW > 0) {
            const pal = badgePalette(item.badgeTone ?? 'accent', t);
            out.push(
              pill(cx + cw - CARD_PAD_X, ry + (ROW_H - 22) / 2, item.badge, {
                ...pal,
                size: 11,
                weight: 600,
                mono: true,
                anchor: 'end',
                padX: 7,
                height: 22,
                minW: effPillW,
              }),
            );
          }
          ry += ROW_H;
        }
      }

      if (card.note) {
        const lines = wrap(card.note, 11, cardInnerW);
        const startY = y + rowH - CARD_PAD_Y - (lines.length - 1) * 15;
        lines.forEach((ln, idx) => {
          out.push(text(hx, startY + idx * 15, ln, { size: 11, fill: t.ink3 }));
        });
      }
      cx += cw + GAP;
    }
    y += rowH + GAP;
  }

  if (spec.footer?.length) {
    y -= 2;
    for (const l of spec.footer.flatMap((paragraph) => wrap(paragraph, 11.5, W - 2 * PAD))) {
      out.push(text(PAD, y + 12, l, { size: 11.5, fill: t.ink3 }));
      y += 16;
    }
  }
  return frame(W, y + PAD - 10, t, { title: spec.title, desc: spec.desc, defs }, out);
}
