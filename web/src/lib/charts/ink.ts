// A series' hue in text. `--chart-N` is a line's ink and `--chart-fill-N` a swatch's: marks, held to 3:1, which in
// light is too pale for 12-13 px text (sky 3.55:1 on white, amber and teal ~3.67:1 on the inset). Text in a series'
// hue (direct labels, JSON tokens) takes `--chart-text-N`, which `design/contrast.mjs` holds to 4.5:1 on every
// surface in both themes (DESIGN_DIRECTION §2.5). Anything that is not a series token (ink, a class stroke) is kept.

const SERIES = /^var\(--chart-(?:fill-)?([1-6])\)$/;

/** The text ink for a series colour: `var(--chart-2)` or `var(--chart-fill-2)` -> `var(--chart-text-2)`. */
export function seriesText(color: string): string {
  const m = SERIES.exec(color.trim());
  return m ? `var(--chart-text-${m[1]})` : color;
}
