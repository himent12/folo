# FOLO design system

One design language for **FOLO Studio** (`web/`) and **FOLO Desktop** (`apps/desktop/`). `tokens.json` is the single
source of truth; both apps generate their theme from it and never hard-code a colour, size or duration.

- Web: `cd web && npm run tokens` writes `web/src/lib/design/tokens.css` (CSS custom properties) and `tokens.ts`
  (typed values for canvas/SVG code). `npm run check` fails if the generated files are stale.
- Contrast: `node design/contrast.mjs` measures every colour pairing documented under Colour below (WCAG 2: text
  4.5:1, non-text marks 3:1), in both themes, with tints and overlays composited over each surface, and that colours
  documented as told apart stay apart under colour-vision deficiency (OKLab ΔE ≥ 7, ×100, under normal vision and
  simulated protan / deutan / tritan: adjacent chart series, the loss components). `npm run check` (CI's web job)
  runs it, so a token change that breaks a pairing fails CI. Add a rule there when you document a new pairing here.
- Desktop: a Rust generator reads `tokens.json` and writes the GPUI theme.

## Character
Precise, fast, instrument-grade: Linear × Vercel × a camera viewfinder. Dense but calm. The content (detection boxes,
charts, images) is the star; chrome stays quiet, neutral and thin. Colour is spent on meaning: the accent for focus,
live state and the primary action; status colours for status; the class palette for classes; nothing decorative.

## Format rules (so any language can read the file)
- Colours: `#RRGGBB` or `#RRGGBBAA` strings, both themes complete. No CSS functions.
- Sizes: plain numbers in px. Tracking: em (multiply by the font size). Durations: ms.
- Easing: cubic-bezier control points `[x1, y1, x2, y2]`.
- Keys starting with `$` are comments; skip them.

## Type
- **Geist** (UI, and every number: metrics, scores, counts, coordinates, axis ticks, legends) and **Geist Mono**
  (only identifiers, paths, filenames, logs, CLI and JSON), weights 400 / 500 / 600 only. A config value is mono only
  when it is an identifier (`folo-s`, a weights path); prose values ("from scratch", "auto") stay in Geist.
  Files (OFL, `fonts/OFL.txt`): `fonts/<Name>-<Weight>.ttf` for desktop and `.woff2` for web.
- Base UI text is `base` (13 / 20). The scale: `micro` 10, `xs` 11, `sm` 12, `base` 13, `md` 14, `lg` 16, `xl` 20 (page
  titles), `2xl` 24, `3xl` 32 (headline metrics), `4xl` 44 (hero metric). Larger steps tighten tracking.
- **Text-size floor: 12 px (`sm`).** Sentences, labels, captions, hints, dialog text, table cells, legends and readouts
  are never smaller. `xs` (11) is for chart tick labels only; `micro` (10) only for label chips drawn on images and
  keycaps' secondary marks. Studio's `npm run check` fails on any use of either step without a `small:` annotation
  naming which (`web/scripts/lint-spacing.mjs`).
- Every metric uses tabular numerals (`font-variant-numeric: tabular-nums`, OpenType `tnum`), so counters that tick
  live never jitter. Units and secondary parts of a number (`%`, `ms`, `/100`) are set one step smaller in
  `text.tertiary`.

## Layout tiers
Four viewport tiers (`breakpoint` in `tokens.json`; a tier starts at its width). Layouts change only at these
boundaries, and each tier has one structure per page:

| Tier | Width | Nav | Columns and rails | Dialogs | Toolbars and KPI rows |
|---|---|---|---|---|---|
| Phone | < 768 | top bar (title, back) + bottom tab bar | one column; rails and secondary panels become sections behind a picker | bottom sheets | list toolbars two rows (filter, then the field; what the list holds is its footer line); KPI rows 2 × 2 or a picker |
| Tablet | 768–1023 | icon sidebar (56 px) | one column; rails stack under the main column or become sections behind a picker | centred | list toolbars two rows; Run's metrics a picker in the chart header; short KPI captions |
| Laptop | 1024–1279 | icon sidebar (opens over the page) | main column + sticky rail (Run, Model, New run, Playground, Settings) | centred | one row; Run's six metric tabs compact (20 px values) |
| Desktop | ≥ 1280 (designed at 1440) | full sidebar (232 px, collapsible) | the same main + rail, wider panels side by side (Overview's Recent runs and Best models) | centred | one row; full-size tabs (24 px values); every table column |

The web writes them as `(min-width: 768px)` / `(max-width: 767px)` (and 1024 / 1279, 1280 / 1279); `npm run tokens`
exports the same queries as `media` in `tokens.ts` for script. A component that needs a threshold of its own (a panel
whose content, not the viewport, decides) uses a container query or an intrinsic layout (`auto-fit`, `minmax`), with a
comment saying why; `lint-spacing.mjs` rejects any other media width.

## Space and shape
- 4 px grid (`space`): 2 / 4 / 6 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48 / 64. Paddings inside panels are 16 (12 when
  dense); gaps between panels are 12 or 16.
- Radii: `xs` 3 (chips), `sm` 4 (small controls), `md` 6 (controls, inputs), `lg` 8 (panels, cards), `xl` 12
  (dialogs, palette). Detection boxes are square (radius 0): they are measurements.
- Controls: heights 20 / 24 / 28 / 32 / 40 (`size.control`). Default 28. Table rows 36, dense 28.
- Lines are 1 px hairlines (`line.subtle` inside panels, `line.default` for panel edges, `line.strong` for hover /
  emphasis and separators). Never 2 px borders, except the focus ring. A line that is the only thing marking a
  control (an input or select edge, an unchecked checkbox or radio, an off switch track, a slider's unfilled rail) is
  `line.control`, ≥ 3:1 on every surface (WCAG 1.4.11); separators and panel edges never use it.

## Surfaces (elevation by lightness, not by shadow)
`bg.app` (window) → `bg.panel` (panels, sidebar) → `bg.raised` (inputs, hovered cards, table headers) →
`bg.overlay` (menus, popovers, palette, dialogs; these also get `shadow.lg`). `bg.inset` is a well darker than the
app: chart plots, image canvases, code/log views. In light mode panels are white on a light-grey app background.
`bg.hover` / `bg.active` are translucent overlays that work on any surface.

## Colour
- Text: `text.primary` for content, `text.secondary` for supporting text, `text.tertiary` for labels, captions and
  metadata. All three meet WCAG AA (4.5:1) on every surface in both themes; `text.disabled` is for disabled controls
  only.
- **Accent "signal" lime** (`accent.solid` #C5F82A): the focus-confirm green of a viewfinder. Used for the primary
  button, focus rings, the live/running state, the selected detection and "best" markers. Never for hero numbers,
  meters, links or decorative fills. On light surfaces the primary button carries a neutral 1 px edge, not a lime one. Text on a lime fill is
  `accent.on` (near black). On light backgrounds lime is too pale for text and strokes, so use `accent.text` /
  `accent.line` (deeper olive) there; on dark they equal the solid. The same applies to **non-text marks that must be
  seen against the surface** (progress bars, live/status dots, slider and switch fills, checkmarks, selection strokes,
  focus rings): use `accent.line` (light #679400 is ≥ 3:1 on every light surface, `accent.on` text on it is 5.4:1).
  `accent.solid` stays for the primary button, whose `accent.on` label carries the contrast; don't use a pale lime fill
  as the only cue for a state. Chart areas are not covered here: they use `chart.series` at `area_opacity`, and the
  series line carries the data.
- Status: `success` (completed, healthy), `warning` (degraded, stopping), `danger` (failed, errors; never
  for navigation such as a link to findings), `info` (informational notices). Waiting states (queued) are neutral
  grey (`text.secondary` on `bg.active`), not a status hue. Each has `solid` (dots, icons, bars; ≥ 3:1 on every
  surface), `text` (readable text; AA on every surface and on its own `muted`) and `muted` (tinted backgrounds for
  badges and banners: the solid at low alpha).
- Charts: `chart.series[0..5]` in this fixed order (sky, amber, magenta, teal, violet, orange), never cycled; the
  order keeps every adjacent pair apart under colour-vision deficiency (checked by `contrast.mjs`). The
  series deliberately exclude the accent lime (reserved for actions, live state, focus and "best" markers). Some
  series share a hue family with status colours (amber, teal), so charts never use colour alone for status: status is
  always labelled. Light series are ≥ 3:1 on both plot surfaces (`bg.panel`, `bg.inset`). Under tritan deficiency
  amber and magenta are the closest pair, which is why lines also carry legends with values. `chart.series` is the
  line ink; a series' fills (legend swatches, run chips, table keys) use `chart.fill`, the exact palette hue in both
  themes with the `chart.swatch_edge` hairline, as class swatches do, so a light theme's amber run never reads brown. Grid lines `chart.grid`, axis lines `chart.axis`, tick labels `chart.label` (11 px, `type.xs`
  Geist, tabular; decimals follow the tick step: "0 2 4 6", "0 0.2 0.4"; tiny values share one power of ten shown
  once, "× 10⁻³"), hover cursor `chart.cursor`. Context curves (many thin lines of one kind under an emphasised one, e.g. every
  class under the mean PR curve) are 1 px `chart.context`, ≥ 3:1 on the panel in both themes and never faded with
  opacity; the emphasised line carries weight and ink, highlighted entities their hue. Area fills under a line use the series colour at `area_opacity`. Loss
  components: total = `text.primary` (thicker), box = series 1 (amber), cls = series 4 (violet, dashed), dfl =
  series 3 (teal, dotted), everywhere a loss is drawn: the three stay apart under every colour-vision type (orange,
  series 5, next to amber did not: they merge under protanopia). Two or more series always get a legend with values, and
  run comparisons assign series slots by run, so a run keeps its colour: sky, amber, magenta, teal, with slots 3
  and 4 also dashed / dotted (no four hues stay apart for every colour-vision type).

## Detection classes
`classes.palette`: 15 saturated, mid-to-high-lightness colours plus one neutral near-white (class 8), chosen to stay distinct from each other and legible on
arbitrary photos. Class `i` → `palette[i % 16]`, in both themes. Adjacent entries stay distinguishable under
colour-vision deficiency, and every box also carries its class name, so colour is never the only cue. The palette
avoids the accent's lime and yellows near it (class 8 is a neutral near-white, separated from the accent and from class
10 under normal and protan/deutan/tritan simulation). A class looks the same everywhere: swatches, chips, legend
dots and class bars use the exact palette hue in both themes, with a 1 px inset edge in `chart.swatch_edge` so pale
hues hold on light surfaces. Only thin strokes (1-2 px curves) on light surfaces use a darker ink of the same hue
(Studio: `classStroke`, mixed toward black until ≥ 3:1). Absent classes show their swatch at 40 % opacity.
- Box: 2 px stroke in the class colour, wrapped by a 1 px outer halo in `classes.halo` so it reads on bright and dark
  images alike. Optional fill at `fill_opacity` on hover/selection.
- Selected box: 3 px `accent.solid` stroke inside the halo, drawn (with its chip) above everything; the other boxes
  drop to 30 % opacity. Lists of the same boxes mark the selected row with a 2 px accent bar.
- Small views (< 480 px): chips 14 px high with 10 px text. At any size, boxes under 20 px on screen show their label
  only on hover or selection.
- Label chip: class colour fill, `chip_text` (near black) text, `xs` medium, radius `xs`, sitting on the box's top-left
  corner outside the box (inside when it would leave the image). Score in tabular Geist after the name.

## Motion
Fast and quiet. `fast` 120 ms for hovers and presses, `base` 160 ms for popovers and tabs, `slow` 200 ms for dialogs
and panels, `chart` 240 ms for data transitions. Enter with `easing.out`, move with `standard`, leave with `in`. No
bounce, no overshoot, no spring. With reduced motion, durations are 0.

## Logo
`logo.svg`: a viewfinder (four corner brackets) around a single dot, the one look. The brackets use `text.primary`;
the dot is `accent.solid`. Wordmark: "FOLO" in Geist SemiBold, tracking −0.02 em, followed by the product name
("Studio" / "Desktop") in `text.tertiary` regular.
