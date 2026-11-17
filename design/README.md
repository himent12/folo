# FOLO design system

One design language for **FOLO Studio** (`web/`) and **FOLO Desktop** (`apps/desktop/`). `tokens.json` is the single
source of truth; both apps generate their theme from it and never hard-code a colour, size or duration. This file
is the reference for tokens, type and colour; `UNIFIED.md` is the one look both apps are built against, screen by
screen (shell, page parts, charts, every screen). `THEMES.md` is the theme system: the named
themes of each mode, custom theme files, and how each app picks and applies one.

The folder is the source of truth for the tokens (`tokens.json`), the fonts (`fonts/`), the logo (the `logo*.svg`
files) and the shared look (`UNIFIED.md`). How the tokens reach each consumer:

- Web: `cd web && npm run tokens` (`web/scripts/tokens.mjs`) writes `web/src/lib/design/tokens.css` (CSS custom
  properties, and `@font-face` rules for the `.woff2` files in `fonts/`) and `tokens.ts` (typed values for canvas/SVG
  code). Both are committed; `web/src/main.tsx` imports the CSS. `npm run check` fails if the generated files are
  stale.
- Desktop: a Rust generator (`apps/desktop/build.rs`, which cargo runs again whenever `tokens.json` changes) reads
  `tokens.json` and writes the theme's tables to `$OUT_DIR/tokens.rs`, which `apps/desktop/src/theme.rs` includes.
- Docs site: `site/scripts/tokens.ts` turns `tokens.json` into CSS variables with the same names as Studio's, and
  `@font-face` rules, each time the site is generated (`npm run dev`, `build` or `check` in `site/`).
- Contrast: `node design/contrast.mjs` measures every colour pairing documented under Colour below (WCAG 2: text
  4.5:1, non-text marks 3:1), in every theme (`THEMES.md`: each built-in theme and each example in `themes/`), with
  tints and overlays composited over each surface, and that colours documented as told apart stay apart under
  colour-vision deficiency (OKLab ΔE ≥ 7, ×100, under normal vision and simulated protan / deutan / tritan: adjacent
  chart series, the loss components). It also checks elevation by lightness and, for the high contrast themes, their
  stricter floors (text 7:1, marks 4.5:1, panel edges 3:1 on the app). `node design/contrast.mjs --theme <file>`
  validates one custom theme file and checks it the same way. The custom format's cases (`theme-cases.json`, which
  the apps' validators share) are replayed on every run, and `node --test design/contrast.test.mjs` tests the theme
  half. `npm run check` (CI's web job) runs it, so a token change that breaks a pairing fails CI. Add a rule there
  when you document a new pairing here.

Related: [web/README.md](../web/README.md) (Studio; Common changes has "Add or change a token"),
[apps/desktop/README.md](../apps/desktop/README.md) (Design tokens), [site/README.md](../site/README.md), the
[root README](../README.md) and [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Character
Precise, fast, instrument-grade: Linear × Vercel × a camera viewfinder. Dense but calm. The content (detection boxes,
charts, images) is the star; chrome stays quiet, neutral and thin. Colour is spent on meaning: the accent for focus,
live state and the primary action; status colours for status; the class palette for classes; nothing decorative.

## Format rules (so any language can read the file)
- Colours: `#RRGGBB` or `#RRGGBBAA` strings, both base palettes (`color.dark`, `color.light`) complete; a theme's
  overrides use the same paths. No CSS functions.
- Sizes: plain numbers in px. Tracking: em (multiply by the font size). Durations: ms.
- Easing: cubic-bezier control points `[x1, y1, x2, y2]`.
- Keys starting with `$` are comments; skip them.

## Type
- **Geist** (UI, and every number: metrics, scores, counts, coordinates, axis ticks, legends) and **Geist Mono**
  (only server ids, paths, filenames, logs, CLI and JSON), weights 400 / 500 / 600 only. A config value is mono only
  when it is a path, a file or a server id (`run_7d1e`, a weights path). An architecture (`folo-s`) is a name, like
  a run's or a model's, and stays in Geist, as do prose values ("from scratch", "auto").
  Files (OFL, `fonts/OFL.txt`): `fonts/<Name>-<Weight>.ttf` for desktop and `.woff2` for web.
- Base UI text is `base` (13 / 20). The scale: `micro` 10, `xs` 11, `sm` 12, `base` 13, `md` 14, `lg` 16, `xl` 20 (page
  titles), `2xl` 24, `3xl` 32 (headline metrics), `4xl` 44 (hero metric). Larger steps tighten tracking.
- **Text-size floor: 12 px (`sm`).** Sentences, labels, captions, hints, dialog text, table cells, legends and readouts
  are never smaller. `xs` (11) is for chart tick labels only; `micro` (10) only for label chips drawn on images and
  keycaps' secondary marks. Studio's `npm run check` fails on any use of either step without a `small:` annotation
  naming which (`web/scripts/lint-spacing.mjs`).
- Every metric uses tabular numerals (`font-variant-numeric: tabular-nums`, OpenType `tnum`), so counters that tick
  live never jitter. Units and secondary parts of a number (`%`, `ms`, `/100`) are set one step smaller in
  `text.tertiary`; beside a KPI tile's large value they are `sm` (`UNIFIED.md` §2, KPI strip).

## Layout tiers
Four viewport tiers (`breakpoint` in `tokens.json`; a tier starts at its width). Layouts change only at these
boundaries, and each tier has one structure per page:

| Tier | Width | Nav | Columns and rails | Dialogs | Toolbars and KPI rows |
|---|---|---|---|---|---|
| Phone | < 768 | top bar (title, back) + bottom tab bar | one column; rails and secondary panels become sections behind a picker | bottom sheets | list toolbars two rows (filter, then the field; what the list holds is its footer line); KPI tiles 2 × 2 |
| Tablet | 768–1023 | icon sidebar (56 px) | one column; rails and side panels stack under the main column | centred | list toolbars two rows; KPI strips wrap to two rows (Run's five tiles three over two); short KPI captions |
| Laptop | 1024–1279 | icon sidebar (opens over the page) | main column + sticky rail (New run, Playground); Run's panels side by side in rows | centred | one row; KPI strips one row |
| Desktop | ≥ 1280 (designed at 1440) | full sidebar (232 px, collapsible) | the same, wider panels side by side (Overview's live run, Recent runs and Best models) | centred | one row; KPI tiles with every caption and trend; every table column |

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
Colours come per theme (`THEMES.md`): `color.dark` and `color.light` are the default Dark and Light, and each named
theme (Midnight, Dim, Paper, the high contrast pair, a user's own file) shifts shades over its mode's base. The roles
and rules below hold in every theme, measured by `contrast.mjs`; the hex values quoted here are the defaults'.

- Text: `text.primary` for content, `text.secondary` for supporting text, `text.tertiary` for labels, captions and
  metadata. All three meet WCAG AA (4.5:1) on every surface in every theme; `text.disabled` is for disabled controls
  only.
- **Accent "signal" lime** (`accent.solid` #C5F82A): the focus-confirm green of a viewfinder. Used for the primary
  button, focus rings, the live/running state and the selected detection. On a chart lime marks only the live end
  of a line, so it always means now; the best epoch is a hollow ring in `text.primary` (`UNIFIED.md` §3). Never for
  hero numbers, meters, links or decorative fills. On light surfaces the primary button carries a neutral 1 px edge, not a lime one. Text on a lime fill is
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
- Window: `window.close` and `window.close_text` are FOLO Desktop's Close button on hover, Windows' own red and
  white in both themes (the caption buttons look like the system's).
- Charts: `chart.series[0..5]` in this fixed order (sky, amber, magenta, teal, violet, orange), never cycled; the
  order keeps every adjacent pair apart under colour-vision deficiency (checked by `contrast.mjs`). The
  series deliberately exclude the accent lime (reserved for actions, live state and focus). Some
  series share a hue family with status colours (amber, teal), so charts never use colour alone for status: status is
  always labelled. Light series are ≥ 3:1 on both plot surfaces (`bg.panel`, `bg.inset`). Under tritan deficiency
  amber and magenta are the closest pair, which is why lines also carry legends with values. `chart.series` is the
  line ink; a series' fills (legend swatches, run chips, table keys) use `chart.fill`, the exact palette hue in both
  themes with the `chart.swatch_edge` hairline, as class swatches do, so a light theme's amber run never reads brown. Text in a series' hue (direct end labels, JSON tokens) uses `chart.text`, held to 4.5:1 on every surface in both themes (in dark it is the series hue itself; in light a darker step of it): `chart.series` is a 3:1 mark, too pale for 12 px text in light (sky 3.55:1 on white). A dashed or dotted series also draws in `chart.text`, 2 px in light, since a pattern carries less ink than a line. Grid lines `chart.grid`, axis lines `chart.axis`, tick labels `chart.label` (11 px, `type.xs`
  Geist, tabular; decimals follow the tick step: "0 2 4 6", "0 0.2 0.4"; tiny values share one power of ten shown
  once, "× 10⁻³"), hover cursor `chart.cursor`. Context curves (many thin lines of one kind under an emphasised one, e.g. every
  class under the mean PR curve) are 1 px `chart.context`, ≥ 3:1 on the panel in every theme and never faded with
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
"folo" in lowercase, drawn from circles and straight stems, with its last o a solid dot: the one look, and the one
thing it found. The letters use `text.primary`; the dot is `accent.solid` in both themes (it is a fill, not a line).

| file | what | where |
|---|---|---|
| `logo-wordmark.svg` | the wordmark, "fol" and the dot (207 × 93) | sidebars and top bars with room, the README |
| `logo-wordmark-light.svg` | the same in the light theme's colours | light backgrounds outside the apps |
| `logo.svg` | the mark: the f and the dot, in a 24-unit box | collapsed sidebars, phones, next to a title |
| `logo-icon.svg` | the mark on a `bg.panel` tile with a `line.strong` edge (64 × 64) | favicons, the Desktop app icon |

- **Construction.** Strokes with flat ends, no outlines to keep in step. Wordmark: stems 16 wide, the f's crossbar 14
  (horizontals look heavier), x-height 60 and ascender 90 above the baseline at y = 91; the o is a ring of outer
  radius 31 and the dot a disc of the same radius, both overshooting the x-height and the baseline by 1. The mark is
  the same f at a quarter of the size beside a dot of the o's proportions.
- **Lockup.** The wordmark, then the product name ("Studio" / "Desktop" / "Docs") in `text.tertiary` regular at
  `md`, 6 px apart, sitting on the wordmark's baseline (the SVG's bottom edge). The wordmark is 16 px tall there.
- **Clear space and size.** Keep half the dot's diameter clear on every side. The wordmark is at least 12 px tall
  and the mark at least 16 px; below that use `logo-icon.svg`.
- **Colour.** Letters in `text.primary` on the theme's surfaces, or everything in `accent.on` on an `accent.solid`
  field. Never recolour the dot, outline the letters, or set "folo" in a font in the logo's place.
- Studio draws it in `web/src/lib/shell/Logo.tsx`, the docs site in `site/scripts/layout.ts`, and Desktop from the
  masks `apps/desktop/assets/icons/logo-mark.svg` and `logo-wordmark.svg` (the letters only; the app draws the dot in
  the theme's accent). `apps/desktop/tools/make-icon.ps1` draws `folo.ico`. Change the geometry in all of them
  together.
