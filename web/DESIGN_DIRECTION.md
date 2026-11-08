# FOLO Studio design direction

Written after round 4 by a design-director pass (Claude Opus 5.5) over every round-4 screenshot and the source.
Round 5 onwards is implemented against this document; per-issue fixes come after the system changes in §5.

## 1. Diagnosis

The screens are competent everywhere and memorable nowhere. Each weak area traces back to the system:

| Area | Local / structural | Why |
|---|---|---|
| Data viz (5.5) | Structural | Four chart implementations (LineChart, Histogram, PrCurves, ConfusionMatrix) with different axis typography; every single-series chart is sky-blue with an area fill, so a run page is six equal-weight charts; hover opens six floating tooltips that repeat the legend values. |
| Typography (6) | Structural | 11 px in 79 places; hard-coded 9/10/15 px; mono leaks into numbers (Playground, PR/histogram/confusion ticks); panel titles 13/500 = body. |
| Colour (6) | Structural | Lime means primary button, running, hero metrics, "best" across incomparable datasets, selected cards, theme previews, sidebar overline. Status hues used as links ("See Health" red, "+1 queued" blue); queued grey on Overview, blue on Runs. |
| Consistency (6) | Structural | No shared table (eight hand-rolled `<table>`s), no chart panel, no page grid. Errors look different per screen. Mobile lists use three card patterns; the top bar shows a title on some screens, only the logo on others. |
| Spacing (6.5) | Mostly structural | Off-grid 10 px (16×), 5, 7, 3 px in components; no column system (8+4, 7+5, thirds), so panel bottoms and inner edges never align. |
| Responsiveness (6.5) | Conceptual | Phone = desktop stacked in one column. The run page on a phone opens with Stop then six charts. |

**The five structural caps**
1. No primitives for the three jobs that make up 80 % of the app: **table, chart panel, page/section grid.** Every screen re-solves them slightly differently.
2. **Colour is decoration, not meaning.** Hue should identify an entity (a run in Compare, a class); right now it is applied by default.
3. **Lime is diluted** (20+ uses), so it no longer reads as "now" or "act".
4. **Type scale too small and too wide**: 7+ sizes per screen, most text at 11–12 px, no panel-title step.
5. **Equal-weight dashboards**: Run, Model, Overview and Compare tile same-weight panels with no single focal object.

## 2. Principles

1. **Density.** List rows 36 px single-line; dense rows (per-class, detections, config, labels) 28 px; two-line rows only on phones (56 px). Panel header a fixed 40 px with 16 px side padding; panel body padding 16 (12 dense); gap between panels 16 everywhere.
2. **Page grid.** Content max 1200 px, 12 columns, 16 px gutters. Allowed splits: 12, 8+4, 6+6, 4+4+4. Panels in a row stretch to equal height; panel titles share a baseline; plot left edges align to column edge + 16. A right rail is always 4 columns and sticky (top 64).
3. **Typographic rhythm.** One job per size, at most six per screen: 20/600 page title · 14/500 panel title · 13/400 body and table cells · 12/400 meta, table headers, legends · 11/400 chart ticks only · 24/600 KPI values. At most one 32/600 hero number per page. No 44 px step, no uppercase overlines in app screens (micro only for key hints). Mono for identifiers, paths, logs, JSON and CLI, never for a number. Helper text 12 px, ≤ 60 characters per line.
4. **Chart frame.** Header (40 px): title 14/500, unit or qualifier tertiary, controls right (at most one segmented group). Legend row (24 px) directly below, starting at the plot's left edge. **Legend values are the readout**: last value at rest, cursor value on hover; one shared vertical cursor across synced charts; no floating tooltip. Plot: horizontal gridlines only (3–5), baseline axis, y ticks right-aligned in a 32 px gutter, 11 px Geist ticks; the x-axis name goes into the header ("by epoch"). A chart panel never contains a second legend, a paragraph, an overline, a floating "best" label (use a 6 px lime ring marker and the value in the legend), or more than one area fill.
5. **Colour by entity.** Hue identifies an entity: a run in Compare, a class in evaluation. A single run's metrics are drawn in ink: the primary metric `text.primary` 1.5 px, secondary metrics `text.tertiary` with dash patterns. Loss: total primary; box / cls / dfl tertiary patterns. Histograms and timing bars neutral ink. Area fill only on the page's single focal series, at 8 %.
6. **Lime means "now" or "the one action".** Allowed: one primary button per viewport, focus ring, running dot and live progress, selected detection, best-epoch marker. Never metric text, meters, nav icons, selected cards (use `line.strong` + `bg.active`, neutral), illustrations, previews. Status colours: dot + word in tables, tinted fills only in banners; queued and stopped neutral. Links are `text.primary` (underline on hover), never a status hue. Class colours only for classes.
7. **Tables.** One `DataTable`: header 32 px, 12/500 tertiary, sentence case; first column 13/500 primary; numbers right-aligned tabular with the unit in the header, not the cell; at most one inline bar column (48 px, neutral); status as dot + word, not a pill; selection = `bg.active` fill and a 2 px accent bar on the left (every list: tables, detections, classes); row checkboxes on hover or once a row is selected.
8. **States** render inside the panel that would hold the data, at its height. Empty: 16 px icon (no frame), one sentence, one button (secondary unless the page has no other primary). Error: same placement, `danger.text` title, a cause sentence, a mono request line, Retry. Loading: skeletons mirror the final geometry exactly; charts show the axis frame without a line. No dashed boxes.
9. **Motion.** Keep the tokens. Drop the page-enter translate. Live values never tween; chart domains re-ease at most once per epoch.
10. **390 px is its own concept.** Top bar always back (or logo) + page title + search; the H1 is not repeated. Lists are 56 px rows (name, one meta line, metric right), never cards. A detail page leads with its focal object; everything else sits behind a segmented control. Dialogs become bottom sheets; rails become sheets opened from a header button.

## 3. Per screen

**Overview.** Focal point: the live run. Remove the demo-data subtitle (the sidebar says it), the Datasets / Playground header buttons (they duplicate nav), the Datasets card row and the System card's two rings (fold GPU and memory into one meta line on the live card: `RTX 4080 SUPER · 94 % · 21.4 / 64 GiB`). Live run full width: epoch hero left (4 cols), mAP chart (8 cols, the page's one area fill). Below: Recent runs (8) + Best models rail (4). One hero, one table.

**Datasets list.** Almost right. Single-line 36 px rows: name, format as tertiary text (no boxed badge), path mono tertiary, split counts as columns, the class-balance strip 160×6 as the signature. Empty state: no in-panel primary (the header has it). Migrate to `DataTable`.

**Dataset detail.** Header: name + one meta line; Delete into a `…` menu. Images: one toolbar (split segmented left, multi-select class picker with swatches, Boxes / S M L right); grid gap 8. Inspector: 28 px label rows, canvas full dialog width. Stats: re-layout. Health issues as a banner list at the top (only when present); class balance full width as the focal point; the two histograms below at 6+6 in neutral ink; KPI strip from six cells to four; "See Health" a neutral link.

**Runs list.** Dataset and model as columns instead of a second line; no always-visible checkboxes; sparklines neutral with a lime endpoint only while running; queued neutral. `DataTable`, 36 px rows.

**Run detail (re-layout).** Header: name + status + one meta line; the completed / failed banner collapses to one line under the header. The KPI strip becomes **KPI tabs** (Vercel analytics): mAP50-95, mAP50, P·R, Loss, LR, img/s, each showing its value; clicking selects the **one main chart** (12 cols × 320 px). Below: Validation samples (8) + rail (4) with a compact per-class list and a collapsed Config; Logs a tab next to Samples. Remove the "Training curves" overline and the standalone LR / throughput panels; the So far / Full run toggle moves into the chart header.

**Compare.** One chart with metric tabs instead of the 2×2 grid (four legends repeating run names). The run chips are the legend, values update on hover. The side-by-side table is the real comparison: differing values in 500 weight, identical rows collapsed ("6 identical settings"), status dot + word. The info banner becomes a 12 px tertiary note under the chips. The fourth series must be visible.

**New run.** Sound. Sticky Summary rail so Start training stays visible; drop per-model metric bars (keep the numbers); selected cards `line.strong` + `bg.active`, no lime dot; CLI block collapsible at the rail's foot; the augmentation preview sticks while the sliders scroll.

**Models list.** Group rows under a 32 px section header per evaluation dataset; the top row of each group marked with tertiary "best" text (no lime across datasets). No per-cell meters, no boxed arch badges. Filters in a toolbar row under the header, as on Runs.

**Model detail.** Hero metric in ink; the four KPI cards with meters become one strip. PR curves: all classes `text.tertiary` 1 px, mean in primary, hovered / selected classes take their hue (fixes 80-class spaghetti). The per-class table is the PR legend (row hover highlights its curve); drop the duplicate AP50 list. Confusion matrix: one sequential luminance ramp, cell numbers in Geist.

**Playground.** The strongest screen. No mono numbers; canvas height viewport − 220; threshold helper paragraphs become tooltips; timing phases in neutral shades. Phones: full-bleed image, thresholds in a bottom sheet, the confidence slider pinned.

**Settings.** Forms max 720 px. Drop the shortcuts table (link the `?` dialog). Theme previews in neutral ink. Reconnect secondary until the URL changes.

## 4. What a 9 looks like here

1. **Vercel Analytics / Observability**: KPI tabs switching one large chart, a single hue, legend as readout, cursor without a floating box → Run detail, Compare, the Overview live card.
2. **Linear issue list**: 36 px single-line rows, muted meta columns, checkboxes on hover, group headers → Runs, Datasets, Models (grouped by dataset).
3. **Linear issue view**: main column + sticky properties rail → Run detail, Model detail, New run.
4. **Vercel deployment page**: status dot + one meta line; failure as a one-line callout, not a big tinted card → run banners, error states.
5. **Raycast**: monochrome icons, exactly one accent moment per view → the lime rule.
6. **Linear empty states**: small, inline, frameless, one action → every empty state.
7. **Vercel Usage**: one big number, supporting numbers in ink → Model and Dataset KPI strips.

## 5. Ranked implementation plan

| # | Change | Touches |
|---|---|---|
| 1 | `ChartPanel` + shared axis module: Geist ticks, legend-as-readout, synced cursor, no tooltip, area off by default | `lib/charts/*`, new `axis.ts`, `ChartPanel.svelte`, `model/PrCurves.svelte` |
| 2 | Colour by entity: ink for single-run metrics, hue only for runs and classes | `design/tokens.json`, `design/README.md`, `run/RunCharts`, `overview/LiveRunCard`, `dataset/StatsTab`, `playground/TimingBar` |
| 3 | Lime audit: one primary per viewport | `ui/Stat`, `models/MetricBar`, `Models`, `Model`, `shell/Sidebar`, `SidebarLive`, `RunNew`, `ui/EmptyState`, settings theme preview |
| 4 | `DataTable` with dot + word status; migrate 8 tables | new `ui/DataTable.svelte`, `StatusBadge`, `Runs`, `Models`, `Overview`, `run/PerClass`, `Model`, `Playground`, `RunCompare`, `Datasets` |
| 5 | Run detail re-layout: KPI tabs, one chart, samples + rail | `Run.svelte`, `run/RunCharts`, `run/Logs`, `run/ConfigPanel` |
| 6 | Type roles + mono ban on numbers; spacing lint for off-grid px | `tokens.json`, `base.css`, files using `--font-mono`, a check in `npm run check` |
| 7 | 12-column grid + fixed page header slots | `shell/Page`, new `ui/Grid`, `Overview`, `Model`, `Dataset`, `RunNew` |
| 8 | Compare: one chart, chips as legend, diff-aware table | `RunCompare.svelte` |
| 9 | Phone concept: one top-bar pattern, rows not cards, run "monitor" | `shell/Topbar`, `TabBar`, `Page`, mobile blocks in `Runs`, `Models`, `Datasets`, `Run` |
| 10 | Unified in-place empty / error / loading states | `ui/EmptyState`, `ErrorState`, `Skeleton`, `Overview`, `Models`, `NotFound`, queued state in `Run` |

Items 1–4 remove the causes behind the data viz, typography, colour and consistency scores; 5 and 8 give the two
densest screens a focal point. Do 1–4 before any further per-issue fixes.
