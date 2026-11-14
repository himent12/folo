# FOLO Desktop: design direction

Written by a design-director pass after round 11, when two reviewers had held at 7.8–8.2 for six rounds. Per-issue
fixes had stopped moving the score; this document sets the structure the screens are built against. Values are
logical px.

## What separates it from a 9

The components are fine; the page structure is not.

- **Density.** Web-style spacing. At 1280 × 800 the Overview hero and the System strip filled the viewport, and the run
  dashboard showed one row of charts. Page headers took about 90 px, and table rows were two-line 52 px rows.
- **Equal weight.** Learning rate got the same area as validation mAP, and six model KPIs had the same size. Nothing
  led the eye.
- **Charts read like matplotlib.**
  - Mono tick labels and lime text inside the plot.
  - Empty bands above the plot.
  - A floating readout over the data.
  - Legends without values.
- **Duplication.** The sidebar live card repeated the nav badge, the Runs row and the Overview hero.
- **Huge empty boxes; Settings rows 1000 px wide; data that jumps instead of easing.**

## Global rules

- **Spacing:** 4 / 8 / 12 / 16 / 24 only. Page gutter 24 (16 below 1120). Panel gap 16. Panel padding 16 × 12 in dense
  areas.
- **Rows:**
  - Tables 36 px, single line; secondary facts get their own columns.
  - Compact lists 32–36 px.
  - Toolbars and panel headers 40 px.
  - Settings rows 56 px.
- **Panel:** 40 px header (title `base`/500, subtitle `sm` tertiary, controls right), hairline, 16 px body.
- **Numbers:**
  - Geist with tabular figures everywhere, chart ticks included; Mono only for paths, ids, code and logs.
  - Metrics 3 decimals.
  - Units one step smaller and tertiary, 2 px gap; durations read "14m 55s".
- **Charts:**
  - A 40 px header with title, subtitle, and a legend with values (the cursor's value while inspecting, else the
    latest).
  - No floating readout. The cursor is a line, dots and an x chip on the axis.
  - One 16 px annotation band for events (close mosaic, failure, warmup).
  - The best epoch is a lime dot on its series, with its value in the legend; no lime text in plots.
  - Fixed y gutter. Primary plots 220 px, secondary 140 px.
- **Lime:**
  - At most one lime-filled button per screen.
  - The live run's dot, rail and badge.
  - Focus rings and the best dot.
  - Running in a table is a ringed lime dot with plain text.
- **Status:** one glyph per state, the same for runs, benchmark entries and loops and cloud instances: a hollow ring
  (queued), a ringed dot (running), a check (completed), a cross (failed), a square (stopped). In a table the glyph
  takes the status hue and the word is plain ink; a detail header shows it in a 20 px badge. Progress in a table is a
  2 px rail (lime only while live) and "42 / 100".
- **Tabs:** a page's sections are underline tabs with their counts ("Images 4,400", "Statistics · 3 findings",
  "Leaderboard 7"); a filter or a view option is a segmented control.
- **Truncation:** every truncating element gets a fixed width per breakpoint; breakpoints drop whole columns rather
  than relying on flex shrink (text truncates against pre-shrink width).
- **Motion:** live points ease in 240 ms, the tab underline 160 ms, hover 120 ms, popovers 160–200 ms. Numbers never
  animate.

## Screens

- **Sidebar:** the live card becomes one 36 px row (dot, name, %) over a 2 px rail.
- **Overview:**
  - The live panel shows a 48 px header with a 2 px progress rail, then from 1280 px a 2×2 KPI block on the left third
    beside the validation-mAP chart (one row of four tiles over the chart below 1280).
  - The machine is a 36 px footer of meters.
  - From 1280 px Recent runs (two thirds) sits beside Best models (one third), both single-line rows above the fold;
    narrower they stack. The Datasets cards come under them.
- **Run dashboard:**
  - One KPI row of five tiles (Epoch, mAP50-95, Precision / Recall, Throughput, Time left).
  - Row A: Validation mAP (3) beside Training loss (2) at 220.
  - Row B: Precision and recall beside Learning rate at 140.
  - Row C: samples beside the log. Rows stack below 1120.
- **Runs:** a 36 px single-line table: Run, Architecture, Dataset, Status, Progress, Best mAP50-95, Trend (a 96 × 20
  mAP50-95 sparkline, neutral, a lime end dot while live), Duration, Created, Image size · batch. Optional columns drop
  whole as the window narrows.
- **Models:** grouped under a 32 px head per evaluation dataset (published and imported weights in groups of their
  own), "best" / "joint best" before the score, sorting inside each group. Columns: Model, Architecture, Classes, Size,
  mAP50-95, mAP50, Source run, Created.
- **Benchmarks:** single-line rows with Images from and Classes columns; a ringed lime dot after a running benchmark's
  name.
- **Datasets:** cards with a 2×2 mosaic and a class-balance strip along the bottom edge.
- **Model detail:**
  - mAP50-95 tile at twice the width, 32 px; other tiles at 20 px.
  - The per-class table is the PR legend: a hovered or focused row isolates its curve and dims the others to 25 %.
- **Dataset detail:** the name with its format tag, one meta line (path, splits, classes, when added), then the Images
  and Statistics tabs with their counts. Images has one toolbar: the split filter with counts, a class select (swatch
  and box count per class), the result count; Boxes, S / M / L and Fit / Fill at the right; the grid straight under
  it. Statistics opens on a one-line health summary over the KPI strip.
- **Settings:** one column of at most 880 px, centred in the page with its header.
- **New run:** each slider and its exact field on one row.

## Priority

1. Chart anatomy.
2. Run dashboard layout.
3. Header compaction and the sidebar row.
4. Overview restructure.
5. Single-line tables.
6. Numeric typography.
7. Model detail.
8. Dataset toolbar.
9. Breakpoints with fixed widths.
10. Empty, error and Playground-empty states.
11. Settings and New run rows.
12. Motion.
