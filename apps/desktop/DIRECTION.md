# FOLO Desktop: design direction

The page structure FOLO Desktop's screens are built against. [README.md](README.md) is the app; `notes/` holds
the review rounds this came out of.

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
  - Units and the secondary part of a value ("img/s", "/ 100", a duration's "m" and "s") in `text.tertiary` on the
    value's baseline: in a KPI tile `sm` (12) under its 24 or 32 px value, elsewhere one step smaller; durations read
    "14m 55s".
- **KPI tiles:** the label and the caption line in `sm` `text.secondary`, the value 24 px 600 (one 32 px headline per
  page at most). Every strip (Run, Model, Overview's live panel, a dataset's Statistics, a benchmark entry) builds
  its tiles from `ui::kpi_label`, `KPI_VALUE`, `KPI_UNIT`, `KPI_HEADLINE` and `kpi_note`, so they read the same.
- **Charts:**
  - A 40 px header with title, subtitle, and a legend with values (the cursor's value while inspecting, else the
    latest).
  - The plot is an inset well (`bg.inset`) under the grid; end marks, the best ring and dots are haloed in its
    colour, and the epochs not trained yet are shaded with the hover tint. The sparkline has no well.
  - No floating readout. The cursor is a line, dots and an x chip on the axis.
  - One 16 px annotation band for events (close mosaic, failure, warmup).
  - The best epoch is a hollow ring in `text.primary` on its series ("○ best 0.656 @ 41" in the legend). A line's end
    says the run's status: while live, the chart's first series (mAP50-95, total loss, precision, the learning rate)
    ends in a lime dot and every other series in a dot of its own hue, so a chart has one lime dot; when completed
    a dot of its hue, a square when stopped, a cross when failed; a best ring at the end goes around the end mark.
    No lime text in plots.
  - A 0–1 metric's axis runs from 0 to its highest value plus headroom, with 3 to 5 round gridlines; under 0.25 the
    chart says it is zoomed in. Loss lines are named at their ends where there is room.
  - Fixed y gutter. Primary plots 220 px, secondary 140 px.
- **Lime:**
  - At most one lime-filled button per screen.
  - The live run's dot, rail and badge.
  - Focus rings and the live end of a chart's first line (one per chart).
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

- **Sidebar:** the live card becomes one 36 px row (name, %) over a 2 px rail, on every page while a run trains.
- **Overview:**
  - The live panel shows a 48 px header with a 2 px progress rail, then from 1280 px a 2×2 KPI block on the left third
    beside the validation-mAP chart (one row of four tiles over the chart below 1280).
  - The machine is a 36 px footer of meters.
  - From 1280 px Recent runs (two thirds) sits beside Best models (one third, at least 380 px, so its three columns
    fit at 1280), both single-line rows above the fold; Recent runs drops whole columns by its own width. Narrower
    they stack. The Datasets cards come under them.
- **Run dashboard:**
  - One KPI row of five tiles (Epoch, mAP50-95, Precision / Recall, Throughput, Time left).
  - Row A: Validation mAP (3) beside Training loss (2) at 220.
  - Row B: Precision and recall beside Learning rate at 140.
  - "So far / All N" in the first chart's header fits every chart's x axis to the epochs trained or the planned run.
  - Row C: validation samples (2/3; Both / Predicted / Truth, epoch, minimum confidence) beside Per class (1/3;
    each class's AP50-95 and its change since the epoch before).
  - Row D: the log (2/3; All / Warn / Error with counts) beside Config (1/3, one column). Rows stack below 1120.
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
  - mAP50-95 tile at twice the width, 32 px; the other tiles as the run page's (24 px values, `sm` units, labels
    and captions in `text.secondary`).
  - One precision-recall panel: the plot (7/12) beside the per-class table (5/12), its legend. Every class is a thin
    context line, the mean in ink, a hovered, focused or pinned class in its hue.
  - The confusion matrix on one luminance ramp (the row's share in each cell), Most confused / Most missed beside it.
  - Details: the model's facts in three columns, full width.
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
