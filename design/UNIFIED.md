# One look: what Studio and Desktop share, screen by screen

FOLO Studio (`web/`) and FOLO Desktop (`apps/desktop/`) grew up in separate design rounds. They share `tokens.json`,
but each made its own choices about layout, charts and status, so the same run looked like two products. This file
is the one design both apps are built against. For every place where they differed it names the version that won
and why. `README.md` (this folder) stays the reference for tokens, type and colour; `web/DESIGN_DIRECTION.md` and
`apps/desktop/DIRECTION.md` keep each app's detail and must not contradict this file.

How the choices were made: the cleaner version wins. Where the two were equally clean, Desktop's wins, because it
is the denser of the two. A feature only one app has (Studio's run comparison, labeling and phone layouts;
Desktop's window chrome and local files) stays in that app and takes the shared look.

Sizes are logical px. "Both" means both apps change; otherwise the app named after the arrow changes.

## 1. Shell

| Part | The one look | From | Changes |
|---|---|---|---|
| Sidebar | 232 wide, `bg.panel`, the logo lockup at the top | both | none |
| Nav rows | 32 high, `md` (14) regular in `text.secondary`; the current row `text.primary` 500 on `bg.active`; 16 px icons | Desktop | → Studio (rows were 28 high, 13 px, 500 / 600) |
| Nav icons | Overview `layout-dashboard`, Datasets `images`, Runs `activity`, Models `box`, Benchmarks `target`, Playground `scan-eye`, Settings `settings` | Desktop | → Studio (Overview was `gauge`, Playground `scan-search`) |
| Runs badge | The number of training and queued runs at the right end of the Runs row: lime text on `accent.muted`, 20 high | Desktop | → Studio |
| Live run | One 36 px row pinned above Settings: the run's name, its percent at the right, a 2 px progress rail under it; it opens the run. No card, no second copy of the epoch and the score. It is there on every page while a run trains, the run's own page included, so the sidebar does not change as you move | Desktop | → Studio (its Training card goes; "+N queued" is the Runs badge) |
| Foot | One status line: a status dot, the data source, the version ("Demo data · v0.1.0"). Studio keeps its collapse button at the line's right end | Desktop | → Studio |
| Top bar | 40 high from 768 px. Left: breadcrumbs on a nested page, nothing on a top-level page (the page's own title says where you are). Right: the search field (280 wide, the Ctrl K keycaps), then Studio's theme button or Desktop's window buttons | Desktop | → Studio (was 48 high, repeated the page title, and showed New run on some pages) |
| New run | The lime primary of Overview and Runs, and a palette command. Never in the top bar | Desktop | → Studio |

Studio's icon sidebar (768–1279 px), phone top bar and tab bar keep their structure and take the same type, icons,
badge and live rail (the rail is a lime dot on the Runs icon where there is no room for a row).

## 2. Parts every screen uses

**Page header.** Title `xl` 600. Under it one line of facts in `sm` `text.secondary`, separated by " · ": what the
page holds right now ("9 runs · 2 active · 1 failed", "5 datasets · 9,400 images"), not a sentence describing the
page. Actions at the right: at most one lime primary. *From Desktop → Studio* (its static descriptions go; help
that mattered moves to an info button or the empty state). A list page shows no totals line under its table; when
a filter or search narrows the list, a line under the table says what is shown ("3 of 10 runs").

**Detail header.** The name, then its status badge on the same line, then one meta line. Actions at the right.
*From Desktop → Studio* (its badge sat on a row above the title). A server id (`run_7d1e`) is not in the header:
it is in the Config panel's header. Rare or destructive actions sit in a "…" menu where the app has one
(Studio); Desktop keeps them as secondary buttons.

**Status.** One glyph per state, the shape carrying the meaning so colour is never the only cue: queued a hollow
ring, running a ringed lime dot, completed a check, failed a cross, stopped a square. In a table the glyph takes
the status hue and the word is plain ink (`text.primary`; `text.secondary` for queued and stopped). On a detail
header the same glyph and word sit in a badge: the status's `muted` fill, its `text` ink, radius `sm`, 20 high.
*From Studio → Desktop* (its tables used a dot for every state and coloured the word).

**Outcome line.** A run or entry that failed, was stopped, or finished with a warning explains itself in one
callout under the header: the cause, the server's words, what to do. A run that completed without a problem shows
no callout: its facts are the KPI tiles and its next steps are the header's buttons (Open model is the primary).
*Callout from both; no completed banner from Desktop → Studio.*

**Panel.** `bg.panel`, a 1 px `line` edge, radius `lg`. A 40 px header: the title in `base` (13) 500, a qualifier
in `sm` `text.tertiary` after it or at the right, controls at the right; a hairline under the header; a 16 px
body. *Header band from Desktop → Studio* (its titles were 14 / 600 with no hairline).

**KPI strip.** One panel cut into tiles by hairlines. A tile: its label in `sm` `text.secondary`, the value in
`2xl` (24) 600 with units one step smaller in `text.tertiary`, one caption line in `sm` `text.secondary`. At most
one tile per page holds a `3xl` (32) value. A tile is not a button. *From Desktop → Studio* (Run's tabs go, §3).

**Tables.** 36 px single-line rows; the head row in `sm` `text.secondary`, sortable heads with a chevron. The
first column is the row's name in `text.primary` 500. Numbers are tabular and right-aligned: the one a list is
ranked by in `text.primary` 500, the rest in `text.secondary`. A second fact gets its own column, never a second
line. A name is never cut while a less important column has room: when a table does not fit its width, whole
columns leave, the least important first, and their facts stay in the row's tooltip and on the detail page.
*Both already said this; the screens in §4 that still had two-line rows change.*

**Progress.** A 2 px rail, lime only while the run is live, `text.tertiary` otherwise, then "42 / 100" with
spaces. *Rail from Studio → Desktop (its table bars were 4 px); spacing from Desktop → Studio.*

**Segmented control and tabs.** A segmented control (a filter, a view option) is one outlined track, 28 high;
its items in `sm` regular, the chosen one `text.primary` 500 on a raised fill with a hairline, a count after a
label in `text.tertiary`. *From Desktop → Studio.* A page's sections (Images / Statistics, Leaderboard / Plan /
Loops) are underline tabs with their counts. *From Studio → Desktop* (its benchmark page used a segmented
control for sections).

**Buttons.** 28 high (32 in a page header), labels 500. Secondary: `bg.raised` with a `line` edge. Primary:
`accent.solid` with `accent.on`. *Weights from Desktop → Studio.*

**Names and identifiers.** An architecture (`folo-s`) is a name: sans, like a run's or a model's name. Mono is
for paths, file names, server ids, commands and logs. *From Desktop → Studio* (it set architectures in mono, which
put a second typeface into every table).

## 3. Charts

**Colour.** A run's charts use the series hues, the same hue for the same quantity everywhere: mAP50-95 sky with
the page's one area fill, mAP50 amber dashed; precision magenta, recall teal dashed; learning rate neutral; loss
as in `README.md` (total ink, box amber, cls violet dashed, dfl teal dotted). *From Desktop → Studio* (its single
run charts were ink only). In a comparison hue still belongs to the run (Studio's Compare), and precision-recall
curves keep the rule below.

**Marks.** The best epoch is a hollow ring in `text.primary` on its series; the legend says "○ best 0.656 @ 41".
While a run is live, the end of a chart's first series (mAP50-95, total loss, precision, the learning rate) is a
filled lime dot, one per chart; every other series ends in a dot of its own hue. A completed run's lines all end
in a dot of their hue, a stopped run's in a square, a failed run's in a cross. So lime on a chart always means
now. *From Studio → Desktop* (it drew the best as a lime dot, which read as the live point).

**Y axis.** A 0–1 metric runs from 0 to its highest value plus headroom, with 3 to 5 round gridlines, never a
fixed 0–1 with the top third empty; an axis zoomed under 0.25 says so. *From Studio → Desktop.* Ticks are `xs`
Geist, tabular.

**X axis.** A run's charts share one x axis and one cursor. While a run is live the axis is the planned run, the
part not trained yet shaded, with the ETA over the boundary; "So far / All N" in the first chart's header fits
the axis to the epochs trained, for all of the page's charts. *Planned axis from Desktop → Studio (its default
was So far); the switch from Studio → Desktop.*

**Frame.** The panel header holds the title and, at the right, a qualifier ("per epoch", "log scale"). The legend
row under it is the readout: the latest values at rest, the cursor's while inspecting. Events ("close mosaic",
"failed during epoch 3") are labels in one band over the plot: the cross at a failed line's end says that it
failed, the label says in which epoch. The plot is an inset well (`bg.inset`). Loss lines are named at their ends
where there is room (Studio's direct labels). *Both.*

**Precision-recall.** Every class a thin context line, the mean in ink, a hovered or pinned class in its class
hue. One rule at 3 classes and at 80. *From Studio → Desktop* (it drew every class in its hue).

**Confusion matrix.** One luminance ramp, the share in each cell, "Most confused" and "Most missed" lists beside
it, a Row % / Counts switch in the header. *Ramp and lists from Studio → Desktop (its cells were green and red);
the switch from Desktop → Studio.*

## 4. Screens

**Overview.** Header: facts line; Open image (Desktop) and the New run primary. The live run is one panel: a
48 px header (name, badge, a meta line that names the dataset, the architecture and the device, Open run) whose
bottom edge is the 2 px progress rail; under it four KPI tiles (Epoch, mAP50-95, Throughput, Time left) in a
2 × 2 block on the left third and the validation chart on the right two thirds, the two flush and divided by
hairlines (one row of four tiles over the chart below 1280 px); a 36 px foot with the machine's meters (GPU
load, CPU, memory) where the server reports them. Under it Recent runs (two thirds: Run, Architecture · dataset
where it fits, Status, Progress, Best mAP50-95, Created) beside Best models (one third: Dataset, Best model,
mAP50-95, one row per evaluation dataset), both single-line rows, then Datasets as the cards below. A panel's
link to its full list is quiet text that names where it goes ("All runs →", "Open run →"), not a button.
*Header rail, machine foot and the Best models table from Desktop; the side-by-side body, the lower grid and the
links from Studio. Both change.*

**Runs.** Columns: Run, Architecture, Dataset, Status, Progress, Best mAP50-95, Trend, Duration, Created. The
trend is a 96 × 20 sparkline of mAP50-95, neutral, with a lime end dot while live. Studio's compare checkboxes
appear on a hovered or focused row and stay once one is ticked. On a narrow window Duration leaves first, then
Dataset (§2, Tables). *Single-line rows and columns from Desktop → Studio; the trend from Studio → Desktop.*

**Run.** Header as §2. Then, top to bottom:
1. One KPI strip of five tiles: Epoch (with the progress rail), mAP50-95 (the change against the last epoch, the
   best), Precision / Recall, Throughput (with its sparkline), Time left (Duration when the run has ended).
2. Validation mAP (three fifths) beside Training loss (two fifths), plots 220 high.
3. Precision / Recall beside Learning rate, in the same three fifths and two fifths so the gap between the
   panels runs straight down, plots 140 high. Studio's Learning rate panel has an LR / Throughput switch, which
   keeps the throughput history its old metric tab showed.
4. Validation samples (two thirds) beside Per class (one third): Both / Predicted / Truth and the epoch picker
   in the samples' header, and the confidence slider there too where it fits (otherwise in the line under the
   header, beside the key); the per-class table with each class's change since the last epoch.
5. Logs (two thirds, with the level filter) beside Config (one third, the server id and Copy command in its
   header).

Rows 2 to 5 stack on a narrow window (Studio below its 1024 px tier, Desktop below 1120 px). On a phone the page is
the KPI tiles, Validation mAP and Training loss, then Studio's section picker with the rest. *Rows 1 to 3 from
Desktop → Studio (its KPI tabs and single chart go); rows 4 and 5 from Studio → Desktop (it had a samples strip, a
full-width log and a three-column config, and no per-class table).*

**Datasets.** A grid of cards, at least 300 wide. A card: a 2 × 2 mosaic of the dataset's first images with their
boxes drawn (each tile 16 : 9), then the name, "2,400 images · 8 classes", the format as a small tag, and the
class-balance strip along the card's bottom edge. *Cards from Desktop → Studio (its table goes; the path, the splits
and the box count are on the dataset's page); the class-balance strip from Studio → Desktop.* Studio keeps its
format filter and search field above the grid.

**Dataset.** Header: the name, the format tag, one meta line (path, splits, classes, when added). Then the tabs
with their counts ("Images 2,400", "Statistics · 3 findings"). Images: one toolbar (the split filter with
counts, the class picker, the result count; Boxes and the S / M / L size at the right) and the grid straight
under it. Statistics: the health line, the KPI strip, class balance, the two histograms. *From Studio → Desktop*
(its KPI strip sat above the tabs and pushed the images down; its class chips row does not hold 80 classes; its
size slider becomes S / M / L). Desktop keeps Fit / Fill.

**Models.** Rows grouped under a 32 px head per evaluation dataset, since scores rank only on the same data; the
best of a group says "best" in `text.tertiary` before its score. Columns: Model, Architecture, Classes, Size,
mAP50-95, mAP50, Source run, Created. *Grouping and single-line rows from Studio → Desktop (its rows were two
lines, ungrouped); the shorter column list from Desktop → Studio (Params, GFLOPs and Started from are on the
model's page).*

**Model.** Header as §2 (Export, and Try in Playground as the primary). A KPI strip of five tiles: mAP50-95 at `3xl`
and twice the width, mAP50, Precision / Recall, Parameters (GFLOPs as its caption), File size. Then one
Precision-recall panel: the plot on the left seven twelfths and the per-class table on the right five (Class, AP50,
AP50-95, Precision, Recall, and Instances where the app has the counts), the table being the plot's legend: hovering
a row isolates its curve, a click pins it. Where the panel is too narrow for both (under about 1100 px in Studio,
940 px in Desktop) the table goes under the plot rather than squeeze it. Then the confusion matrix panel (§3). Then
Details, a full-width panel of facts in three columns (architecture, input size, how it was trained, where its
weights came from, what it was evaluated on, its id, its measured and reference latency). *Strip and the
plot-with-its-legend panel from Desktop → Studio (its separate Per class panel and sticky Details rail go); the
wider class table, the curve colours, the matrix and the Details facts from Studio → Desktop.*

**Benchmarks.** Single-line rows: Benchmark (a ringed lime dot after the name while an entry is running), Images
from, Images, Classes, Conditions, Entries, Leader (the score in `text.primary` 500, then the entry's name),
Created. *Rows from Studio → Desktop; the leader cell from Desktop → Studio.* A benchmark's page: its sections
are underline tabs (§2); everything else keeps each app's content and takes the shared parts.

**Playground.** One screen without scrolling at 1280 × 800. Header: title, one line, Open image. Left: the image
panel, as tall as the window allows, its 40 px header holding the file, the timing, the count shown and the zoom
controls. Right, a 320 px rail in this order: Model and Run on; Thresholds (Confidence, and Overlap and Max
detections where the model uses them, their explanations as tooltips); the Labels / Scores / Ground truth switches;
Detections (class, size, a confidence bar and its value; Studio's Classes filter and JSON as further tabs of this
section; selecting a row selects its box); Timing. Zoom controls are Desktop's only. *Arrangement from Desktop →
Studio (its model picker sat in the page header, and the detections table under the image made the page two screens
long).* Studio's phone layout keeps its bottom sheet.

**Settings.** One column of at most 880 px, centred with its header. Each section a panel; each setting a row at
least 56 high: its name and one line of description on the left, its control or its fact on the right. Theme is
a row with a Dark / Light / System segmented control. Measured throughput is a table (Device, Architecture,
Image size, Batch, Precision, the figure, its range where measured, Source, Updated) with Training / Inference
in the panel's header.
*From Desktop → Studio* (its section rail, theme preview cards and two-line throughput lists go).

**New run.** Each app keeps its own container: Studio a page with the sticky summary rail, Desktop a dialog.
Both take the shared fields, segmented controls and buttons. No structural change.

**Everything else** (Compare, labeling, benchmark entries and loops, cloud panels, dialogs, the palette, empty
and error states) keeps its structure and takes the parts of §2 and §3.

## 5. What does not change

- `tokens.json`: no colour, size or duration changes for this.
- Studio's tiers (`README.md`, Layout tiers) and its phone concepts.
- What each screen can do. This is a change of look and arrangement; no action, field or fact is removed from
  an app except where §4 says where it moved.
