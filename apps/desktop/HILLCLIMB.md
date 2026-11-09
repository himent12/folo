# FOLO Desktop design hill-climb

Each round: capture every scenario (`tools/shoot-all.ps1`), get two independent design reviews (Claude Opus 5.5 and
GPT-6.1 Sol) against `REVIEW_RUBRIC.md`, fix, repeat. Stop condition: both reviewers score ≥ 9 on every line in the
same round.

Lines: 1 hierarchy · 2 typography · 3 colour · 4 spacing · 5 components · 6 dataviz · 7 info/copy · 8 consistency ·
9 interaction · 10 accessibility · 11 responsiveness · 12 polish.

## Round 0 — foundation (self-review, no external scores)

Built: app shell (custom title bar with native caption buttons, sidebar with live-run card that collapses to an icon
rail below 1120 px), theme generated from `design/tokens.json`, mock backend with a live training simulator, all six
screens, command palette, New run dialog, stop confirmation, image inspector (zoom/pan, box list), toasts, 28 scenarios
plus minimum/wide sizes.

Fixed during self-review: inspector closing itself while images loaded; KPI strip overflowing at minimum width (now a
3 × 2 grid); cut-short runs squashed into a 100-epoch axis; queued runs counted as live; dataset toolbar overflow;
New run form layout (equal-width segments); inconsistent tick decimals and unaligned plot gutters; noisy throughput
sparkline; LR axis not starting at zero; the confusion matrix's background/background cell.

## Round 1 — first external review

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 7 | 6 | 7 | 7 | 7 | 6 | 6 | 7 | 7 | 6 | 5 | 7 | 6.5 |
| GPT-6.1 Sol | 8 | 7 | 6 | 7 | 7 | 7 | 7 | 8 | 6 | 4 | 5 | 7 | 6.6 |

Main issues: Playground side panel collapsing at minimum height; chart annotations colliding and readouts hiding them;
uppercase overlines mangling "mAP50-95"; error state subtitle still "Loading…" and literal backticks in copy;
skeletons not filling their boxes; clipped class names (confusion matrix, Classes KPI); Reduce motion a no-op;
light-theme lime fills under 3:1; no modal focus scope (Tab escaped dialogs, inspector keys needed a click);
breadcrumbs and dialog/toast close buttons not focusable; palette selection not scrolled into view; empty-select
modulo-by-zero; hover dimming hid context; tertiary text under AA (token; raised with the web thread).

Round 2 changes: whole-panel scrolling Playground sidebar with focusable detection rows and a no-models state;
annotation lanes on panel-coloured pills, readouts below them; `caption()` (case-preserving) for metric labels;
`rich()` text renders `code` as chips; error subtitle and Connection settings action; skeletons fill their parent,
relative-width loading rows; full class names with ellipsis, "background" spelled out, matrix text colour from cell
luminance; Reduce motion stops pulses/shimmers and switch animation; `accent_fill()` (olive in light) for non-text
fills, switch knobs with real contrast; modal focus trap (Tab cycles inside the dialog/inspector/palette), stop
confirmation focuses "Keep training", navigation shortcuts inert behind modals; focusable breadcrumbs, dialog close and
toast dismiss; palette keeps the selection scrolled into view; select guards empty lists; dimmed boxes keep 60 %
strength and labels; label chips avoid each other; legends on the Overview mAP chart, axis titles on the PR curve;
loss axes from zero; "mAP50-95 · latest/final" with no red delta on finished runs; GPU utilisation history fills the
System panel on a fixed 0–100 % scale; segmented items no longer shift; "Close mosaic" in the New run form; inspector
says "Fit"; mock file names match image ids; the sidebar's live card hides on Overview.

## Round 2

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 7 | 8 | 7 | 7 | 8 | 7 | 7 | 7 | 7 | 7 | 7 | 7.3 |
| GPT-6.1 Sol | 8 | 7 | 7 | 8 | 7 | 8 | 7 | 8 | 7 | 6 | 7 | 7 | 7.3 |

Main issues: lists not sorted by time; clipped class summaries; Playground latency below the fold, no zoom; lime on
plain magnitude bars and links; marker labels still crossing data; time units at full size; mixed metric precision;
uppercase overlines without tracking; Overview hero dead space at wide sizes; wrapped log lines; weak focus ring on
lime buttons; no pressed states on chips/segments/switches; global shortcuts and the palette leaking over modals,
focus not returned to the invoker; "New run from this config" only kept the dataset; stop dialog text overflow;
dataset defects not actionable; System panel too prominent at minimum size; confusion matrix "Normalized" unclear;
inspector toolbar overlapping the image; Windows paths with forward slashes. Tertiary text contrast was fixed in
design/tokens.json by the web thread (now #80868F dark, #646B75 light).

Round 3 changes: runs and models newest first; class summaries fit whole names ("circle, square +3"); Playground
latency pinned in a footer and in the stage header, wheel zoom / drag pan / Fit toolbar on the stage; neutral
magnitude bars and links (lime only for live, focus and primary); marker labels in a reserved band above the plot,
an end tick when the range overshoots (120-epoch runs); durations with small units ("12 m 02 s"); three-decimal
detection metrics; readout order follows the legend; sentence-case group headings; Overview hero KPIs in four equal
columns with dividers, a hero-shaped loading skeleton, "Collecting…" for an empty GPU history, a compact System
strip below 1200 px; single-line log entries; focus ring in the text colour on lime buttons; pressed states on
chips, segments and switches; palette and global shortcuts inert behind modals, focus returned to the invoker;
"New run from this config" restores the whole config (halving the batch after out-of-memory, with a note); stop
dialog copy within its padding; Empty images tile filters the grid; tiles crop to fill; native path separators;
export toast with "Show in folder"; confusion matrix as row percentages with Actual/Predicted labels; AP50 legend
heading; inspector image clear of its zoom bar; 1-based image ids; two-pass box painting so strokes never cover
label chips; full-width Settings; "Vulkan" casing; the collapsed sidebar's live button shows its percentage.

## Round 3

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 7 | 7 | 7 | 7 | 7 | 8 | 7 | 7 | 7 | 6 | 7 | 7.1 |
| GPT-6.1 Sol | 8 | 8 | 7 | 7 | 7 | 7 | 8 | 8 | 7 | 6 | 6 | 7 | 7.2 |

Shared issues: confusion matrix overflowing the gutter; Playground sidebar at minimum size (controls crowd out
detections, filename dropped); charts pointer-only; slider-only numeric entry in New run; selects committing while
browsing; class palette reused for non-class series and too faint as chart strokes on white; no disabled button
state (New run live with no datasets / no server); PR curves diving to recall 1.0; ragged final ticks; marker bands
misaligning plot tops in a row; readout covering series ends; palette rows compressed; text fields not scrolling to
the caret; duplicated empty states on Overview; Settings rows too wide; section headings in New run quieter than
field labels.

Round 4 changes:
- **Charts:** keyboard chart groups (←/→ walk the shared cursor, Shift ×10, Home/End, Esc); loss axes start at 0; all four dashboard charts reserve the same marker band.
- **New run:** typed epoch and learning-rate fields beside the sliders, with validation and hints; disabled buttons that say why; selects browse with arrows and commit on Enter; section headings above hairlines.
- **Inputs and palette:** palette rows never shrink; text fields scroll to keep the caret visible.
- **Overview:** a single "Get started" panel on first run; New run disabled without a server or dataset; Datasets in a panel; aligned "View all".
- **Playground:** compact side panel with help in tooltips; neutral latency bars; long filenames truncate; +/−/0 zoom.
- **Neutral magnitude colour:** CPU, memory and split bars no longer use series colours.
- **Datasets:** a Fit/Fill toggle (Fit by default) and arrow-key grid navigation.
- **Settings:** two columns on wide windows; key chips in the theme row.
- **Runs:** square validation tiles; roomier log.

## Round 4

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 7 | 8 | 7 | 8 | 7 | 6 | 8 | 8 | 8 | 7 | 7 | 7.4 |
| GPT-6.1 Sol | 8 | 7 | 8 | 7 | 8 | 7 | 7 | 8 | 8 | 7 | 7 | 7 | 7.4 |

Shared issues:
- Mojibake in the Playground: a PowerShell Get-Content/Set-Content round trip double-encoded the UTF-8. A source test now fails on it.
- Validation-sample tiles collapsed to zero height, because the aspect ratio was set on a flex-basis-0 item.
- Legends ran together: the PR legend and the latency legend.
- Epoch axes showed fractional ticks labelled as integers ("0 2 5" for a 3-epoch run).
- Marker bands left dead space above the plots.
- The confusion-matrix ramp flattened 2–25 %, and its background row and column were ambiguous.
- Scrollable areas gave no cue that more content was below.
- Zoomed images could not be panned from the keyboard.
- The training estimate named the GPU even with CPU selected.
- Pale class colours made the confidence bars weak in light mode.
- "Show them" looked like a caption rather than an action.
- Captures caught mid-load on a cold start.

Round 5 changes:
- **Charts:** integer epoch ticks ending on the run's length; failed runs zoom to the epochs they ran; one marker lane.
- **Run page:** validation tiles fixed (aspect ratio on a definite-width box); log lines wrap.
- **Model page:**
  - PR legend as a heading plus a three-column grid.
  - Square-root confusion ramp; "missed" and "extra" labels.
  - Three-decimal precision and recall.
- **Scrolling and keyboard:** thin scroll thumbs on pages, the Playground panel, dialogs and the palette; keyboard pan when zoomed.
- **Detail fixes:**
  - Real GPU in the New run estimate, with honest CPU copy.
  - Class-stroke confidence bars.
  - "Show them" styled as a link.
  - Placeholder clear of the caret.
  - A mojibake source test and a warm-up capture.

## Round 5

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7.9 |
| GPT-6.1 Sol | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 7 | 8 | 8 | 7.8 |

Shared issues:
- "Broken labels" had no action.
- Smoothing on the loss chart went unexplained.
- Confusion-matrix caption and ramp: the caption was wrong in Counts mode, and 1–4 % cells were indistinguishable.
- Playground at minimum height: latency crowded out detections.
- Overview columns: the gutters didn't line up between rows.
- Uneven end ticks (100/120).
- Header actions at three heights.
- Toasts looked like panels.
- The stop glyph read as a checkbox.
- Readout covering the live tip of the curve.
- Mismatched sparkline colours.
- Duplicate "hexagon" detection rows.
- Histograms reachable only by pointer.
- The grid-tile focus ring was clipped.
- The model file always said f32.

Round 6 changes:
- **Ticks:** exact-divisor ticks (0 30 60 90 120); the y-axis top lands on a tick; sparse series show points.
- **Charts:**
  - Marker labels try the other side of their line before stacking.
  - The readout flips relative to the data's end; its title says "smoothed", and the loss header says "smoothed · EMA 0.94".
  - Neutral sparklines.
- **Overview and chrome:**
  - Overview columns share a 360 px right column; System becomes a strip below 1360 px.
  - Header actions sit on the title line.
  - Toasts are elevated and inset 16 px.
  - A filled stop glyph.
- **Model page:** a confusion ramp from the surface, blank cells under 1 %, "missed" and "extra" set apart, and a caption for each mode.
- **Datasets:**
  - Box-size bins as ranges; histograms are keyboard stops.
  - Grid tiles keep their focus ring and show metadata on focus; the Fit letterbox uses the raised surface.
  - "Broken labels" copies the `folo dataset check` command, since the API can't list the skipped files.
- **Playground and runs:**
  - Detection rows show box size; the latency ramp is neutral; the latency footer hides on short windows.
  - An inert "No trained models" field.
  - Queued-run toasts say "Queued".
  - mAP50-95 named everywhere.
  - f16/f32 inferred from file size.

## Round 6

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 7 | 7 | 8 | 8 | 8 | 7 | 8 | 8 | 7 | 8 | 7.7 |
| GPT-6.1 Sol | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 7 | 7 | 8 | 7.9 |

Some captures were stale: `dataset-min.png` predated the last fix. From now on every round is a full recapture.

Shared issues:
- Header actions sat at three heights: the detail pages had their own header code.
- The readout covered curves.
- Stacked marker labels.
- The confusion matrix: blanks for zero and under 1 %, the amber ramp, and the term "extra".
- The Broken-labels tile overflowed at minimum width.
- Recent runs mixed three kinds of value in one column.
- The titlebar repeated the page title.
- The disabled primary button looked like a dimmed hover.
- Naming drift: "Mock data" vs "Built-in demo", "(vulkan)" vs "Vulkan".
- Toasts: they didn't respect the gutter, actionable ones expired, and hover didn't pause them.
- The New run dialog didn't say a run would queue.
- No epoch caption on the x-axis.

Round 7 changes:
- **Headers:** one `detail_header` for every page, with actions on the title line; no titlebar crumb on top-level pages.
- **Charts:**
  - The readout picks the least-covering corner beside the cursor.
  - Close markers pack into one lane.
  - An "epoch" caption sits in the axis corner.
  - The overview loss axis ends on a tick.
- **Confusion matrix:** a blue sequential ramp, "·" for zero, "<1%" shown, "Missed" and "False positives" labels, a short caption.
- **Recent runs:** fixed progress and best-mAP50-95 columns.
- **Buttons:** disabled is neutral (inset surface, border, disabled text); pressed is a real colour step.
- **Toasts:** a 24 px gutter, actionable ones persist, hover pauses expiry; queued runs offer "Open run".
- **New run and Settings:** "Queue training" plus a subtitle when the GPU is busy; "Switch to server" / "Reconnect".
- **Naming:** "Built-in demo" everywhere; device shown as "RTX 4080 SUPER · Vulkan".
- **Smaller fixes:**
  - The Playground header keeps the file name and drops dimensions when narrow.
  - The latency ramp uses three distinct steps.
  - Hidden legend items show a hollow swatch.
  - Compact KPI tiles when the strip wraps.
  - A neutral "best" marker on failed runs.

## Round 7

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 7.9 |
| GPT-6.1 Sol | 8 | 8 | 9 | 7 | 8 | 8 | 8 | 9 | 7 | 8 | 6 | 8 | 7.8 |

Shared issues:
- Overview at minimum width: Best models overlapped Recent runs. A regression from the fixed 380 px column.
- The "best" label was packed away from its marker.
- Lime on tile links and the GPU meter.
- Total loss hidden by default on the dashboard.
- The light-theme latency and split bars were too heavy.
- Settings copy didn't match its buttons.
- The status column order differed between Overview and Runs.
- The empty Playground kept a dead rail.
- Selects didn't close on an outside click, and Home/End didn't scroll.
- The "/" in the Playground file name shrank.
- The inspector repeated the file name.

Round 8 changes:
- **Overview:** stacks Recent runs and Best models below 1200 px, and panels clip; name before status in Recent runs; neutral GPU meter (amber above 95 %).
- **Charts:** marker labels stay on their line (next lane down when they collide), every chart in the row reserves the same lanes, and "best" carries its epoch; total loss is on by default.
- **Neutral colour:** quiet tile links (text_secondary); neutral latency and split ramps.
- **Settings:** "Connect" with copy that matches it.
- **Playground and inspector:**
  - The empty Playground gives the stage the full width, with Open image disabled.
  - The "/" separator never shrinks.
  - The inspector shows the split badge instead of a second file name.
- **Selects:** close on an outside mouse-down; Home/End reveal the option.

## Round 8

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 7.9 |
| GPT-6.1 Sol | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 7 | 7 | 8 | 7.9 |

Shared issues:
- Text clipped mid-letter at minimum size. GPUI truncates text against the width available before flex shrink and caches that, so a shrunken label clips with no ellipsis; tight spots now get fixed widths and shorter labels instead.
- Series colours repeated across the dashboard's charts.
- Plot tops misaligned when one chart used a second label lane.
- Class-chip counts ignored the split filter.
- Playground latency vanished on short windows.
- The inspector's coordinates had low contrast and no header.
- Toasts stacked over content.
- Chart keyboard paths weren't discoverable.
- Status dots used lime.
- Format drift ("42/100", "640px").

Round 9 changes:
- **Series colours:** total loss and the learning rate are neutral; box/cls/dfl take series slots 2–4; precision and recall share slot 5, recall dashed, with a dashed legend swatch.
- **Chart labels:**
  - Every grid chart reserves two label lanes.
  - Markers keep their given order, so "best" sits on top.
  - Failed runs mark "failed · epoch N".
  - The finished-run throughput sparkline is kept.
- **Datasets:** chip counts follow the split; the strip narrows below 1120 px.
- **Playground:**
  - The latency scrolls with the panel on short windows.
  - The compact toolbar keeps file and size.
  - The empty-state subtitle is honest; the object-count dot is gone.
- **Inspector:** a coordinate header and readable highlighted rows; no duplicate split badge.
- **Toasts:** the newest shows in full, older ones fold into "N more · Dismiss all".
- **Charts:** a key hint under a keyboard-focused chart group; a fade under the log header.
- **Consistency:** connection dots use success green; one format ("42 / 100", "640 px").
- **Tooling:** `shoot-all.ps1` retries a capture once when a window is slow to appear.

## Round 9

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7.9 |
| GPT-6.1 Sol | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 9 | 8 | 8.2 |

Shared issues:
- The Runs table wrapped "120 / 120".
- Marker labels at different heights in sibling charts.
- Empty bands above the plots.
- The Recent runs gap at wide widths.
- box and dfl were neighbouring hues.
- The folded toast row couldn't expand and floated as its own card.
- "Following" on the log wasn't true once scrolled.
- Help icons were pointer-only.
- Chip counts didn't say they count boxes.
- Slider thumbs had no hover state.
- Raw loss traces competed with the smoothed lines.
- The KPI block was heavy.
- The failure marker sat past the data.
- Shortcuts were split mid-group.
- Dimmed label chips lost contrast.
- The light-theme letterbox disappeared.

Round 10 changes:
- **Charts:**
  - Markers have pinned lanes (lane 0 for events, lane 1 for the schedule), so siblings match.
  - Less padding above plots; raw traces at 9 %.
  - dfl dashed, with a dashed legend swatch.
  - The failure marker sits at the last trained step.
- **Runs:** fixed 72 px no-wrap fractions; one-step-smaller KPIs when the strip wraps.
- **Log:** follows only at the bottom, with "Resume following" otherwise.
- **Recent runs:** a fixed 170 px name column with a spacer after the status.
- **Toasts:** a slim strip attached to the stack: "Show N earlier" / "Dismiss all".
- **Playground:** focusable help icons that show their text on keyboard focus; latency follows the samples in one scrolling column, and the toolbar drops the duplicate ms.
- **Datasets:** "Boxes" before the class chips; a visible letterbox cell in both themes.
- **Controls:** slider-thumb hover.
- **Inspector:** label chips stay full strength when other boxes dim.
- **Settings:** shortcuts in "Actions" and "Go to" groups.
- **New run:** a fade above the footer.

## Round 10

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 8 | 7 | 8 | 7.8 |
| GPT-6.1 Sol | 8 | 8 | 9 | 8 | 8 | 7 | 8 | 8 | 8 | 7 | 8 | 8 | 7.9 |

Shared issues:
- Failed-run labels collided: best and failure were both pinned to lane 0.
- The dataset filter row wrapped awkwardly at minimum width.
- Scroll thumbs were paint-only.
- The confusion matrix's single blue ramp made errors look like hits.
- The System strip at 1280 was mostly empty.
- Image tabs and the split filter used the same control.
- The Playground rail was too wide at minimum size.
- Close mosaic had no number field.
- The palette cut its last row.
- Toasts expired under keyboard focus.
- Reduce motion ignored the OS setting.
- Copy drift ("at epoch" vs "· epoch"; pill before vs after the name).
- "Server unreachable" appeared three times.

Round 11 changes:
- **Charts:** a pinned lane falls back to another lane on a collision; "failed in epoch N".
- **Scroll thumbs:** drawn from `ThumbGeo` and draggable (app-level drag state); a track click jumps there.
- **Confusion matrix:** blue only on the diagonal, errors and margins on a neutral ramp; the bottom row is explained as the false-positive split.
- **Dataset detail:**
  - Underline tabs for Images / Statistics.
  - Class chips on their own row below 1100 px.
  - The grid defaults to Fill.
- **Overview:** System meters flex; the hero pill comes after the name; a quiet subtitle on error; the hero loss axis ends on a tick.
- **Playground:** a 280 px rail and latency in the toolbar below 1100 px.
- **New run:** a Close mosaic number field, validated against the epoch count.
- **Overlays and accessibility:**
  - A bottom fade on the palette.
  - Toasts wait while their buttons have keyboard focus.
  - Reduce motion starts from Windows' animation setting (`SPI_GETCLIENTAREAANIMATION`).

## Round 11

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 8 | 7.9 |
| GPT-6.1 Sol | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 7 | 8 | 8 | 8.1 |

Rounds 5–11 held at 7.8–8.2. Per-issue fixes had reached diminishing returns, so round 12 is a structural redesign
against `DIRECTION.md`, written by a design-director pass. The review brief now also requires a "To make it a 9:" for
every line below 9.

Round 12 changes (structural):
- **Chart anatomy:** value legends instead of the floating readout; a cursor x-chip on the axis; Geist tabular ticks;
  one annotation band; the best epoch as a lime dot on its series; no "epoch" corner caption.
- **Run dashboard:**
  - Five KPIs in one row.
  - Validation mAP (3) beside Training loss (2) at 220 px, with loss toggles and values on their own row.
  - Precision / recall beside the learning rate at 140 px.
  - Rows stack below 1120 px.
- **Overview and sidebar:**
  - The sidebar live card becomes a 36 px row with a 2 px rail.
  - The Overview hero has a 48 px header with a progress rail, a 2×2 KPI column, one mAP chart with a value legend,
    and a system-meter footer.
  - Lists are single-line 36 px rows with dot statuses; meta columns appear only at 1500 px and up.
- **Runs and model detail:**
  - The Runs table is single-line 36 px rows with Model, Dataset and Size · batch columns (wide only).
  - Durations read "14m 55s".
  - Model detail has a double-width 32 px mAP50-95 tile; the per-class table isolates PR curves on hover or focus.
- **Dataset, Settings and New run:**
  - Dataset detail puts tabs, split, Fit/Fill, Boxes and size on one line; class chips go on a row with the count.
  - Settings is one 760 px column.
  - New run puts each slider and its exact field on one row.
- **Header meta:** a 12 px line.
- **Richer demo data:** 5 datasets, 9 runs, 6 models.

## Round 12 (structural)

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 7 | 8 | 6 | 7 | 7 | 8 | 7 | 8 | 8 | 8 | 7 | 7.4 |
| GPT-6.1 Sol | 8 | 8 | 8 | 7 | 8 | 7 | 8 | 8 | 8 | 8 | 7 | 8 | 7.8 |

The redesign landed, but its seams cost points:
- Settings controls were ragged, with the address field clipped.
- The three dashboard rows used three different gutters.
- Linked charts showed different ticks, and the cursor chip collided with tick labels.
- The PR legend was below the fold.
- Units sat below the baseline.
- The loading skeleton didn't match the loaded page.
- Off-diagonal confusion cells were flat.
- Sol again reported missing light-theme Playground boxes; the capture has them.

Round 13 changes:
- **Settings:** label columns flex, controls right-aligned in full-width rows (the panels are flex columns: block panels don't stretch children in GPUI).
- **Run dashboard:**
  - One 3:2 grid for all three rows.
  - Every chart card is header + 28 px legend row + plot, so plot tops align.
  - Six shared x ticks; tick labels hidden under the cursor chip.
  - Three y ticks on the small charts.
  - Square cropped validation samples; compact charts on short windows.
- **Type and copy:** units bottom-aligned and lifted to the numeral's baseline (`baseline_lift`); "best X @ N" everywhere.
- **Model page:**
  - PR class chips with AP50 in the panel header; hover or keyboard focus isolates a curve.
  - Danger-tinted ramp for confusion errors, scaled to 20 %.
  - A detail line on every model KPI.
- **Models table:** 36 px single-line rows with a From run column.
- **Overview:** the loading skeleton mirrors the hero; the chart tints the epochs not reached yet, labelled with the ETA.
- **Polish:**
  - Dialogs and the palette fade in over 160 ms (instant under Reduce motion).
  - Grid tiles show a hover edge.
  - Taller palette and log edge fades.
  - Tabular Geist for coordinates and number fields.

## Round 13

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8.0 |
| GPT-6.1 Sol | 8 | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 8.2 |

The seams from round 12 were closed. What was left was mostly measurable and local:
- Models table columns overflowed at narrow and default widths, and headers sat 3–7 px off their values (header cells could shrink).
- PR isolation could stick after focus left. Export had its own menu rules.
- The evaluation row pushed the per-class table below the fold. Errors above 20 % shared one colour.
- At 960 × 620 the run dashboard showed only one primary chart.
- Loading Overview had a separate system strip, so the page jumped 44 px.
- 0–1 axes used three different tick schemes. The PR legend wrapped and never named AP50.
- Best models overflowed its column at wide sizes.
- Precision and recall shared one hue.

Round 14 changes:
- **Models table:**
  - Header and value cells share a flush-right fixed cell, and header cells no longer shrink.
  - Params drops below 1360 px, From run below 1240.
  - mAP50-95, Size and Created sort (click or Enter, 12 px chevron).
- **Model page:**
  - The KPI strip shares caption, value and detail lines, and Precision and Recall gain detail lines.
  - The PR legend is a list beside the plot ("Class / AP50", best first).
  - Hover and focus isolate a class temporarily; Enter or a click pins it, derived at render.
  - Confusion errors use a continuous square-root ramp over 0–100 %, with a key. "<1%" and zero cells are neutral.
  - 32 px cells, and the table head doubles as its title, so four class rows sit above the fold at 1280 × 800.
- **Export:** a select-style menu with arrows, Home/End and Enter, closed by Escape, Tab or a click outside.
- **Run dashboard:** short or narrow windows get 64 px KPI tiles, 32 px chart headers and 140 px plots, and the primary pair stays side by side. Recall has its own hue.
- **Axes:** every 0–1 axis uses 0.25 steps with two decimals (three ticks on small charts).
- **Overview:**
  - The loading hero has the loaded hero's geometry and no separate strip.
  - The header drops the duplicate percentage.
  - Best models is clipped to its column.
- **Datasets:** tabs have hover and pressed states, and the filter count is per split ("564 of 800 val images match").
- **Playground:** the detections list has Class, Size (px) and Confidence labels.
- **Code review of #43:**
  - Close mosaic revalidates when the epochs change.
  - Tick labels give way only to a chip that is actually drawn.
  - Log following tracks a revision, so it keeps following at the 2,000-line cap.

## Round 14

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8.0 |
| GPT-6.1 Sol | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 9 | 8 | 8.1 |

Sol gave its first 9s on hierarchy and responsiveness. Remaining items:
- Divergences from the shared design rules:
  - The loss encoding.
  - The selected-box treatment.
  - Queued shown in a status hue.
  - Mono scores on the detection chips.
- Keyboard: Tab didn't reveal off-screen controls in scrolling forms. The stop dialog's X and scrim didn't restore focus. Disabled buttons weren't Tab stops.
- Live charts jumped instead of easing.
- Measurable slips:
  - Near-black "<1%" cells.
  - Precision–recall axes and ticks.
  - Header and column alignment in Recent runs.
  - The datasets skeleton's padding.
  - Orphan-row card widths.
  - Run KPIs at minimum size.

Round 15 changes:
- **Shared rules:**
  - Loss: box = series 1, cls = series 4 dashed, dfl = series 5 dotted, total = text.primary and thicker; the legend samples match.
  - Selected box: 3 px accent stroke, the others at 30 %, and a 2 px accent bar on its list row (inspector and Playground). The inspector's list is sorted by class, then area.
  - Queued is neutral.
  - Chip scores use tabular Geist.
- **Keyboard:**
  - A keyboard focus move scrolls the focused control and its ring into view, with an 8 px margin, in the dialog, the Playground rail or the page.
  - Every stop-dialog dismissal restores focus.
  - Disabled buttons are Tab stops that show their reason while focused.
- **Live charts:** new data eases in over 240 ms. Shared points move and new points slide out. It applies only to appends and is instant under Reduce motion.
- **Model page:**
  - Precision–recall has axis titles and 0.25 ticks on both axes.
  - The legend's hint moves to the panel header.
  - Confusion cells and the ramp sit on the panel surface, and the key gains a "correct" swatch.
- **Run dashboard:** the mAP chart has the future span with its ETA, like the Overview. P&R uses 0.25 ticks. KPI tiles keep their detail lines at every size, and the headline value uses the row's size.
- **Overview:** Recent runs gains a column-head row on the row grid, with Progress right after Status. Dataset · arch shows whenever there's room; Created drops when the list sits beside Best models. The datasets skeleton has its padding.
- **Other:**
  - Equal-third dataset cards.
  - The empty Playground drop zone is capped at 420 px and centred.
  - 56 px Settings rows.
  - Copy: the Models subtitle and the server-address help.
- **Code review of #45:**
  - Tab closes Export.
  - The Export trigger only toggles; Enter commits the highlighted option.
  - Pins reset between models and are validated.
  - The Models order is cached per data revision and column.

## Round 15

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 9 | 8 | 9 | 8 | 9 | 8 | 8 | 8 | 8.3 |
| GPT-6.1 Sol | 8 | 9 | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8.3 |

Both reviewers gave their first 9s on several lines. The remaining items were mostly local:
- Runs columns at 1280 and minimum size.
- Icon tooltips missing on keyboard focus.
- A selected chip still in its class colour.
- The PR chart's floating readout.
- Unreadable confidence on chips.
- Best models dropping its architecture column at 1280.
- Dead space in the Overview hero.
- The confusion-matrix key crowding the column labels.
- The empty Playground placement.
- The class table's layout.

Between rounds, a real-server smoke test (#52) wired up dataset creation and resume, and fixed the display of unknown GPU memory and load.

Round 16 changes:
- **Runs:** columns are budgeted per breakpoint. Dataset shows from 1240, Size · batch from 1360, Created from 1440, and header cells don't shrink.
- **Keyboard:** icon buttons show their tooltip on keyboard focus. Switches show hover in both positions.
- **Chips:**
  - The selected chip takes `accent.solid` / `accent.on`, matching its outline.
  - Confidence uses the chip ink at full strength, set apart by weight (passes AA).
  - Detection bars use the class colour.
- **Model page:**
  - The PR legend shows precision at the cursor (no floating readout; an x chip on the axis).
  - The confusion key sits under the matrix.
  - The class table has a 200 px name column and lists the best AP50 first.
- **Overview:** the hero KPIs are centred and the ETA label no longer repeats. Best models shows five rows with architecture.
- **Smaller fixes:** the empty Playground drop zone is top-aligned, and the thumbnail-size control is labelled as such.

## Round 16

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 9 | 9 | 8 | 9 | 8 | 8 | 9 | 9 | 9 | 8 | 9 | 8.6 |
| GPT-6.1 Sol | 8 | 8 | 9 | 9 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Opus is at 9 on seven lines. What was left:
- **Failed runs:** recovery advice contradicted itself (Resume after an OOM, no primary action).
- **Charts:**
  - Edge tick labels were clamped off their ticks.
  - The cursor chip was unnamed ("36").
  - Loss endpoint dots overlapped.
  - The LR caption didn't mention the epoch axis.
  - The PR cursor column had no heading.
- **Inspector:** dimmed boxes kept full-strength chips.
- **Overview:** Recent runs dropped Created at 1280 while showing it at 960, and the hero KPIs weren't aligned.
- **Labels:** class names and Playground filenames had no width contract.
- **Small items:**
  - The loss-smoothing explanation was pointer-only.
  - PR rows had no pressed state.
  - The Best models skeleton had 4 rows against 5 loaded.
  - The copy icon gap.

Round 17 changes:
- **Failed runs:**
  - "New run from this config" is the primary action.
  - The OOM banner says the new run halves the batch, and OOM failures don't offer Resume.
  - Elsewhere the button reads "Resume from epoch N" (from the last finished epoch).
- **Charts:**
  - The first and last x labels anchor to their ticks.
  - The cursor chip reads "epoch 36".
  - Live endpoint dots within 6 px of another are skipped.
  - The LR caption reads "per step · by epoch".
  - Dimmed boxes dim their chips too.
- **Model page:**
  - The PR legend has a "Precision" column ("–" until inspecting) and 84 px truncating names.
  - Rows show pressed feedback.
  - Class table names truncate at a known width.
- **Overview:**
  - Recent runs tightens every column beside Best models, so all six fit at 1280.
  - The hero KPIs are top-aligned with the chart title, rows 24 apart.
  - The Best models skeleton has 5 rows.
- **Playground:** a filename budget per breakpoint, with the full name on hover.
- **Small fixes:**
  - The loss legend gets a focusable help icon.
  - The log has a taller top fade.
  - A 4 px gap before the copy icon.

## Round 17

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8.1 |
| GPT-6.1 Sol | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 9 | 7 | 8 | 8.3 |

A regression cost Opus a point on most lines. The PR legend's line keys had collapsed to zero width (no `flex_none` beside fixed-width columns), so the curves couldn't be told apart.

Other items:
- The Precision column read "–" at rest.
- The future band showed on one chart of four.
- Validation samples had no chips.
- Precision shared a hue with dfl.
- "Running" wasn't accented in tables.
- The model page had no narrow layout.
- Narrow gutters were 24 px.
- New run was available before datasets loaded.
- Broken labels weren't emphasised.
- Export size estimates were unconditional.
- Light-mode primary buttons and class swatches were missing their edges.
- LR ticks needed a shared power of ten.

Round 18 changes:
- **PR legend:**
  - Line keys are `flex_none`.
  - At rest the precision column shows the conf 0.25 operating point ("P @ 0.25"); while inspecting it shows "P @ R 0.62".
  - An "All classes" row with a hairline sits under the classes.
- **Run charts:**
  - Every linked chart shades the remaining epochs (the ETA label stays on mAP).
  - Precision is magenta and recall teal dashed, hues no loss part uses.
  - LR ticks read "0 5 10" with "× 10⁻³", and the legend reads "6.58 × 10⁻³".
- **Validation samples:** score chips; boxes under 20 px show labels only when selected (design/README).
- **Shared design rules:**
  - "Running" uses accent text.
  - Class swatches get a `chart.swatch_edge` inset.
  - Light-mode primary buttons get a neutral 1 px edge.
- **Layout:** below 1120 px, model detail stacks PR over the confusion matrix and page gutters are 16 px.
- **Guards and copy:**
  - New run is disabled ("Loading datasets…") until datasets load, and Ctrl+N waits too.
  - A non-zero broken-label count reads as a warning.
  - Export sizes are "≈" estimates from the parameter count (f16 2 B and ONNX f32 4 B per weight).
- **Overview:** the hero KPIs spread to the chart's height.

## Round 18

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 8 | 9 | 8 | 8 | 8 | 8.0 |
| GPT-6.1 Sol | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Both reviews centred on the model page's cramped PR plot (~175 px beside its legend) and on box labels:
- A chip floated over a cropped box.
- Score-only chips had no class key.
- Small-image chips used the full size.
- Dimmed chips were brighter than their boxes.
- Labels were suppressed by height only.

Other items:
- The confusion error ramp started with a jump and peaked at α 0.45.
- The Overview KPI column had a void, Best models had no heads, and the loading meter band was empty.
- P&R had 3 ticks (a leftover override) and "close mosaic" was printed four times.
- A class filter didn't show inside tiles.
- Select menus never flipped; pan was unbounded.
- Training estimates were attributed to the detected GPU.

Round 19 changes:
- **Model page:**
  - PR gets 62 % of the row, so the plot is ~280 px; the matrix keeps 400 px and the label column is 96 px. The legend is 252 px and the x ticks read 0 / 0.5 / 1.
  - Confusion errors use α = 0.8·√(v / max error), and the key reads "errors 0 → 25% (max)".
  - Precision / Recall detail reads "best-F1 · IoU 0.50".
- **Box labels:**
  - No chip for boxes cropped out of view; chips clamp to the visible part of their box.
  - Chips keep a 2 px gap from each other.
  - Views under 480 px use 14 px chips with 10 px text.
  - Dimmed chips are at 30 %, like their boxes.
  - The small-box rule tests width or height.
  - Validation samples get a class key.
- **Run charts:** P&R uses the 0.25 grid. "close mosaic" is named on the mAP chart only; the others keep the line.
- **Overview:**
  - KPIs have a fixed 24 px gap.
  - Best models gets column heads that share row lines with Recent runs.
  - The loading hero has meter skeletons.
- **Controls:**
  - Select menus open above the trigger when there's no room below, capped to the space available with an 8 px margin.
  - Image pan keeps at least a 48 px strip in view.
  - Long detection names show in full on hover and on keyboard focus.
- **Datasets:** a class filter drops the other classes' boxes to 30 % in each tile.
- **Copy:** New run calls its time a reference estimate.

## Round 20

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 8 | 8 | 7.9 |
| GPT-6.1 Sol | 8 | 9 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.2 |

Opus's accessibility 7 rests partly on "run charts have no keyboard path". That's wrong: `chart_group` in `controls.rs` gives each chart group a Tab stop with ← / → / Shift / Home / End / Esc. The review brief now points reviewers at it.

Real items:
- Score-only sample chips.
- The LR multiplier floated on its own line, and the legend read "lr 6.58".
- P/R was labelled differently on the run and model pages.
- Log timestamps used the disabled token.
- The inspector label list had no focus-reveal path, and its key hints were pinned far below.
- Sortable headers looked static.
- Latest-point halos competed with the cursor.
- The New run shortcut read as a third control.
- Dimmed chip text was hard to read.
- The empty Overview had a void.

Round 21 changes:
- **Run dashboard:**
  - Sample chips show "circle 0.86" where the view is at least 140 px wide.
  - The LR caption carries "× 10⁻³"; the legend and axis are in that power (`LineChart.pow_caption`).
  - P/R reads "best-F1 · IoU 0.50 · val" and "best-F1 · validation".
  - Latest-point halos hide while the cursor inspects.
  - "close mosaic" is labelled on the top row.
  - Log timestamps and debug tags use `text.tertiary`.
  - dfl dots are spaced 1.5 / 3.
- **Inspector:** the label list has its own scroll handle for focus reveal, and the key hints follow the list.
- **Models table:** sortable headers show a faint chevron and a "Sort by …" tooltip.
- **New run:** the shortcut is quiet text before Cancel.
- **Overview:** the first-run page gets an "Or from a terminal" panel (synth, train, serve).
- **Copy:** "Boxes by class" on dataset chips; "failed during epoch N"; dimmed chip text at 85 %.

## Round 21

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 8 | 8.3 |
| GPT-6.1 Sol | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8.2 |

Opus now gives 9 for interaction, accessibility and responsiveness; with the review brief pointing at `chart_group`, its keyboard checks came back clean. Both reviewers flagged chips still overlapping in dense sample thumbnails (an occupied-spot fallback). Other items:
- Squashed loss components under the total.
- The stop dialog promised the epoch in progress as a checkpoint.
- Keyboard chart inspection started out in the future epochs.
- Grid boxes had no class names.
- No visible default sort, and mAP50 wasn't sortable.
- Recent runs showed progress only for the running row.
- Mixed label styles on the confusion axes, and mixed bar encodings in the class table.
- Plain-text shortcuts, sentence-case helper copy, and a narrow Settings page.
- Missing pressed states; a silent refusal on the last loss toggle.
- "mock data" copy.

Round 22 changes:
- **Chips:**
  - No fallback onto an occupied spot: three more spots down inside the box are tried, then the chip is skipped (the selected box's chip always shows).
  - Dataset tiles get class-name chips; narrow views drop names and never draw empty chips.
- **Charts:**
  - Training loss is on a log axis (powers of two: 8 4 2 1 0.5).
  - Keyboard chart inspection starts at the latest recorded point and never steps into unrecorded epochs (`chart_group` takes `end`).
- **Copy:**
  - The stop dialog names the last finished epoch's checkpoint and says the epoch in progress is discarded.
  - "choose Built-in demo in Settings".
- **Models:** the default sort is Created, descending (shown); mAP50 is sortable.
- **Overview:** every Recent runs row has a progress cell ("37/100").
- **Model page:**
  - Both confusion axes use XS 500.
  - Per-class bars are neutral.
- **New run:** the shortcut is keycaps again (left of Cancel); helper copy is fragments ("last 10 epochs train without mosaic").
- **Settings:** full content width.
- **Pressed states:**
  - `bg.active` on hoverable rows, tiles and toggles.
  - The last visible loss toggle explains "Keep at least one series visible".

## Round 22

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 8 | 9 | 9 | 8 | 8 | 8.3 |
| GPT-6.1 Sol | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Both reviewers found the confusion-matrix heading "hexagon" hard-clipped (flex cells truncated without a known width), and both found the chip piles on the dataset-card montages too dense. Other items:
- The log loss axis wasn't disclosed.
- Unsorted columns used the sorted glyph.
- Progress read "42/100" in one place and "42 / 100" in another; queued runs read "waiting" in one place and "Waiting for GPU" in another.
- A dead band under the hero KPIs.
- A toast-stack header that read as a second toast.
- No hover on selected chips.
- Keyboard zoom wasn't disclosed in the inspector.
- A wide sparkline in the wrapped KPI tile.
- Loud raw-loss underlays.
- Tall chrome at the minimum size: the first dataset image row started at 349 px, and New run hid Schedule.
- Settings width: one reviewer wants the 760 px column back. This is now a settled decision in the brief, capped at 880 px.

Round 23 changes:
- **Confusion matrix:**
  - Headings are fitted to the measured column width; a name that doesn't fit shows a short form ("hex.") with a tooltip. "Missed" stays whole.
  - The matrix takes 42 % of the row (PR 58 %).
- **Dataset cards:** box outlines only, no chips (the dataset page names every box).
- **Models table:** unsorted columns show a faint up-down mark; the active column's label and arrow are text.primary.
- **Copy:**
  - "42 / 100" everywhere, and "Waiting for GPU" on Overview too.
  - The loss caption reads "log scale · EMA 0.94".
- **Charts:**
  - Raw underlays at 5 %.
  - The throughput sparkline is capped at 160 px.
- **Overview hero:** the KPI rows span the chart's height; Epoch gets its progress bar.
- **Toasts:** "Show N earlier · Dismiss all" is a compact right-aligned pill; the earlier toasts' edges peek out behind the latest.
- **Selected chips:** a hover tint over the selected fill.
- **Inspector hints:** add "+ − zoom" and "0 fit".
- **Short windows (< 700 px tall):**
  - The dataset path joins the title line; page gaps are 12 px; the stats strip uses XL values. The first image row moved from 349 to 305 px.
  - New run tightens its rhythm and drops restating helpers, so Epochs, the learning rate, Optimizer and Close mosaic are all in view.
- **Settings:** capped at 880 px, left-aligned.
- **Review brief:** a "settled decisions" paragraph (Settings width, KPI gaps, "close mosaic" labelling, LR exponent placement).

## Round 23

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8.1 |
| GPT-6.1 Sol | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 7 | 9 | 8 | 8.2 |

Opus found an 8 px dark band between live data and the future tint on every live chart. Both reviewers found the Overview hero's KPI column left a gap either way it was aligned. Sol found that the Overview chart had no keyboard path. Other items:
- The empty Playground's New run was a dead end with no datasets.
- Thumbnail focus rings were clipped.
- Dimmed chips were unreadable over photos.
- The colour-only connection dot in the collapsed rail.
- The matrix diagonal reused the circle class's cyan.
- A toast seam.
- Missing pressed states.
- An unsortable Runs table.
- No region jump.
- Cramped P/R ticks.
- Copy drift.
- Model names cut at 200 px.
- Left-aligned per-class numbers.
- End tick labels off their ticks.

Round 24 changes:
- **Charts:**
  - The future tint starts at the latest point drawn, behind a 1 px "now" rule.
  - Every x label is centred on its tick.
  - The run P/R chart and the short Overview chart tick at 0 / 0.50 / 1.00.
- **Overview hero:** one row of four KPIs above a full-width chart, so there is no side column to leave a gap; the loading skeleton matches. The chart is a `chart_group` (keyboard cursor).
- **Confusion matrix:**
  - Correct cells use the success ramp.
  - The caption fits one line.
- **Playground:**
  - With no models and no datasets, the primary action is "Add a dataset" (to Datasets).
  - Sample thumbnails clip inside, so their focus ring shows whole, and they have a pressed state.
  - The PR legend reads "P @ best F1" at rest.
- **Dimmed chips:** an opaque muted class tint with contrast-picked text.
- **Collapsed rail:** the connection shows a check, refresh or x glyph with a tooltip.
- **Pressed states (`bg.active`):** nav, breadcrumbs, search, live-mini, toast links, inspector rows, select trigger and options, dataset KPI tiles, cards and grid tiles.
- **Runs table:** sortable by Best mAP50-95, Duration and Created (the shared `sort_th_on`).
- **Models table:** names use the width the other columns leave, with the full name on hover.
- **Per-class table:** numbers right-aligned.
- **F6 / Shift+F6:** cycle the sidebar (current section), search and the page's first control.
- **New run:** the architecture letters are in Geist Mono.
- **Toasts:** the peek edge uses line.default; the pill docks 2 px above it.
- **Copy:**
  - The run P/R caption reads "best-F1 · IoU 0.50 · val".
  - Best models drops "by mAP50-95".
  - "then Add dataset → choose ./shapes".
  - The auto-start help says it applies at startup.
  - A shorter stop dialog.
- **Cleanup:** dead leftovers removed (`chart_block`, `let _ =` placeholders); the disabled-button comment now matches its behaviour.
- **Review brief:** the selection colour is now a settled decision.

## Round 24

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 9 | 8 | 8 | 8.3 |
| GPT-6.1 Sol | 8 | 9 | 8 | 9 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Both reviewers are at 8.3 with seven lines at 9 between them. Opus found a real bug: `alpha()` replaces a translucent token's alpha, so the toast peek edge (`line.default` at alpha 1) drew opaque white. Other items:
- Pages and breadcrumbs disagreed on the narrow gutter.
- Chart legends were outdented 6 px from their titles.
- The false-positive row set the top of the error ramp.
- 24 px progress stubs.
- Misaligned KPI-wrap dividers.
- Uneven short log ticks (8 4 1 0.25).
- Missing end dots (mAP50, recall).
- A contradictory empty Playground.
- Overview's legend placed unlike the run screen's.

Sol asked for:
- grayscale-safe series (mAP50 dashed);
- surface-safe confidence bars in light mode;
- hover on the selected segment;
- help where short windows drop helper rows;
- "last validated" for stopped and failed runs;
- a keyboard caption on the PR panel;
- a shorter compact hero.

Sol's dimmed-outline opacity (0.55) contradicts design/README's 30 % rule, so the brief now lists it as settled. Grid-tile chips stay as they are: an earlier round asked for class names on grid boxes.

Round 25 changes:
- **`theme::fade`** scales a token's own alpha; the toast edge uses it.
- **Gutter:** one `gutter()` (16 / 24) for every page, the dataset detail, Playground and the breadcrumbs.
- **Charts:**
  - Run chart legends start at 16 px, under their titles.
  - Log axes raise their top to the next power on the tick stride (`pow2_top`), so steps stay even: 16 4 1 0.25.
  - mAP50 is dashed (charts and legends) and gets an end dot.
  - Near end dots both show (only a dot within 3 px of another is dropped).
- **Confusion matrix:** the false-positive row is off the error ramp, on its own neutral ramp, 8 px below. The key's max is now the largest class-to-class confusion. The n/a cell is plain text.
- **Overview:**
  - Progress shows the number only where a bar would be under 48 px.
  - The hero legend is a left-aligned row under its title (skeleton too).
  - The short-window hero is lower (76 px plot, 12 px padding).
- **Run page:**
  - The KPI wrap is on a 3-column grid: Throughput spans two columns, so the dividers line up.
  - The mAP KPI reads "last validated" for stopped and failed runs.
- **Playground:**
  - The empty state's headline, body and button agree; the drop zone is dashed only when a drop works.
  - Confidence bars use `class_stroke`.
- **Controls:**
  - Selected segments get a hover tint.
  - Short windows keep Batch size and Device help as icons.
- **PR panel:** the caption shows keys while the chart or a class row has keyboard focus.
- **Empty Overview cards:** 16 px padding.

## Round 25

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 8 | 8 | 8.4 |
| GPT-6.1 Sol | 9 | 9 | 9 | 9 | 8 | 8 | 8 | 9 | 9 | 8 | 8 | 8 | 8.5 |

The best round so far: Sol has seven lines at 9. Both reviewers found Playground latency hidden at the default size (and a row cut by the rail's bottom edge). Other items:
- Help popovers could leave the window.
- Selection on class chips was colour-only.
- "Boxes by class" counted boxes but filtered images.
- The FP row's denominator wasn't stated.
- "Throughput" on finished runs read as current.
- The toast stack still let the page show through.
- The 1280 Progress column dropped its bar.
- The short log axis grew 15 % empty headroom.
- The ETA label sat outside the marker band.
- "hex." was shortened although it fit.
- The PR panel had a dead band.
- Contradictory comments.

Opus also measured the light theme's warning text at 4.44:1. That is a token in design/, so it went to the orchestrator.

Round 26 changes:
- **Orchestrator follow-ups from #69:**
  - Registering a duplicate folder says "already added".
  - Local GPU OOM uses the `gpu_out_of_memory` code, with a "Run on Local CPU" action.
  - Playground and New run show server hints (`device_not_found`, OOM).
  - One shared OOM check for failed runs.
  - The Broken labels tile splits its count by severity (issue counts).
  - The README Desktop screenshots are refreshed.
- **Playground:**
  - Latency in the toolbar at every width.
  - The rail fades out at its bottom edge (12 px of trailing room).
  - The filename's budget comes from the toolbar's remaining width.
  - The Run on note: "Local GPU / CPU need a model from folo serve".
- **Help popovers:** anchored and snapped 8 px inside the window.
- **Class chips:**
  - Selected chips show a check in the class colour in the swatch's slot.
  - The row is "Filter images by class", with box counts in each chip's tooltip.
- **Confusion matrix:**
  - The FP row reads "Share of all FP" in Row % mode.
  - Headings that fit render whole (2 px overhang allowed), so "hexagon" is no longer "hex.".
- **PR panel:** the header caption is "IoU 0.50"; the keyboard- or pointer-aware instruction moves under the class list.
- **Run page:**
  - Finished runs read "Final throughput".
  - Log axes count the tick stride down from the top power, so the domain stays the data's and the steps stay even (8 2 0.5).
  - The ETA label sits in the marker band when lane 0 is free there.
- **Overview:** Progress is 112 px wide in the 1280 band (name 124, best 80), so its bar shows.
- **Toasts:** opaque peek edges docked to the pill, and the stack occludes.
- **Comments:** the misplaced ones are fixed.

## Round 26

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 9 | 8 | 8 | 9 | 8 | 9 | 9 | 9 | 8 | 8 | 8.6 |
| GPT-6.1 Sol | 9 | 9 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Opus is at 8.6 with eight lines at 9. Sol found four behaviour bugs:
- New run errors rendered below the fold.
- Sliders and typed values took different ranges.
- The Empty images and class filters could combine into an impossible filter.
- In the inspector, a stale pointer hover outranked keyboard focus.

Both reviewers also flagged:
- the toast stack's floating pill;
- PR curves told apart by colour only;
- unlabelled chip counts and the copy action;
- the inspector's hint layout;
- the compact Playground rail;
- the New run fade over controls;
- the Overview chart without its caption or ETA;
- the error state's missing subtitle;
- a redundant "Open run".

Round 27 changes:
- **New run:**
  - Errors lead the form; a failed submit scrolls there, and an empty name focuses Name.
  - Sliders draw (and map positions back with) a range widened to any typed value, recorded per slider in `slider_ranges`.
  - 36 px of bottom room under the fade.
- **Dataset page:**
  - Empty images and a class filter are mutually exclusive.
  - The chip row reads "Filter images by class · counts are boxes".
  - The Broken labels tile shows the severity split and then a named "Copy check command".
- **Inspector:**
  - While the keyboard is in use, the focused label row decides the highlighted box.
  - Key hints are a grid with a 96 px key column, one action per row.
- **PR panel:** curves and legend swatches are solid, dashed or dotted by class.
- **Toasts:** "Show N earlier · Dismiss all" sits in the newest card's action row; no floating pill.
- **Queued-run toast:** offers "View queue" (Runs, Active), since the run's page opens anyway.
- **Overview:**
  - The chart has its "per epoch" caption and ETA label.
  - The error state keeps a subtitle ("Not connected · <server>").
- **Run page:** the wrapped KPI row splits evenly (no stretched sparkline tile).
- **Short windows:** a tighter Playground control block.

## Round 27

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 9 | 9 | 8 | 8 | 8.4 |
| GPT-6.1 Sol | 9 | 9 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Both reviewers asked for confusion-matrix cells to be readable (hover and keyboard) and for PR patterns that don't repeat (three patterns for five classes). Sol found:
- Slider ends quantised into invalid values.
- Playground's hover outranking keyboard focus.
- Toast controls not holding expiry.
- Disabled-option reasons that were pointer-only.
- Histograms without Shift stepping.
- Latency placed below the sample picker.

Opus found:
- Slider tracks inset from the column.
- The chip-row label too long.
- Precision named three ways.
- The inspector header's "w × h" against its values' "w×h".
- Two empty-state surfaces.
- The cursor chip hiding the tick under it. That is by design, and now a settled decision in the brief.

#85's code review fixes (slider ranges frozen mid-drag and exact ends, pointer precedence, separate toast-control rows) shipped with round 27.

Round 28 changes:
- **Confusion matrix:**
  - Cells read out on hover, or via one keyboard Tab stop (arrows move a cursor cell; Esc clears).
  - The readout spells out counts and denominators ("circle predicted as square: 41 · 4% of circle objects").
  - The selected cell gets a text.primary border.
- **PR curves:** five patterns: solid, dashed, dotted, long dash (`LONG_DASH`), dash-dot (`DASH_DOT`; `Series::pattern`). The legend swatches draw the same patterns.
- **Histograms:** Shift+arrow steps ten bins.
- **Playground:** latency follows Detections, before the sample images.
- **Segmented controls:** with keyboard focus, they show their disabled options' reasons.
- **Slider tracks:** span the column, aligned with the fields above; the thumb travels inset by its radius.
- **Copy:**
  - The run chart is "Precision / Recall".
  - The inspector header is "x, y · w×h (px)".
  - The chip row is "Classes · box counts".
- **Empty Playground drop zone:** uses the panel surface.

## Round 28

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 9 | 8 | 8 | 9 | 9 | 9 | 9 | 9 | 9 | 8 | 8.8 |
| GPT-6.1 Sol | 8 | 9 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Opus is at 8.8, its best yet, with nine lines at 9. Two rendering defects decided its remaining lines:
- The PR legend rows needed 250 px in a 236 px column.
- A 1 px seam at the slider fill's lead.

Opus also flagged:
- The wrapped KPI grid.
- The ragged Overview KPI row.
- The palette's grey Stop.
- The `<1%` matrix cells' jump.
- The inspector's floating zoom pill.

Sol flagged:
- The compact New run help icons, outside the modal's keyboard scope.
- A dismissed New run losing its configuration.
- "Server" shown for the demo.
- The fade over visible form controls.
- The picker below the fold.
- Rail icons without focus labels.
- No pressed state on palette rows.

Round 29 changes:
- **New run:**
  - Esc, the scrim and Cancel keep the form as a draft; the next plain New run reopens it. Opening it for a dataset, from a run or on a new server starts fresh.
  - The compact help icons are `nr-` handles, so Tab reaches them.
- **Playground:**
  - With the built-in demo, Run on reads "Demo", with "Built-in demo · simulated inference".
  - The validation-image picker follows the controls, ahead of the results.
- **Collapsed rail:** a focused icon shows its name and shortcut.
- **Model page:**
  - The PR legend fits its 220 px row: 14 px swatch, 68 px name, 70 px P, 44 px AP50, three 8 px gaps. The summary row's spacer matches the swatch.
  - `<1%` confusion cells sit on the error ramp at their value.
- **Run page:** the wrapped KPI row is back on a 3-column grid. Throughput spans two columns, and its sparkline fills that tile, so no hole opens.
- **Overview:** Throughput and Time left get "avg N" and "N elapsed" lines.
- **Controls:**
  - The slider's lead fill overlaps the proportional fill by 1 px (no seam).
  - Palette rows have a pressed state, and Stop is in danger text.
- **Inspector:** the zoom controls sit in the top bar before Close, as in Playground.
- **Review brief:** the toolbar latency in Playground is a settled decision (both reviewers asked for it in round 25).

## Round 29

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 8 | 8 | 8.4 |
| GPT-6.1 Sol | 9 | 8 | 8 | 9 | 8 | 8 | 8 | 9 | 9 | 8 | 8 | 8 | 8.3 |

Opus fell from 8.8 to 8.4, mostly over one new defect: Playground's "Run on" caption was hard-clipped mid-word at the rail edge, which cost typography, copy and polish. Both reviewers also flagged:
- The PR curves' end-dots, which matched no number in the legend.
- The confusion matrix key ("errors 0 … (max)"), which named neither its measure nor its unit.

Opus also flagged the Overview live-run KPI row, whose four tiles left 40–50 % of the card empty, and the Overview chart squashed to about 63 px at minimum size.

Sol flagged:
- `<1%` matrix labels below AA on their cells.
- "Start folo serve automatically" hiding that it starts a mock server.
- The "Close mosaic (epochs)" label.
- "Add a dataset" in the empty Playground not launching the picker.
- Focus resetting to the page on every navigation.
- Fewer than three detection rows visible in Playground at minimum size.

The caption (it wraps now) and the matrix ink (every nonzero cell picks its ink from its fill) were fixed on #90 after review.

Round 30 changes:
- **Model page:**
  - The PR chart marks each class's recall and precision at the best-F1 threshold (one threshold shared by all classes: the "P @ best F1" column) instead of the curve ends. A footnote says so; isolating a class keeps only its dot.
  - The matrix key reads "Error share 0% … N% max" (or "Errors 0 … N max" when unnormalised).
- **Overview:**
  - The live run's KPI tiles share the row equally, with dividers, as on the run page.
  - The short-window chart is 108 px (was 76), with compact card padding below 120 px.
  - The loading skeleton's tiles match.
  - With no server, Open image is disabled with "Not connected", like New run.
- **Copy:**
  - Settings: "Start a demo server automatically", with help that names `folo serve --mock`.
  - New run: "Final epochs without mosaic".
- **Empty states:** "Add a dataset" (Overview, getting started, Playground) opens the folder picker on the Datasets page, as its label says, rather than only navigating there.

## Round 30

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 8 | 8 | 9 | 9 | 9 | 9 | 8 | 8 | 8.5 |
| GPT-6.1 Sol | 8 | 9 | 9 | 9 | 8 | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 8.4 |

Both reviewers moved up a notch. Sol now gives typography, colour, spacing and dataviz a 9.

Opus's main defects:
- The inspector image sat off-centre (a stale 72 px reserve for the old bottom zoom bar).
- Best-F1 dots hid each other.
- A "Fit" readout beside a full-screen-looking fit icon.
- Glyphs over the Playground rail's bottom border.
- The run P/R chart's empty lower half.
- Flat New run section heads.

Opus also asked to move the dfl loss hue away from box. That colour mapping lives in design/README, so it went to the orchestrator.

Sol's main defects were behavioural:
- A refused stop left the run "Stopping".
- Segmented-control arrows reached the page (panning a zoomed image).
- The log scrollback had no keyboard path.
- A fourth toast evicted actionable ones.
- Long focus tips ran across the window.
- At minimum size Playground showed barely one detection row.

The PR dots stay at each class's real P and R. The curves are the interpolated precision envelope, which runs at or above it, so a dot on its curve would contradict the "P @ best F1" column.

Round 31 changes:
- **Behaviour:**
  - A refused stop restores the run's status.
  - Navigation keys stay inside a segmented control.
  - The log is one Tab stop (arrows, Page Up/Down, Home/End).
  - Toasts with an action outlast passive ones; at most five wait, and the focused toast stays.
- **Focus tips:** they wrap at 320 px, and options turned off for the same reason share a line.
- **Playground below 700 px tall:**
  - The sample picker folds to one line.
  - Confidence and IoU each take one row, so four detection rows show at 960 × 620.
  - The rail's content clips inside its border.
- **Zoom:** the level ("Fit" / "2.4×") is the fit button in Playground and the inspector, and the fit icon is gone. The inspector image is centred.
- **Charts:**
  - PR dots within 8 px of each other are nudged apart, with a one-line footnote.
  - The run P/R chart's axis starts a tenth under its lowest value (0 while either value is low).
- **New run:**
  - Section heads are 14 px semibold.
  - Number boxes match the segmented controls' 28 px, so the helpers align.
  - The close-mosaic helper names its epochs ("Epochs 91–100 train on whole images").
  - The bottom fade shows only while more of the form lies below.
- **Copy:**
  - The GPU line says "VRAM".
  - Settings shows command names as code chips.
  - The inspector says "top-left x, y".
  - The narrow dataset action says "Copy check".
  - The getting-started commands have copy buttons.

## Round 31

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 9 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 9 | 8 | 8.6 |
| GPT-6.1 Sol | 8 | 9 | 8 | 9 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Opus reached its best yet, 8.6. Seven lines are at 9, including responsiveness ("minimum size now looks designed") and accessibility. Its remaining defects:
- An unlabelled raised floor on the run P/R axis. Two ticks gave 0.50 and 1.00 only, contradicting the code's own comment.
- A "Fit" button that did nothing while fitted.
- The toast footer inset from the card's padding.
- Progress drawn twice on the Overview live card.
- No start time in the Runs list at 1280.
- Mixed-case New run helpers.

Sol dipped to 8.3 on new findings rather than regressions:
- A stale PR hover beat keyboard focus.
- Switch and slider tracks around 1.5:1 against panels. That needs a design token, so it went to the orchestrator.
- The New run Schedule cut by the footer at 960 × 620.
- Truncated PR class names with no full name.
- No visible pin.
- An unnamed thumbnail slider.
- F6 missing from Settings.
- "GPU 0" without its adapter.
- Bars on every per-class metric.

The orchestrator agreed the loss colours with Studio: dfl moves to series 3 (teal) once design/README lands it.

Dogfood report 2 (#93) asked Desktop for:
- Studio's binary sizes.
- Local log times.
- `/` in log paths.
- A legend value that reads without the axis caption.

Its release-build crash (D1) is the engine fix agent's.

Round 32 changes:
- **PR legend and per-class table:**
  - In keyboard mode the focused row isolates its class, and a pointer move hands back.
  - A pinned class shows a pin in its name slot, and the hint says how to unpin.
  - Names cut short show whole on hover and focus.
  - Only AP50-95 keeps a comparison bar.
- **Run page:**
  - The P/R chart labels a raised floor (0.30 / 0.65 / 1.00), through a new `y_tick_values`.
  - The LR legend value carries its power ("6.58×10⁻³").
- **Log:**
  - Log times are local.
  - Paths read with `/`.
  - A late history answer only scrolls the run on screen.
- **Overview:**
  - Progress is drawn once (the header line); the Epoch tile says "42% complete".
  - The loading strip has the loaded tiles' dividers.
- **Runs:** Created shows from 1280 (Size · batch from 1440).
- **New run:**
  - At 960 × 620 the whole Schedule fits above the footer: the dataset summary sits on its label row, with tighter rhythm.
  - Device names the GPU.
  - Helpers are in sentence case.
- **Zoom:** "Fit" is plain text while fitted; fitting moves keyboard focus to zoom-in.
- **Toast:** one footer row, with the action on the left and the stack links ending under the close button.
- **Sizes:** binary units (KiB, MiB, GiB), like Studio. The New run estimate is in GiB.
- **Small items:**
  - The Playground picker toggle hovers like other rows.
  - The thumbnail slider names its value.
  - The broken-labels tooltip says a click copies.
  - Settings lists F6 / Shift F6.

## Round 32

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 8 | 9 | 9 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 8 | 8 | 8.4 |
| GPT-6.1 Sol | 9 | 8 | 9 | 9 | 8 | 9 | 8 | 9 | 8 | 8 | 9 | 8 | 8.5 |

Sol reached its best yet, 8.5, with dataviz, responsiveness and hierarchy at 9. Opus dipped to 8.4. Hierarchy fell to 8 because Playground's rail stacked its inputs over the result: at 1280 × 800 the detections sat below the fold. Opus also flagged:
- A 0.30 / 0.65 / 1.00 P/R axis.
- Opacity-only presses on chips, the switch and tiles.
- A grey "Share of all FP" ramp with no key.
- Icon-shaped loading rows.
- The empty band under Name in New run.
- A stretched sparkline at minimum size.

Sol flagged:
- An oversized chart group revealed by its bottom.
- No Retry after a failed prediction.
- An optimizer helper that described the schedule.
- Switches animating on first draw.
- A pointer-only connection line.
- "16" sample images where 8 show.
- "When the GPU is free" for any device.
- Proportional digits in age columns.
- Undiscoverable pan and fit keys.
- Ambiguous demo-server wording.

Opus measured the light "Running" badge at about 4.5:1. The contrast test covers that pair and passes, so the reading came from anti-aliased pixels. Three reversals went into the brief's settled decisions:
- "Copy check".
- The toast's stack links.
- Loss ticks counted from the top.

Round 33 changes:
- **Playground:**
  - The sample picker folds below 900 px, so all detections and the latency show at 1280 × 800.
  - A help icon gives zoom, pan and fit.
  - A failed prediction offers Retry (and Local CPU after an OOM).
- **New run:**
  - The Optimizer helper says what Auto picks for this run (the engine's rule: SGD past 10,000 steps of a 64-image nominal batch, else AdamW).
  - The learning-rate helper gives the schedule, and under Auto the rate it uses.
  - The dataset summary sits on its label row at every size.
  - The queue waits on the current run, not "the GPU".
- **Charts:**
  - The P/R floor snaps to 0.2 steps (0.40 / 0.70 / 1.00).
  - The FP share row is plain numbers.
  - The sparkline is capped at 160 px, or 240 in the wrapped grid.
- **Controls:**
  - Switches animate only after a change.
  - Pressed is a fill on chips, the switch and dataset cards, or an edge on image tiles and samples.
  - The connection line is a Tab stop; Enter opens Settings, and its status shows on focus.
- **Focus:** a target taller than the view reveals its top.
- **Copy:**
  - The demo-server help is clearer.
  - "8 of the 16" validation images.
  - Ages use tabular digits.
- **Loading rows:** shaped like table rows, with no icon.

## Round 33

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8 | 8.25 |
| GPT-6.1 Sol | 9 | 9 | 8 | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 9 | 8 | 8.4 |

Both reviewers put the same issue first: under Auto, New run showed an editable learning rate (0.010) beside a helper saying the engine uses another (0.00111). Sol also found that an unparsed rate blocked an Auto run, and the code review caught that Retry stayed live during its own rerun.

Opus flagged:
- Segmented labels shifting when the selection turns Medium.
- "16.0 GiB VRAM ▬ 94%" reading as memory use.
- The 0.20 / 0.60 / 1.00 P/R axis.
- "0.10×10⁻³".
- Header actions enabled differently while loading.
- Sparse per-class columns.
- Overlays without motion. The palette and dialogs already fade in; toasts and the inspector didn't.

Sol flagged:
- Stale IoU results shown as current.
- Selected chip counts at 4.43:1.
- A picker that skipped the first 17 samples.
- "Start training" when the run would queue.
- "objects" against "Detections".
- No keyboard hint on the matrix.
- An unconditional rail fade.

The sparkline cap is now a settled decision: one round called it a noise band, the next an empty one.

Round 34 changes:
- **New run:**
  - Under Auto the rate is a read-only "Auto · 0.00111" (3 significant figures), with "Set by Auto; pick SGD or AdamW to choose your own".
  - An unparsed rate no longer blocks Auto.
  - "Queue training" / "Queuing…" holds in every state while another run is training.
- **Playground:**
  - "Updating…" heads the detections while a rerun is out.
  - Changing where it runs shows "Detecting…", so an OOM error and Retry no longer stay live during the CPU fallback.
  - Samples are the first four with boxes.
  - The toolbar says "detections".
  - The rail fade shows only with more below, and the rail ends with 16 px of padding.
- **Charts:**
  - P/R ticks run every 0.2 from a floor of 0.4 or more.
  - The LR legend is normalised ("1.00×10⁻⁴").
  - Per-class metrics sit in fixed 120 px columns.
  - The confusion matrix shows "Arrows move · Home / End row ends · Esc clears" on keyboard focus.
- **Controls:**
  - Segment labels keep one weight.
  - Selected chip counts use secondary ink.
  - Toasts and the inspector fade in (160 ms; none under Reduce motion).
- **Consistency:**
  - The GPU meter reads "94% load".
  - Open image needs a model in every state, and says why when disabled.
  - The Overview KPI is "mAP50-95 · latest", as on the run page.
  - The palette says "N commands" with an empty query.
  - Queued runs read "Waiting to start", not "Waiting for GPU" (the queue waits on the current run).

## Round 34

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 8 | 8 | 8 | 9 | 9 | 9 | 9 | 9 | 8.6 |
| GPT-6.1 Sol | 9 | 8 | 9 | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 9 | 8 | 8.3 |

Opus matched its best, 8.6, with polish at 9 for the first time ("feels like an instrument"). Its remaining items:
- New run's section heads outweighing the title.
- An axis-power legend mismatch on the learning-rate card.
- The PR legend's P column repeating the class table.
- The Auto rate drawn as an input well.
- Checkbox-like slider end icons.
- Counts reading as part of segment labels.

Opus also caught that five round-34 shots predated the last commit. A partial reshoot had mixed builds; round 35 is one build again.

Sol's 8.3 was mostly mixed-input focus:
- A slider's own arrow handler swallowed keys before keyboard mode turned on.
- Clicks didn't move focus to the clicked control.

Sol also flagged:
- Unbounded palette titles.
- Optimizer placed after the rate it governs.
- R missing from the PR operating point.
- An unkeyed best dot on Overview.
- "Try in Playground" on the index.
- "Model" over architecture names.
- A play icon on Queue training.
- "pre / model / NMS".

#100's re-review listed three places where a failed model list read as loading, not found or empty.

Design #98 (Studio round 16) landed `line.control`, at 3:1 or better on every surface, and moved dfl to series 3 (teal).

Three more reversals were settled in the brief:
- The PR legend keeps P.
- The LR legend is written in the axis's power.
- Section heads stay 14 px and the dialog title grows to 20 px.

Round 35 changes:
- **Tokens:**
  - Switch tracks, slider rails, input and select borders and unselected chip outlines use `line.control` (a new test checks 3:1 on app, panel, raised, overlay and inset in both themes). Hover steps to text.tertiary.
  - dfl is teal (series 3), dotted.
- **Focus:**
  - Keyboard mode turns on in the capture phase for arrows, Home, End and Page keys, so a slider's first keyboard step shows its ring.
  - A click focuses the clicked control before its action.
- **New run:**
  - The title is 20 px semibold.
  - Optimizer leads the Schedule, with the rate beside it.
  - Auto's rate reads "Auto · 1.11×10⁻³" in an outline.
  - Queue training shows a clock.
- **Learning rate:** the legend is in the axis's power ("0.10×10⁻³"), one power per card.
- **Failed model list:**
  - The page subtitles say so.
  - The model page shows "Models didn't load" with Retry, not "not found".
  - Playground offers Retry and its picker says the models didn't load.
- **Copy:**
  - Palette titles truncate.
  - Overview keys its best dot ("best 0.656 @ 41").
  - The Models index says "Open Playground".
  - The Runs column reads "Arch".
  - Latency stages read Preprocess / Inference / NMS.
  - Legend tooltips give the operating point's P and R.
- **Controls:**
  - Segment counts the caller marks ("All 9") use tertiary tabular digits.
  - The thumbnail slider ends in image glyphs.

## Round 35

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 8 | 8 | 9 | 8 | 9 | 9 | 8 | 8 | 8.4 |
| GPT-6.1 Sol | 9 | 8 | 8 | 9 | 8 | 8 | 8 | 9 | 8 | 8 | 8 | 8 | 8.3 |

Both reviewers' top defect was the Playground latency legend: the full stage names clipped "NMS 0.2 ms" at the rail edge at 1280. The code review found the same at 960. The entries now wrap whole; this was fixed on #102 before merge.

Opus flagged:
- Wide word gaps wherever inline code appears (`ui::rich` laid out one flex item per word).
- A stop dialog whose commit weighed no more than "Keep training".
- Ctrl Enter shown beside Cancel.
- Three near-identical grey latency stages.
- Segmented controls still on the weaker edge beside `line.control` inputs.
- A confusion matrix whose small errors all read as alarm.
- An Overview chart without the run page's header rule.

Sol flagged:
- Selects and segments that don't take focus on click.
- Ctrl+O bypassing Open image's disabled rule.
- An off-switch hover that weakened the track.
- Faint selected segments.
- Unexplained architecture letters.
- "30 commands" for a list that holds datasets, runs and models.
- A seconds-precise reference estimate.

The engine's dogfood-2 fixes (#99) moved the GPU wait into inference. Desktop passes the engine's timings through, so its Local GPU split is right without a change.

Round 36 changes:
- **Text:** `ui::rich` is one styled text run (Geist, with code spans in Geist Mono on a chip fill), so word spacing is body spacing.
- **Stop dialog:** the commit is a new `DangerCommit` button, the Danger tint with a danger edge. A solid red fill fails AA with white text in both themes, and there's no on-danger token.
- **New run:**
  - Ctrl Enter follows Queue training.
  - The Architecture helper names the size ("folo-s, small · …").
  - Estimates are in whole minutes ("≈ 10 min").
- **Latency:**
  - Inference is in secondary ink; the other two stages are tertiary at two strengths, with 8 px swatches.
  - The third stage is "Postprocess", as the engine and Studio name it.
- **Controls:**
  - Segmented containers and selected segments are on `line.control`.
  - Off-switch hover and press strengthen the track.
  - Selects and segments take focus on click.
- **Ctrl+O and the palette** share Open image's rule (`open_image_blocked`) and say why when blocked.
- **Confusion matrix:** the error ramp is linear from a 0.1 floor, replacing the square root that lifted 1–4 % toward full red.
- **Overview:** the chart section opens on a full-width rule.
- **Palette:** the unfiltered count says "items".

## Round 36

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 9 | 9 | 8 | 9 | 9 | 9 | 9 | 9 | 9 | 7 | 8 | 8.7 |
| GPT-6.1 Sol | 9 | 9 | 9 | 8 | 9 | 9 | 8 | 9 | 9 | 9 | 7 | 8 | 8.6 |

Both reviewers reached their best yet, with nine or ten lines at 9. Both held responsiveness at 7 and polish at 8 for one regression: on the first-run Overview, the third getting-started step overflowed the panel and the window. Making `ui::rich` one text run let each step size to its unwrapped text. The step columns now shrink (`min_w_0`, zero flex basis) and wrap; this was fixed on #105 before merge. The empty Overview is now captured at minimum and wide sizes too.

Opus also flagged:
- Latency's last lines below the rail's fold at 1280 × 800.
- Image size's help dropped in the short New run.
- Two near-identical grey latency swatches.
- Dead space in the inspector's side column.

Sol flagged:
- The minimum Overview showing one Recent runs row.
- Model rankings without their evaluation dataset.
- No in-app list of broken labels.
- "IoU" read as evaluation IoU.
- A failure banner whose message column could overflow.
- No copy action in the error state.

The code reviews caught:
- Open image's rule treating a failed runs list as "not connected", fixed to depend on models only.
- Drag-and-drop skipping that rule.

#101 (AMP) landed `amp` in TrainConfig. The server can't yet say whether it has tensor cores; a capability field is coming to `/api/system`.

Round 37 changes:
- **Mixed precision:**
  - New run has a switch at the end of Schedule, stating its need for tensor cores (`folo serve --tensor-cores`). It is inert, and never sent, on the CPU.
  - A run that fails with `not_supported` gets its own banner hint and isn't offered Resume. New run from this config turns mixed precision off with a note.
- **Playground at 1280 × 800:**
  - The Detections head is 32 px and the column row 20 px.
  - Device and fps ride on the Latency row, so all detections, the latency bar and its first legend line fit.
  - Postprocess is an outline in the bar and its key.
  - The IoU slider reads "NMS IoU".
- **New run:**
  - The short layout keeps Image size's help as an (i).
  - Rows with an (i) stay as tall as plain label rows, so neighbours line up.
- **Run page:** the failure banner's text column shrinks, so long backend messages wrap.
- **Capture set:** the empty Overview at 960 × 620 and 1680 × 1050.

## Round 37

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | avg |
|------|---|---|---|---|---|---|---|---|---|----|----|----|-----|
| Opus 5.5 | 9 | 8 | 9 | 8 | 9 | 9 | 9 | 9 | 9 | 9 | 8 | 8 | 8.7 |
| GPT-6.1 Sol | 9 | 8 | 8 | 8 | 9 | 8 | 8 | 9 | 9 | 8 | 8 | 8 | 8.3 |

The new empty-Overview captures (960 and 1680) showed two first-run defects:
- The inline `folo dataset synth` chip split across lines into two boxes. GPUI's line wrapper breaks at any non-word character, no-break spaces included, so no character keeps a run whole.
- The step cards ended at different heights.

Opus also flagged:
- The Latency key wrapping 2 + 1, so its last line sits under the fade at 1280 × 800.
- The empty page stretching across 1660 px.
- Queued and Stopped sharing a grey dot.
- "Loading…" against the sidebar's "Connecting…".

Sol flagged:
- Tertiary text on active fills (palette detail 3.87:1, active detection sizes 4.41:1).
- No dataset named on the model's evaluation.
- Terminal steps that didn't say how to connect.
- The CLI training row reading as part of the sequence.
- "YOLO folder" leaving out COCO.
- Awkward PR and stop-dialog copy.

The code review caught two blocking AMP bugs, fixed on #106 before merge:
- `carry_over` copied `amp` over the form.
- Live status events dropped `error_code`.

Round 38 changes:
- **Getting started:**
  - Step 1's command is a chip on its own line.
  - The step cards stretch to one height (each card `flex_1().min_w_0()` inside a flex wrapper).
  - The empty page is capped at 1040 px.
  - Code spans in `rich` use no-break spaces, though GPUI may still wrap at them.
- **Terminal panel:**
  - "start it, then Settings → folo serve → Connect".
  - An "Independent CLI training" caption, saying that run won't appear in Desktop.
  - Step 1 says "YOLO or COCO".
- **Contrast:** the selected palette detail and the active detection size use secondary ink.
- **Status:** Queued is a hollow ring.
- **Model page:** names the evaluated dataset ("Validation · shapes-hard · 800 images").
- **Overview:** says "Connecting…" until the server answers.
- **Copy:**
  - The PR legend says "Hover a legend or table row".
  - The stop dialog says "Progress after epoch N is lost".
- **Runs:** an active status event clears a run's old error and code.
