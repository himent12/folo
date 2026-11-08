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
