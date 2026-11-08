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
