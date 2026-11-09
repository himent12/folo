# FOLO Desktop — design review brief

You are a demanding senior product designer reviewing a native desktop app. **Do not edit any files.** Look and report.

## What it is
FOLO Desktop is the native app of FOLO ("FOLO Only Looks Once"), a from-scratch Rust vision framework (a YOLO-style
object detector with training). Built in Rust with GPUI (Zed's UI framework) on Windows. Screens: Overview, Datasets
(index + detail with an image grid and statistics), Runs (list + live training dashboard), Models (list + evaluation
detail), Playground (run a model on an image), Settings. Global: command palette, New run dialog, image inspector,
toasts, dark and light themes. All data is deterministic mock data; judge the UI, not the numbers.

The target design language (shared with the web app, from `design/README.md`): precise, fast, instrument-grade —
Linear × Vercel × a camera viewfinder. Dense but calm; content (boxes, charts, images) is the star; chrome stays
quiet; colour is spent on meaning (signal-lime accent for focus/live/primary, status colours for status, a 16-colour
class palette for detection classes). Geist + Geist Mono, 13 px base. Detection boxes: 2 px class-colour stroke with a
1 px dark halo, square corners, label chips on the top-left corner.

## Screenshots
The folder in your task holds PNGs captured from the running app at 150 % display scale (1920 × 1200 px = a
1280 × 800 logical window). `*-min.png` are the minimum window size (960 × 620 logical); `*-wide.png` 1680 × 1050.
Names: screen first, then modifiers (`-light` light theme, `-loading`, `-empty`, `-error` (server unreachable),
`-hover` (a chart hover readout), `-filter`). `palette`, `new-run`, `stop-confirm`, `inspector`, `toast` are overlays;
`kb-focus` shows keyboard focus on the New run button. The other `-focus` shots (`run-live-charts-focus`,
`run-done-sample-focus`, `model-table-focus`, `playground-canvas-focus`) show keyboard focus on a control with its own keys and
its key hint; `run-done-bottom` is the run page scrolled to its end (log and configuration). **Open every PNG**, at full resolution where details matter
(zoom in before calling something clipped).

Also read the source (path in your task) for states screenshots can't show: hover, pressed, focus rings, keyboard
paths (`src/controls.rs`, `src/app.rs`, `src/overlays.rs`), motion, resizing logic.

Known platform limits (don't score these against the app): GPUI has no letter-spacing; the window's own caption buttons
are drawn by the app.

Keyboard paths that live outside the screen files (check before reporting one missing): the linked run charts, the
PR chart and the dataset histograms are one Tab stop each through `chart_group` in `src/controls.rs` (← / → step,
Shift = ×10, Home / End, Esc clears, same cursor and legend values as the mouse); every row, chip and button goes
through `bind` / `bind_ring` (Enter / Space, focus ring).

Settled decisions (reviewers have argued both ways; the current choice stays unless it causes a defect, so spend
your issues elsewhere): Settings is capped at 880 px, left-aligned; the run KPI strip keeps fixed gaps rather than
spreading tiles; "close mosaic" is named on the top-row charts only; the learning-rate power of ten sits in the card
caption; the selected detection box and its chip take accent.solid (design/README's selection rule), not a class colour
with an extra ring, and the other boxes' outlines drop to 30 % (the same rule; their chips stay opaque and readable); the chart
cursor's chip sits on the x axis and the tick labels under it give way (it names the cursor's epoch where the axis
would); Playground's image toolbar shows the latency beside the detection count (asked for by both reviewers in
round 25), even though the rail also has the breakdown; the narrow Broken labels tile says "Copy check" (one round
asked for it, the next for "Copy command"; its tooltip names the command); the newest toast carries the stack's
"Show N earlier · Dismiss all" (the only way to the folded toasts and to clear them); log-scale loss ticks thinned at
small sizes are counted from the top power, so the top is labelled; the Throughput sparkline is a glance, capped
(160 px in the strip, 240 in the wrapped grid at minimum size): one round called it a noise band at full width, the
next an empty band when capped; the PR legend keeps its "P @ best F1" column beside AP50 (one round asked to add R
there, the next to drop P as a repeat of the class table); the learning-rate legend value is written in the axis's
power ("0.10×10⁻³"), so a card has one power and the value still reads alone (the dogfood report asked for that);
New run's section heads stay 14 px semibold and the dialog title is 20 px semibold above them (one round asked for
larger heads, the next said they outweighed the title). The run page's "Precision / Recall" card and the model page's
"Precision–recall" card are different charts: two metrics over epochs, and one curve of precision against recall.
Their names differ on purpose. The first-run Overview is capped at 1040 px and left-aligned like Settings. One round said the uncapped
cards stretched three short steps across 1,600 px, the next asked them to fill a wide window. The Playground rail's sliders
are inline rows ("label · slider · value") for every model, so the rail never reflows when the model changes (asked for
in round 44; round 47 asked for stacked rows again). New run's End-to-end and Mixed precision rows keep to the left column, their
switch beside the label (round 42 asked for that; round 48 asked for them to span the dialog again). A dialog's destructive commit
(`Btn::DangerCommit`) is the danger tint with a danger edge, not a solid red fill: white on the danger red fails AA in
both themes and the tokens have no on-danger ink (round 52 asked for a solid fill). Run log lines wrap rather than
truncate: every line stays readable without a tooltip. Pages cap at Desktop's own `layout.desktop_content_max` (design/tokens.json, 1440) and
centre; Studio's `size.content_max` no longer applies to Desktop.

## Scoring
Use `REVIEW_RUBRIC.md` (path in your task): 12 lines, 1–10, where **9 means "would ship as-is at Linear / Raycast / Zed
quality"**. Be calibrated and strict: no 9 unless you would genuinely ship that line unchanged. Return the exact
output format at the end of the rubric: a score and one-line reason per line, then the **top 10 concrete issues
ranked by impact** (screenshot + region, problem, specific fix with px / token / copy), then bugs and rendering
artifacts.

**For every line you score below 9, add "To make it a 9:"** — the specific, verifiable change(s) that would earn the 9
(what to change, where, by how much), not a general direction like "tighten spacing". If nothing short of a redesign
of a screen would do it, say which screen and what the new structure should be. Keep it under ~1500 words.
