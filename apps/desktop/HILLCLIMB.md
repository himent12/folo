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
