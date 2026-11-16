# Stress round lane reports

One report per lane of the stress round of FOLO Studio and FOLO Desktop (2026-11-14 and 15). Each lane used one area
as a hard user would, fixed what it found with regression tests and merged one pull request. The round's summary is
[DOGFOOD-4](../DOGFOOD-4.md). Findings are counted from each lane's `FINDINGS.md` (blocker / annoying / polish),
with the status the lane gave them; "deferred" includes findings left open and findings another lane fixed
(s5-label-5, by s1-datasets), duplicates of another lane's finding are in neither column.

| lane | what it covered | PR | blocker / annoying / polish | fixed | deferred |
|---|---|---|---|---:|---:|
| [s1-datasets](s1-datasets.md) | Studio: datasets list, dataset grid, removal, outages | #267 | 0 / 4 / 4 | 5 | 3 |
| [s2-runs](s2-runs.md) | Studio: Runs list, New run, row menus | #275 | 0 / 5 / 4 | 6 | 2 (1 duplicate) |
| [s3-live](s3-live.md) | Studio: the Run page live, Run compare | #269 | 0 / 3 / 3 | 3 | 3 |
| [s4-models](s4-models.md) | Studio: models, model page, Playground | #266 | 0 / 2 / 2 | 3 | 1 |
| [s5-label](s5-label.md) | Studio: labeling jobs and benchmarks | #271 | 0 / 2 / 3 | 3 | 2 |
| [s6-shell](s6-shell.md) | Studio: shell, prefs, router, answers of the wrong shape | #268 | 2 / 2 / 3 | 4 | 3 |
| [d1-datasets](d1-datasets.md) | Desktop: datasets, image cache and inspector | #277 | 1 / 3 / 2 | 5 | 1 |
| [d2-runs](d2-runs.md) | Desktop: runs, New run, run charts | #282 | 0 / 4 / 4 | 7 | 1 |
| [d3-models](d3-models.md) | Desktop: models, model page, Playground | #272 | 0 / 3 / 4 | 3 | 4 |
| [d4-bench-settings](d4-bench-settings.md) | Desktop: benchmarks, settings, title bar | #276 | 0 / 3 / 1 | 4 | 0 |
| [d5-window-input](d5-window-input.md) | Desktop: window, keyboard, text fields, arguments, state file | #280 | 0 / 3 / 4 | 7 | 0 |
| [d6-connection](d6-connection.md) | Desktop: finding and starting `folo serve`, transport, event streams | #273 | 3 / 1 / 2 | 4 | 2 |
| [x1-parity](x1-parity.md) | Studio against Desktop on the same data | #270 | 1 / 4 / 2 | 4 | 3 |
| [x2-newcomer](x2-newcomer.md) | both apps: a newcomer's first steps | #274 | 0 / 2 / 5 | 3 | 3 (1 duplicate) |
| [x3-server](x3-server.md) | the server under both apps: registration, registries, logs | #278 | 2 / 3 / 4 | 7 | 1 (1 duplicate) |
| [x4-soak](x4-soak.md) | both apps: long sessions, hundreds of runs, big datasets | #281 | 0 / 3 / 0 | 3 | 0 |
| **total** | | | **9 / 47 / 47** | **71** | **29** (3 duplicates) |

d1-datasets-5 and x4-soak-3 were fixed in part and count as fixed; d3-models-2 counts as deferred. s1-datasets-7 is
in the lane's `FINDINGS.md`, not in its report's table. s5-label-2 is polish in the lane's `FINDINGS.md`, annoying in
its report's table; the count follows `FINDINGS.md`.
