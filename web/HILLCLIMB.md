# FOLO Studio design hill-climb

Each round: `npm run shoot -- r<N>` (every screen and state, dark + light, 1440 / 1024 / 390), then two independent
design reviewers (Claude Opus 5.5 and GPT-6.1 Sol, fresh each round, no previous scores) score
[`REVIEW_RUBRIC.md`](REVIEW_RUBRIC.md) from the screenshots and the source. Stop condition: **both reviewers ≥ 9 on
every line in the same round.**

Lines: 1 hierarchy & layout · 2 typography · 3 colour & theming · 4 spacing rhythm · 5 component craft & states ·
6 data visualisation · 7 information design & copy · 8 consistency · 9 interaction & motion · 10 accessibility ·
11 responsiveness · 12 polish & delight.

## Rounds

| Round | Reviewer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | GPT-6.1 Sol | 7 | 8 | 7 | 7 | 7 | 5 | 6 | 7 | 7 | 6 | 5 | 7 | 5 |
| r1 | Claude Opus 5.5 | 7 | 7 | 6 | 7 | 7 | 7 | 7 | 7 | 8 | 5 | 5 | 7 | 5 |
| r2 | GPT-6.1 Sol | 7 | 8 | 7 | 8 | 8 | 7 | 6 | 8 | 7 | 6 | 5 | 7 | 5 |
| r2 | Claude Opus 5.5 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 8 | 6 | 5 | 7 | 5 |
| r3 | GPT-6.1 Sol | 7 | 8 | 8 | 8 | 8 | 7 | 7 | 8 | 8 | 7 | 7 | 8 | 7 |
| r3 | Claude Opus 5.5 | 7 | 7 | 6 | 7 | 7 | 6 | 8 | 7 | 8 | 7 | 7 | 7 | 6 |
| r4 | GPT-6.1 Sol | 7 | 8 | 8 | 8 | 8 | 7 | 7 | 8 | 8 | 7 | 7 | 7 | 7 |
| r4 | Claude Opus 5.5 | 7 | 6 | 6 | 6.5 | 6.5 | 5.5 | 7 | 6 | 7.5 | 7 | 6.5 | 6.5 | 5.5 |
| r5 | GPT-6.1 Sol | 7.5 | 8 | 8 | 8 | 8 | 7.5 | 7.5 | 8 | 8 | 7.5 | 7 | 7.5 | 7 |
| r5 | Claude Opus 5.5 (71 of 132 shots seen) | 7.5 | 7.5 | 8 | 7.5 | 8 | 8 | 8 | 7.5 | 8 | 7.5 | 7 | 7.5 | 7 |
| r6 | GPT-6.1 Sol | 7.5 | 8 | 8 | 8 | 7.5 | 7 | 7.5 | 7.5 | 8 | 7.5 | 7 | 7.5 | 7 |
| r7 | GPT-6.1 Sol | 7.5 | 8 | 8 | 8 | 7.5 | 7.5 | 8 | 7.5 | 8 | 8 | 7 | 8 | 7 |
| r7 | Claude Opus 5.5 (78 of 134 shots seen) | 8 | 7.5 | 8 | 7.5 | 8 | 8 | 8.5 | 7.5 | 8.5 | 7.5 | 8 | 8 | 7.5 |
| r8 | GPT-6.1 Sol | 8 | 8 | 8 | 8.5 | 8 | 7 | 7.5 | 8 | 8.5 | 8 | 8 | 8 | 7 |
| r9 | GPT-6.1 Sol | 8 | 8 | 8 | 8 | 8 | 7.5 | 7.5 | 8 | 8.5 | 8 | 7 | 7.5 | 7 |
| r9 | Claude Opus 5.5 (81 of 151 shots seen) | 8 | 8 | 7.5 | 8 | 8.5 | 8 | 8.5 | 8 | 8 | 7.5 | 8 | 8 | 7.5 |
| r10 | GPT-6.1 Sol | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8 | 8.5 | 8.5 | 9 | 8.5 | 8 | 8.5 | 8 |
| r10 | Claude Opus 5.5 (82 of 172 shots seen) | 8.5 | 8 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8.5 | 8 | 8.5 | 8 |
| r12 | GPT-6.1 Sol | 8.5 | 8.5 | 9 | 9 | 8.5 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8 |
| r12 | Claude Opus 5.5 (all 184 via contact sheets) | 8 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8 |
| r15 | GPT-6.1 Sol (57 sheets, 26 full-size) | 8 | 8.5 | 9 | 8.5 | 8.5 | 8 | 7.5 | 8.5 | 8 | 8.5 | 8 | 8.5 | 7.5 |
| r15 | Claude Opus 5.5 (57 sheets, 6 full-size) | 8 | 7.5 | 8 | 8 | 8 | 7.5 | 8 | 7.5 | 8 | 8 | 8 | 8 | 7.5 |
| r16 | GPT-6.1 Sol (78 sheets, 32 full-size) | 8 | 8.5 | 8.5 | 8 | 8 | 8 | 8.5 | 8.5 | 9 | 8.5 | 7.5 | 8 | 7.5 |
| r16 | Claude Opus 5.5 (78 sheets, 7 full-size) | 8 | 8 | 8.5 | 8 | 7.5 | 8.5 | 8.5 | 7.5 | 8.5 | 8 | 7.5 | 8 | 7.5 |
| r17 | GPT-6.1 Sol (90 sheets, 34 full-size) | 8 | 9 | 9 | 9 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8 |
| r17 | Claude Opus 5.5 (90 sheets, ~18 full-size) | 8 | 8 | 8.5 | 8 | 7.5 | 8 | 8.5 | 8 | 8.5 | 8 | 7.5 | 8 | 7.5 |

## Per-line minimum (lower of the two reviewers)

Regressions show up here first: any line that drops from the previous round is fixed or reverted before new work.
Rounds with one reviewer are marked; their minimum is that reviewer's row.

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 7 | 7 | 6 | 7 | 7 | 5 | 6 | 7 | 7 | 5 | 5 | 7 | 5 |
| r2 | 7 | 7 | 7 | 7 | 7 | 7 | 6 | 7 | 7 | 6 | 5 | 7 | 5 |
| r3 | 7 | 7 | 6 | 7 | 7 | 6 | 7 | 7 | 8 | 7 | 7 | 7 | 6 |
| r4 | 7 | 6 ↓ | 6 | 6.5 ↓ | 6.5 ↓ | 5.5 ↓ | 7 | 6 ↓ | 7.5 ↓ | 7 | 6.5 ↓ | 6.5 ↓ | 5.5 |
| r5 | 7.5 ↑ | 7.5 ↑ | 8 ↑ | 7.5 ↑ | 8 ↑ | 7.5 ↑ | 7.5 ↑ | 7.5 ↑ | 8 ↑ | 7.5 ↑ | 7 ↑ | 7.5 ↑ | 7 |
| r6 (Sol only) | 7.5 | 8 | 8 | 8 | 7.5 | 7 | 7.5 | 7.5 | 8 | 7.5 | 7 | 7.5 | 7 |
| r7 | 7.5 | 7.5 | 8 | 7.5 | 7.5 | 7.5 | 8 | 7.5 | 8 | 7.5 | 7 | 8 | 7 |
| r8 (Sol only) | 8 | 8 | 8 | 8.5 | 8 | 7 | 7.5 | 8 | 8.5 | 8 | 8 | 8 | 7 |
| r9 | 8 | 8 | 7.5 | 8 | 8 | 7.5 | 7.5 | 8 | 8 | 7.5 | 7 | 7.5 | 7 |
| r10 | 8.5 ↑ | 8 | 8.5 ↑ | 8 | 8.5 ↑ | 8 ↑ | 8.5 ↑ | 8 | 8.5 | 8.5 ↑ | 8 ↑ | 8.5 ↑ | 8 |
| r12 | 8 ↓ | 8.5 ↑ | 8.5 | 8.5 ↑ | 8.5 | 8.5 ↑ | 8 ↓ | 8.5 ↑ | 8.5 | 8.5 | 8 | 8.5 | 8 |
| r15 | 8 | 7.5 ↓ | 8 ↓ | 8 ↓ | 8 ↓ | 7.5 ↓ | 7.5 ↓ | 7.5 ↓ | 8 ↓ | 8 ↓ | 8 | 8 ↓ | 7.5 |
| r16 | 8 | 8 ↑ | 8.5 ↑ | 8 | 7.5 ↓ | 8 ↑ | 8.5 ↑ | 7.5 | 8.5 ↑ | 8 | 7.5 ↓ | 8 | 7.5 |
| r17 | 8 | 8 | 8.5 | 8 | 7.5 | 8 | 8.5 | 8 ↑ | 8.5 | 8 | 7.5 | 8 | 7.5 |

r4 lowest lines: 6 data visualisation (5.5), 2 typography, 3 colour, 8 consistency (6). Diagnosis: 2, 3 and 8 are
structural (no single rule for mono vs sans, class colours that changed between legend and box, two accents in light),
fixed in round 4 by system rules in `design/README.md` rather than per-screen patches; 6 is partly structural (run
comparison and dense COCO evaluation screens need a different layout, not tweaks).

r5 (after the design-director pass, `DESIGN_DIRECTION.md`): every line up, no line below 7 for the first time. Lowest:
11 responsiveness (7 from both). Opus could open only 71 of 132 shots (image request limit), so its r5 row is partial;
from r6 reviewers pace image loading and retry.

r10: no line below 8 from either reviewer for the first time (min 8, up from 7). Both reviewers converge on what is
left: phone-width loose ends (truncated names, the long phone model page, row heights), drift between screens (two
run-list patterns, four device names, desktop banners vs phone callouts) and type sizes off the scale. Rounds 11–12
take these as system rules where possible (one end-label placement, one ETA formatter, status marks by shape).

r11 was not reviewed separately (its fixes shipped with r12's screenshots). r12: the first round where Opus saw every
screenshot (four-up contact sheets, 46 images). Lines 2, 4, 6, 8 rose to 8.5; 1 and 7 dropped half a point on
issues the r12 shots expose rather than regressions in code: the 1024 px tier collapsing to phone patterns (Model
rail, New run summary, run KPI select) and mobile model choice without guidance. Both name the same weakest areas:
the laptop tier, mobile Compare and the mobile confusion matrix.

r13 and r14 were reviewed for code (Sol), not scored. r15 is the first round scored against the "what exactly would
make this a 9" brief: every reviewer lists, per line below 9, the concrete changes that would make it a 9. The drop
from r12 is partly calibration (227 shots now, including r14's new error states: device not found, out of memory,
the Playground error card) and partly real: those new states are where much of the r15 criticism lands (a technical
red paragraph at Device, an error card over the image). Lowest (7.5): 2 typography (10–11 px supporting labels,
mixed decimals), 6 data visualisation (a 0–1 y domain everywhere, a 170 px Compare chart, unlabelled histogram
density), 7 information design (the issue browser living inside the "broken label files" finding) and 8 consistency
(nine hand-written tables). The structural moves for r15 → r16: one shared table component, fitted chart domains,
the dataset check as its own section, route focus, and the compact phone preamble on Model and Dataset pages.

r16 (#98: fitted chart domains, one table system, the dataset check as its own section, route focus, compact phone
headers, `line.control`, loss hues): seven lines up or level, Sol's 9 on interaction the first 9 since r12. The two
that dropped (5 states, 11 responsiveness) drop on one new bug both reviewers found first: at 1024 the Models table
hid body cells but not the `SortHeader` cells (component-scoped hide classes), so every value sat under the wrong
heading. The rest of what holds the lines at 7.5–8: undesigned states (an invalid field shows only the focus ring,
images drawn as box outlines before they load, the Playground failure below the fold on phones with a stale
"Detections appear here"), state frames held at desktop height on phones, Compare (faded or hidden series, short runs
crushed on a 0–150 epoch axis), the confusion matrix ramp (invisible low cells in dark), the loss chart's two control
groups, and header actions that move between pages at 1024.

r17 (#103: shared column hiding, invalid fields, states in frame, Compare z-order and progress axis, confusion ramp,
one loss control, thumbnails that load): Sol's first three 9s (typography, colour, spacing), Sol's minimum up to 8.
Opus holds the floor at 7.5 on 5 (states) and 11 (responsiveness), on bugs rather than taste: the inspector never draws
the selected box (`selected` not passed through DatasetImage), Runs clips at 820 once a run is selected (the checkbox
column appears), and the Playground out-of-memory state loses focus and scroll at 1024. From r18 the shoot reports
horizontal overflow per scene, and every round gets a bug pass over the sheets before the reviewers see them.
