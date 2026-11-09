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
| r18 | GPT-6.1 Sol (109 sheets, 30 full-size) | 8.5 | 8.5 | 9 | 9 | 9 | 8 | 8 | 9 | 8.5 | 8.5 | 8.5 | 8.5 | 8 |
| r18 | Claude Opus 5.5 (109 sheets, 16 full-size) | 8 | 8 | 8.5 | 8.5 | 8.5 | 7.5 | 8 | 8 | 8.5 | 8.5 | 8 | 8 | 7.5 |
| r19 | GPT-6.1 Sol (120 sheets, 34 full-size) | 7.5 | 8.5 | 8.5 | 8.5 | 7.5 | 9 | 8 | 8 | 8 | 8 | 6.5 | 7.5 | 6.5 |
| r19 | Claude Opus 5.5 (120 sheets, 23 full-size) | 7.5 | 7.5 | 8 | 7.5 | 7 | 8 | 7.5 | 8 | 8.5 | 8 | 6.5 | 7.5 | 6.5 |
| r20 | GPT-6.1 Sol (128 sheets, 36 full-size) | 8.5 | 8.5 | 9 | 9 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8 |
| r20 | Claude Opus 5.5 (128 sheets, 11 full-size) | 8 | 8 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8.5 | 8 |
| r21 | GPT-6.1 Sol (134 sheets, 30 full-size) | 8 | 9 | 9 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 9 | 8 | 8.5 | 8 |
| r21 | Claude Opus 5.5 (134 sheets, ~106 full-size) | 7.5 | 8 | 8 | 7.5 | 8 | 7.5 | 7.5 | 7.5 | 8 | 8.5 | 8 | 8 | 7.5 |
| r22 | GPT-6.1 Sol (152 sheets, 42 full-size) | 8 | 8 | 9 | 9 | 8 | 8 | 8 | 8.5 | 8.5 | 8 | 8 | 8 | 8 |
| r22 | Claude Opus 5.5 (152 sheets, 8 full-size) | 8 | 8 | 8.5 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8.5 | 8 | 8 |
| r23 | GPT-6.1 Sol (169 sheets, 50 full-size) | 8 | 8.5 | 8.5 | 9 | 9 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8 | 8.5 | 8 |
| r23 | Claude Opus 5.5 (169 sheets, 32 full-size) | 8 | 8 | 8 | 8.5 | 8.5 | 8.5 | 8 | 8 | 8.5 | 8.5 | 8.5 | 8 | 8 |
| r24 | GPT-6.1 Sol (175 sheets, 47 full-size) | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8 |
| r24 | Claude Opus 5.5 (175 sheets, 36 full-size) | 8 | 8 | 8 | 8 | 8 | 8.5 | 8.5 | 8 | 8.5 | 8.5 | 8 | 8 | 8 |
| r25 | GPT-6.1 Sol (192 sheets, 50 full-size) | 8 | 8.5 | 9 | 8.5 | 8.5 | 8.5 | 8 | 9 | 9 | 8.5 | 8 | 8.5 | 8 |
| r25 | Claude Opus 5.5 (192 sheets, 33 full-size) | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 9 | 9 | 8 | 8.5 | 8 |
| r26 | GPT-6.1 Sol (195 sheets, 50 full-size) | 8.5 | 9 | 9 | 9 | 9 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 |
| r26 | Claude Opus 5.5 (195 sheets, 33 full-size) | 8 | 8.5 | 9 | 8.5 | 8.5 | 8.5 | 8.5 | 8.5 | 9 | 9 | 8 | 8.5 | 8 |
| r27 | GPT-6.1 Sol (204 sheets, 52 full-size) | 8 | 8.5 | 9 | 9 | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8 | 8 | 8 |
| r27 | Claude Opus 5.5 (204 sheets, 30 full-size) | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8 | 8 | 8.5 | 9 | 9 | 8 | 8.5 | 8 |

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
| r18 | 8 | 8 | 8.5 | 8.5 ↑ | 8.5 ↑ | 7.5 ↓ | 8 ↓ | 8 | 8.5 | 8.5 ↑ | 8 ↑ | 8 | 7.5 |
| r19 | 7.5 ↓ | 7.5 ↓ | 8 ↓ | 7.5 ↓ | 7 ↓ | 8 ↑ | 7.5 ↓ | 8 | 8 ↓ | 8 ↓ | 6.5 ↓ | 7.5 ↓ | 6.5 |
| r20 | 8 ↑ | 8 ↑ | 8.5 ↑ | 8 ↑ | 8.5 ↑ | 8 | 8.5 ↑ | 8.5 ↑ | 8.5 ↑ | 8.5 ↑ | 8.5 ↑ | 8.5 ↑ | 8 |
| r21 | 7.5 ↓ | 8 | 8 ↓ | 7.5 ↓ | 8 ↓ | 7.5 ↓ | 7.5 ↓ | 7.5 ↓ | 8 ↓ | 8.5 | 8 ↓ | 8 ↓ | 7.5 |
| r22 | 8 ↑ | 8 | 8.5 ↑ | 8.5 ↑ | 8 | 8 ↑ | 8 ↑ | 8.5 ↑ | 8.5 ↑ | 8 ↓ | 8 | 8 | 8 |
| r23 | 8 | 8 | 8 ↓ | 8.5 | 8.5 ↑ | 8.5 ↑ | 8 | 8 ↓ | 8.5 | 8.5 ↑ | 8 | 8 | 8 |
| r24 | 8 | 8 | 8 | 8 ↓ | 8 ↓ | 8.5 | 8 | 8 | 8.5 | 8.5 | 8 | 8 | 8 |
| r25 | 8 | 8.5 ↑ | 8.5 ↑ | 8.5 ↑ | 8.5 ↑ | 8.5 | 8 | 8.5 ↑ | 9 ↑ | 8.5 | 8 | 8.5 ↑ | 8 |
| r26 | 8 | 8.5 | 9 ↑ | 8.5 | 8.5 | 8.5 | 8.5 ↑ | 8.5 | 8.5 ↓ | 8.5 | 8 | 8.5 | 8 |
| r27 | 8 | 8.5 | 9 | 8.5 | 8.5 | 8 ↓ | 8 ↓ | 8.5 | 8.5 | 8.5 | 8 | 8 ↓ | 8 |

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

r18 (#107: inspector selection, Runs at 820, the overflow check in the shoot (clean), engine short-run warnings, ONNX
switches, one-row phone headers, the shortcuts switch): Sol's best round (five 9s: colour, spacing, states,
consistency; nothing under 8), and the r17 bugs are gone (5 and 11 up to 8.5 / 8). The floor moved to 6 (data
visualisation, 7.5): the 3–5 tick cap made the PR curves' quarter steps read "0.3 / 0.8" because PrCurves forced one
decimal (fixed before merge, 97ff017), plus the box-size histogram's uneven bins and colliding end labels. 7 dropped
to 8 on copy: four names for "from scratch", batches vs optimizer steps, the engine's warning as a technical
paragraph.

r19 (#113: Mixed precision, the Settings System panel, one-line run banners, categorical histogram): a regression
round. The full-height sidebar wrapped the shell's column in `.rail`, and the shell's phone rule hid every `.rail`,
including the Model and Run pages' own rails, so three phone sections (Model Details, Run Classes, the queued run's
Config) rendered blank in the shots. Both reviewers found it first, and it drives 1, 5, 9, 11 and 12 (fixed before
merge, 9af99df, as `.shell-rail`). Data visualisation reached Sol's 9. Opus is stricter on text size this round
(10–11 px in dialogs, the LR exponent). From r20: the bug pass opens every phone section scene, and the shoot flags a
picked section that renders nothing.

r20 (#120: blank-section QA in the shoot, one remediation voice for f16 / tensor cores / AMP, Playground and Models
fixes): the best round so far. Every line is at 8 or above, and responsiveness and component craft climbed from
6.5 / 7 to 8.5 for both reviewers. Lowest lines: 1 hierarchy, 2 typography and 4 spacing (Opus 8: columns that end
at different heights, mono used for ids in some places but not others, the theme cards' "Current" mark), and 6
data visualisation (Sol 8: Compare's "By progress" divides live runs by their planned epochs but stopped ones by
their actual epochs; benchmark ratios stated across different conditions). Round 21 pairs the structural work (one
set of breakpoint tiers and a text-size floor, both linted; QA in CI) with the reviewers' items.

r21 (#126: layout tiers and the text floor, linted; QA in CI; Ident, --line-panel, Remedy, collision-aware tooltips;
#123's stored precision): Sol rose (2, 3 and 10 at 9, min 8 on 1 and 11), but Opus opened about 106 full-size shots
through four sub-reviewers and listed about 40 small rough edges, so its lines fell half a point (min 7.5 on 1, 4, 6, 7
and 8). The causes it names are systemic rather than per-screen: panels holding an empty state stretch to their
neighbour (queued Logs, Playground's tablet drop zone, Settings' gap); y domains that leave two-thirds of a plot
empty; detection overlays too heavy in thumbnails and a selection that does not stand out; counts and names that
disagree between places (Health's issues vs checks, a checkpoint's name vs its id); KPI values off their baseline;
lime on the active nav icon. Round 22 takes them as rules, plus #129 (operating point, resolved config, queue
control, ONNX).

r22 (#135: the round-21 causes as rules, #129 and #130 in the UI, six branches merged by hand): the floor is back to
8 on every line, with 3, 4, 8 and 9 at 8.5 (Sol 9 on 3 and 4). What holds 1, 2, 5, 6, 7, 10, 11 and 12 at 8 is now
mostly truth and fit, not styling: an out-of-memory retry that the estimator says will fail again; demo exports that
claim a saved file; per-class AP that does not average to the headline; phone rows that cut facts ("8 clas…",
"shapes-hard…", Compare's first pair headers broken per letter); feedback and tooltips that sit behind an open
dialog; help that needs hover; desktop canvases left half empty (lists, Settings, the empty Playground).

r23 (#141: one primary per state, the out-of-memory fit, truthful demo exports, class means in the mock, facts never
cut on phones, one tick family, dialog-contained feedback and tooltips, tap-to-open help, dogfood-3's CLI command and
close-mosaic cap): craft, data visualisation and accessibility rose to 8.5 (Sol 9 on 4 and 5), but 1, 2, 3, 7, 8, 11
and 12 stay at 8. What remains is narrower and more specific: small coloured text under AA in light (JSON tokens,
end labels, a lime metric, a red value), a second focus style on inputs, phone sections that put summaries before
the work (the KPI grid above every Model section, Compare's legend, a queued run's full config), prose columns in
wide frames, a New run preset that its own forecast says will not fit (the oom-retry scene's hard-coded batch), and
demo facts that disagree on one screen (a toast against the live run, 37.5 % against 37.9 %).

r24 (#143: #138's evaluation provenance; contrast-safe chart text ink, one focus style, status hue on marks; the work
before the summary on phones; prose and dialog measures; the shared sheet handle; truthful demo facts): Sol's best
round (eleven lines at 8.5 or 9; only 7 at 8, for scores that lose their evaluation context in summaries: Overview's
best models, the Models list, pickers). Opus stays at 8 on most lines, and some of what it names came from this
round's own changes: the best-epoch ring next to the live end ring (an "oo"), a failed run ending on a lime ring
with no failure mark, a tablet bottom sheet without a scrim, list toolbars that vanish in empty and error states.
The rest are laptop and tablet edges (cut model descriptions, the estimate box, wrapped callout buttons, tooltips
over rows) and naming drift (loss series named three ways, raw vs smoothed unexplained).

r25 (#148: scores keep their evaluation context, one mark per meaning, states and tiers, #140's final fields): the
best round so far. Nine lines are at 8.5 or more for both reviewers and interaction is at 9 for both; Opus gives 9
on 9 and 10, Sol on 3, 8 and 9. The floor of 8 is on 1, 7 and 11, and both reviewers' first issue is the same phone
defect: the completion callout keeps "Try in Playground" on the text's row, which cuts "Completed in 53m 30s". The
rest are phone structure (Compare's preamble before the plot, tracing a class from Curves, New run's long Model
section, Settings' strip clipped and its scrollspy on the wrong section), facts cut in the Playground model select,
two lime rings at best and latest, lime still marking selection, the raw amp refusal and "per step" batch wording.

r26 (#151: nothing clips on phones, one mark per meaning, ink selection, precise words): Sol's minimum reached 8.5
with five 9s, and colour is 9 for both. The floor of 8 is Opus's 1 and 11: Runs' header wrapping while selecting at
820, Compare's y axis from 0 squeezing the differences into the top quarter (and labels colliding at 820), phone
error callouts stacking three buttons under a lime primary on a red tint, frameless Overview error and Compare empty
states, New run's rail wrapping raggedly at 1024, and mostly empty panels for few classes or a near-perfect curve.
Sol adds the stale Overview not saying what is current, touch inputs under 16 px, grey PR curves that cannot be
identified where they are drawn, and the palette's idle list losing active runs to a case mismatch.

r27 (#158: headers, callouts, frames and tiers; charts that show the differences; states that say what is current):
colour is 9 for both, and Opus gives 9 on interaction and accessibility, but the floor of 8 widened to lines 1, 6, 7,
11 and 12. Both reviewers name the same defects: Compare's stopped and failed runs labelled in the right gutter far
from where their lines end, the published Model's Band / All curves toggle clipped under the Trace bar on phones,
"mAP50-…" in the phone Compare select and "shapes-har…" in the Models list, the phone Run page repeating one value
three times above its chart, Compare's phone header stack, and one narrow column of findings in tablet Health. Opus
adds New run showing the previous settings' estimate as current, the sidebar's "Last known" beside a live hero,
"mAP" unnamed in four places, Fit's domain leaving a quarter of the plot empty, and the tablet inspector and
Playground empty state; Sol adds class strips that do not look like disclosures, "mosaic off" read as current, and
the live log announcing without its own pause.

## Final state (stopped after round 28)

The hill-climb was stopped at the user's request after round 28 (#163), with the UI accepted as it is. Round 28 was
merged without new design reviews, so the last scored round is 27:

| Line | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-6.1 Sol | 8 | 8.5 | 9 | 9 | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8 | 8 |
| Claude Opus 5.5 | 8.5 | 8.5 | 9 | 8.5 | 8.5 | 8 | 8 | 8.5 | 9 | 9 | 8 | 8.5 |
| Minimum | 8 | 8.5 | 9 | 8.5 | 8.5 | 8 | 8 | 8.5 | 8.5 | 8.5 | 8 | 8 |

Round 28 addressed the round-27 reviews' items (listed in #163), but no reviewer has scored it. Still open, as named
by the reviewers and not verified since:
- Compare: a failed run that ends inside the bundle of other lines gets a swatch entry in the gutter, not a label at
  its end; the phone chart keeps ~45 px of headroom between the caveat and the top tick.
- The Inspector at 820 letterboxes the image heavily now that the dialog takes the viewport's height.
- The stuck Compare pair header's edge is `line.subtle`; if it still reads as edgeless, it needs a stronger line.
- Phone Playground's out-of-memory notice is three stacked 44 px buttons (the one remedy layout), taller than before.
- Sol's round-27 code-review note: the vite client's "ResizeObserver loop completed with undelivered notifications"
  during `npm run qa` is not attributed (a lead: `dataset/StatsTab.svelte`'s observer, which predates round 27).
- Lines 1, 6, 7, 11 and 12 were last scored 8; nothing was scored 9 or above on every line by both reviewers.
