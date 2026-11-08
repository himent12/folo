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
| r10 (Sol; Opus pending) | 8.5 | 8.5 | 8.5 | 9 | 8.5 | 8 | 8.5 | 8.5 | 9 | 8.5 | 8 | 8.5 | 8 |

r4 lowest lines: 6 data visualisation (5.5), 2 typography, 3 colour, 8 consistency (6). Diagnosis: 2, 3 and 8 are
structural (no single rule for mono vs sans, class colours that changed between legend and box, two accents in light),
fixed in round 4 by system rules in `design/README.md` rather than per-screen patches; 6 is partly structural (run
comparison and dense COCO evaluation screens need a different layout, not tweaks).

r5 (after the design-director pass, `DESIGN_DIRECTION.md`): every line up, no line below 7 for the first time. Lowest:
11 responsiveness (7 from both). Opus could open only 71 of 132 shots (image request limit), so its r5 row is partial;
from r6 reviewers pace image loading and retry.
