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
