# FOLO Studio design review rubric

Score each line 1–10 from the screenshots (and the source where a line needs it). Be demanding: the bar is the best
product UIs shipping today.

**Calibration**
- **10**: best in class. Nothing to improve that a world-class design team would bother with.
- **9**: would ship as-is at Linear / Vercel / Raycast quality. Only nitpicks remain.
- **7–8**: good, clearly professional, but a careful reviewer finds several real issues.
- **5–6**: competent but generic, inconsistent, or with visible rough edges.
- **3–4**: obvious problems a user would notice (broken layouts, poor contrast, confusing flows).
- **1–2**: broken.

A score of 9 or 10 needs that quality on **every** screen and state captured (dark and light, 1440 / 1024 / 390), not
only the best one.

| # | Line | What a 9 looks like |
|---|------|---------------------|
| 1 | **Visual hierarchy & layout** | Each screen has one obvious focal point; grouping, alignment and density guide the eye; grids hold at every width; nothing feels cramped or empty by accident. |
| 2 | **Typography** | A real type scale used consistently; weights and sizes carry hierarchy; tabular numerals for metrics; mono only where it means something; no orphans or awkward truncation. |
| 3 | **Colour, contrast & theming** | Restrained palette; the accent means something (focus, live, primary); status colours only for status; text meets WCAG AA; dark and light both look designed, not inverted. |
| 4 | **Spacing & alignment rhythm** | Everything sits on the 4/8 px grid; consistent paddings inside panels and gaps between them; baselines and edges line up across neighbouring components. |
| 5 | **Component craft & states** | Buttons, inputs, tables, badges, tabs, dialogs, tooltips feel finished: hover, focus, active, disabled, loading, empty and error states all designed. |
| 6 | **Data visualisation** | Charts are legible and honest: sensible axes and domains, recessive grids, consistent series colours that follow the entity, legends/direct labels, crosshair readouts; detection boxes legible on any photo. |
| 7 | **Information design & copy** | The right information at the right density; units and context on every number; copy is short, specific and helpful (errors say what to do next); no jargon without need. |
| 8 | **Consistency across screens** | Same patterns for the same jobs everywhere: page headers, panels, tables, filters, empty states, actions placement. |
| 9 | **Interaction & motion feel** | (From code and states.) Fast, purposeful transitions (120–200 ms, no bounce); clear affordances; keyboard paths everywhere (palette, shortcuts, arrow keys); live data updates without jank. |
| 10 | **Accessibility** | Visible focus, semantic structure, labels for icon buttons and charts, colour never the only cue, reduced motion respected, touch targets adequate on mobile. |
| 11 | **Responsiveness** | 1440, 1024 and 390 px all feel intentional: layouts reflow, nothing overflows or truncates badly, mobile navigation and dialogs work. |
| 12 | **Overall polish & delight** | It feels crafted and memorable: a distinctive identity (viewfinder, signal lime), small moments of delight, no rough edges left. |

## Reviewer output format
1. A table: rubric line, score (1–10), one-line reason.
2. The top 10 concrete issues ranked by impact: screen (and theme/viewport), element, what is wrong, the fix.
