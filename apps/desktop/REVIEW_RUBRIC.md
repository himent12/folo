# FOLO Desktop design review rubric

Score each line 1–10 from the screenshots (and the source for states screenshots can't show: hover, focus, pressed,
motion, keyboard paths). One line of reason per score.

**Calibration.** 10 = best in class; nothing to change. **9 = would ship as-is at Linear / Raycast / Zed quality**: a
demanding design lead signs off without asking for changes in that area. 8 = very good, a few specific fixes away.
7 = good but clearly improvable; a design lead would send it back. 6 = acceptable, visibly unpolished. ≤ 5 =
problems a user would notice in the first minute. Score what is on screen, not intent; a 9 needs no open issue in
that line that you would ask to fix before shipping.

| # | Line | What a 9 looks like |
|---|------|---------------------|
| 1 | **Visual hierarchy & layout** | Every screen has one obvious focal point (the live run, the image, the chart that matters). Primary actions are findable in under a second. Panels group related information; nothing competes. |
| 2 | **Typography** | One type scale used consistently (Geist / Geist Mono, 400/500/600). Clear title → section → label → detail steps. Tabular numerals for every live or tabular number; units one step smaller and quieter. No awkward wraps, truncations or orphan words. |
| 3 | **Colour, contrast & theming** | Colour is spent on meaning (accent for focus/live/primary, status colours for status, class palette for classes). Both dark and light are equally finished; text meets WCAG AA; hairlines and surfaces layer cleanly in both themes. |
| 4 | **Spacing & alignment rhythm** | A visible 4 px grid. Panel paddings, gaps and row heights repeat. Edges align across panels, charts and tables; baselines line up in metric rows. No crowding, no dead zones. |
| 5 | **Component craft & states** | Buttons, segmented controls, chips, sliders, switches, selects, badges, tables, tooltips and menus look like one family and have complete states (rest, hover, pressed, focus, selected, disabled). Details are crisp: radii, hairlines, icon sizes and alignment. |
| 6 | **Data visualisation** | Charts are legible and honest: sensible axes and ticks, consistent decimals, readable legends, a useful hover readout, restrained gridlines. Detection boxes and labels read on any image. Confusion matrix, PR curves and histograms are instantly understandable. |
| 7 | **Information design & copy** | The right numbers are shown at the right precision with units. Labels are short, specific and consistent ("mAP50-95", "img/s"). Empty, loading and error states explain what happened and what to do next. |
| 8 | **Consistency across screens** | The same patterns everywhere: page header, panel header, list → detail with breadcrumbs, table rows, KPI strips, empty states. Nothing looks like it came from a different app. |
| 9 | **Interaction & motion feel** | (From code and states.) Hover/pressed feedback on everything interactive; live data updates without jank; drag, zoom, pan and sliders feel direct; motion is fast and quiet (no bounce). |
| 10 | **Accessibility** | Full keyboard path: Tab order follows layout, a visible focus ring on every focusable element, Enter/Space/arrows/Esc work, shortcuts are discoverable. Contrast passes; meaning never relies on colour alone. |
| 11 | **Window resizing & responsiveness** | The minimum size (960 × 620) and a wide window (1680 × 1050) both look designed: content reflows, nothing overflows or clips, density adapts (collapsed sidebar, wrapped KPI grid). |
| 12 | **Overall polish & delight** | It feels like a precise instrument: fast, calm, confident. Small touches (live pulse, linked chart cursors, the logo's signal dot, smart defaults) make it memorable. You would show it off. |

## Output format for reviewers

```
SCORES
1 hierarchy: N — reason
2 typography: N — reason
3 colour: N — reason
4 spacing: N — reason
5 components: N — reason
6 dataviz: N — reason
7 info/copy: N — reason
8 consistency: N — reason
9 interaction: N — reason
10 accessibility: N — reason
11 responsiveness: N — reason
12 polish: N — reason
AVERAGE: N.N

TOP 10 ISSUES (ranked by impact)
1. [line #] screenshot + region: the problem → the concrete fix (px, token, copy)
...

BUGS / RENDERING ARTIFACTS
- ...
```
