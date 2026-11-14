# tools/charts: the charts of the README and docs

The bar and line charts in [README.md](../../README.md), [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) and
[examples/coco](../../examples/coco/README.md) are static SVG files in `docs/img/charts/`, drawn by this generator.
It is plain Node (no dependencies, no `npm install`) and takes its colours from `design/tokens.json`. The rules for
changing a number are in [docs/BENCHMARKS.md, Charts](../../docs/BENCHMARKS.md#charts).

| file | what it does |
|---|---|
| [`data.json`](data.json) | every number the charts show; each group's `source` names the document section or the run records it was copied from (`coco_curves`: the runs' `metrics.jsonl`) |
| [`render.mjs`](render.mjs) | the chart definitions; writes `docs/img/charts/<name>-light.svg` and `<name>-dark.svg`, or with `--check` compares them |
| [`bars.mjs`](bars.mjs) | the layout of the horizontal grouped bar charts |
| [`lines.mjs`](lines.mjs) | the layout of the line chart (validation mAP by epoch) |
| [`lib.mjs`](lib.mjs) | shared pieces: the theme read from `design/tokens.json`, SVG primitives, a text-width estimate |

`render.mjs` draws five charts, each in a light and a dark file: `speed-inference`, `speed-training` and `accuracy`
(used by the README), `speed-by-size` (docs/BENCHMARKS.md, which also shows `accuracy` and `coco-curves`) and
`coco-curves` (examples/coco).

## Changing a number

1. Change it in the document it comes from (the `source` of its group in `data.json`).
2. Change it in `data.json`. A row's `text` is the value as printed, and must equal its `values` entry rounded to
   the digits it shows; `render.mjs` refuses to run otherwise.
3. Run `node tools/charts/render.mjs` from the repository root and commit the changed SVG files with the data.

`node tools/charts/render.mjs --check` exits 1 when a committed chart is missing, differs from what the data gives,
or is a file in `docs/img/charts/` that the script does not draw. CI runs it in `.github/workflows/site.yml`, step
"charts", on every pull request. The README, docs and the alt text of each `<img>` carry the same numbers as prose,
and `--check` does not read those: keep them in step by hand.

[Up: tools/](../README.md)
