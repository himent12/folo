// Draws the charts of the README and docs from tools/charts/data.json, in the colours of design/tokens.json:
// one SVG per chart and theme in docs/img/charts (<name>-light.svg, <name>-dark.svg).
//   node tools/charts/render.mjs            write the files
//   node tools/charts/render.mjs --check    exit 1 if the committed files are not exactly what the data gives (CI)
// No dependencies. The numbers live in data.json with the document each one was copied from; the layouts are in
// grid.mjs, bars.mjs and lines.mjs.

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { barsFigure } from './bars.mjs';
import { dashboardFigure } from './grid.mjs';
import { theme } from './lib.mjs';
import { linesFigure } from './lines.mjs';

const data = JSON.parse(readFileSync(new URL('./data.json', import.meta.url), 'utf8'));
const OUT = new URL('../../docs/img/charts/', import.meta.url);

// A row's `text` is what the chart prints for its `values`, written out so the precision is a choice (12.74 prints
// as "12.7"): it has to be that value, rounded to the digits it shows.
for (const group of [data.inference.latency, data.inference.throughput, data.training.throughput, data.training.memory, data.accuracy.coco.rows, data.accuracy.finetune.rows]) {
  for (const row of group) {
    row.values.forEach((value, i) => {
      const digits = (row.text[i].split('.')[1] ?? '').length;
      if (value.toFixed(digits) !== row.text[i]) throw new Error(`data.json: "${row.label}" prints ${row.text[i]} for ${value}`);
    });
  }
}

const times = (ratio) => `${ratio.toFixed(1)}×`;
/** How many times better the first value is than the second. */
const speedup = ([folo, other], better) => (better === 'lower' ? other / folo : folo / other);
const withSpeedBadge = (rows, better) =>
  rows.map((row) => ({ ...row, badge: times(speedup(row.values, better)), badgeTone: 'accent' }));

const FOLO_N = { label: 'folo-n', colour: 'series0', strong: true };
const YOLO_N = { label: 'yolo26n (best config per row)', colour: 'reference' };

// Short sub-labels keep the 2×2 dashboard crisp while preserving exact numbers from data.json.
const compactRow = (row, label, detail) => ({ ...row, label, detail });

const gpuLatency = withSpeedBadge(
  [
    compactRow(data.inference.latency[0], 'GPU forward', 'f32, on device'),
    compactRow(data.inference.latency[1], 'GPU end to end', 'image in, boxes out'),
  ],
  'lower',
);
const cpuLatency = withSpeedBadge(
  [
    compactRow(data.inference.latency[2], 'CPU forward', 'f32, best threads'),
    compactRow(data.inference.latency[3], 'CPU end to end', 'image in, boxes out'),
  ],
  'lower',
);
const gpuThroughput = withSpeedBadge(
  [
    compactRow(data.inference.throughput[0], 'Forward, f32', 'yolo26n uses TF32'),
    compactRow(data.inference.throughput[1], 'Forward, f16', 'FOLO tensor cores'),
    compactRow(data.inference.throughput[2], 'End to end, f16', 'decoded images in'),
  ],
  'higher',
);
const trainThroughput = withSpeedBadge(
  [
    compactRow(data.training.throughput[0], 'Train step, f32', 'fwd + loss + bwd + SGD'),
    compactRow(data.training.throughput[1], 'Train step, AMP', 'mixed precision'),
    compactRow(data.training.throughput[2], 'Whole trainer, f32', 'loader + mosaic, 8w'),
    compactRow(data.training.throughput[3], 'Whole trainer, AMP', 'sustained / epoch'),
  ],
  'higher',
);
const trainMemory = data.training.memory.map((row, i) => ({
  ...compactRow(row, i === 0 ? 'Train step, f32' : 'Train step, AMP', 'peak live tensors'),
  badge: `−${Math.round((1 - row.values[0] / row.values[1]) * 100)}%`,
  badgeTone: 'accent',
}));

const figures = {
  speed: (t) =>
    dashboardFigure(
      {
        title: 'Speed & memory: folo-n vs yolo26n',
        subtitle: `${data.machine} · Ultralytics 8.4.198`,
        desc:
          '2x2 dashboard comparing folo-n against yolo26n. Latency at batch 1 (GPU on 0-7 ms scale, CPU on 0-25 ms scale): ' +
          data.inference.latency.map((r) => `${r.label} ${r.text[0]} vs ${r.text[1]} ms`).join(', ') +
          '. Batch-16 GPU inference throughput: ' +
          data.inference.throughput.map((r) => `${r.label} ${r.text[0]} vs ${r.text[1]} img/s`).join(', ') +
          '. Batch-16 training throughput: ' +
          data.training.throughput.map((r) => `${r.label} ${r.text[0]} vs ${r.text[1]} img/s`).join(', ') +
          '. Peak GPU memory: ' +
          data.training.memory.map((r) => `${r.label} ${r.text[0]} vs ${r.text[1]} MiB`).join(', ') +
          '.',
        series: [FOLO_N, YOLO_N],
        grid: [
          [
            {
              title: 'Latency · batch 1',
              hint: '↓ ms, lower is better',
              unit: 'ms',
              labelW: 114,
              pillW: 48,
              groups: [
                { title: 'GPU', scale: '0–7 ms', max: 7, rows: gpuLatency },
                { title: 'CPU', scale: '0–25 ms', max: 25, rows: cpuLatency },
              ],
            },
            {
              title: 'Training · batch 16',
              hint: '↑ img/s, higher is better',
              unit: 'img/s',
              labelW: 128,
              pillW: 48,
              groups: [{ rows: trainThroughput }],
            },
          ],
          [
            {
              title: 'GPU inference · batch 16',
              hint: '↑ img/s, higher is better',
              unit: 'img/s',
              labelW: 114,
              pillW: 48,
              groups: [{ rows: gpuThroughput }],
            },
            {
              title: 'Peak GPU memory · batch 16',
              hint: '↓ MiB, lower is better',
              unit: 'MiB',
              labelW: 114,
              pillW: 52,
              note: 'Peak allocated by live tensors; both allocators reserve more.',
              groups: [{ rows: trainMemory }],
            },
          ],
        ],
        footer: [
          'FOLO: medians of alternated runs (2026-11-12/13; end to end on trained weights). YOLO26: fastest config per row (2026-11-07). Full method: docs/BENCHMARKS.md.',
          data.warmup,
        ],
      },
      t,
    ),

  accuracy: (t) =>
    dashboardFigure(
      {
        title: 'Accuracy: FOLO vs Ultralytics YOLO26',
        subtitle: 'AP@[.5:.95] (pycocotools) · higher is better · 0–1 shared scale · whiskers show 3-seed min–max',
        desc:
          'Two-panel chart of AP on one shared 0 to 1 scale. COCO val2017 (not like for like): ' +
          data.accuracy.coco.rows.map((r) => `${r.label} ${r.text[0]} vs ${r.text[1]}`).join(', ') +
          '. Fine-tuned on four small datasets (folo-n vs yolo26n, 3-seed mean): ' +
          data.accuracy.finetune.rows.map((r) => `${r.label} ${r.text[0]} vs ${r.text[1]} (${r.verdict})`).join(', ') +
          '.',
        series: [
          { label: 'FOLO', colour: 'series0', strong: true },
          { label: 'Ultralytics YOLO26', colour: 'reference' },
        ],
        grid: [
          [
            {
              title: 'COCO val2017',
              hint: 'not like for like',
              flex: 1,
              labelW: 118,
              pillW: 56,
              sharedMax: 1.0,
              note: 'FOLO trained on COCO alone from scratch (128 / 56 ep). YOLO26 pretrained on Objects365 and trained longer.',
              groups: [
                {
                  rows: data.accuracy.coco.rows.map((row, i) => ({
                    ...row,
                    label: i === 0 ? 'folo-n v5 vs 26n' : 'folo-s vs 26s',
                    detail: i === 0 ? '128 ep, distilled' : '56 ep, scratch',
                    badge: row.delta,
                    badgeTone: 'muted',
                  })),
                },
              ],
            },
            {
              title: 'Fine-tuned on small datasets',
              hint: 'folo-n vs yolo26n · 3 seeds',
              flex: 1,
              labelW: 118,
              pillW: 56,
              sharedMax: 1.0,
              note: '100 epochs from own COCO weights (folo-n-coco-v2), one pycocotools scorer.',
              groups: [
                {
                  rows: data.accuracy.finetune.rows.map((row) => ({
                    ...row,
                    label: row.label.replace(' (BCCD)', '').replace(' (MRI)', ''),
                    detail: row.detail.replace(' training images', ' train imgs'),
                    badge: row.verdict === 'level' ? 'level' : row.delta,
                    badgeTone: row.verdict === 'level' ? 'neutral' : 'warn',
                  })),
                },
              ],
            },
          ],
        ],
        footer: [
          '"level": inside 3-seed spread or saturated (Signatures). Opt-in SEBF + DFL-CGR (--iou 0.78 --max-det 1000) adds +0.0065–0.0076 AP on COCO; full tables in docs/BENCHMARKS.md.',
        ],
      },
      t,
    ),

  'speed-by-size': (t) =>
    barsFigure(
      {
        title: 'Speed-up over the YOLO26 model of the same size',
        subtitle: data.machine,
        desc:
          'Grouped bars of how many times faster FOLO is than the YOLO26 model of the same size, for the nano and the small models. ' +
          data.by_size.rows.map((r) => `${r.label}, ${r.detail}: ${times(speedup(r.n, r.better))} for folo-n, ${times(speedup(r.s, r.better))} for folo-s.`).join(' '),
        series: [
          { label: 'folo-n vs yolo26n', colour: 'series0', strong: true },
          { label: 'folo-s vs yolo26s', colour: 'series1', strong: true },
        ],
        panels: [
          {
            title: 'Times faster',
            hint: 'latency: YOLO26 / FOLO · throughput: FOLO / YOLO26',
            refLine: { value: 1, label: '1× = YOLO26' },
            rows: data.by_size.rows.map((row) => {
              const values = [speedup(row.n, row.better), speedup(row.s, row.better)];
              return { label: row.label, detail: row.detail, values, text: values.map(times) };
            }),
          },
        ],
        footer: [
          "Ratios from the FOLO targets table in docs/BENCHMARKS.md: FOLO's median vs YOLO26's best config per row (measured 2026-11-12/13 and 2026-11-07).",
          data.warmup,
        ],
      },
      t,
    ),

  'coco-curves': (t) =>
    linesFigure(
      {
        title: 'COCO from scratch: validation mAP50-95 by epoch',
        subtitle: "The trainer's own validation after every epoch, 640 px, one GPU",
        desc:
          'Line chart of validation mAP50-95 after each epoch for three training runs on COCO from random weights. ' +
          data.coco_curves.series.map((s) => `${s.label}: ${s.values.at(-1).toFixed(3)} after ${s.values.length} epochs.`).join(' '),
        xTitle: 'epoch',
        xTicks: [0, 10, 20, 30, 40, 50, 60],
        xMax: 64,
        yTicks: [0, 0.1, 0.2, 0.3, 0.4],
        yMax: 0.45,
        yFormat: (v, end) => (end ? v.toFixed(3) : v === 0 ? '0' : v.toFixed(1)),
        series: data.coco_curves.series,
        footer: [
          `Each run was still rising at its last epoch. ${data.coco_curves.final_ap} Runs & commands: examples/coco/README.md.`,
        ],
      },
      t,
    ),
};

const check = process.argv.includes('--check');
const expected = new Map();
for (const [name, draw] of Object.entries(figures)) {
  for (const mode of ['light', 'dark']) expected.set(`${name}-${mode}.svg`, draw(theme(mode)));
}

if (!check) {
  mkdirSync(OUT, { recursive: true });
  for (const [file, svg] of expected) writeFileSync(new URL(file, OUT), svg);
  console.log(`wrote ${expected.size} files to docs/img/charts`);
} else {
  const problems = [];
  for (const [file, svg] of expected) {
    const url = new URL(file, OUT);
    const current = existsSync(url) ? readFileSync(url, 'utf8').replace(/\r\n/g, '\n') : null;
    if (current !== svg) problems.push(`${fileURLToPath(url)} is ${current === null ? 'missing' : 'out of date'}`);
  }
  // A chart that was renamed or dropped leaves its old files behind, still linked from somewhere or not.
  for (const file of existsSync(OUT) ? readdirSync(OUT) : []) {
    if (!expected.has(file)) problems.push(`${fileURLToPath(new URL(file, OUT))} is not a chart this script draws`);
  }
  if (problems.length) {
    console.error(problems.join('\n'));
    console.error(`\ndocs/img/charts does not match tools/charts/data.json: run \`node tools/charts/render.mjs\` and commit the result.`);
    process.exit(1);
  }
  console.log('charts are up to date');
}
