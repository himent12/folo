// Draws the charts of the README and docs from tools/charts/data.json, in the colours of design/tokens.json:
// one SVG per chart and theme in docs/img/charts (<name>-light.svg, <name>-dark.svg).
//   node tools/charts/render.mjs            write the files
//   node tools/charts/render.mjs --check    exit 1 if the committed files are not exactly what the data gives (CI)
// No dependencies. The numbers live in data.json with the document each one was copied from; the layouts are in
// bars.mjs and lines.mjs.

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { barsFigure } from './bars.mjs';
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
const withSpeedup = (rows, better) => rows.map((row) => ({ ...row, note: times(speedup(row.values, better)), noteSub: 'faster' }));

const FOLO_N = { label: 'folo-n', colour: 'series0', strong: true };
const YOLO_N = { label: 'yolo26n (Ultralytics 8.4.198), its best configuration for each row', colour: 'reference' };

const figures = {
  'speed-inference': (t) =>
    barsFigure(
      {
        title: 'Inference speed: folo-n against yolo26n',
        subtitle: data.machine,
        desc:
          'Grouped bars. Latency at batch 1 in milliseconds and throughput at batch 16 in images per second, on the GPU and the CPU, for folo-n and yolo26n. ' +
          [...data.inference.latency.map((r) => `${r.label}: ${r.text[0]} ms against ${r.text[1]} ms.`), ...data.inference.throughput.map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]} images per second.`)].join(' '),
        series: [FOLO_N, YOLO_N],
        panels: [
          { title: 'Latency at batch 1', hint: 'milliseconds, lower is better', unit: 'ms', notes: true, rows: withSpeedup(data.inference.latency, 'lower') },
          { title: 'Throughput at batch 16', hint: 'images per second, higher is better', unit: 'img/s', notes: true, rows: withSpeedup(data.inference.throughput, 'higher') },
        ],
        footer: [data.method, data.warmup],
      },
      t,
    ),

  'speed-training': (t) =>
    barsFigure(
      {
        title: 'Training speed: folo-n against yolo26n',
        subtitle: data.machine,
        desc:
          'Grouped bars. Training throughput at batch 16 in images per second and peak GPU memory allocated by a training step in MiB, for folo-n and yolo26n. ' +
          [...data.training.throughput.map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]} images per second.`), ...data.training.memory.map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]} MiB.`)].join(' '),
        series: [FOLO_N, YOLO_N],
        panels: [
          { title: 'Throughput at batch 16', hint: 'images per second, higher is better', unit: 'img/s', notes: true, rows: withSpeedup(data.training.throughput, 'higher') },
          {
            title: 'Peak GPU memory allocated at batch 16',
            hint: 'MiB, lower is better',
            unit: 'MiB',
            notes: true,
            rows: data.training.memory.map((row) => ({ ...row, note: `${Math.round((1 - row.values[0] / row.values[1]) * 100)}%`, noteSub: 'less' })),
          },
        ],
        footer: [data.method, 'Memory is what live tensors hold at the peak; both allocators reserve more than that (row 11 there).'],
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
          { label: 'folo-n against yolo26n', colour: 'series0', strong: true },
          { label: 'folo-s against yolo26s', colour: 'series1', strong: true },
        ],
        panels: [
          {
            title: 'Times faster',
            hint: 'latency: YOLO26 over FOLO; throughput: FOLO over YOLO26',
            refLine: { value: 1, label: '1× = YOLO26' },
            rows: data.by_size.rows.map((row) => {
              const values = [speedup(row.n, row.better), speedup(row.s, row.better)];
              return { label: row.label, detail: row.detail, values, text: values.map(times) };
            }),
          },
        ],
        footer: ["Each bar divides two numbers of the FOLO targets table in docs/BENCHMARKS.md: FOLO's median and the best YOLO26 configuration of that row (measured 2026-11-12 and 2026-11-07).", data.warmup],
      },
      t,
    ),

  accuracy: (t) =>
    barsFigure(
      {
        title: 'Accuracy: FOLO against YOLO26',
        subtitle: 'AP@[.5:.95], pycocotools',
        desc:
          'Grouped bars of AP on one scale. On COCO val2017, FOLO trained on COCO alone is behind the published YOLO26 models. Fine-tuned on four small datasets, folo-n and yolo26n are level on three and yolo26n is ahead on one. ' +
          [...data.accuracy.coco.rows, ...data.accuracy.finetune.rows].map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]}.`).join(' '),
        sharedScale: true,
        series: [
          { label: 'FOLO', colour: 'series0', strong: true },
          { label: 'Ultralytics YOLO26', colour: 'reference' },
        ],
        panels: [
          {
            title: 'COCO val2017',
            hint: 'higher is better. Not like for like: see below',
            notes: true,
            rows: data.accuracy.coco.rows.map((row) => ({ ...row, note: row.delta, noteSub: 'AP' })),
          },
          {
            title: 'Fine-tuned on small datasets',
            hint: 'folo-n and yolo26n, one scorer for both: mean and range of 3 seeds',
            notes: true,
            rows: data.accuracy.finetune.rows.map((row) => ({ ...row, note: row.delta, noteSub: row.verdict })),
          },
        ],
        footer: [
          'COCO: FOLO trained on COCO alone from random weights, 56 and 64 epochs (folo-n: the second half distilled from folo-s). The YOLO26 numbers are the published ones, from models pretrained on Objects365 and trained for far longer.',
          'Fine-tuning: each from its own COCO weights, at its defaults, 100 epochs; the differences are of the unrounded means. "level": inside the spread of the seeds, or on a set both saturate (signatures). Details: docs/BENCHMARKS.md, Accuracy.',
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
          `Each run was still rising at its last epoch: v2 in mid-schedule, the other two at the end of their learning-rate decay. ${data.coco_curves.final_ap}`,
          'Runs and commands: examples/coco/README.md.',
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
