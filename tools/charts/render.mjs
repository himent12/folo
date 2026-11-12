// Draws the charts of the README and docs from tools/charts/data.json, in the colours of design/tokens.json:
// one SVG per chart and theme in docs/img/charts (<name>-light.svg, <name>-dark.svg).
//   node tools/charts/render.mjs            write the files
//   node tools/charts/render.mjs --check    exit 1 if a committed file differs from what the data gives (CI)
// No dependencies. The numbers live in data.json with the document each one was copied from; the layouts are in
// bars.mjs and lines.mjs.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { barsFigure } from './bars.mjs';
import { theme } from './lib.mjs';
import { linesFigure } from './lines.mjs';

const data = JSON.parse(readFileSync(new URL('./data.json', import.meta.url), 'utf8'));
const OUT = new URL('../../docs/img/charts/', import.meta.url);

const times = (ratio) => `${ratio.toFixed(1)}×`;
/** How many times better the first value is than the second. */
const speedup = ([folo, other], better) => (better === 'lower' ? other / folo : folo / other);
const withSpeedup = (rows, better) => rows.map((row) => ({ ...row, note: times(speedup(row.values, better)), noteSub: 'faster' }));

const FOLO_N = { label: 'folo-n', colour: 'series0', strong: true };
const YOLO_N = { label: 'yolo26n (Ultralytics 8.4.198), its best configuration for each row', colour: 'reference' };
const METHOD = 'Medians of alternated runs, 2026-11-12. Method, noise and every configuration: docs/BENCHMARKS.md.';

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
        footer: [METHOD],
      },
      t,
    ),

  'speed-training': (t) =>
    barsFigure(
      {
        title: 'Training speed: folo-n against yolo26n',
        subtitle: data.machine,
        desc:
          'Grouped bars. Training throughput at batch 16 in images per second and peak GPU memory of a training step in MiB, for folo-n and yolo26n. ' +
          [...data.training.throughput.map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]} images per second.`), ...data.training.memory.map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]} MiB.`)].join(' '),
        series: [FOLO_N, YOLO_N],
        panels: [
          { title: 'Throughput at batch 16', hint: 'images per second, higher is better', unit: 'img/s', notes: true, rows: withSpeedup(data.training.throughput, 'higher') },
          {
            title: 'Peak GPU memory at batch 16',
            hint: 'MiB, lower is better',
            unit: 'MiB',
            notes: true,
            rows: data.training.memory.map((row) => ({ ...row, note: `${Math.round((1 - row.values[0] / row.values[1]) * 100)}%`, noteSub: 'less' })),
          },
        ],
        footer: [METHOD],
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
        footer: ['Each bar divides two medians of the FOLO targets table in docs/BENCHMARKS.md (the best YOLO26 configuration per row).'],
      },
      t,
    ),

  accuracy: (t) =>
    barsFigure(
      {
        title: 'Accuracy: FOLO against YOLO26',
        subtitle: 'COCO AP (pycocotools), one scorer for both sides',
        desc:
          'Grouped bars of AP. On COCO val2017, FOLO trained from scratch is behind the published YOLO26 models. Fine-tuned on four small datasets, folo-n and yolo26n are level on three and yolo26n is ahead on one. ' +
          [...data.accuracy.coco.rows, ...data.accuracy.finetune.rows].map((r) => `${r.label}: ${r.text[0]} against ${r.text[1]}.`).join(' '),
        series: [
          { label: 'FOLO', colour: 'series0', strong: true },
          { label: 'Ultralytics YOLO26', colour: 'reference' },
        ],
        panels: [
          {
            title: 'COCO val2017',
            hint: 'AP, higher is better. Not like for like: see below',
            notes: true,
            rows: data.accuracy.coco.rows.map((row) => ({ ...row, note: row.delta, noteSub: 'AP' })),
          },
          {
            title: 'Fine-tuned on small datasets',
            hint: 'folo-n and yolo26n: AP, mean and range of 3 seeds',
            notes: true,
            rows: data.accuracy.finetune.rows.map((row) => ({ ...row, note: row.delta, noteSub: row.verdict })),
          },
        ],
        footer: [
          'COCO: FOLO trained from scratch for 56 to 62 epochs. The YOLO26 numbers are the published ones, from models',
          'pretrained on Objects365 and trained for far longer. Fine-tuning: each from its own COCO weights, at its',
          'defaults, 100 epochs. "level": inside the spread of the seeds. Details: docs/BENCHMARKS.md, Accuracy.',
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
        series: data.coco_curves.series.map((s, i) => ({ ...s, short: ['folo-n', 'folo-n v2', 'folo-s'][i], labelAt: i === 0 ? 'below' : 'right' })),
        footer: ['Every epoch improved on the one before in all three runs: none had reached a plateau when it stopped.', 'Runs and commands: examples/coco/README.md.'],
      },
      t,
    ),
};

const check = process.argv.includes('--check');
let stale = 0;
if (!check) mkdirSync(OUT, { recursive: true });
for (const [name, draw] of Object.entries(figures)) {
  for (const mode of ['light', 'dark']) {
    const file = new URL(`${name}-${mode}.svg`, OUT);
    const svg = draw(theme(mode));
    if (!check) {
      writeFileSync(file, svg);
      continue;
    }
    const current = existsSync(file) ? readFileSync(file, 'utf8').replace(/\r\n/g, '\n') : null;
    if (current !== svg) {
      stale++;
      console.error(`${fileURLToPath(file)} is ${current === null ? 'missing' : 'out of date'}`);
    }
  }
}
if (stale) {
  console.error(`\n${stale} chart file(s) differ from tools/charts/data.json: run \`node tools/charts/render.mjs\` and commit the result.`);
  process.exit(1);
}
console.log(check ? 'charts are up to date' : `wrote ${Object.keys(figures).length * 2} files to docs/img/charts`);
