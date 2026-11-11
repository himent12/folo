// Data for the charts and detection sections of the ui gallery (sections/Chart*.*, sections/Detect*.*), shared by
// both twins so only the components differ. Deterministic shapes like the mock's: a run's per-epoch metrics, a
// comparison of runs that ended differently, per-step losses on a log axis, a benchmark's curves, box-size bins, a
// confusion matrix, detections on an image.
import type { Marker, Series, XMark } from '$lib/charts/lineSeries';
import type { XYMark, XYRef, XYSeries, XYStrip } from '$lib/charts/xy';

const epochs = (n: number, f: (e: number) => number) => Array.from({ length: n }, (_, i) => ({ x: i + 1, y: +f(i + 1).toFixed(4) }));
/** A mAP curve that climbs and saturates, with a small wobble (no randomness: both twins draw the same line). */
const climb = (top: number, rate: number, wobble = 0.006) => (e: number) => top * (1 - Math.exp(-e / rate)) + wobble * Math.sin(e * 1.7);

/** Compare: four runs, one still live, one stopped, one failed early, one complete. */
export const compare: Series[] = [
  { id: 'r1', label: 'fasteners-s', color: 'var(--chart-1)', end: 'live', points: epochs(18, climb(0.62, 6)) },
  { id: 'r2', label: 'fasteners-n', color: 'var(--chart-2)', pattern: 'dash', points: epochs(30, climb(0.55, 7)) },
  { id: 'r3', label: 'shapes-hard-s', color: 'var(--chart-3)', pattern: 'dot', end: 'stopped', points: epochs(22, climb(0.48, 5)) },
  { id: 'r4', label: 'a-very-long-run-name-indeed', color: 'var(--chart-4)', end: 'failed', points: epochs(9, climb(0.4, 4)) },
];
export const compareMarkers: Marker[] = [
  { x: 18, y: compare[0]!.points[17]!.y, label: 'best', quiet: true },
  { x: 27, y: compare[1]!.points[26]!.y, label: 'best 0.55' },
];
export const mosaicOff: XMark = { x: 25, label: 'mosaic off' };

/** A run's loss terms per step (log axis): raw box loss as a ghost under its EMA, cls dashed, dfl dotted. */
const steps = (n: number, f: (s: number) => number) => Array.from({ length: n }, (_, i) => ({ x: i / 10, y: +f(i).toFixed(5) }));
export const loss: Series[] = [
  { id: 'box-raw', label: 'box (raw)', color: 'var(--chart-1)', ghost: true, points: steps(200, (s) => 2.2 * Math.exp(-s / 70) + 0.35 + 0.12 * Math.sin(s * 2.3)) },
  { id: 'box', label: 'box', color: 'var(--chart-1)', area: true, points: steps(200, (s) => 2.2 * Math.exp(-s / 70) + 0.35) },
  { id: 'cls', label: 'cls', color: 'var(--chart-2)', pattern: 'dash', points: steps(200, (s) => 3.1 * Math.exp(-s / 50) + 0.5) },
  { id: 'dfl', label: 'dfl', color: 'var(--chart-3)', pattern: 'dot', front: true, points: steps(200, (s) => 1.4 * Math.exp(-s / 90) + 0.8) },
];

/** Per-class PR-style curves as context lines, their mean emphasised, and the spread as a band. */
const curve = (k: number) => Array.from({ length: 21 }, (_, i) => ({ x: i / 20, y: +Math.max(0, 1 - Math.pow(i / 20, k)).toFixed(4) }));
export const perClass: Series[] = [
  { id: 'spread', label: 'spread', color: 'var(--chart-1)', points: curve(5), lower: curve(1.5) },
  ...[1.6, 2.2, 3, 4.4].map((k, i) => ({ id: `c${i}`, label: `class ${i}`, color: 'var(--chart-context)', context: true, points: curve(k) })),
  { id: 'mean', label: 'mean', color: 'var(--chart-1)', emphasis: true, points: curve(2.6) },
];

/** A benchmark's plot (XYChart): two entities' accuracy over a stress level, with intervals, references and a strip. */
const levels = [0, 0.25, 0.5, 0.75, 1];
export const xySeries: XYSeries[] = [
  {
    id: 'folo',
    label: 'folo-s',
    color: 'var(--chart-1)',
    dots: true,
    points: levels.map((x, i) => ({ x, y: +(0.82 - x * 0.3).toFixed(3), lo: +(0.78 - x * 0.31).toFixed(3), hi: +(0.86 - x * 0.29).toFixed(3), hollow: i === 4 })),
  },
  { id: 'yolo', label: 'yolo26n', color: 'var(--chart-2)', pattern: 'dot', points: levels.map((x, i) => ({ x, y: i === 2 ? null : +(0.76 - x * 0.4).toFixed(3) })) },
  { id: 'size', label: 'size', color: 'var(--text-tertiary)', thin: true, line: true, points: levels.map((x) => ({ x, y: +(0.3 + x * 0.2).toFixed(3), r: 2 + x * 4 })) },
];
export const xyRefs: XYRef[] = [{ axis: 'diagonal', label: 'y = x' }, { axis: 'y', value: 0.5, label: 'chance', below: true }, { axis: 'x', value: 0.6, label: 'threshold' }];
export const xyMarks: XYMark[] = [{ x: 0.5, y: 0.67, label: 'knee', side: 'above' }];
export const xyStrip: XYStrip = { items: levels.slice(0, 4).map((x, i) => ({ x0: x, x1: levels[i + 1]!, v: [120, 80, 45, 12][i]! })), unit: 'detections' };
export const xyTable = {
  caption: 'Accuracy by stress level',
  head: ['Level', 'folo-s', 'yolo26n'],
  rows: levels.map((x, i) => [String(x), String(xySeries[0]!.points[i]!.y), String(xySeries[1]!.points[i]!.y ?? '—')]),
};

/** Box sizes (Histogram): finer bins near the small end, an open last one. */
export const histEdges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
export const histCounts = [12, 48, 230, 410, 380, 150, 30];
export const histBin = (lo: number, hi: number) => (hi >= 1 ? `≥${lo * 100}%` : `${lo * 100}–${hi * 100}%`);

/** Confusion matrices: three classes, and nine (pages sideways on a phone). */
export const cm3 = {
  classes: ['bolt', 'nut', 'washer'],
  matrix: [
    [40, 2, 0, 3],
    [1, 30, 5, 2],
    [2, 5, 50, 3],
    [1, 0, 4, 0],
  ],
};
const names9 = ['bolt', 'nut', 'washer', 'screw', 'bearing', 'spring', 'pin', 'rivet', 'clip'];
export const cm9 = {
  classes: names9,
  matrix: Array.from({ length: 10 }, (_, r) => Array.from({ length: 10 }, (_, c) => (r === c ? (r === 9 ? 0 : 40 + r * 3) : (r * 7 + c * 3) % 11 < 3 ? (r + c) % 6 : 0))),
};

/** An image with detections (DetectionView): an inline SVG, so both twins load the same pixels with no network. */
export const sceneSrc = `data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#3a4250"/><circle cx="160" cy="180" r="70" fill="#c8a060"/><rect x="330" y="220" width="180" height="120" fill="#7088a8"/><circle cx="560" cy="80" r="8" fill="#e0e0e0"/></svg>',
)}`;
export const sceneBoxes = [
  { class: 0, x1: 90, y1: 110, x2: 230, y2: 250, score: 0.91 },
  { class: 1, x1: 330, y1: 220, x2: 510, y2: 340, score: 0.78 },
  { class: 2, x1: 551, y1: 71, x2: 569, y2: 89, score: 0.42 },
];
export const sceneTruth = [{ class: 1, x1: 320, y1: 212, x2: 515, y2: 345 }];
export const sceneClasses = ['bolt', 'nut', 'washer'];

/**
 * The API the sparklines and dataset images read, fixed and immediate (`hub.api` wrapped by the section that needs it,
 * when it renders): every run's mAP history is the same curve, image `ok` is the scene, `slow` never answers (the
 * skeleton), any other image fails (Image unavailable). Everything else is the hub's own API.
 */
export function galleryApi<A extends object>(base: A): A {
  const own: Record<string, unknown> = {
    runMetrics: async (id: string) => ({
      epochs: Array.from({ length: id === 'f' ? 6 : 20 }, (_, i) => ({ val: { map50_95: 0.5 * (1 - Math.exp(-(i + 1) / 5)) } })),
    }),
    imageUrl: (_: string, id: string) => (id === 'ok' ? Promise.resolve(sceneSrc) : id === 'slow' ? new Promise(() => {}) : Promise.reject(new Error('offline'))),
  };
  return new Proxy(base, { get: (t, k, r) => (typeof k === 'string' && k in own ? own[k] : Reflect.get(t, k, r)) });
}
