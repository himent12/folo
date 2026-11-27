// Mock datasets: four procedurally generated detection datasets (one with no validation split). Every image (size,
// boxes, classes) is a pure function of (dataset, index), so the grid, stats, samples and predictions agree.

import type { Box, Dataset, DatasetImage, DatasetIssue, DatasetStats, Split } from '../types';
import { gauss, hash, range, rng, weighted } from './rng';

export type Shape = 'round' | 'square' | 'long';

export interface MockDatasetSpec {
  id: string;
  name: string;
  path: string;
  format: 'yolo' | 'coco';
  classes: string[];
  /** Relative frequency of each class. */
  classWeights: number[];
  /** Geometry of each class's boxes. */
  classShape: Shape[];
  /** Explicit rendered shape glyph per class when generated class names include custom labels or varied casing. */
  renderShapes?: readonly string[];
  splits: Record<Split, number>;
  /** Every image's size, unless `frames` gives each its own. */
  width: number;
  height: number;
  /**
   * Photos of several shapes (a labeling job's uploads from phones: landscape, portrait, a panorama): each image's
   * size picked from these by its own seed (by `weight`), its boxes placed within it. Absent: every image is
   * `width × height`.
   */
  frames?: { width: number; height: number; weight: number }[];
  /** Mean boxes per image and the probability that an image is empty. */
  meanBoxes: number;
  emptyRate: number;
  /** Relative box side, as a fraction of the shorter image side: [min, max]. */
  sizeRange: [number, number];
  brokenLabels: number;
  createdAgoS: number;
  /** How ./render draws it: coloured shapes, fasteners on a conveyor, or fish and frogs in a pond (a benchmark's). */
  style: 'shapes' | 'fasteners' | 'pond';
  /** Generation seed for synthetic datasets; built-in datasets derive theirs from the id. */
  seed?: number;
}

/** Base seed of a dataset's randomness: the requested seed for synthetic sets, else the id. */
export function seedOf(s: MockDatasetSpec): number {
  return s.seed !== undefined ? hash(`seed:${s.seed}`) : hash(s.id);
}

export const specs: MockDatasetSpec[] = [
  {
    id: 'ds_fasteners',
    name: 'fasteners',
    path: 'D:/data/fasteners',
    format: 'coco',
    classes: ['bolt', 'nut', 'washer', 'screw', 'spring', 'rivet', 'pin', 'bearing'],
    classWeights: [16, 18, 22, 14, 5, 11, 9, 3],
    classShape: ['long', 'round', 'round', 'long', 'long', 'round', 'long', 'round'],
    splits: { train: 1920, val: 360, test: 120 },
    width: 1024,
    height: 768,
    meanBoxes: 6.5,
    emptyRate: 0.01,
    sizeRange: [0.07, 0.3],
    brokenLabels: 0,
    createdAgoS: 86400 * 6 + 3600 * 3,
    style: 'fasteners',
  },
  {
    id: 'ds_shapes',
    name: 'shapes',
    path: 'D:/data/shapes',
    format: 'yolo',
    classes: ['circle', 'square', 'triangle'],
    classWeights: [1, 1, 1],
    classShape: ['round', 'square', 'square'],
    splits: { train: 800, val: 200, test: 0 },
    width: 640,
    height: 640,
    meanBoxes: 4.2,
    emptyRate: 0.012,
    sizeRange: [0.08, 0.42],
    brokenLabels: 0,
    createdAgoS: 86400 * 13,
    style: 'shapes',
  },
  {
    id: 'ds_shapes_hard',
    name: 'shapes-hard',
    path: 'D:/data/shapes-hard',
    format: 'yolo',
    classes: ['circle', 'square', 'triangle', 'star', 'hexagon', 'cross'],
    classWeights: [10, 9, 9, 4, 6, 2],
    classShape: ['round', 'square', 'square', 'square', 'round', 'square'],
    splits: { train: 4000, val: 800, test: 200 },
    width: 640,
    height: 640,
    meanBoxes: 9,
    emptyRate: 0.0074,
    sizeRange: [0.03, 0.3],
    brokenLabels: 3,
    createdAgoS: 86400 * 2 + 3600 * 5,
    style: 'shapes',
  },
  {
    // A quick hand-made set with no validation images: Health says so, and /eval of a model with its classes falls
    // back to its training split (docs/API.md, Split fallback; the Model page's `model-eval-train-fallback` scene).
    id: 'ds_fasteners_mini',
    name: 'fasteners-mini',
    path: 'D:/data/fasteners-mini',
    format: 'yolo',
    classes: ['bolt', 'nut'],
    classWeights: [11, 9],
    classShape: ['long', 'round'],
    splits: { train: 24, val: 0, test: 0 },
    width: 1024,
    height: 768,
    meanBoxes: 6,
    emptyRate: 0,
    sizeRange: [0.08, 0.3],
    brokenLabels: 0,
    createdAgoS: 86400 * 4 + 3600 * 7,
    style: 'fasteners',
  },
];

export function total(s: MockDatasetSpec) {
  return s.splits.train + s.splits.val + s.splits.test;
}

export function splitOf(s: MockDatasetSpec, index: number): Split {
  if (index < s.splits.train) return 'train';
  if (index < s.splits.train + s.splits.val) return 'val';
  return 'test';
}

export const imageId = (index: number) => `img_${String(index).padStart(6, '0')}`;
export const imageIndex = (id: string) => Number.parseInt(id.replace(/^img_/, ''), 10);

/** The size of the image at `index`: the spec's, or one of its `frames` by the image's own seed. */
export function frameOf(s: MockDatasetSpec, index: number): { width: number; height: number } {
  if (!s.frames?.length) return { width: s.width, height: s.height };
  const f = s.frames[weighted(rng(seedOf(s) ^ hash(`frame:${index}`)), s.frames.map((x) => x.weight))]!;
  return { width: f.width, height: f.height };
}

/** The image at `index`: boxes are placed with light overlap avoidance so labels stay readable. */
export function image(s: MockDatasetSpec, index: number): DatasetImage {
  const r = rng(seedOf(s) ^ Math.imul(index + 1, 2654435761));
  const { width: W, height: H } = frameOf(s, index);
  const split = splitOf(s, index);
  const sub = split === 'train' ? index : split === 'val' ? index - s.splits.train : index - s.splits.train - s.splits.val;
  const ext = s.format === 'coco' ? 'jpg' : 'png';
  const file = `images/${split}/${String(sub).padStart(6, '0')}.${ext}`;
  const boxes: Box[] = [];
  if (r() >= s.emptyRate) {
    const n = Math.max(1, Math.round(s.meanBoxes + gauss(r) * s.meanBoxes * 0.55));
    const short = Math.min(W, H);
    for (let tries = 0; boxes.length < n && tries < n * 8; tries++) {
      const cls = weighted(r, s.classWeights);
      const shape = s.classShape[cls] ?? 'square';
      // Log-uniform sizes: many small objects, fewer large ones.
      const [lo, hi] = s.sizeRange;
      const side = short * lo * Math.pow(hi / lo, r() ** 1.35);
      let w = side;
      let h = side;
      if (shape === 'long') {
        const aspect = range(r, 2.4, 3.8);
        if (r() < 0.5) {
          w = side * Math.sqrt(aspect);
          h = side / Math.sqrt(aspect);
        } else {
          h = side * Math.sqrt(aspect);
          w = side / Math.sqrt(aspect);
        }
      } else if (shape === 'square') {
        w = side * range(r, 0.85, 1.15);
      }
      w = Math.min(w, W * 0.9);
      h = Math.min(h, H * 0.9);
      const x1 = range(r, 4, W - w - 4);
      const y1 = range(r, 4, H - h - 4);
      const box = { class: cls, x1, y1, x2: x1 + w, y2: y1 + h };
      if (boxes.some((b) => iou(b, box) > 0.08)) continue;
      boxes.push(roundBox(box));
    }
  }
  return { id: imageId(index), file, split, width: W, height: H, boxes };
}

function roundBox<T extends Box>(b: T): T {
  return { ...b, x1: Math.round(b.x1), y1: Math.round(b.y1), x2: Math.round(b.x2), y2: Math.round(b.y2) };
}

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

// Relative box size = sqrt(box area / image area), in the server's bins (folo-data BOX_SIZE_EDGES, docs/API.md).
const SIZE_EDGES = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
const PER_IMAGE_EDGES = [0, 1, 2, 4, 8, 16, 32, 64];

function bin(edges: number[], v: number): number {
  for (let i = 0; i < edges.length - 1; i++) if (v < (edges[i + 1] ?? Infinity)) return i;
  return edges.length - 2;
}

// Caches are keyed by the spec object itself: ids can repeat across mock instances with different specs.
const statsCache = new WeakMap<MockDatasetSpec, DatasetStats>();

export function stats(s: MockDatasetSpec): DatasetStats {
  const hit = statsCache.get(s);
  if (hit) return hit;
  const class_counts = s.classes.map(() => 0);
  const size = new Array<number>(SIZE_EDGES.length - 1).fill(0);
  const per = new Array<number>(PER_IMAGE_EDGES.length - 1).fill(0);
  let boxes = 0;
  let empty = 0;
  for (let i = 0; i < total(s); i++) {
    const img = image(s, i);
    const area = img.width * img.height;
    if (img.boxes.length === 0) empty++;
    per[bin(PER_IMAGE_EDGES, img.boxes.length)]!++;
    for (const b of img.boxes) {
      boxes++;
      class_counts[b.class]!++;
      size[bin(SIZE_EDGES, Math.sqrt(((b.x2 - b.x1) * (b.y2 - b.y1)) / area))]!++;
    }
  }
  const st: DatasetStats = {
    boxes,
    empty_images: empty,
    broken_labels: s.brokenLabels,
    class_counts,
    box_size_hist: { edges: SIZE_EDGES, counts: size },
    boxes_per_image_hist: { edges: PER_IMAGE_EDGES, counts: per },
  };
  statsCache.set(s, st);
  return st;
}

export function toDataset(s: MockDatasetSpec, now: number, withStats: boolean): Dataset {
  return {
    id: s.id,
    name: s.name,
    path: s.path,
    format: s.format,
    classes: [...s.classes],
    created: new Date(now - s.createdAgoS * 1000).toISOString(),
    splits: { ...s.splits },
    ...(withStats ? { stats: stats(s) } : {}),
  };
}

/** Index of images of `split` that contain `cls`, built lazily per (dataset, split, class). */
const filterCache = new WeakMap<MockDatasetSpec, Map<string, number[]>>();

export function indices(s: MockDatasetSpec, split: Split | undefined, cls: number | undefined): number[] | null {
  if (split === undefined && cls === undefined) return null;
  const key = `${split ?? ''}|${cls ?? ''}`;
  let byKey = filterCache.get(s);
  if (!byKey) filterCache.set(s, (byKey = new Map()));
  const hit = byKey.get(key);
  if (hit) return hit;
  const out: number[] = [];
  const lo = split === undefined || split === 'train' ? 0 : split === 'val' ? s.splits.train : s.splits.train + s.splits.val;
  const hi = split === undefined ? total(s) : lo + s.splits[split];
  for (let i = lo; i < hi; i++) {
    if (cls === undefined || image(s, i).boxes.some((b) => b.class === cls)) out.push(i);
  }
  byKey.set(key, out);
  return out;
}


const issuesCache = new WeakMap<MockDatasetSpec, DatasetIssue[]>();

/**
 * `folo dataset check` findings for a mock dataset, consistent with its stats: one error per broken label file, an
 * `empty_image` info per image without boxes, and a few deterministic warnings, each with the cleanup `folo dataset
 * check --fix-dry-run` would suggest (none for infos). Dataset order (train, val, test).
 */
export function issues(s: MockDatasetSpec): DatasetIssue[] {
  const hit = issuesCache.get(s);
  if (hit) return hit;
  const r = rng(seedOf(s) ^ 0x51ed);
  const n = total(s);
  const labelOf = (file: string) => file.replace(/^images\//, 'labels/').replace(/\.[a-z]+$/, '.txt');
  const out: { i: number; issue: DatasetIssue }[] = [];
  const BROKEN: [DatasetIssue['kind'], (line: number) => string][] = [
    ['broken_label', (l) => `line ${l}: expected 5 values (class cx cy w h), found 4`],
    ['class_overflow', () => `class id ${s.classes.length + 2} is out of range (${s.classes.length} classes)`],
    ['out_of_range_box', () => 'cx 412.0 is outside [0, 1]: pixel coordinates instead of normalised'],
  ];
  for (let k = 0; k < s.brokenLabels; k++) {
    const i = Math.floor(r() * s.splits.train);
    const [kind, msg] = BROKEN[k % BROKEN.length]!;
    const line = 1 + Math.floor(r() * 9);
    out.push({ i, issue: { kind, severity: 'error', file: labelOf(image(s, i).file), line, message: msg(line), fix: `delete line ${line}` } });
  }
  for (let k = 0; k < 2 && n > 0; k++) {
    const i = Math.floor(r() * n);
    const line = 1 + Math.floor(r() * 6);
    out.push({ i, issue: { kind: 'clipped_box', severity: 'warning', file: labelOf(image(s, i).file), line, message: 'box extends 0.004 past the right edge; clipped on load', fix: `clip the box on line ${line} to the image` } });
  }
  if (n > 2) {
    const a = Math.floor(r() * (n - 1));
    const b = Math.min(n - 1, a + 1 + Math.floor(r() * (n - a - 1)));
    out.push({ i: b, issue: { kind: 'duplicate_image', severity: 'warning', file: image(s, b).file, line: null, message: `same bytes as ${image(s, a).file}`, fix: `remove this copy (and its label file); keep ${image(s, a).file}` } });
  }
  for (let i = 0; i < n; i++) {
    const img = image(s, i);
    if (img.boxes.length === 0) out.push({ i, issue: { kind: 'empty_image', severity: 'info', file: img.file, line: null, message: 'no boxes (a background image)' } });
  }
  const list = out.sort((x, y) => x.i - y.i).map((o) => o.issue);
  issuesCache.set(s, list);
  return list;
}
