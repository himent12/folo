// The Playground's session for the tab: the image, its result and the thresholds outlive the page, so leaving it
// (a model page, the run, the sidebar) and coming back, Back / Forward included, finds them as they were. It is kept
// in memory (an uploaded file cannot go anywhere else); what can be named is also in the URL (`?model=…&dataset=…&
// image=…&split=…&at=…&conf=…&iou=…&max=…`), so a link or a reload restores a dataset image and the thresholds. What
// the URL names wins over what is in memory, field by field.

import { hub } from '$lib/api/index';
import type { Dataset, DatasetImage as Img, Prediction, Split } from '$lib/api/types';
import { DEFAULTS } from './predict';
import type { Source } from './source';

/** A source as kept: a file without its object URL (revoked with the page; a new one is made on the way back). */
export type KeptSource = { kind: 'file'; file: Blob; name: string; width: number; height: number } | { kind: 'dataset'; ds: Dataset; img: Img; at?: number };

export interface Session {
  source: KeptSource | null;
  /** The result for `source` (and its model, `pred.model_id`) on `runOn`: shown again without asking twice. */
  pred: Prediction | null;
  runOn: string;
  conf: number;
  iou: number;
  maxDet: number;
  labels: boolean;
  scores: boolean;
  truth: boolean;
  hidden: number[];
}

let kept: Session | null = null;

/** The tab's session, if the Playground has been open in it. */
export const session = {
  get: (): Session | null => kept,
  set: (s: Session | null) => void (kept = s),
};

/** A source as the session keeps it. */
export function keep(s: Source | null): KeptSource | null {
  if (!s) return null;
  return s.kind === 'file' ? { kind: 'file', file: s.file, name: s.name, width: s.width, height: s.height } : s;
}

/** A kept source made a source again (a file gets a fresh object URL). */
export function restore(s: KeptSource): Source {
  return s.kind === 'file' ? { ...s, url: URL.createObjectURL(s.file) } : s;
}

/** A dataset image as the URL names it. */
export interface ImageRef {
  dataset: string;
  image: string;
  split: Split;
  /** Its place in its split when picked: one request finds it again unless the dataset changed. */
  at: number | null;
}

/** What the URL says: a dataset image and thresholds (each absent when not named, or out of range). */
export interface FromUrl {
  image: ImageRef | null;
  conf: number | null;
  iou: number | null;
  maxDet: number | null;
}

const SPLITS: readonly string[] = ['train', 'val', 'test'];

function number(q: URLSearchParams, key: string, min: number, max: number): number | null {
  const raw = q.get(key);
  const v = raw === null || raw.trim() === '' ? NaN : Number(raw);
  return Number.isFinite(v) && v >= min && v <= max ? v : null;
}

export function fromUrl(q: URLSearchParams): FromUrl {
  const dataset = q.get('dataset');
  const image = q.get('image');
  const split = q.get('split') ?? 'val';
  const at = number(q, 'at', 0, Number.MAX_SAFE_INTEGER);
  return {
    image: dataset && image && SPLITS.includes(split) ? { dataset, image, split: split as Split, at: at === null ? null : Math.floor(at) } : null,
    conf: number(q, 'conf', 0.01, 0.95),
    iou: number(q, 'iou', 0.05, 0.95),
    maxDet: number(q, 'max', 1, 300),
  };
}

/** The Playground's URL for this state: the model, a dataset image, and the thresholds that differ from the defaults. */
export function playgroundPath(s: { model: string | null; source: Source | null; conf: number; iou: number; maxDet: number }): string {
  const q = new URLSearchParams();
  if (s.model) q.set('model', s.model);
  if (s.source?.kind === 'dataset') {
    q.set('dataset', s.source.ds.id);
    q.set('image', s.source.img.id);
    q.set('split', s.source.img.split);
    if (s.source.at !== undefined) q.set('at', String(s.source.at));
  }
  if (s.conf !== DEFAULTS.conf) q.set('conf', String(s.conf));
  if (s.iou !== DEFAULTS.iou) q.set('iou', String(s.iou));
  if (s.maxDet !== DEFAULTS.maxDet) q.set('max', String(s.maxDet));
  const qs = String(q);
  return `/playground${qs ? `?${qs}` : ''}`;
}

const PAGE = 200;
/** How far a link's image is looked for when it is not at its place any more (the dataset changed). */
const MAX_PAGES = 25;

/** The dataset image a link names, or null when it is gone (or the dataset is). */
export async function findImage(ref: ImageRef, signal?: AbortSignal): Promise<{ ds: Dataset; img: Img; at: number } | null> {
  const ds = await hub.api.dataset(ref.dataset, signal);
  if (ref.at !== null) {
    const page = await hub.api.datasetImages(ds.id, { split: ref.split, offset: ref.at, limit: 1 }, signal);
    const img = page.images[0];
    if (img?.id === ref.image) return { ds, img, at: ref.at };
  }
  for (let p = 0; p < MAX_PAGES; p++) {
    const page = await hub.api.datasetImages(ds.id, { split: ref.split, offset: p * PAGE, limit: PAGE }, signal);
    const i = page.images.findIndex((x) => x.id === ref.image);
    if (i >= 0) return { ds, img: page.images[i]!, at: p * PAGE + i };
    if ((p + 1) * PAGE >= page.total) break;
  }
  return null;
}
