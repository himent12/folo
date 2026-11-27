// The Playground's session for the tab: the image, its result and the thresholds outlive the page, so leaving it
// (a model page, the run, the sidebar) and coming back, Back / Forward included, finds them as they were. It is kept
// in memory (an uploaded file cannot go anywhere else); what can be named is also in the URL (`?model=…&dataset=…&
// image=…&split=…&at=…&conf=…&iou=…&max=…`), so a link or a reload restores a dataset image and the thresholds. What
// the URL names wins over what is in memory, field by field, and over what is shown when the URL changes while the
// page is open (./keepSession). It belongs to the server it was made against (`hub.server`): another server, or the
// mock for a live one, starts without it.

import { hub } from '$lib/api/index';
import type { Dataset, DatasetImage as Img, Prediction, Split } from '$lib/api/types';
import type { InferenceConfig } from '$lib/inference/browser';
import type { CompareViewMode } from './compare';
import { normalizePerClassConf, roundConf, type PerClassConfMap } from './perClassConf';
import { DEFAULTS } from './predict';
import type { Source } from './source';

/** A source as kept: a file without its object URL (revoked with the page; a new one is made on the way back). */
export type KeptSource = { kind: 'file'; file: Blob; name: string; width: number; height: number } | { kind: 'dataset'; ds: Dataset; img: Img; at?: number };

export interface Session {
  source: KeptSource | null;
  /** The result for `source` (and its model, `pred.model_id`) on `runOn`: shown again without asking twice. */
  pred: Prediction | null;
  runOn: string;
  /** Active inference options (standard / SAHI / TTA) for the tab. */
  inference?: InferenceConfig;
  conf: number;
  iou: number;
  maxDet: number;
  perClassConf?: PerClassConfMap;
  /** Model ID and class list that `perClassConf` indices belonged to (for safe cross-model remapping on restore). */
  perClassModelId?: string | null;
  perClassClasses?: string[];
  labels: boolean;
  scores: boolean;
  truth: boolean;
  hidden: number[];
}

let kept: { server: string; session: Session } | null = null;

/** The tab's session, if the Playground has been open in it against the server in use. */
export const session = {
  get: (): Session | null => (kept?.server === hub.server ? kept.session : null),
  set: (s: Session | null) => void (kept = s && { server: hub.server, session: s }),
};

/** One-shot calibrated thresholds handoff staged by Model Detail (`Apply in Playground`). */
export interface ThresholdHandoff {
  modelId: string;
  modelName?: string;
  globalConf: number;
  perClassConf: PerClassConfMap;
  classes: string[];
}

/** How long a staged Model Detail handoff stays valid before expiring. */
export const HANDOFF_TTL_MS = 30_000;

let stagedHandoff: { server: string; stagedAt: number; handoff: ThresholdHandoff } | null = null;

/** Stages calibrated thresholds for the next Playground mount/navigation on this server. */
export function stageThresholdHandoff(h: ThresholdHandoff): void {
  stagedHandoff = {
    server: hub.server,
    stagedAt: Date.now(),
    handoff: {
      ...h,
      globalConf: roundConf(h.globalConf),
      perClassConf: normalizePerClassConf(h.perClassConf),
      classes: [...h.classes],
    },
  };
}

/**
 * Consumes a staged calibrated threshold handoff matching `modelId`. Always clears any pending handoff
 * (even on server mismatch, TTL expiry, or mismatched/absent modelId) so an abandoned handoff can never
 * hijack a later unrelated Playground visit.
 */
export function consumeThresholdHandoff(modelId: string | null): ThresholdHandoff | null {
  const pending = stagedHandoff;
  stagedHandoff = null;
  if (!pending) return null;
  if (pending.server !== hub.server) return null;
  if (Date.now() - pending.stagedAt > HANDOFF_TTL_MS) return null;
  if (!modelId || pending.handoff.modelId !== modelId) return null;
  return pending.handoff;
}

const MAX_PCC_URL_LEN = 1200;
const CANONICAL_CLASS_ID = /^(?:0|[1-9]\d*)$/;

/** Serializes `perClassConf` into a compact `0:0.42,1:0.35` query parameter, or null when empty or too long. */
export function serializePerClassParam(map: Readonly<Record<number, number>> | null | undefined): string | null {
  const norm = normalizePerClassConf(map);
  const ids = Object.keys(norm)
    .map(Number)
    .sort((a, b) => a - b);
  if (!ids.length) return null;
  const encoded = ids.map((id) => `${id}:${norm[id]!}`).join(',');
  return encoded.length <= MAX_PCC_URL_LEN ? encoded : null;
}

/**
 * Parses `pcc` from URL query parameters:
 * - `null` when `pcc` is absent or contains only malformed tokens;
 * - `{}` when `pcc=` is explicitly present and empty (clears any saved per-class overrides on mount);
 * - `{ [id]: conf }` when one or more valid `id:conf` entries are present.
 */
export function perClassFromUrl(q: URLSearchParams): PerClassConfMap | null {
  if (!q.has('pcc')) return null;
  const raw = q.get('pcc') ?? '';
  if (!raw.trim()) return {};
  const out: PerClassConfMap = {};
  for (const token of raw.split(',')) {
    const trimmed = token.trim();
    if (!trimmed) continue;
    const colon = trimmed.indexOf(':');
    if (colon <= 0) continue;
    const k = trimmed.slice(0, colon).trim();
    const vStr = trimmed.slice(colon + 1).trim();
    if (!CANONICAL_CLASS_ID.test(k) || !vStr) continue;
    const id = Number(k);
    const v = Number(vStr);
    if (!Number.isSafeInteger(id) || id > 0xffff_ffff || !Number.isFinite(v) || v < 0 || v > 1) continue;
    out[id] = roundConf(v);
  }
  return Object.keys(out).length ? out : null;
}

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

/** What the URL says: a model, a dataset image and thresholds (each absent when not named, or out of range). */
export interface FromUrl {
  model: string | null;
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
    model: q.get('model') || null,
    image: dataset && image && SPLITS.includes(split) ? { dataset, image, split: split as Split, at: at === null ? null : Math.floor(at) } : null,
    conf: number(q, 'conf', 0, 1),
    iou: number(q, 'iou', 0.05, 0.95),
    maxDet: number(q, 'max', 1, 300),
  };
}

/** Compare A/B URL parameters (absent in single-model URLs). */
export interface CompareFromUrl {
  compare: boolean;
  modelB: string | null;
  view: CompareViewMode | null;
}

export function compareFromUrl(q: URLSearchParams): CompareFromUrl {
  const compare = q.get('compare') === '1' || q.get('compare') === 'true';
  const modelB = q.get('modelB') || null;
  const rawView = q.get('view');
  const view: CompareViewMode | null = rawView === 'split' || rawView === 'side-by-side' ? rawView : null;
  return { compare, modelB, view };
}

/** The Playground's URL for this state: the model, a dataset image, and the thresholds that differ from the defaults. */
export function playgroundPath(s: {
  model: string | null;
  source: Source | null;
  imageRef?: ImageRef | null;
  conf: number;
  iou: number;
  maxDet: number;
  perClassConf?: Readonly<Record<number, number>> | null;
  /** Keep `conf` in the URL even when equal to `DEFAULTS.conf` (so threshold links never inherit unrelated saved conf). */
  explicitConf?: boolean;
  /** Keep `pcc=` in the URL even when `perClassConf` is empty (so threshold links explicitly clear saved overrides). */
  explicitPerClass?: boolean;
  compare?: boolean;
  modelB?: string | null;
  view?: CompareViewMode;
}): string {
  const q = new URLSearchParams();
  if (s.model) q.set('model', s.model);
  if (s.compare) {
    q.set('compare', '1');
    if (s.modelB) q.set('modelB', s.modelB);
    if (s.view && s.view !== 'side-by-side') q.set('view', s.view);
  }
  if (s.imageRef) {
    q.set('dataset', s.imageRef.dataset);
    q.set('image', s.imageRef.image);
    q.set('split', s.imageRef.split);
    if (s.imageRef.at !== null) q.set('at', String(s.imageRef.at));
  } else if (s.source?.kind === 'dataset') {
    q.set('dataset', s.source.ds.id);
    q.set('image', s.source.img.id);
    q.set('split', s.source.img.split);
    if (s.source.at !== undefined) q.set('at', String(s.source.at));
  }
  const pcc = serializePerClassParam(s.perClassConf);
  if (s.conf !== DEFAULTS.conf || s.explicitConf || pcc !== null) q.set('conf', String(s.conf));
  if (s.iou !== DEFAULTS.iou) q.set('iou', String(s.iou));
  if (s.maxDet !== DEFAULTS.maxDet) q.set('max', String(s.maxDet));
  if (pcc !== null) q.set('pcc', pcc);
  else if (s.explicitPerClass) q.set('pcc', '');
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
