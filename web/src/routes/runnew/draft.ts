// New run's draft: what was typed, kept for the tab's session (sessionStorage), so leaving the form and coming back,
// Back / Forward and a reload find it as it was. One draft per tab; a successful start clears it.
//
// The draft remembers the link it was made under (`query`). Opening that same link again (Back, a reload) restores
// the draft whole, since it already holds what the link set plus the edits since. A different link (`?dataset=`,
// `?from=` a run, `?imgsz=`) wins for the fields it sets; the draft fills in the rest. It also remembers the server
// it was made against (`server`, as `hub.server` names it): against another one, or the mock for a live one, its
// datasets, weights and cloud offers mean nothing, so it is dropped.

import type { Arch, Augment, Optimizer } from '$lib/api/types';

const KEY = 'folo.runNew.draft';
const ARCHS: readonly string[] = ['folo-n', 'folo-s', 'folo-m', 'folo-l', 'folo-x'] satisfies Arch[];
const OPTIMIZERS: readonly string[] = ['auto', 'sgd', 'adamw', 'musgd'] satisfies Optimizer[];

/** Run on's choice, as `CloudPick` holds it. */
export interface CloudDraft {
  target: string;
  offerId: string | null;
  offerChosen: boolean;
  hours: number | null;
  hoursEdited: boolean;
}

export interface Draft {
  /** The server the draft was made against (`hub.server`). */
  server: string;
  /** The link the draft was made under (the route's query, as a string). */
  query: string;
  datasetId: string;
  arch: Arch;
  /** Number inputs bind `null` when cleared: kept as typed, so the field says what is wrong again. */
  epochs: number | null;
  batch: string;
  imgsz: string;
  imgszChosen: boolean;
  device: string;
  optimizer: Optimizer;
  amp: boolean;
  e2e: boolean;
  lr0: number | null;
  /** As typed (lib/seed): checked as text, never rounded by a number field. */
  seed: string;
  /** The seed was left empty because the copied run's is over what Studio holds exactly (runnew/jump `copiedSeed`). */
  seedOver: boolean;
  weights: string;
  name: string;
  nameEdited: boolean;
  augment: Augment;
  closeEdited: boolean;
  advancedOpen: boolean;
  augOpen: boolean;
  cloud: CloudDraft;
}

type Fields = Record<string, unknown>;
const isObject = (v: unknown): v is Fields => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
/** A number field as the form keeps it: a number as typed (its error says what is wrong), or `null` when cleared. */
const typed = (v: unknown) => v === null || finite(v);
const strings = (d: Fields, keys: string[]) => keys.every((k) => typeof d[k] === 'string');
const flags = (d: Fields, keys: string[]) => keys.every((k) => typeof d[k] === 'boolean');

/** Augmentation the trainer accepts (folo-data `Augment::validate`); close mosaic, as typed. */
function isAugment(a: unknown): a is Augment {
  if (!isObject(a)) return false;
  const prob = (v: unknown) => finite(v) && v >= 0 && v <= 1;
  const gain = (v: unknown) => finite(v) && v >= 0;
  const hsv = Array.isArray(a.hsv) && a.hsv.length === 3 && a.hsv.every(gain);
  const scale = finite(a.scale) && a.scale >= 0 && a.scale < 1;
  return [a.mosaic, a.mixup, a.fliplr].every(prob) && gain(a.translate) && scale && hsv && typed(a.close_mosaic);
}

function isCloud(c: unknown): c is CloudDraft {
  if (!isObject(c)) return false;
  return typeof c.target === 'string' && (c.offerId === null || typeof c.offerId === 'string') && typed(c.hours) && flags(c, ['offerChosen', 'hoursEdited']);
}

/** Every field of a draft, of its type and in its range: what the form can restore without a check of its own. */
export function isDraft(d: unknown): d is Draft {
  return (
    isObject(d) &&
    strings(d, ['server', 'query', 'datasetId', 'seed', 'batch', 'imgsz', 'device', 'weights', 'name']) &&
    flags(d, ['imgszChosen', 'seedOver', 'amp', 'e2e', 'nameEdited', 'closeEdited', 'advancedOpen', 'augOpen']) &&
    ARCHS.includes(d.arch as string) &&
    OPTIMIZERS.includes(d.optimizer as string) &&
    [d.epochs, d.lr0].every(typed) &&
    isAugment(d.augment) &&
    isCloud(d.cloud)
  );
}

/**
 * The tab's draft made against `server`, or null (none, unreadable, or storage off). A draft from another version of
 * the form (or hand-edited), or made against another server, is dropped whole rather than half applied, and removed,
 * so the form opens with its defaults.
 */
export function readDraft(server: string, store: () => Storage = () => sessionStorage): Draft | null {
  let raw: string | null;
  try {
    raw = store().getItem(KEY);
  } catch {
    return null;
  }
  if (!raw) return null;
  try {
    const d: unknown = JSON.parse(raw);
    if (isDraft(d) && d.server === server) return d;
  } catch {
    // Not JSON: dropped as below.
  }
  clearDraft(store);
  return null;
}

/** Keeps `d` for the tab; returns it as stored, so the form can later clear only its own draft (`clearOwnDraft`). */
export function saveDraft(d: Draft, store: () => Storage = () => sessionStorage): string {
  const raw = JSON.stringify(d);
  try {
    store().setItem(KEY, raw);
  } catch {
    // Storage full or off: the form still works, it is just not kept.
  }
  return raw;
}

/**
 * Clears the tab's draft only while it is still `raw`, the one a form wrote: a start answered after the user left and
 * opened New run again must not take the newer form's edits with it (one draft per tab).
 */
export function clearOwnDraft(raw: string, store: () => Storage = () => sessionStorage) {
  try {
    if (store().getItem(KEY) === raw) store().removeItem(KEY);
  } catch {
    // As below.
  }
}

export function clearDraft(store: () => Storage = () => sessionStorage) {
  try {
    store().removeItem(KEY);
  } catch {
    // As above.
  }
}
