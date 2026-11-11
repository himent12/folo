// New run's draft: what was typed, kept for the tab's session (sessionStorage), so leaving the form and coming back,
// Back / Forward and a reload find it as it was. One draft per tab; a successful start clears it.
//
// The draft remembers the link it was made under (`query`). Opening that same link again (Back, a reload) restores
// the draft whole, since it already holds what the link set plus the edits since. A different link (`?dataset=`,
// `?from=` a run, `?imgsz=`) wins for the fields it sets; the draft fills in the rest.

import type { Arch, Augment, Optimizer } from '$lib/api/types';

const KEY = 'folo.runNew.draft';

/** Run on's choice, as `CloudPick` holds it. */
export interface CloudDraft {
  target: string;
  offerId: string | null;
  offerChosen: boolean;
  hours: number | null;
  hoursEdited: boolean;
}

export interface Draft {
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
  seed: number | null;
  weights: string;
  name: string;
  nameEdited: boolean;
  augment: Augment;
  closeEdited: boolean;
  advancedOpen: boolean;
  augOpen: boolean;
  cloud: CloudDraft;
}

/** The tab's draft, or null (none, unreadable, or storage off). */
export function readDraft(store: () => Storage = () => sessionStorage): Draft | null {
  try {
    const raw = store().getItem(KEY);
    const d = raw ? (JSON.parse(raw) as Partial<Draft>) : null;
    // A draft from another version of the form (or hand-edited) is dropped rather than half applied.
    return d && typeof d.query === 'string' && typeof d.name === 'string' && typeof d.arch === 'string' && d.augment && d.cloud ? (d as Draft) : null;
  } catch {
    return null;
  }
}

export function saveDraft(d: Draft, store: () => Storage = () => sessionStorage) {
  try {
    store().setItem(KEY, JSON.stringify(d));
  } catch {
    // Storage full or off: the form still works, it is just not kept.
  }
}

export function clearDraft(store: () => Storage = () => sessionStorage) {
  try {
    store().removeItem(KEY);
  } catch {
    // As above.
  }
}
