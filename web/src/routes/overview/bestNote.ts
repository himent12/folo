// The foot of Overview's Best models: what the list leaves out, and why, in words a newcomer reads without the docs.

import { count } from '$lib/format';
import type { ScoreContext } from '$lib/scoreSummary';

export interface BestNote {
  /** Datasets whose best model is ranked but past the six shown. */
  more: number;
  /** Models of `imported` or `published` context: a file copied in, weights pulled; trained elsewhere. */
  elsewhere: number;
  /** Models of a run whose split is unknown (its run or dataset could not be read). */
  unknown: number;
  /** Why those are unknown: "evaluation split unknown", "could not load runs". */
  why: string;
}

/** Counts the models a ranking left out by why (scoreSummary `scoreContext`). */
export function unrankedKinds(contexts: readonly ScoreContext[]): Pick<BestNote, 'elsewhere' | 'unknown'> {
  const elsewhere = contexts.filter((c) => c.kind === 'imported' || c.kind === 'published').length;
  return { elsewhere, unknown: contexts.length - elsewhere };
}

/**
 * The note, one short sentence per reason, or '' when nothing is left out: "3 more datasets in All models.", "2 models
 * trained elsewhere (imported or downloaded) are not ranked.", "1 model not ranked: evaluation split unknown."
 */
export function bestNote(n: BestNote): string {
  return [
    n.more ? `${count(n.more, 'more dataset')} in All models.` : '',
    n.elsewhere ? `${count(n.elsewhere, 'model')} trained elsewhere (imported or downloaded) ${n.elsewhere === 1 ? 'is' : 'are'} not ranked.` : '',
    n.unknown ? `${count(n.unknown, 'model')} not ranked: ${n.why}.` : '',
  ]
    .filter(Boolean)
    .join(' ');
}
