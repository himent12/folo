// Datasets added or removed in this tab, so a list on screen (or the next one shown) reads them again.

import { signal } from 'fw';

/** Bumped by every dataset action that changes the list (register, generate, remove). */
export const datasetChanges = signal(0);

/** Says the list of datasets changed. */
export function datasetsChanged() {
  datasetChanges.set(datasetChanges.peek() + 1);
}
