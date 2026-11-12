// The Playground's model went away while it was open (its run was deleted, its file removed), or a link names one that
// is not there: the picker falls back to the best model, and the page says so once instead of switching without a
// word.

import { effect, untrack } from 'fw';
import type { Model } from '$lib/api/types';
import { toast } from '$lib/ui/toast';

/** The model that was shown and is gone from `list` now that `next` shows instead; null when nothing vanished. */
export function vanished(prev: Model | null, next: Model | null, list: readonly Model[]): Model | null {
  if (!prev || !next || prev.id === next.id || list.some((m) => m.id === prev.id)) return null;
  return prev;
}

/** The model a link (or a pick) asks for when `list` has no such model: its id; null when it is there or none is asked. */
export function missing(wanted: string | null, list: readonly Model[]): string | null {
  return wanted && list.length && !list.some((m) => m.id === wanted) ? wanted : null;
}

/**
 * Follows the shown model (`model`, picked from `list` by `wanted`): when it vanishes, an info toast names both models;
 * when the link names one that is not there (deleted before it was opened, or a wrong id), a toast names the one shown
 * instead. Either way `adopt` makes the replacement the chosen one (so the URL stops naming a model that does not
 * exist). Call in a component.
 */
export function followVanished(model: () => Model | null, list: () => readonly Model[], wanted: () => string | null, adopt: (id: string) => void): void {
  let shown: Model | null = null;
  effect(() => {
    const next = model();
    const all = list();
    const asked = wanted();
    untrack(() => {
      const gone = vanished(shown, next, all);
      shown = next;
      if (!next) return;
      if (gone) toast.info(`${gone.name} no longer exists`, `Its run or file was deleted. Showing ${next.name} instead.`);
      else if (missing(asked, all)) toast.info('The linked model was not found', `It was deleted or the link is wrong. Showing ${next.name} instead.`);
      else return;
      adopt(next.id);
    });
  });
}
