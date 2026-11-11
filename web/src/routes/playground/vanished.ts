// The Playground's model went away while it was open (its run was deleted, its file removed): the picker falls back
// to the next model, and the page says so once instead of switching without a word.

import { effect, untrack } from 'fw';
import type { Model } from '$lib/api/types';
import { toast } from '$lib/ui/toast';

/** The model that was shown and is gone from `list` now that `next` shows instead; null when nothing vanished. */
export function vanished(prev: Model | null, next: Model | null, list: readonly Model[]): Model | null {
  if (!prev || !next || prev.id === next.id || list.some((m) => m.id === prev.id)) return null;
  return prev;
}

/**
 * Follows the shown model (`model`, picked from `list`): when it vanishes, an info toast names both models and `adopt`
 * makes the replacement the chosen one (so the URL stops naming a model that does not exist). Call in a component.
 */
export function followVanished(model: () => Model | null, list: () => readonly Model[], adopt: (id: string) => void): void {
  let shown: Model | null = null;
  effect(() => {
    const next = model();
    const all = list();
    untrack(() => {
      const gone = vanished(shown, next, all);
      shown = next;
      if (!gone || !next) return;
      toast.info(`${gone.name} no longer exists`, `Its run or file was deleted. Showing ${next.name} instead.`);
      adopt(next.id);
    });
  });
}
