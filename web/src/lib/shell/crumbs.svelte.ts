// Per-page chrome: breadcrumbs shown in the top bar, and whether the page owns a create action. Pages call
// `setCrumbs` (and `ownCreateAction`) from their script.

import { untrack } from 'svelte';

export interface Crumb {
  label: string;
  path?: string;
}

class PageMeta {
  crumbs = $state<Crumb[]>([]);
  /** Mounted pages with a create action of their own (a count, so the order pages mount and unmount in never matters). */
  creators = $state(0);
  /** The top bar's New run steps aside: the page's own create action is the one in that corner. */
  get ownsCreate() {
    return this.creators > 0;
  }
}

export const page = new PageMeta();

/** Sets the breadcrumbs for the current page and the document title. */
export function setCrumbs(...crumbs: Crumb[]) {
  page.crumbs = crumbs;
  if (typeof document !== 'undefined') {
    const last = crumbs.at(-1)?.label;
    document.title = last ? `${last} · FOLO Studio` : 'FOLO Studio';
  }
}

/**
 * Called from the script of a page whose header holds a create action (Runs' New run, Datasets' Add dataset): while it
 * is mounted the top bar hides its own New run, so one corner never holds two create buttons (or, offline, two disabled
 * ones with the reason between them).
 */
export function ownCreateAction() {
  // Untracked: the effect writes the count, so reading it as a dependency would re-run it forever.
  $effect(() => {
    untrack(() => page.creators++);
    return () => {
      untrack(() => page.creators--);
    };
  });
}
