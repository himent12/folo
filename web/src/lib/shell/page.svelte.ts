// Per-page chrome: breadcrumbs shown in the top bar. Pages call `setCrumbs` from their script.

export interface Crumb {
  label: string;
  path?: string;
}

class PageMeta {
  crumbs = $state<Crumb[]>([]);
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
