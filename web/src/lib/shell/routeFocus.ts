// Focus on navigation. A new page (a path change) moves focus to its heading, or to the main region while the page
// has none yet (a detail page still loading), so keyboard and screen-reader users land on what changed. A change of
// only the query (a filter, a tab) is the same page: focus stays where it is.

/** True when `next` is another page than `prev` (paths differ; queries do not count). */
export function isNewPage(prev: string, next: string): boolean {
  return prev !== next;
}

/** Where focus lands on a new page: the element marked `data-route-focus` (Page's h1), else the main region. */
export function routeFocusTarget(main: HTMLElement): HTMLElement {
  return main.querySelector<HTMLElement>('[data-route-focus]') ?? main;
}

/** Moves focus to the new page and starts it at the top. Programmatic focus shows no ring unless the keyboard led here. */
export function focusRoute(main: HTMLElement): HTMLElement {
  const target = routeFocusTarget(main);
  target.focus({ preventScroll: true });
  if (typeof window !== 'undefined') window.scrollTo?.(0, 0);
  return target;
}

