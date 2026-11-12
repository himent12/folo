// Focus on navigation. A new page (a path change) moves focus to its heading, or to the main region while the page
// has none yet (a detail page still loading), so keyboard and screen-reader users land on what changed. A page whose
// URL asks for a part of it (Settings' `?section=units`) scrolls there itself and marks that part's heading
// `data-route-landing`: focus lands there and the scroll is left where the page put it, not reset to the top. A change of only the query (a filter, a tab) is the same
// page: focus stays where it is.

import { restoringScroll } from './entryMemory';

/** True when `next` is another page than `prev` (paths differ; queries do not count). */
export function isNewPage(prev: string, next: string): boolean {
  return prev !== next;
}

/**
 * Where focus lands on a new page: the part its URL asks for (`data-route-landing`), else the element marked
 * `data-route-focus` (Page's h1), else the main region.
 */
export function routeFocusTarget(main: HTMLElement): HTMLElement {
  return main.querySelector<HTMLElement>('[data-route-landing]') ?? main.querySelector<HTMLElement>('[data-route-focus]') ?? main;
}

/**
 * Marks `el` (a heading, focusable) as where the page lands (see the top): the one such mark in the document. Call as
 * soon as it is rendered, and scroll to it; the route focus that follows keeps to it instead of resetting the page to
 * its top.
 */
export function landOn(el: HTMLElement) {
  for (const old of document.querySelectorAll('[data-route-landing]')) if (old !== el) old.removeAttribute('data-route-landing');
  el.setAttribute('data-route-landing', '');
}

/**
 * Moves focus to the new page and starts it at the top, unless Back / Forward is bringing it back to where it was
 * (./entryMemory) or the page lands on a part of it (`landOn`: it scrolled there). The title takes no ring
 * (shell/Page): it is not a control; nor does a landing heading.
 */
export function focusRoute(main: HTMLElement): HTMLElement {
  const target = routeFocusTarget(main);
  const landing = target.hasAttribute('data-route-landing');
  target.focus({ preventScroll: true, ...(landing ? { focusVisible: false } : {}) } as FocusOptions);
  if (typeof window !== 'undefined' && !restoringScroll() && !landing) window.scrollTo?.(0, 0);
  return target;
}

/**
 * `focusRoute`, then keeps the focus on the page while it settles: a detail page swaps its loading heading for the
 * loaded one (the focused element is removed, focus falls to the body), so the new heading takes focus, as long as
 * the user has not moved focus anywhere else. Stops on the user's own focus, after `ms`, or when the returned
 * cleanup runs (the next navigation).
 */
export function holdRouteFocus(main: HTMLElement, ms = 15000): () => void {
  let target = focusRoute(main);
  const lost = () => {
    const a = document.activeElement;
    return !target.isConnected && (a === null || a === document.body || a === main);
  };
  const observer = new MutationObserver(() => {
    if (!lost()) return;
    target = routeFocusTarget(main);
    target.focus({ preventScroll: true });
  });
  observer.observe(main, { childList: true, subtree: true });
  // Focus the user moves elsewhere ends the hold; focus we set on the page's heading does not.
  const onFocus = (e: FocusEvent) => {
    if (e.target !== target && e.target !== main) stop();
  };
  document.addEventListener('focusin', onFocus);
  const timer = setTimeout(() => stop(), ms);
  function stop() {
    observer.disconnect();
    document.removeEventListener('focusin', onFocus);
    clearTimeout(timer);
  }
  return stop;
}

