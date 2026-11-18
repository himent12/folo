// Stylesheets of lazy chunks that failed to load (a dropped connection, a server restarting) are fetched again before
// the next chunk load, so Retry on a screen that could not load, the next screen that needs the same stylesheet and
// the demo's mock module all get their styles without a reload.
//
// Why the app does it: the bundler's preload helper (Vite's, in the built entry) adds a `<link rel=stylesheet>` for
// every stylesheet a lazy chunk needs and waits for it, but remembers every file it has asked for once that request
// settles, failed or not, and never asks for it again in this page. A stylesheet that failed once stayed missing: the
// screen showed unstyled, and with the shared stylesheet (vite.config.ts) every later screen too. A failed link applies
// nothing, so taking it out changes no pixel.

/** The stylesheets (absolute URLs) whose last load failed and that no load has brought in since. */
const failed = new Set<string>();
/** A refetch of each, while one is out. */
const refetching = new Map<string, Promise<void>>();

/** Starts watching `doc` for stylesheet links whose load fails (once per document). Returns a stop function (tests). */
export function watchFailedStyles(doc: Document = document): () => void {
  const onError = (e: Event) => {
    const el = e.target;
    if (!(el instanceof HTMLLinkElement) || el.rel !== 'stylesheet' || !el.href) return;
    failed.add(el.href);
    // The preload helper listens on the link itself and still hears this error after the link has left.
    el.remove();
  };
  // `error` does not bubble: caught on its way down.
  doc.addEventListener('error', onError, true);
  return () => {
    doc.removeEventListener('error', onError, true);
    failed.clear();
    refetching.clear();
  };
}

/**
 * Loads again every stylesheet whose load failed; resolves once all have loaded, rejects when one fails again (it stays
 * on the list for the next try). Called before a lazy chunk is imported (src/routes.ts, the mock in lib/api).
 */
export function refetchFailedStyles(doc: Document = document): Promise<void> {
  if (!failed.size) return Promise.resolve();
  return Promise.all(
    [...failed].map((href) => {
      let p = refetching.get(href);
      if (!p) {
        p = new Promise<void>((ok, no) => {
          const link = doc.createElement('link');
          link.rel = 'stylesheet';
          // As the preload helper links them (the same request, the same cache entry).
          link.crossOrigin = '';
          link.href = href;
          link.addEventListener('load', () => {
            failed.delete(href);
            ok();
          });
          // watchFailedStyles removes it and keeps the file on the list.
          link.addEventListener('error', () => no(new Error(`Unable to load the stylesheet ${href}`)));
          doc.head.appendChild(link);
        }).finally(() => refetching.delete(href));
        refetching.set(href, p);
      }
      return p;
    }),
  ).then(() => {});
}
