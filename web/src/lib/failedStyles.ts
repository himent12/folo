// Stylesheets of lazy chunks that failed to load (a dropped connection, a server restarting) are asked for again when
// the next screen or the demo's mock module loads: best effort, never in the way.
//
// Why: the bundler's preload helper (Vite's, in the built entry) adds a `<link rel=stylesheet>` for every stylesheet a
// lazy chunk needs and waits for it, but remembers every file it has asked for once that request settles, failed or not,
// and never asks for it again in this page. On its own, a stylesheet that failed once stays missing until a reload:
// Retry shows the screen without it (the helper skips it), and so does every later screen that needs it.
//
// The rules:
// - The failed link stays where the helper put it (a failed link applies nothing). A retry is a copy of that link,
//   every attribute included, put in its place: the stylesheets keep the order a clean load gives them, so rules of
//   equal specificity resolve as they would have.
// - Nobody waits for a retry: the screen or the mock loads exactly as it would without this module, and the stylesheet
//   applies when it arrives. A retry that fails leaves its copy where the link was and is tried again next time.

/** The link of each stylesheet whose last load failed (by URL), until a load of it succeeds. */
const failed = new Map<string, HTMLLinkElement>();
/** The retry of each, while one is out. */
const retrying = new Map<string, Promise<void>>();

const isSheet = (el: EventTarget | null): el is HTMLLinkElement =>
  el instanceof HTMLLinkElement && el.rel === 'stylesheet' && !!el.href;

/** Starts watching `doc` for stylesheet links whose load fails (once per document). Returns a stop function (tests). */
export function watchFailedStyles(doc: Document = document): () => void {
  // `error` and `load` do not bubble: caught on their way down.
  const onError = (e: Event) => {
    if (isSheet(e.target)) failed.set(e.target.href, e.target);
  };
  const onLoad = (e: Event) => {
    if (isSheet(e.target)) failed.delete(e.target.href);
  };
  doc.addEventListener('error', onError, true);
  doc.addEventListener('load', onLoad, true);
  return () => {
    doc.removeEventListener('error', onError, true);
    doc.removeEventListener('load', onLoad, true);
    failed.clear();
    retrying.clear();
  };
}

/**
 * Asks again for every stylesheet whose last load failed, each in the place of its failed link and with its
 * attributes. Never rejects; resolves once each retry has loaded or failed (tests). Callers start it and do not wait:
 * `src/routes.ts` before a screen's chunk, `lib/api` before the mock's.
 */
export function retryFailedStyles(): Promise<void> {
  if (!failed.size) return Promise.resolve();
  return Promise.all(
    [...failed].map(([href, dead]) => {
      let p = retrying.get(href);
      if (p) return p;
      // Taken out by something else: there is no place left to put it back in.
      if (!dead.isConnected) {
        failed.delete(href);
        return Promise.resolve();
      }
      p = new Promise<void>((done) => {
        const link = dead.cloneNode(false) as HTMLLinkElement;
        // watchFailedStyles hears the outcome first (capture): a failure puts this link on the list in place of `dead`.
        link.addEventListener('load', () => done(), { once: true });
        link.addEventListener('error', () => done(), { once: true });
        dead.replaceWith(link);
      }).finally(() => retrying.delete(href));
      retrying.set(href, p);
      return p;
    }),
  ).then(() => {});
}
