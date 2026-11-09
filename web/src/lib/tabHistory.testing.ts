// A tab's history for tests. happy-dom keeps one URL and no entries, so Back, Forward and a followed link cannot be
// tried with it; this stands in for the browser: `pushState` / `replaceState` / `go` work on a list of entries, and
// a move tells the page as a browser does (`popstate`, then `hashchange` when the hash changed), in a later task.
// Restore with `vi.restoreAllMocks()`.

import { vi } from 'vitest';

export interface TabHistory {
  /** A link is followed (or the URL edited): a new entry without state, and the browser says so. */
  follow(hash: string): void;
  back(): void;
  forward(): void;
  /** The hashes of the entries, oldest first, and which one the tab is on. */
  hashes(): string[];
  at(): number;
  /** Lets the moves under way happen. */
  settle(): Promise<void>;
}

export function tabHistory(): TabHistory {
  const push = history.pushState.bind(history);
  const replace = history.replaceState.bind(history);
  let entries: { url: string; state: unknown }[] = [{ url: location.href, state: history.state }];
  let at = 0;

  const add = (state: unknown, url: string | URL | null | undefined) => {
    push(state, '', url);
    entries = [...entries.slice(0, at + 1), { url: location.href, state }];
    at++;
  };
  vi.spyOn(history, 'pushState').mockImplementation((state, _unused, url) => add(state, url));
  vi.spyOn(history, 'replaceState').mockImplementation((state, _unused, url) => {
    replace(state, '', url ?? location.href);
    entries[at] = { url: location.href, state };
  });
  vi.spyOn(history, 'go').mockImplementation((delta = 0) => {
    const to = at + delta;
    if (to < 0 || to >= entries.length || delta === 0) return;
    setTimeout(() => {
      const before = location.hash;
      const entry = entries[(at = to)]!;
      replace(entry.state, '', entry.url);
      window.dispatchEvent(new PopStateEvent('popstate', { state: entry.state }));
      if (location.hash !== before) window.dispatchEvent(new HashChangeEvent('hashchange'));
    }, 0);
  });

  return {
    follow(hash) {
      add(null, hash);
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    },
    back: () => history.go(-1),
    forward: () => history.go(1),
    hashes: () => entries.map((e) => new URL(e.url).hash),
    at: () => at,
    settle: () => new Promise((done) => setTimeout(done, 20)),
  };
}
