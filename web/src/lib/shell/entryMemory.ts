// What Studio keeps with each history entry, so Back and Forward return to where the user was (shell F3).
//
// - Scroll: the page's scroll position is recorded per entry (`router.entry`, the entry's place in the tab's history)
//   as the user scrolls. A new navigation starts at the top (routeFocus); Back / Forward to an entry scrolls it back to
//   where it was once the screen is tall enough for it, that is once its data has rendered (a list, a detail page, the
//   dataset's virtualised grid, whose height is set by its image count). A press, a wheel turn or a key during that wait
//   leaves the page where the user puts it. A reload restores the entry's position the same way.
// - `entryState(key)`: any small value a screen wants back with its entry (typed list filters, a selection), the same
//   way: `const picked = entryState<string[]>('runs.compare')`, then `picked.get()` when the screen mounts and
//   `picked.set(ids)` when it changes. It belongs to the entry the screen is on at that moment; a new entry starts
//   without it, Back / Forward give it back. State that should survive a shared link belongs in the URL query instead.
//
// Both live in memory and in sessionStorage (written when the tab is hidden or closed), so a reload keeps them.

import { router, type EntryChange } from '$lib/router';

const KEY = 'folo.entries';
/** How long a restore waits for the screen to be tall enough before it settles for the nearest position. */
export const RESTORE_MS = 3000;
/** Once there, how long the page's height must stay put before the restore lets go (data still arriving moves it). */
const SETTLE_MS = 400;

type Kept = { y?: number; state?: Record<string, unknown> };

let kept = new Map<number, Kept>();
let restoring: { until: number; frame: number } | null = null;

function load() {
  try {
    const raw = sessionStorage.getItem(KEY);
    kept = raw ? new Map((JSON.parse(raw) as [number, Kept][]).filter(([p]) => typeof p === 'number')) : new Map();
  } catch {
    kept = new Map();
  }
}

function persist() {
  try {
    sessionStorage.setItem(KEY, JSON.stringify([...kept]));
  } catch {
    // Storage full or blocked: memory still serves this page's life.
  }
}

const entry = (pos: number): Kept => {
  let k = kept.get(pos);
  if (!k) kept.set(pos, (k = {}));
  return k;
};

/** True while a Back / Forward is bringing the page back to its position (routeFocus then leaves the scroll alone). */
export function restoringScroll(): boolean {
  return restoring !== null;
}

function stopRestoring() {
  if (!restoring) return;
  cancelAnimationFrame(restoring.frame);
  restoring = null;
  entry(router.entry).y = window.scrollY;
}

/**
 * Scrolls to `y` as soon as the page is tall enough, and holds it there until the page's height has settled (up to
 * RESTORE_MS in all; then the nearest position it allows).
 */
function restore(y: number) {
  stopRestoring();
  if (y <= 0) {
    window.scrollTo(0, 0);
    return;
  }
  const r = { until: performance.now() + RESTORE_MS, frame: 0 };
  restoring = r;
  let height = -1;
  let steady = 0;
  const step = () => {
    if (restoring !== r) return;
    const now = performance.now();
    const h = document.documentElement.scrollHeight;
    if (h !== height) (height = h), (steady = now);
    const max = Math.max(0, h - window.innerHeight);
    if (Math.abs(window.scrollY - Math.min(y, max)) > 1) window.scrollTo(0, Math.min(y, max));
    if (now >= r.until || (max >= y && now - steady >= SETTLE_MS)) stopRestoring();
    else r.frame = requestAnimationFrame(step);
  };
  step();
}

function onEntry(e: EntryChange) {
  if (restoring) {
    cancelAnimationFrame(restoring.frame);
    restoring = null;
  }
  if (!e.traverse) {
    // A new entry at a place an old one had (after going back): it starts with nothing kept.
    kept.delete(e.to);
    return;
  }
  restore(kept.get(e.to)?.y ?? 0);
}

/**
 * Starts recording and restoring the scroll per history entry (App, once). Takes over from the browser's own scroll
 * restoration, which runs before a hash route's screen has rendered and so always lands at the top.
 */
export function installEntryMemory(): () => void {
  load();
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  const onScroll = () => {
    if (!restoring) entry(router.entry).y = window.scrollY;
  };
  // The user takes over: a restore still waiting gives way.
  const onInput = () => stopRestoring();
  const onHide = () => {
    if (document.visibilityState === 'hidden') persist();
  };
  const offEntry = router.onEntry(onEntry);
  window.addEventListener('scroll', onScroll, { passive: true });
  for (const ev of ['wheel', 'touchstart', 'pointerdown', 'keydown'] as const) window.addEventListener(ev, onInput, { passive: true, capture: true });
  document.addEventListener('visibilitychange', onHide);
  window.addEventListener('pagehide', persist);
  // A reload: the entry's position, as Back would.
  const y = kept.get(router.entry)?.y;
  if (y) restore(y);
  return () => {
    stopRestoring();
    offEntry();
    window.removeEventListener('scroll', onScroll);
    for (const ev of ['wheel', 'touchstart', 'pointerdown', 'keydown'] as const) window.removeEventListener(ev, onInput, { capture: true });
    document.removeEventListener('visibilitychange', onHide);
    window.removeEventListener('pagehide', persist);
  };
}

/** A value kept with the current history entry (see the top of this file). */
export function entryState<T>(key: string): { get(): T | undefined; set(v: T | undefined): void } {
  return {
    get: () => kept.get(router.entry)?.state?.[key] as T | undefined,
    set: (v) => {
      const k = entry(router.entry);
      k.state = { ...k.state, [key]: v };
      if (v === undefined) delete k.state[key];
    },
  };
}
