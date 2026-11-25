// The fw app's route table: which screen each route shows, each screen its own chunk (loaded with a dynamic import).
//
// One line per route, sorted by route name: the route and the file in src/routes/ that renders it. A screen is live as
// soon as its port exists: `routes/Runs.tsx` (default export: the screen's component) is picked up by the glob below,
// so porting a screen needs no edit here, and a file that does not exist yet never breaks the build (the route shows
// the shell alone until it does). Change a line only to point a route at another file.
//
// Loading: the current route's chunk is fetched at startup; any other is fetched when a link to it is hovered,
// focused or touched (`preloadHref`, App), when the G of a "go to" shortcut is pressed or the palette opens
// (`preloadAll`), and in idle time after the first screen has shown. A route whose chunk is not there yet keeps the
// screen on display, with the route it shows, until it is (App, `router.holdWhile`); one whose chunk failed to load
// shows an error with Retry, and its screen as soon as any later try brings the chunk in.

import { signal, untrack, type Component, type Signal } from 'fw';
import { retryFailedStyles } from '$lib/failedStyles';
import { parse, type RouteName } from '$lib/router';

/** Every route but not-found (App renders that one itself), and the file under src/routes/ (no extension) for it. */
export const ROUTES: Record<Exclude<RouteName, 'not-found'>, string> = {
  'bench-compare': 'BenchCompare',
  'bench-entry': 'BenchEntry',
  'bench-loop': 'BenchLoop',
  benchmark: 'Benchmark',
  benchmarks: 'Benchmarks',
  dataset: 'Dataset',
  datasets: 'Datasets',
  'label-job': 'LabelJob',
  'label-jobs': 'LabelJobs',
  model: 'Model',
  models: 'Models',
  overview: 'Overview',
  playground: 'Playground',
  run: 'Run',
  'run-compare': 'RunCompare',
  'run-new': 'RunNew',
  runs: 'Runs',
  settings: 'Settings',
};

export type ScreenModule = { default: Component<Record<string, never>> };
export type ScreenName = Exclude<RouteName, 'not-found'>;
type ScreenComponent = Component<Record<string, never>>;

/** A screen's chunk as App sees it: loaded (its component), or failed (the error of the last try). */
interface ScreenState {
  comp?: ScreenComponent;
  failed?: { error: unknown };
}

/** Top-level navigation routes preloaded ahead of detail screens during idle staggered preloading. */
const NAV_ORDER: readonly ScreenName[] = ['overview', 'datasets', 'runs', 'models', 'benchmarks', 'playground', 'settings'];

const STAGGER_ORDER: readonly ScreenName[] = [
  ...NAV_ORDER,
  ...(Object.keys(ROUTES) as ScreenName[]).filter((name) => !NAV_ORDER.includes(name)),
];

export interface StaggerOptions {
  /** Routes to fetch per idle step (default 2). */
  batch?: number;
  /** Pause between batches in milliseconds (default 250) so active API calls and interactions take priority. */
  delayMs?: number;
  /** Explicit route order (default: top-level nav screens first, then detail screens). */
  order?: readonly ScreenName[];
}

/**
 * The screens' chunks, each loaded once on demand. Whether a chunk has loaded or failed is reactive: App (and the
 * router's hold on the displayed route) read it, so whichever request brings a chunk in (App's own, a link hovered, a
 * Retry), the screen shows; a failure shows an error until a later try succeeds.
 */
export class Screens {
  readonly #loaderOf: (name: ScreenName) => (() => Promise<ScreenModule>) | undefined;
  readonly #states = new Map<ScreenName, Signal<ScreenState>>();
  readonly #loading = new Map<ScreenName, Promise<ScreenComponent | null>>();

  /** `loaderOf(name)`: the import of the route's chunk, or undefined when the route has no screen. */
  constructor(loaderOf: (name: ScreenName) => (() => Promise<ScreenModule>) | undefined) {
    this.#loaderOf = loaderOf;
  }

  #state(name: ScreenName): Signal<ScreenState> {
    let s = this.#states.get(name);
    if (s === undefined) this.#states.set(name, (s = signal<ScreenState>({})));
    return s;
  }

  /**
   * The screen of a route: its component when loaded, null when the route has no screen (the shell alone), undefined
   * while its chunk has not loaded (call `preload`). Reactive.
   */
  screenOf(name: ScreenName): ScreenComponent | null | undefined {
    if (!this.#loaderOf(name)) return null;
    return this.#state(name).get().comp;
  }

  /** Why a route's chunk could not be loaded, while it could not (a later successful load clears it). Reactive. */
  failure(name: ScreenName): { error: unknown } | undefined {
    if (!this.#loaderOf(name)) return undefined;
    const s = this.#state(name).get();
    return s.comp ? undefined : s.failed;
  }

  /**
   * Loads a route's chunk (once; again after a failure); resolves with its component, or null when the route has no
   * screen. Its state changes only when the import settles, never synchronously (App calls this from a computed).
   */
  preload(name: ScreenName): Promise<ScreenComponent | null> {
    let p = this.#loading.get(name);
    if (p) return p;
    const load = this.#loaderOf(name);
    p = load
      ? load().then(
          (m) => {
            this.#state(name).set({ comp: m.default });
            return m.default;
          },
          (e: unknown) => {
            // A failed fetch (offline, a deploy replaced the chunk) is tried again on the next request.
            this.#loading.delete(name);
            if (!untrack(() => this.#state(name).get().comp)) this.#state(name).set({ failed: { error: e } });
            throw e;
          },
        )
      : Promise.resolve(null);
    this.#loading.set(name, p);
    return p;
  }

  /** Preloads the screen an app link (`#/runs/run_7d1e`) leads to; anything else is ignored. */
  preloadHref(href: string | null | undefined): void {
    if (!href || !href.startsWith('#/')) return;
    const name = parse(href).name;
    if (name !== 'not-found') void this.preload(name).catch(() => {});
  }

  /** Preloads every screen (a keyboard jump or the palette may lead anywhere). */
  preloadAll(): void {
    for (const name of Object.keys(ROUTES) as ScreenName[]) void this.preload(name).catch(() => {});
  }

  /**
   * Preloads remaining screens gently in small idle batches (top-level nav screens first) so startup idle preloading
   * never floods HTTP connections or competes with the active screen's API calls. Returns a cancel function.
   */
  preloadStaggered(opts: StaggerOptions = {}): () => void {
    const batch = Math.max(1, opts.batch ?? 2);
    const delayMs = opts.delayMs ?? 250;
    const queue = (opts.order ?? STAGGER_ORDER).filter((name) => !!this.#loaderOf(name));
    let i = 0;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let idleId: number | undefined;
    const g = globalThis as {
      requestIdleCallback?: (f: () => void, o?: { timeout: number }) => number;
      cancelIdleCallback?: (id: number) => void;
    };

    const nextIdle = () => {
      if (cancelled || i >= queue.length) return;
      if (g.requestIdleCallback) {
        idleId = g.requestIdleCallback(
          () => {
            idleId = undefined;
            step();
          },
          { timeout: 2000 },
        );
      } else {
        step();
      }
    };

    const step = () => {
      if (cancelled) return;
      const pending: Promise<unknown>[] = [];
      while (i < queue.length && pending.length < batch) {
        const name = queue[i++]!;
        if (this.#loading.has(name) || untrack(() => this.#state(name).get().comp) !== undefined) continue;
        pending.push(this.preload(name).catch(() => {}));
      }
      if (i >= queue.length) return;
      void Promise.allSettled(pending).then(() => {
        if (cancelled) return;
        // Brief pause between batches so user interactions and API requests always win the connection pool.
        timer = setTimeout(() => {
          timer = undefined;
          nextIdle();
        }, delayMs);
      });
    };

    nextIdle();
    return () => {
      cancelled = true;
      if (timer !== undefined) clearTimeout(timer);
      if (idleId !== undefined) g.cancelIdleCallback?.(idleId);
    };
  }
}

/** The screens that exist (ported), by file: each a chunk of its own. */
const files = import.meta.glob<ScreenModule>(['./routes/*.tsx', '!./routes/*.test.tsx', '!./routes/*.testing.tsx', '!./routes/NotFound.tsx']);

/**
 * The app's screens. Each load first asks again for the stylesheets that failed to load earlier (lib/failedStyles: the
 * bundler's preload helper never asks for a file twice), without waiting for them: the chunk loads as it would anyway.
 */
export const screens = new Screens((name) => {
  const load = files[`./routes/${ROUTES[name]}.tsx`];
  return (
    load &&
    (() => {
      void retryFailedStyles();
      return load();
    })
  );
});

/** Loads a route's chunk (see `Screens.preload`). */
export const preload = (name: ScreenName) => screens.preload(name);
