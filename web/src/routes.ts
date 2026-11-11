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
// screen on display until it is (App).

import type { Component } from 'fw';
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

type ScreenModule = { default: Component<Record<string, never>> };

/** The screens that exist (ported), by file: each a chunk of its own. */
const files = import.meta.glob<ScreenModule>(['./routes/*.tsx', '!./routes/*.test.tsx', '!./routes/*.testing.tsx', '!./routes/NotFound.tsx']);

type ScreenName = Exclude<RouteName, 'not-found'>;

const loaded = new Map<ScreenName, Component<Record<string, never>>>();
const loading = new Map<ScreenName, Promise<Component<Record<string, never>> | null>>();

const loaderOf = (name: ScreenName) => files[`./routes/${ROUTES[name]}.tsx`];

/**
 * The screen of a route: its component when loaded, null when the route has no screen yet (not ported: the shell
 * alone), undefined while its chunk has not loaded (call `preload`).
 */
export function screenOf(name: ScreenName): Component<Record<string, never>> | null | undefined {
  const c = loaded.get(name);
  if (c) return c;
  return loaderOf(name) ? undefined : null;
}

/** Loads a route's chunk (once); resolves with its component, or null when the route has no screen. */
export function preload(name: ScreenName): Promise<Component<Record<string, never>> | null> {
  let p = loading.get(name);
  if (p) return p;
  const load = loaderOf(name);
  p = load
    ? load().then(
        (m) => (loaded.set(name, m.default), m.default),
        (e) => {
          // A failed fetch (offline, a deploy replaced the chunk) is tried again on the next request.
          loading.delete(name);
          throw e;
        },
      )
    : Promise.resolve(null);
  loading.set(name, p);
  return p;
}

/** Preloads the screen an app link (`#/runs/run_7d1e`) leads to; anything else is ignored. */
export function preloadHref(href: string | null | undefined): void {
  if (!href || !href.startsWith('#/')) return;
  const name = parse(href).name;
  if (name !== 'not-found') void preload(name).catch(() => {});
}

/** Preloads every screen (a keyboard jump or the palette may lead anywhere; idle time after startup). */
export function preloadAll(): void {
  for (const name of Object.keys(ROUTES) as ScreenName[]) void preload(name).catch(() => {});
}
