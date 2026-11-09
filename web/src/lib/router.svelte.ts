// A tiny hash router. Hash URLs (`#/runs/run_7d1e`) work behind any static file server, including `folo serve`; a
// path URL (`/runs/run_7d1e`) is moved to its hash route on load.

export type RouteName =
  | 'overview'
  | 'datasets'
  | 'dataset'
  | 'label-jobs'
  | 'label-job'
  | 'runs'
  | 'run-new'
  | 'run-compare'
  | 'run'
  | 'models'
  | 'model'
  | 'playground'
  | 'settings'
  | 'not-found';

export interface Route {
  name: RouteName;
  params: Record<string, string>;
  query: URLSearchParams;
  path: string;
}

const table: [RouteName, RegExp, string[]][] = [
  ['overview', /^\/$/, []],
  ['datasets', /^\/datasets$/, []],
  // Labeling jobs live under Datasets, ahead of a dataset's page (whose id would otherwise take `label`).
  ['label-jobs', /^\/datasets\/label$/, []],
  ['label-job', /^\/datasets\/label\/([^/]+)$/, ['id']],
  ['dataset', /^\/datasets\/([^/]+)$/, ['id']],
  ['runs', /^\/runs$/, []],
  ['run-new', /^\/runs\/new$/, []],
  ['run-compare', /^\/runs\/compare$/, []],
  ['run', /^\/runs\/([^/]+)$/, ['id']],
  ['models', /^\/models$/, []],
  ['model', /^\/models\/([^/]+)$/, ['id']],
  ['playground', /^\/playground$/, []],
  ['settings', /^\/settings$/, []],
];

/** Parses `#/path?query` into a route. Exported for tests. */
export function parse(hash: string): Route {
  const raw = hash.replace(/^#/, '') || '/';
  const [pathPart = '/', queryPart = ''] = raw.split('?', 2);
  const path = pathPart.length > 1 ? pathPart.replace(/\/+$/, '') : '/';
  const query = new URLSearchParams(queryPart);
  for (const [name, re, keys] of table) {
    const m = re.exec(path);
    if (m) {
      const params: Record<string, string> = {};
      try {
        keys.forEach((k, i) => (params[k] = decodeURIComponent(m[i + 1] ?? '')));
      } catch {
        // Malformed percent-encoding (e.g. `%ZZ`): treat as an unknown page rather than crash.
        return { name: 'not-found', params: {}, query, path };
      }
      return { name, params, query, path };
    }
  }
  return { name: 'not-found', params: {}, query, path };
}

/** The top-level section a route belongs to (for nav highlighting). */
export function section(name: RouteName): string {
  if (name === 'dataset' || name === 'label-jobs' || name === 'label-job') return 'datasets';
  if (name === 'run' || name === 'run-new' || name === 'run-compare') return 'runs';
  if (name === 'model') return 'models';
  return name;
}

/** The query keys the page itself reads from `location.search` (lib/api: the demo and its scenario), not a route. */
const PAGE_KEYS = new Set(['mock', 'scenario']);

/**
 * Where a path URL Studio was opened at belongs (`/datasets` → `/#/datasets`): the server answers any path with the
 * app, which routes by hash only. A known route's path moves into the hash with its query (`/runs/new?from=run_a` →
 * `/#/runs/new?from=run_a`), the page's own keys (`?mock=1`) stay in the page's query, and a hash route already there
 * wins (`/datasets#/runs` → `/#/runs`). Null when nothing needs to change: the app's root, or a path that is no route
 * (a missing file). `base` is the app's base path. Exported for tests.
 */
export function hashUrl(loc: { pathname: string; search: string; hash: string }, base = '/'): string | null {
  if (!loc.pathname.startsWith(base)) return null;
  const rest = loc.pathname.slice(base.length).replace(/\/+$/, '');
  if (!rest || rest === 'index.html') return null;
  const path = `/${rest}`;
  if (parse(`#${path}`).name === 'not-found') return null;
  if (loc.hash && loc.hash !== '#') return `${base}${loc.search}${loc.hash}`;
  const page = new URLSearchParams();
  const route = new URLSearchParams();
  for (const [k, v] of new URLSearchParams(loc.search)) (PAGE_KEYS.has(k) ? page : route).append(k, v);
  const q = (p: URLSearchParams) => (String(p) ? `?${p}` : '');
  return `${base}${q(page)}#${path}${q(route)}`;
}

// Before the router reads the hash: a path URL becomes its hash route, in place (no history entry).
if (typeof location !== 'undefined' && typeof history !== 'undefined') {
  const base = import.meta.env?.BASE_URL?.startsWith('/') ? import.meta.env.BASE_URL : '/';
  const to = hashUrl(location, base);
  if (to) history.replaceState(history.state, '', to);
}

class Router {
  route = $state<Route>(parse(typeof location === 'undefined' ? '' : location.hash));

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('hashchange', () => {
        this.route = parse(location.hash);
      });
    }
  }

  /** Navigate to an app path like `/runs/run_7d1e`. */
  go(path: string, opts: { replace?: boolean } = {}) {
    const hash = `#${path}`;
    if (opts.replace) {
      history.replaceState(null, '', hash);
      this.route = parse(hash);
    } else if (location.hash !== hash) {
      location.hash = hash;
    }
  }
}

export const router = new Router();

/** `href` for an app path. */
export const href = (path: string) => `#${path}`;
