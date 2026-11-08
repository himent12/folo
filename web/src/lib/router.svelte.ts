// A tiny hash router. Hash URLs (`#/runs/run_7d1e`) work behind any static file server, including `folo serve`.

export type RouteName =
  | 'overview'
  | 'datasets'
  | 'dataset'
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
      keys.forEach((k, i) => (params[k] = decodeURIComponent(m[i + 1] ?? '')));
      return { name, params, query, path };
    }
  }
  return { name: 'not-found', params: {}, query, path };
}

/** The top-level section a route belongs to (for nav highlighting). */
export function section(name: RouteName): string {
  if (name === 'dataset') return 'datasets';
  if (name === 'run' || name === 'run-new' || name === 'run-compare') return 'runs';
  if (name === 'model') return 'models';
  return name;
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
