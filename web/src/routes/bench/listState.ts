// The Benchmarks list's view state (source/status filter, sort column/direction, search query), kept in the page's
// URL query so Back from a benchmark, a reload, or a shared link restores the list as it was left:
// `?source=active&sort=leader&dir=desc&q=haze`. Defaults are omitted so the default view keeps a bare `#/benchmarks`,
// and transient `?new=1` params are preserved.

export type BenchFilter = 'all' | 'dataset' | 'label_job' | 'generated' | 'active';
export type BenchSortKey = 'name' | 'source' | 'images' | 'classes' | 'conditions' | 'entries' | 'leader' | 'created';
export type Dir = 'asc' | 'desc';

export interface BenchListState {
  filter: BenchFilter;
  sort: BenchSortKey;
  dir: Dir;
  q: string;
}

const FILTERS: readonly BenchFilter[] = ['all', 'dataset', 'label_job', 'generated', 'active'];
const SORTS: readonly BenchSortKey[] = ['name', 'source', 'images', 'classes', 'conditions', 'entries', 'leader', 'created'];
const TEXT_SORTS: ReadonlySet<BenchSortKey> = new Set(['name', 'source']);
const FILTER_LABELS: Record<Exclude<BenchFilter, 'all'>, string> = {
  dataset: 'dataset',
  label_job: 'labeling-job',
  generated: 'generated',
  active: 'running',
};

export const DEFAULT_STATE: BenchListState = { filter: 'all', sort: 'created', dir: 'desc', q: '' };

/** Reads view state from URL query params; unknown or invalid values fall back to defaults. */
export function readState(query: URLSearchParams): BenchListState {
  const rawFilter = query.get('source');
  const rawSort = query.get('sort');
  const rawDir = query.get('dir');
  const sort: BenchSortKey = rawSort && SORTS.includes(rawSort as BenchSortKey) ? (rawSort as BenchSortKey) : 'created';
  const defaultDir: Dir = TEXT_SORTS.has(sort) ? 'asc' : 'desc';
  const dir: Dir = rawDir === 'asc' || rawDir === 'desc' ? rawDir : defaultDir;
  return {
    filter: rawFilter && FILTERS.includes(rawFilter as BenchFilter) ? (rawFilter as BenchFilter) : 'all',
    sort,
    dir,
    q: (query.get('q') ?? '').replace(/[\r\n]+/g, ''),
  };
}

/**
 * Serializes `s` to a query string with defaults omitted. When `preserve` is given, unrelated transient params
 * (such as `?new=1`) are kept intact.
 */
export function writeState(s: BenchListState, preserve?: URLSearchParams): string {
  const q = new URLSearchParams();
  if (preserve) {
    for (const [k, v] of preserve.entries()) {
      if (k !== 'source' && k !== 'sort' && k !== 'dir' && k !== 'q') q.append(k, v);
    }
  }
  if (s.filter !== 'all') q.set('source', s.filter);
  if (s.q) q.set('q', s.q);
  if (s.sort !== 'created' || s.dir !== 'desc') {
    q.set('sort', s.sort);
    q.set('dir', s.dir);
  }
  return q.toString();
}

/** Describes the active filters when 0 benchmarks match, so empty-state recovery explains what narrowed the list. */
export function emptyFilterSummary(s: Pick<BenchListState, 'filter' | 'q'>): string {
  const term = s.q.trim();
  const kind = s.filter === 'all' ? null : FILTER_LABELS[s.filter];
  if (kind && term) return `No ${kind} benchmarks match "${term}".`;
  if (kind) return `No ${kind} benchmarks right now.`;
  if (term) return `No benchmarks match "${term}".`;
  return 'Try another source or search term.';
}

let leftFor: { entry: number; query: string } | null = null;

/** Remembers the list's query when navigating into a benchmark page, so Escape restores it as Back would. */
export function leftForBench(entry: number, query: string): void {
  leftFor = query ? { entry, query } : null;
}

/** Restores the remembered query once when returning to a bare `/benchmarks` right after that benchmark page. */
export function returnQuery(entry: number, query: URLSearchParams): string | null {
  const l = leftFor;
  leftFor = null;
  return l && l.query && !query.toString() && entry === l.entry + 1 ? l.query : null;
}
