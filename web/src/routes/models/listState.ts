// The Models list's view state (evaluation group filter, sort column/direction, search query), kept in the page's URL
// query so Back from a model, a reload, or a shared link restores the list as it was left:
// `?group=~published&sort=bytes&dir=asc&q=folo-n`. Defaults are omitted so the default view keeps a bare `#/models`.

export type ModelSortKey = 'name' | 'arch' | 'classes' | 'bytes' | 'map50_95' | 'map50' | 'source' | 'created';
export type Dir = 'asc' | 'desc';

export interface ModelListState {
  group: string;
  sort: ModelSortKey;
  dir: Dir;
  q: string;
}

const SORTS: readonly ModelSortKey[] = ['name', 'arch', 'classes', 'bytes', 'map50_95', 'map50', 'source', 'created'];
const TEXT_SORTS: ReadonlySet<ModelSortKey> = new Set(['name', 'arch', 'source']);

export const DEFAULT_STATE: ModelListState = { group: 'all', sort: 'map50_95', dir: 'desc', q: '' };

/** Reads view state from URL query params; unknown sort/dir values fall back to defaults. */
export function readState(query: URLSearchParams): ModelListState {
  const rawGroup = query.get('group');
  const rawSort = query.get('sort');
  const rawDir = query.get('dir');
  const sort: ModelSortKey = rawSort && SORTS.includes(rawSort as ModelSortKey) ? (rawSort as ModelSortKey) : 'map50_95';
  const defaultDir: Dir = TEXT_SORTS.has(sort) ? 'asc' : 'desc';
  const dir: Dir = rawDir === 'asc' || rawDir === 'desc' ? rawDir : defaultDir;
  return {
    group: rawGroup && rawGroup.trim() ? rawGroup : 'all',
    sort,
    dir,
    q: query.get('q') ?? '',
  };
}

/** Serializes `s` to a query string with defaults omitted. */
export function writeState(s: ModelListState): string {
  const q = new URLSearchParams();
  if (s.group && s.group !== 'all') q.set('group', s.group);
  if (s.q) q.set('q', s.q);
  if (s.sort !== 'map50_95' || s.dir !== 'desc') {
    q.set('sort', s.sort);
    q.set('dir', s.dir);
  }
  return q.toString();
}

/** Describes the active filters when 0 models match, naming the active group (if any) and/or search term. */
export function emptyFilterSummary(s: Pick<ModelListState, 'group' | 'q'>, groupLabel?: string | null): string {
  const term = s.q.trim();
  const hasGroup = s.group !== 'all';
  if (hasGroup && term) {
    return groupLabel ? `No models in ${groupLabel} match "${term}".` : `No models match the selected group filter and "${term}".`;
  }
  if (hasGroup) {
    return groupLabel ? `No models in ${groupLabel} right now.` : 'No models match the selected group filter.';
  }
  if (term) return `No models match "${term}".`;
  return 'Try another dataset or search term.';
}

let leftFor: { entry: number; query: string } | null = null;

/** Remembers the list's query when navigating into a model page, so Escape restores it as Back would. */
export function leftForModel(entry: number, query: string): void {
  leftFor = query ? { entry, query } : null;
}

/** Restores the remembered query once when returning to a bare `/models` right after that model page. */
export function returnQuery(entry: number, query: URLSearchParams): string | null {
  const l = leftFor;
  leftFor = null;
  return l && l.query && !query.toString() && entry === l.entry + 1 ? l.query : null;
}
