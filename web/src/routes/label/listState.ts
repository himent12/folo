// The Labeling jobs list's view state (status filter and search query), kept in the page's URL query so Back from a
// job, a reload, or a shared link restores the list as it was left: `?status=failed&q=bolt`. Defaults are omitted so
// the default view keeps a bare `#/datasets/label`, and transient `?new=1` params are preserved.

export type LabelFilter = 'all' | 'active' | 'failed' | 'idle';

export interface LabelListState {
  filter: LabelFilter;
  q: string;
}

const FILTERS: readonly LabelFilter[] = ['all', 'active', 'failed', 'idle'];
const FILTER_LABELS: Record<Exclude<LabelFilter, 'all'>, string> = {
  active: 'labeling',
  failed: 'failed',
  idle: 'idle',
};

export const DEFAULT_STATE: LabelListState = { filter: 'all', q: '' };

/** Reads view state from URL query params; unknown or invalid values fall back to defaults. */
export function readState(query: URLSearchParams): LabelListState {
  const raw = query.get('status');
  return {
    filter: raw && FILTERS.includes(raw as LabelFilter) ? (raw as LabelFilter) : 'all',
    q: (query.get('q') ?? '').replace(/[\r\n]+/g, ''),
  };
}

/**
 * Serializes `s` to a query string with defaults omitted. When `preserve` is given, unrelated transient params
 * (such as `?new=1`) are kept intact.
 */
export function writeState(s: LabelListState, preserve?: URLSearchParams): string {
  const q = new URLSearchParams();
  if (preserve) {
    for (const [k, v] of preserve.entries()) {
      if (k !== 'status' && k !== 'q') q.append(k, v);
    }
  }
  if (s.filter !== 'all') q.set('status', s.filter);
  if (s.q) q.set('q', s.q);
  return q.toString();
}

/** Describes the active filters when 0 labeling jobs match, so empty-state recovery explains what narrowed the list. */
export function emptyFilterSummary(s: Pick<LabelListState, 'filter' | 'q'>): string {
  const term = s.q.trim();
  const status = s.filter === 'all' ? null : FILTER_LABELS[s.filter];
  if (status && term) return `No ${status} labeling jobs match "${term}".`;
  if (status) return `No ${status} labeling jobs right now.`;
  if (term) return `No labeling jobs match "${term}".`;
  return 'Try another status or search term.';
}

let leftFor: { entry: number; query: string } | null = null;

/** Remembers the list's query when navigating into a labeling job page, so Escape restores it as Back would. */
export function leftForLabelJob(entry: number, query: string): void {
  leftFor = query ? { entry, query } : null;
}

/** Restores the remembered query once when returning to a bare `/datasets/label` right after that job page. */
export function returnQuery(entry: number, query: URLSearchParams): string | null {
  const l = leftFor;
  leftFor = null;
  return l && l.query && !query.toString() && entry === l.entry + 1 ? l.query : null;
}
