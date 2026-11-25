// The Datasets list's view state (format filter, sort, search query), kept in the page's URL query so Back from a
// dataset, a reload, or a shared link restores the list as it was left: `?format=yolo&sort=images&q=fast`.
// Defaults are omitted so the default view keeps a bare `#/datasets` URL, and transient `?add=` params are preserved.

export type Format = 'all' | 'yolo' | 'coco';
export type SortKey = 'name' | 'images' | 'boxes' | 'classes';

export interface DatasetListState {
  format: Format;
  sort: SortKey;
  q: string;
}

const FORMATS: readonly Format[] = ['all', 'yolo', 'coco'];
const SORTS: readonly SortKey[] = ['name', 'images', 'boxes', 'classes'];
const FORMAT_LABELS: Record<Exclude<Format, 'all'>, string> = {
  yolo: 'YOLO',
  coco: 'COCO',
};

export const DEFAULT_STATE: DatasetListState = { format: 'all', sort: 'name', q: '' };

/** Reads view state from URL query params; unknown or invalid values fall back to defaults. */
export function readState(query: URLSearchParams): DatasetListState {
  const format = query.get('format');
  const sort = query.get('sort');
  return {
    format: format && FORMATS.includes(format as Format) ? (format as Format) : 'all',
    sort: sort && SORTS.includes(sort as SortKey) ? (sort as SortKey) : 'name',
    q: (query.get('q') ?? '').replace(/[\r\n]+/g, ''),
  };
}

/**
 * Serializes `s` to a query string with defaults omitted. When `preserve` is given, unrelated transient params
 * (such as `?add=folder` while the Add dataset dialog is open) are kept intact.
 */
export function writeState(s: DatasetListState, preserve?: URLSearchParams): string {
  const q = new URLSearchParams();
  if (preserve) {
    for (const [k, v] of preserve.entries()) {
      if (k !== 'format' && k !== 'sort' && k !== 'q') q.append(k, v);
    }
  }
  if (s.format !== 'all') q.set('format', s.format);
  if (s.sort !== 'name') q.set('sort', s.sort);
  if (s.q) q.set('q', s.q);
  return q.toString();
}

/** Describes the active filters when 0 datasets match, so empty-state recovery explains what is narrowing the list. */
export function emptyFilterSummary(s: Pick<DatasetListState, 'format' | 'q'>): string {
  const term = s.q.trim();
  const fmt = s.format === 'all' ? null : FORMAT_LABELS[s.format];
  if (fmt && term) return `No ${fmt} datasets match "${term}".`;
  if (fmt) return `No ${fmt} datasets found.`;
  if (term) return `No datasets match "${term}".`;
  return 'Try another format or search term.';
}

let leftFor: { entry: number; query: string } | null = null;

/** Remembers the list's query when navigating into a dataset page, so Escape restores it as Back would. */
export function leftForDataset(entry: number, query: string): void {
  leftFor = query ? { entry, query } : null;
}

/** Restores the remembered query once when returning to a bare `/datasets` right after that dataset page. */
export function returnQuery(entry: number, query: URLSearchParams): string | null {
  const l = leftFor;
  leftFor = null;
  return l && l.query && !query.toString() && entry === l.entry + 1 ? l.query : null;
}
