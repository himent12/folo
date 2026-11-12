// The Runs list's view state (filter, search, sort, the runs ticked for Compare), kept in the page's URL query so
// Back from a run, a reload or a shared link shows the list as it was left: `?status=failed&q=bccd&sort=map&dir=asc
// &sel=run_a,run_b`. Defaults are left out of the URL (newest first, every status, nothing typed or ticked).

import { runProgress, trainingSeconds } from '$lib/api/progress';
import type { Run, RunStatus } from '$lib/api/types';

export type Filter = 'all' | 'active' | 'completed' | 'failed';
export const groups: Record<Filter, RunStatus[]> = {
  all: ['queued', 'running', 'stopping', 'stopped', 'completed', 'failed'],
  active: ['queued', 'running', 'stopping'],
  completed: ['completed', 'stopped'],
  failed: ['failed'],
};

export type SortKey = 'created' | 'map' | 'progress' | 'duration';
export type Dir = 'asc' | 'desc';

export interface ListState {
  filter: Filter;
  q: string;
  sort: SortKey;
  dir: Dir;
  /** Run ids ticked for Compare, in the order they were ticked (at most 4). */
  sel: string[];
}

const SORTS: SortKey[] = ['created', 'map', 'progress', 'duration'];
export const DEFAULT_STATE: ListState = { filter: 'all', q: '', sort: 'created', dir: 'desc', sel: [] };

/** The state a query holds; anything it does not name (or names wrongly, `?status=stopped`) is the default. */
export function readState(query: URLSearchParams): ListState {
  const status = query.get('status');
  const sort = query.get('sort');
  const dir = query.get('dir');
  return {
    filter: status && Object.hasOwn(groups, status) ? (status as Filter) : 'all',
    q: query.get('q') ?? '',
    sort: sort && SORTS.includes(sort as SortKey) ? (sort as SortKey) : 'created',
    dir: dir === 'asc' || dir === 'desc' ? dir : 'desc',
    sel: (query.get('sel') ?? '').split(',').filter(Boolean).slice(0, 4),
  };
}

/** The query of `s`, defaults left out (the list as it opens has a bare `#/runs`). */
export function writeState(s: ListState): string {
  const q = new URLSearchParams();
  if (s.filter !== 'all') q.set('status', s.filter);
  if (s.q) q.set('q', s.q);
  if (s.sort !== 'created' || s.dir !== 'desc') {
    q.set('sort', s.sort);
    q.set('dir', s.dir);
  }
  if (s.sel.length) q.set('sel', s.sel.join(','));
  // Commas in run ids lists read better unescaped.
  return q.toString().replaceAll('%2C', ',');
}

/** The value a run sorts by; null sorts last in either direction (a run with no score yet, one that never started). */
function key(r: Run, by: SortKey, now: number): number | null {
  switch (by) {
    case 'created':
      return Date.parse(r.created);
    case 'map':
      return r.best ? r.best.map50_95 : null;
    case 'progress':
      return runProgress(r);
    case 'duration':
      return trainingSeconds(r, now);
  }
}

/** `runs` sorted by `by` (a stable sort: ties keep the server's order, newest first). */
export function sortRuns(runs: readonly Run[], by: SortKey, dir: Dir, now = Date.now()): Run[] {
  const sign = dir === 'asc' ? 1 : -1;
  return runs
    .map((r, i) => ({ r, i, k: key(r, by, now) }))
    .sort((a, b) => (a.k === null ? (b.k === null ? a.i - b.i : 1) : b.k === null ? -1 : (a.k - b.k) * sign || a.i - b.i))
    .map((x) => x.r);
}

/**
 * The list's query when it was left for a run's page, with that page's place in the tab's history (router.entry):
 * Escape on the run page (shell/shortcuts) opens a bare `/runs` as a new entry rather than going Back, and that entry
 * shows the list as it was left there, ticks included, as Back would (shell F3).
 */
let leftFor: { entry: number; query: string } | null = null;

/** The list was left for the page at history place `entry` (a run's), its query `query`. */
export function leftForRun(entry: number, query: string): void {
  leftFor = { entry, query };
}

/**
 * The query to show the list with when it opens at history place `entry` with `query`: the one it was left with, when
 * this is a bare `/runs` opened right after that run's page; null otherwise. Asked once per opening.
 */
export function returnQuery(entry: number, query: URLSearchParams): string | null {
  const l = leftFor;
  leftFor = null;
  return l && l.query && !query.toString() && entry === l.entry + 1 ? l.query : null;
}
