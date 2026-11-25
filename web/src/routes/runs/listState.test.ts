import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { DEFAULT_STATE, emptyFilterSummary, leftForRun, readState, returnQuery, sortRuns, writeState } from './listState';

const run = (id: string, o: Partial<Run> = {}) =>
  ({ id, status: 'completed', created: '2026-11-11T10:00:00Z', started: null, finished: null, best: null, epoch: 0, epochs: 5, step: 0, steps_per_epoch: 0, ...o }) as Run;

describe('the Runs list state in the URL', () => {
  it('round-trips, defaults left out', () => {
    const s = { filter: 'failed', q: 'bccd copy', sort: 'map', dir: 'asc', sel: ['run_a', 'run_b'] } as const;
    const q = writeState({ ...s, sel: [...s.sel] });
    expect(q).toBe('status=failed&q=bccd+copy&sort=map&dir=asc&sel=run_a,run_b');
    expect(readState(new URLSearchParams(q))).toEqual(s);
    expect(writeState(DEFAULT_STATE)).toBe('');
  });

  it('reads what it does not know as the default, strips CR/LF in q, and keeps at most four ticked runs', () => {
    expect(readState(new URLSearchParams('status=stopped&sort=name&dir=up&q=bc%0Acd&sel=a,b,c,d,e'))).toEqual({
      ...DEFAULT_STATE,
      q: 'bccd',
      sel: ['a', 'b', 'c', 'd'],
    });
  });

  it('deduplicates sel in insertion order before applying the four-run cap', () => {
    expect(readState(new URLSearchParams('sel=run_a,run_a')).sel).toEqual(['run_a']);
    expect(readState(new URLSearchParams('sel=,run_a,,run_a,')).sel).toEqual(['run_a']);
    expect(readState(new URLSearchParams('sel=a,a,b,a,c,b,d,e')).sel).toEqual(['a', 'b', 'c', 'd']);
    expect(writeState({ ...DEFAULT_STATE, sel: ['a', 'a', 'b', 'c', 'b', 'd', 'e'] })).toBe('sel=a,b,c,d');
  });

  it('summarizes active status and search filters for empty-state recovery', () => {
    expect(emptyFilterSummary({ filter: 'failed', q: 'bccd' })).toBe('No failed runs match "bccd".');
    expect(emptyFilterSummary({ filter: 'active', q: '  ' })).toBe('No active runs right now.');
    expect(emptyFilterSummary({ filter: 'all', q: 'bccd' })).toBe('No runs match "bccd".');
    expect(emptyFilterSummary({ filter: 'all', q: '' })).toBe('Try another status filter or search term.');
  });
});

describe('sortRuns', () => {
  const runs = [
    run('a', { best: { epoch: 1, map50_95: 0.2 }, created: '2026-11-11T10:00:03Z' }),
    run('b', { created: '2026-11-11T10:00:02Z' }),
    run('c', { best: { epoch: 2, map50_95: 0.5 }, created: '2026-11-11T10:00:01Z' }),
  ];
  it('sorts by score either way, runs with none last', () => {
    expect(sortRuns(runs, 'map', 'desc').map((r) => r.id)).toEqual(['c', 'a', 'b']);
    expect(sortRuns(runs, 'map', 'asc').map((r) => r.id)).toEqual(['a', 'c', 'b']);
    expect(sortRuns(runs, 'created', 'asc').map((r) => r.id)).toEqual(['c', 'b', 'a']);
  });

  it('sorts non-finite dates and scores last in both directions', () => {
    const withBad = [
      run('bad-date', { created: 'not-a-date', best: { epoch: 1, map50_95: Number.NaN } }),
      ...runs,
    ];
    expect(sortRuns(withBad, 'created', 'asc').map((r) => r.id)).toEqual(['c', 'b', 'a', 'bad-date']);
    expect(sortRuns(withBad, 'created', 'desc').map((r) => r.id)).toEqual(['a', 'b', 'c', 'bad-date']);
    expect(sortRuns(withBad, 'map', 'asc').map((r) => r.id)).toEqual(['a', 'c', 'bad-date', 'b']);
    expect(sortRuns(withBad, 'map', 'desc').map((r) => r.id)).toEqual(['c', 'a', 'bad-date', 'b']);
  });
});

describe('returning from a run', () => {
  it('shows the list as it was left only on a bare /runs opened right after the run page, once', () => {
    leftForRun(4, 'sel=run_a,run_b');
    expect(returnQuery(5, new URLSearchParams())).toBe('sel=run_a,run_b');
    expect(returnQuery(5, new URLSearchParams())).toBeNull();
    // A link with its own state, or a visit two pages later, is not a return.
    leftForRun(4, 'sel=run_a,run_b');
    expect(returnQuery(5, new URLSearchParams('q=bccd'))).toBeNull();
    leftForRun(4, 'sel=run_a,run_b');
    expect(returnQuery(6, new URLSearchParams())).toBeNull();
  });
});
