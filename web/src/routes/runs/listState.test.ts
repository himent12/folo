import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { DEFAULT_STATE, readState, sortRuns, trainingSeconds, writeState } from './listState';

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

  it('reads what it does not know as the default, and keeps at most four ticked runs', () => {
    expect(readState(new URLSearchParams('status=stopped&sort=name&dir=up&sel=a,b,c,d,e'))).toEqual({ ...DEFAULT_STATE, sel: ['a', 'b', 'c', 'd'] });
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

  it('measures a duration from start to end, or to now while it trains', () => {
    expect(trainingSeconds(run('x', { started: '2026-11-11T10:00:00Z', finished: '2026-11-11T10:01:30Z' }))).toBe(90);
    expect(trainingSeconds(run('x', { started: '2026-11-11T10:00:00Z' }), Date.parse('2026-11-11T10:00:10Z'))).toBe(10);
    expect(trainingSeconds(run('x'))).toBeNull();
  });
});
