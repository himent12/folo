import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { aheadLine, aheadOf, startedToast } from './queue';

const run = (name: string, status: Run['status'], queue_position: number | null = null, extra: Partial<Run> = {}) =>
  ({ id: `run_${name}`, name, status, queue_position, eta_s: null, config: {}, ...extra }) as Run;

describe('a new run and the queue', () => {
  it('starts at once on a free machine, or in the cloud', () => {
    expect(aheadOf([run('done', 'completed'), run('old', 'failed')], false)).toBeNull();
    expect(aheadOf([run('q-one', 'running')], true)).toBeNull();
    // A cloud run elsewhere holds nothing here.
    expect(aheadOf([run('far', 'running', null, { cloud: {} as Run['cloud'] })], false)).toBeNull();
  });

  it('waits behind the run training now and the runs already waiting, and says when the first should end', () => {
    const one = aheadOf([run('q-one', 'running', null, { eta_s: 30 })], false)!;
    expect(one.position).toBe(1);
    expect(aheadLine(one)).toBe('q-one is training: this run waits in the queue and starts when it ends (it ends in about 30s).');
    const three = aheadOf([run('q-one', 'running'), run('q-two', 'queued', 1), run('q-three', 'queued', 2)], false)!;
    expect(three.position).toBe(3);
    expect(aheadLine(three)).toBe('q-one is training and 2 runs wait: this run will be 3rd in the queue.');
    // Taken by the worker and still being built: out of the queue, but holding the slot.
    expect(aheadLine(aheadOf([run('q-one', 'queued')], false)!)).toBe('q-one is starting: this run waits in the queue and starts when it ends.');
  });

  it('says Queued only for a run that really waits', () => {
    const busy = aheadOf([run('q-one', 'running')], false);
    expect(startedToast(run('q-two', 'queued', 1), busy)).toEqual({ title: 'Queued q-two', place: '1st in queue' });
    // First in the queue of a free machine: the worker is about to take it.
    expect(startedToast(run('q-two', 'queued', 1), null)).toEqual({ title: 'Started q-two', place: null });
    expect(startedToast(run('q-two', 'queued', null), busy)).toEqual({ title: 'Started q-two', place: null });
    expect(startedToast(run('q-three', 'queued', 2), null)).toEqual({ title: 'Queued q-three', place: '2nd in queue' });
  });
});
