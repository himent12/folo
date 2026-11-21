// A run that ends while Studio is open is announced once; history and the user's own Stop are not news.
import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { runEndMessage, runEnds } from './runWatch';

const run = (id: string, status: Run['status'], extra: Partial<Run> = {}) =>
  ({ id, name: id, status, epoch: 23, epochs: 100, error: null, best: null, ...extra }) as unknown as Run;

describe('runEnds', () => {
  it('says nothing on the first poll', () => {
    expect(runEnds(undefined, [run('a', 'completed')])).toEqual([]);
  });

  it('finds runs that went from training to completed, failed or out of memory', () => {
    const prev = [run('a', 'running'), run('b', 'running'), run('c', 'stopping'), run('d', 'running')];
    const next = [
      run('a', 'completed'),
      run('b', 'failed', { error: 'boom' }),
      run('c', 'failed', { error: 'the GPU ran out of memory (wgpu backend: Out of Memory)' }),
      run('d', 'running'),
    ];
    expect(runEnds(prev, next).map((e) => [e.run.id, e.kind])).toEqual([
      ['a', 'completed'],
      ['b', 'failed'],
      ['c', 'oom'],
    ]);
  });

  it('ignores a stop, a run that was queued, and runs that were already over', () => {
    const prev = [run('a', 'running'), run('b', 'queued'), run('c', 'completed')];
    const next = [run('a', 'stopped'), run('b', 'failed'), run('c', 'completed')];
    expect(runEnds(prev, next)).toEqual([]);
  });
});

describe('runEndMessage', () => {
  // #382: "best mAP50-95 0.000 at epoch 3" for a run that scored 0.00042.
  it('never says a best score that is not zero as zero', () => {
    const r = run('a', 'completed', { best: { epoch: 3, map50_95: 0.00042 } });
    expect(runEndMessage({ kind: 'completed', run: r }).detail).toEqual(['Finished training', 'best mAP50-95 0.00042 at epoch 3']);
  });
});
