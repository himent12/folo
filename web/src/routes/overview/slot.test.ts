// The Training card's run: training first, else the run starting, never a run still waiting behind another (shell F13).
import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { slotRun } from './slot';

const run = (id: string, status: Run['status'], extra: Partial<Run> = {}) => ({ id, status, ...extra }) as unknown as Run;

describe('slotRun', () => {
  it('is the run training or stopping', () => {
    expect(slotRun([run('a', 'completed'), run('b', 'running'), run('c', 'queued', { queue_position: 1 })])?.id).toBe('b');
    expect(slotRun([run('a', 'stopping'), run('c', 'queued', { queue_position: null })])?.id).toBe('a');
  });

  it('is the starting run while the slot is free', () => {
    // Taken by the worker (no place in the queue) beats the front of the queue.
    expect(slotRun([run('a', 'queued', { queue_position: 1 }), run('b', 'queued', { queue_position: null })])?.id).toBe('b');
    expect(slotRun([run('a', 'queued', { queue_position: 2 }), run('b', 'queued', { queue_position: 1 })])?.id).toBe('b');
    // An older server reports no position.
    expect(slotRun([run('a', 'completed'), run('b', 'queued')])?.id).toBe('b');
  });

  it('is nothing when the slot is free and no local run starts', () => {
    expect(slotRun(undefined)).toBeUndefined();
    expect(slotRun([run('a', 'completed'), run('b', 'queued', { queue_position: 2 })])).toBeUndefined();
    expect(slotRun([run('a', 'queued', { queue_position: null, cloud: {} as Run['cloud'] })])).toBeUndefined();
  });
});
