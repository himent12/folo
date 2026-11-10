import { describe, expect, it } from 'vitest';
import { applyQueueMove, hasFeature, ordinal, withQueueMove } from './queue';
import type { Run } from './types';

describe('applyQueueMove (docs/API.md, POST /api/runs/{id}/queue)', () => {
  const q = ['a', 'b', 'c', 'd'];

  it('moves to an end, one place, or to a place; the others keep their order', () => {
    expect(applyQueueMove(q, 'c', { action: 'top' })).toEqual(['c', 'a', 'b', 'd']);
    expect(applyQueueMove(q, 'b', { action: 'bottom' })).toEqual(['a', 'c', 'd', 'b']);
    expect(applyQueueMove(q, 'c', { action: 'up' })).toEqual(['a', 'c', 'b', 'd']);
    expect(applyQueueMove(q, 'b', { action: 'down' })).toEqual(['a', 'c', 'b', 'd']);
    expect(applyQueueMove(q, 'd', { position: 2 })).toEqual(['a', 'd', 'b', 'c']);
  });

  it('start puts the run at the front (the worker takes it when the slot is free)', () => {
    expect(applyQueueMove(q, 'd', { action: 'start' })).toEqual(['d', 'a', 'b', 'c']);
  });

  it('is a no-op at the end it moves towards; a place past the end means the back', () => {
    expect(applyQueueMove(q, 'a', { action: 'up' })).toEqual(q);
    expect(applyQueueMove(q, 'd', { action: 'down' })).toEqual(q);
    expect(applyQueueMove(q, 'a', { position: 99 })).toEqual(['b', 'c', 'd', 'a']);
  });

  it('is null for a run that does not wait', () => {
    expect(applyQueueMove(q, 'x', { action: 'top' })).toBeNull();
  });
});

describe('withQueueMove', () => {
  const run = (id: string, queue_position: number | null): Run => ({ id, queue_position }) as Run;

  it('renumbers the waiting runs and leaves the others alone', () => {
    const list = [run('live', null), run('b', 2), run('a', 1), run('done', null)];
    const next = withQueueMove(list, 'b', { action: 'start' });
    expect(next.map((r) => [r.id, r.queue_position])).toEqual([
      ['live', null],
      ['b', 1],
      ['a', 2],
      ['done', null],
    ]);
    expect(list[1]!.queue_position).toBe(2);
  });
});

describe('queue words', () => {
  it('says the place as people do', () => {
    expect([1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101].map(ordinal)).toEqual(['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '23rd', '101st']);
  });

  it('reads optional features from /api/system, absent on older servers', () => {
    expect(hasFeature({ features: ['queue_control'] } as never, 'queue_control')).toBe(true);
    expect(hasFeature({} as never, 'queue_control')).toBe(false);
    expect(hasFeature(undefined, 'weights_onnx')).toBe(false);
  });
});
