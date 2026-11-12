import { describe, expect, it } from 'vitest';
import { queueItems, queueWait, REMOVE_LABEL, removeDescription } from './queueActions';

const labels = (...a: Parameters<typeof queueItems>) => queueItems(...a).map((i) => i.label);
const noop = () => {};

describe('queueItems: one queue action set, one order, wherever it appears', () => {
  it('lists Start next, Move up, Move down, then Remove from queue, the last destructive', () => {
    expect(labels(2, 3, { name: 'live' }, noop, noop)).toEqual(['Start next', 'Move up', 'Move down', REMOVE_LABEL]);
    expect(queueItems(1, 1, null, noop, noop).at(-1)?.danger).toBe(true);
  });

  it('leaves out only the moves that do not apply, never one the page also shows as a button', () => {
    expect(labels(1, 2, { name: 'live' }, noop, noop)).toEqual(['Move down', REMOVE_LABEL]);
    // The run page's ⋯ for the 2nd of two (its callout also shows Start next): the same set as its Runs row.
    expect(labels(2, 2, { name: 'live' }, noop, noop)).toEqual(['Start next', 'Move up', REMOVE_LABEL]);
    // Nothing trains: the first can start now.
    expect(labels(1, 1, null, noop, noop)).toEqual(['Start now', REMOVE_LABEL]);
  });
});

describe('queueWait: how long a waiting run waits', () => {
  it('sums what the training run has left and the lengths in front', () => {
    expect(queueWait(420, [{ lo: 3000, hi: 3600 }])).toEqual({ lo: 3420, hi: 4020, known: true });
    // Nothing trains: the queued runs alone.
    expect(queueWait(undefined, [{ lo: 60, hi: 120 }])).toEqual({ lo: 60, hi: 120, known: true });
  });

  it('is a floor, never an estimate, when any part is unknown', () => {
    expect(queueWait(420, [null])).toEqual({ lo: 420, hi: 420, known: false });
    expect(queueWait(null, [{ lo: 60, hi: 120 }])).toEqual({ lo: 60, hi: 120, known: false });
  });

  it('is null when nothing is known', () => {
    expect(queueWait(null, [null])).toBeNull();
    expect(queueWait(undefined, [])).toBeNull();
  });
});

describe('removeDescription: what Remove from queue does', () => {
  it('deletes a run that never started, and keeps what a resumed run trained', () => {
    expect(removeDescription({ started: null })).toMatch(/nothing is lost: it leaves the queue and the runs list/);
    expect(removeDescription({ started: '2026-11-07T06:00:03Z' })).toMatch(/keeps what it trained so far/);
  });
});
