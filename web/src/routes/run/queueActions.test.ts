import { describe, expect, it } from 'vitest';
import { queueItems, REMOVE_LABEL } from './queueActions';

const labels = (...a: Parameters<typeof queueItems>) => queueItems(...a).map((i) => i.label);
const noop = () => {};

describe('queueItems: one queue action set, one order, wherever it appears', () => {
  it('lists Start next, Move up, Move down, then Remove from queue', () => {
    expect(labels(2, 3, { name: 'live' }, noop, { remove: noop })).toEqual(['Start next', 'Move up', 'Move down', REMOVE_LABEL]);
  });

  it('leaves out the moves that do not apply and the actions the page already shows as buttons', () => {
    expect(labels(1, 2, { name: 'live' }, noop, { remove: noop })).toEqual(['Move down', REMOVE_LABEL]);
    // The run page: Start next is the banner's button and Remove from queue the header's.
    expect(labels(2, 3, { name: 'live' }, noop, { start: false })).toEqual(['Move up', 'Move down']);
  });

  it('marks Remove from queue destructive', () => {
    expect(queueItems(1, 1, null, noop, { remove: noop }).at(-1)?.danger).toBe(true);
  });
});
