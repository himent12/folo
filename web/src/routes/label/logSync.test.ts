import { describe, expect, it } from 'vitest';
import { AppendLog } from '$lib/api/appendLog';
import { syncLog } from './logSync';

describe('syncLog', () => {
  it('appends what a new list adds, trims what its cap dropped, and replaces on anything else', () => {
    const [a, b, c, d, e, x] = ['a', 'b', 'c', 'd', 'e', 'x'].map((t) => ({ t }));
    const log = new AppendLog<{ t: string }>();
    syncLog(log, [a!, b!]);
    let r = log.read(null);
    expect(r.added).toEqual([a, b]);
    // A line added: an append of one.
    syncLog(log, [a!, b!, c!]);
    r = log.read(r.cursor);
    expect([r.reset, r.dropped, r.added]).toEqual([false, 0, [c]]);
    // Two added while the cap drops the first: a trim and an append.
    syncLog(log, [b!, c!, d!, e!]);
    r = log.read(r.cursor);
    expect([r.reset, r.dropped, r.added]).toEqual([false, 1, [d, e]]);
    // A line merged in between (an older time): the log is replaced.
    syncLog(log, [b!, x!, c!, d!, e!]);
    r = log.read(r.cursor);
    expect([r.reset, r.added]).toEqual([true, [b, x, c, d, e]]);
    expect(log.items).toEqual([b, x, c, d, e]);
    // Nothing new: nothing to read.
    syncLog(log, [b!, x!, c!, d!, e!]);
    r = log.read(r.cursor);
    expect([r.reset, r.dropped, r.added]).toEqual([false, 0, []]);
  });
});
