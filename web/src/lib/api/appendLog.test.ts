import { describe, expect, it } from 'vitest';
import { AppendLog } from './appendLog';

describe('AppendLog', () => {
  it('gives a reader what was appended since its cursor, and a reset when the list was replaced', () => {
    const log = new AppendLog<number>();
    let r = log.read(null);
    expect(r).toMatchObject({ reset: true, added: [], dropped: 0 });
    log.push(1, 2);
    log.items.push(3);
    r = log.read(r.cursor);
    expect(r).toMatchObject({ reset: false, added: [1, 2, 3], dropped: 0 });
    const items = log.items;
    // A replacement that starts with the same objects is an append of the rest.
    log.replace([...items, 4]);
    expect(log.items).toBe(items);
    r = log.read(r.cursor);
    expect(r).toMatchObject({ reset: false, added: [4] });
    log.replace([1, 2]);
    r = log.read(r.cursor);
    expect(r).toMatchObject({ reset: true, added: [1, 2] });
    expect(log.read(r.cursor).added).toEqual([]);
  });

  it('says how many items left the front, and resets a reader that fell behind them', () => {
    const log = new AppendLog<number>();
    log.push(1, 2, 3, 4);
    const c = log.read(null).cursor;
    log.push(5);
    log.trimFront(2);
    expect(log.items).toEqual([3, 4, 5]);
    expect(log.read(c)).toMatchObject({ reset: false, added: [5], dropped: 2 });
    const behind = log.read(null).cursor;
    log.push(6, 7, 8, 9);
    log.trimFront(5);
    expect(log.read(behind)).toMatchObject({ reset: true, added: [8, 9] });
  });
});
