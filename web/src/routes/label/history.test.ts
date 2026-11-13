import { describe, expect, it } from 'vitest';
import { History, HISTORY_MAX } from './history';

describe('the editor history', () => {
  it('undoes and redoes edits in order, and a new edit drops what could be redone', () => {
    const h = new History<string>();
    h.record('a');
    h.record('b');
    expect(h.undo('c')).toBe('b');
    expect(h.undo('b')).toBe('a');
    expect(h.undo('a')).toBeUndefined();
    expect(h.redo('a')).toBe('b');
    expect(h.redo('b')).toBe('c');
    expect(h.redo('c')).toBeUndefined();
    h.undo('c');
    h.record('b');
    expect(h.future).toEqual([]);
  });

  it('joins a run of edits of one kind into one step, and starts a new one after an undo or another kind', () => {
    const h = new History<number>();
    h.record(0, 'move 1');
    h.record(1, 'move 1');
    h.record(2, 'move 1');
    h.record(3, 'move 2');
    expect(h.past).toEqual([0, 3]);
    expect(h.undo(4)).toBe(3);
    h.record(3, 'move 2');
    expect(h.past).toEqual([0, 3]);
    h.record(5, null);
    h.record(6, null);
    expect(h.past).toEqual([0, 3, 5, 6]);
  });

  it('keeps the newest steps', () => {
    const h = new History<number>();
    for (let i = 0; i < HISTORY_MAX + 5; i++) h.record(i);
    expect(h.past.length).toBe(HISTORY_MAX);
    expect(h.past[0]).toBe(5);
  });
});
