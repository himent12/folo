// @vitest-environment happy-dom
import { createRoot, flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import { toast } from '$lib/ui/toast';
import { followVanished, vanished } from './vanished';

const m = (id: string) => ({ id, name: `${id}-best` }) as Model;

describe('vanished', () => {
  it('is the shown model once it left the list and another took its place', () => {
    expect(vanished(m('a'), m('b'), [m('b')])?.id).toBe('a');
    // The user picked another model: the first one is still there.
    expect(vanished(m('a'), m('b'), [m('a'), m('b')])).toBeNull();
    expect(vanished(null, m('b'), [m('b')])).toBeNull();
    expect(vanished(m('a'), null, [])).toBeNull();
  });
});

describe('followVanished', () => {
  afterEach(() => (toast.items = []));

  it('says so once and adopts the replacement', () => {
    const list = signal<Model[]>([m('a'), m('b')]);
    const adopted: string[] = [];
    const stop = createRoot((dispose) => {
      followVanished(
        () => list.get()[0] ?? null,
        () => list.get(),
        (id) => adopted.push(id),
      );
      return dispose;
    });
    flushSync();
    list.set([m('b')]);
    flushSync();
    list.set([m('b'), m('c')]);
    flushSync();
    expect(adopted).toEqual(['b']);
    expect(toast.items.map((t) => t.title)).toEqual(['a-best no longer exists']);
    stop();
  });
});
