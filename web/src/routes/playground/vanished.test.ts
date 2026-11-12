// @vitest-environment happy-dom
import { createRoot, flush as flushSync, signal } from 'fw';
import { afterEach, describe, expect, it } from 'vitest';
import type { Model } from '$lib/api/types';
import { toast } from '$lib/ui/toast';
import { followVanished, missing, vanished } from './vanished';

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

describe('missing', () => {
  it('is the asked-for id when the loaded list has no such model', () => {
    expect(missing('m_gone', [m('a')])).toBe('m_gone');
    expect(missing('a', [m('a')])).toBeNull();
    // Nothing asked, or the list not in yet: nothing to say.
    expect(missing(null, [m('a')])).toBeNull();
    expect(missing('m_gone', [])).toBeNull();
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
        () => null,
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

  // v-models-label N6: a link to a deleted model showed another one without a word.
  it('names the model a link to a missing one shows instead, once, and adopts it', () => {
    const list = signal<Model[]>([]);
    const wanted = signal<string | null>('m_gone');
    const adopted: string[] = [];
    const stop = createRoot((dispose) => {
      followVanished(
        () => list.get().find((x) => x.id === wanted.get()) ?? list.get()[0] ?? null,
        () => list.get(),
        () => wanted.get(),
        (id) => (adopted.push(id), wanted.set(id)),
      );
      return dispose;
    });
    flushSync();
    expect(toast.items).toEqual([]);
    list.set([m('a'), m('b')]);
    flushSync();
    list.set([m('a'), m('b'), m('c')]);
    flushSync();
    expect(adopted).toEqual(['a']);
    expect(toast.items.map((t) => [t.title, t.detail])).toEqual([['The linked model was not found', 'It was deleted or the link is wrong. Showing a-best instead.']]);
    stop();
  });
});
