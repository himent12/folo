// @vitest-environment happy-dom
// Sprint 11 regression tests for fw runtime: ComputedNode._subscribe notification ordering,
// ReactiveMap/ReactiveSet keySignal re-registration after unwatched mutation, spread with nullish
// props + style: directives, and toClass with empty-string directive key.
import { describe, expect, it, vi } from 'vitest';
import { computed, createRoot, effect, flush, signal, watchedSignal } from './core';
import { toClass } from './dom';
import { ReactiveMap, ReactiveSet } from './reactivity';
import { spread } from './spread';

describe('fw Sprint 11 runtime hardening', () => {
  it('delivers synchronous notifications from watchedSignal watch(true) through ComputedNode._subscribe', () => {
    let bumpOnWatch = false;
    let watches = 0;
    const src = watchedSignal(0, (on) => {
      watches += on ? 1 : -1;
      if (on && bumpOnWatch) src.set(src.peek() + 1);
    });
    const derived = computed(() => src.get() * 10);
    // Prime derived unwatched so it holds `src` in `_sources` with no targets yet.
    expect(derived.peek()).toBe(0);
    expect(watches).toBe(0);

    bumpOnWatch = true;
    const seen: number[] = [];
    const stop = createRoot((dispose) => {
      effect(() => {
        seen.push(derived.get());
      });
      return dispose;
    });
    flush();
    // Subscribing triggers watch(true) which bumps src to 1 before refresh, so the effect sees 10 immediately
    // and stays cleanly subscribed for subsequent writes.
    expect(seen).toEqual([10]);
    expect(watches).toBe(1);

    src.set(2);
    flush();
    expect(seen).toEqual([10, 20]);

    stop();
    expect(watches).toBe(0);
  });

  it('unsubscribes earlier sources when a later source throws during ComputedNode._subscribe', () => {
    let firstWatches = 0;
    let shouldThrow = false;
    const first = watchedSignal(1, (on) => {
      firstWatches += on ? 1 : -1;
    });
    const second = watchedSignal(2, (on) => {
      if (on && shouldThrow) throw new Error('watch-boom');
    });
    const derived = computed(() => first.get() + second.get());
    expect(derived.peek()).toBe(3);
    expect(firstWatches).toBe(0);

    shouldThrow = true;
    const stop = createRoot((dispose) => {
      effect(() => {
        void derived.get();
      });
      return dispose;
    });
    expect(() => flush()).toThrow('watch-boom');
    stop();
    // Because `subscribeLink(this, node)` ran before recursing into sources, disposing the effect unsubscribes `first`
    // cleanly instead of leaking `firstWatches === 1` forever.
    expect(firstWatches).toBe(0);
  });

  it('keeps ReactiveMap and ReactiveSet keySignals registered when re-subscribed after an unwatched mutation', () => {
    const map = new ReactiveMap<string, number>([['a', 1]]);
    const positive = computed(() => (map.get('a') ?? 0) > 0);
    // First watch and unwatch so the keySignal snapshots `1`.
    const stop1 = createRoot((dispose) => {
      effect(() => void positive.get());
      return dispose;
    });
    flush();
    stop1();

    // Mutate while unwatched so `!same` holds on re-subscribe, while `positive` stays `true`.
    map.set('a', 2);
    const mapSeen: boolean[] = [];
    const stop2 = createRoot((dispose) => {
      effect(() => {
        mapSeen.push(positive.get());
      });
      return dispose;
    });
    flush();
    expect(mapSeen).toEqual([true]);

    // Subsequent mutation to the same key must still notify the active subscriber.
    map.set('a', 0);
    flush();
    expect(mapSeen).toEqual([true, false]);
    stop2();

    // Same for ReactiveSet.
    const set = new ReactiveSet<string>(['x']);
    const hasX = computed(() => set.has('x'));
    const s1 = createRoot((dispose) => {
      effect(() => void hasX.get());
      return dispose;
    });
    flush();
    s1();

    set.delete('x');
    const setSeen: boolean[] = [];
    const s2 = createRoot((dispose) => {
      effect(() => {
        setSeen.push(hasX.get());
      });
      return dispose;
    });
    flush();
    expect(setSeen).toEqual([false]);

    set.add('x');
    flush();
    expect(setSeen).toEqual([false, true]);
    s2();
  });

  it('spread with style directives does not throw when props object is null or undefined', () => {
    const el = document.createElement('div');
    const props = signal<Record<string, unknown> | null | undefined>(null);
    const stop = createRoot((dispose) => {
      spread(
        el,
        () => props.get() as Record<string, unknown>,
        undefined,
        undefined,
        () => ({ color: 'red' }),
      );
      return dispose;
    });
    flush();
    expect(el.style.color).toBe('red');

    props.set(undefined);
    flush();
    expect(el.style.color).toBe('red');

    props.set({ style: 'display: flex' });
    flush();
    expect(el.style.display).toBe('flex');
    expect(el.style.color).toBe('red');
    stop();
  });

  it('toClass ignores an empty-string directive key instead of looping infinitely or adding trailing spaces', () => {
    expect(toClass('alpha beta', undefined, { '': false, alpha: false })).toBe('beta');
    expect(toClass('alpha', 'fw-s', { '': false })).toBe('alpha fw-s');
    expect(toClass('alpha', undefined, { '': true })).toBe('alpha');
    const spy = vi.fn(() => toClass('solo', undefined, { '': 0 }));
    expect(spy()).toBe('solo');
  });
});
