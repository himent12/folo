// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';
import { computed, createRoot, deep, effect, flush, MediaQuery, ReactiveMap, ReactiveSet, snapshot, Tween } from 'fw';

const track = <T>(fn: () => T) => {
  const seen: T[] = [];
  createRoot(() => effect(() => void seen.push(fn())));
  flush();
  return seen;
};

describe('deep', () => {
  it('tracks nested properties, array mutation methods and length', () => {
    const s = deep({ user: { name: 'a' }, list: [1, 2] });
    const names = track(() => s.user.name);
    const sums = track(() => s.list.reduce((a, b) => a + b, 0));
    const lens = track(() => s.list.length);
    s.user.name = 'b';
    s.list.push(3);
    flush();
    s.list.splice(0, 1);
    flush();
    s.list[0] = 10;
    flush();
    expect(names).toEqual(['a', 'b']);
    expect(sums).toEqual([3, 6, 5, 13]);
    expect(lens).toEqual([2, 3, 2]);
  });

  it('notifies on added and deleted keys; snapshot gives plain data', () => {
    const s = deep<Record<string, number>>({ a: 1 });
    const keys = track(() => Object.keys(s).join(','));
    const hasB = track(() => 'b' in s);
    s.b = 2;
    flush();
    delete s.a;
    flush();
    expect(keys).toEqual(['a', 'a,b', 'b']);
    expect(hasB).toEqual([false, true]);
    expect(snapshot(s)).toEqual({ b: 2 });
  });
});

describe('ReactiveMap / ReactiveSet', () => {
  it('map reads track per key and size', () => {
    const m = new ReactiveMap<string, number>([['a', 1]]);
    const a = track(() => m.get('a'));
    const b = track(() => m.has('b'));
    const size = track(() => m.size);
    m.set('b', 2);
    flush();
    m.set('a', 5);
    flush();
    m.delete('b');
    flush();
    expect(a).toEqual([1, 5]);
    expect(b).toEqual([false, true, false]);
    expect(size).toEqual([1, 2, 2, 1]);
  });

  it('set reads track', () => {
    const s = new ReactiveSet<number>();
    const has1 = track(() => s.has(1));
    const all = track(() => [...s].join(','));
    s.add(1);
    flush();
    s.add(2);
    flush();
    s.clear();
    flush();
    expect(has1).toEqual([false, true, false]);
    expect(all).toEqual(['', '1', '1,2', '']);
  });
});

describe('MediaQuery', () => {
  it('reads the query live and wraps bare queries in parentheses', () => {
    const q = new MediaQuery('min-width: 1px');
    expect(typeof q.current).toBe('boolean');
    expect(q.current).toBe(matchMedia('(min-width: 1px)').matches);
    const c = computed(() => q.current);
    expect(c.get()).toBe(q.current);
  });
});

describe('Tween', () => {
  it('eases to the target and resolves', async () => {
    const t = new Tween(0, { duration: 30 });
    const done = t.set(10);
    expect(t.target).toBe(10);
    await done;
    expect(t.current).toBe(10);
    await t.set(20, { duration: 0 });
    expect(t.current).toBe(20);
  });
});

describe('MediaQuery per instance', () => {
  it('asks matchMedia at construction, so a stub applies, and listens only while tracked', () => {
    const real = window.matchMedia;
    let matches = true;
    const listeners = new Set<() => void>();
    window.matchMedia = ((q: string) =>
      ({
        media: q,
        get matches() {
          return matches;
        },
        addEventListener: (_: string, f: () => void) => listeners.add(f),
        removeEventListener: (_: string, f: () => void) => listeners.delete(f),
      }) as unknown as MediaQueryList) as typeof window.matchMedia;
    try {
      const q = new MediaQuery('(max-width: 767px)');
      expect(q.current).toBe(true);
      expect(listeners.size).toBe(0);
      const seen = track(() => q.current);
      expect(listeners.size).toBe(1);
      matches = false;
      for (const f of listeners) f();
      flush();
      expect(seen).toEqual([true, false]);
    } finally {
      window.matchMedia = real;
    }
  });
});

describe('MediaQuery in a computed', () => {
  it('listens while an effect reads the computed, and stops when the last reader goes (no leak per screen visit)', () => {
    const real = window.matchMedia;
    const listeners = new Set<() => void>();
    let matches = false;
    window.matchMedia = ((q: string) =>
      ({
        media: q,
        get matches() {
          return matches;
        },
        addEventListener: (_: string, f: () => void) => listeners.add(f),
        removeEventListener: (_: string, f: () => void) => listeners.delete(f),
      }) as unknown as MediaQueryList) as typeof window.matchMedia;
    try {
      const q = new MediaQuery('(max-width: 767px)');
      // Read in a computed nothing watches: nothing to listen for.
      const loose = computed(() => (q.current ? 'phone' : 'wide'));
      expect(loose.get()).toBe('wide');
      expect(listeners.size).toBe(0);
      for (let visit = 0; visit < 5; visit++) {
        const seen: string[] = [];
        const dispose = createRoot((d) => {
          const c = computed(() => (q.current ? 'phone' : 'wide'));
          effect(() => void seen.push(c.get()));
          return d;
        });
        flush();
        expect(listeners.size).toBe(1);
        matches = !matches;
        for (const f of [...listeners]) f();
        flush();
        expect(seen.length).toBe(2);
        dispose();
        expect(listeners.size).toBe(0);
      }
    } finally {
      window.matchMedia = real;
    }
  });
});
