// @vitest-environment happy-dom
import { describe, expect, it, vi } from 'vitest';
import { computed, createRoot, deep, effect, flush, MediaQuery, memo, onCleanup, ReactiveMap, ReactiveSet, renderEffect, signal, snapshot, Tween } from 'fw';

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

  it('prunes unsubscribed key signals in ReactiveMap and ReactiveSet when the last reader disposes', () => {
    const deleteSpy = vi.spyOn(Map.prototype, 'delete');
    const setSpy = vi.spyOn(Map.prototype, 'set');
    try {
      const m = new ReactiveMap<string, number>([['a', 1]]);
      const s = new ReactiveSet<string>(['x']);

      // Unwatched computeds do not retain entries in #keys, yet still stay fresh across mutations.
      setSpy.mockClear();
      const looseMap = computed(() => m.get('loose'));
      const looseSet = computed(() => s.has('loose'));
      expect(looseMap.get()).toBeUndefined();
      expect(looseSet.get()).toBe(false);
      expect(setSpy.mock.calls.filter(([k]) => k === 'loose')).toHaveLength(0);

      m.set('loose', 7);
      s.add('loose');
      expect(looseMap.get()).toBe(7);
      expect(looseSet.get()).toBe(true);

      // Two subscribers on the same key: disposing the first keeps the key alive; disposing the second prunes it.
      const seenA1: (number | undefined)[] = [];
      const seenA2: (number | undefined)[] = [];
      const seenX1: boolean[] = [];
      const seenX2: boolean[] = [];
      const d1 = createRoot((d) => {
        effect(() => void seenA1.push(m.get('a')));
        effect(() => void seenX1.push(s.has('x')));
        return d;
      });
      const d2 = createRoot((d) => {
        effect(() => void seenA2.push(m.get('a')));
        effect(() => void seenX2.push(s.has('x')));
        return d;
      });
      flush();

      deleteSpy.mockClear();
      d1();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'a' || k === 'x')).toHaveLength(0);

      m.set('a', 2);
      s.delete('x');
      flush();
      expect(seenA1).toEqual([1]);
      expect(seenA2).toEqual([1, 2]);
      expect(seenX1).toEqual([true]);
      expect(seenX2).toEqual([true, false]);

      deleteSpy.mockClear();
      d2();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'a')).toHaveLength(1);
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'x')).toHaveLength(1);

      // Re-subscribing after pruning works normally.
      const seenA3: (number | undefined)[] = [];
      const d3 = createRoot((d) => {
        effect(() => void seenA3.push(m.get('a')));
        return d;
      });
      flush();
      m.set('a', 3);
      flush();
      expect(seenA3).toEqual([2, 3]);
      d3();
    } finally {
      deleteSpy.mockRestore();
      setSpy.mockRestore();
    }
  });

  it('keeps computeds and owning memos fresh across watch/unwatch cycles and preserves per-key granularity', () => {
    const deleteSpy = vi.spyOn(Map.prototype, 'delete');
    try {
      const m = new ReactiveMap<string, number>([['a', 1]]);
      let cRuns = 0;
      const c = computed(() => {
        cRuns++;
        return m.get('a');
      });

      // 1. Watched -> unwatched -> mutated -> unwatched read & re-watch.
      const out1: (number | undefined)[] = [];
      const d1 = createRoot((d) => {
        effect(() => void out1.push(c.get()));
        return d;
      });
      flush();
      expect(out1).toEqual([1]);
      expect(cRuns).toBe(1);

      // Unrelated key does not recompute c.
      m.set('b', 10);
      flush();
      expect(cRuns).toBe(1);

      // Disposing unsubscribes and prunes 'a'.
      deleteSpy.mockClear();
      d1();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'a')).toHaveLength(1);

      // Mutate while unwatched, then read unwatched and re-watch.
      m.set('a', 2);
      expect(c.get()).toBe(2);

      // Mutate again while unwatched, then re-watch under an effect alongside a second subscriber.
      m.set('a', 3);
      const out2: (number | undefined)[] = [];
      const out3: (number | undefined)[] = [];
      const d2 = createRoot((d) => {
        effect(() => void out2.push(m.get('a')));
        return d;
      });
      const d3 = createRoot((d) => {
        effect(() => void out3.push(c.get()));
        return d;
      });
      flush();
      expect(out2).toEqual([3]);
      expect(out3).toEqual([3]);

      m.set('a', 4);
      flush();
      expect(out2).toEqual([3, 4]);
      expect(out3).toEqual([3, 4]);

      // Neither prunes until both dispose.
      deleteSpy.mockClear();
      d2();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'a')).toHaveLength(0);
      m.set('a', 5);
      flush();
      expect(out3).toEqual([3, 4, 5]);
      d3();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'a')).toHaveLength(1);

      // 2. Owning memo (JSX branch) keeps per-key granularity from mount and prunes on dispose.
      let branchRuns = 0;
      const outMemo: (number | undefined)[] = [];
      const dMemo = createRoot((d) => {
        const read = memo(() => {
          branchRuns++;
          return m.get('target');
        });
        effect(() => void outMemo.push(read()));
        return d;
      });
      flush();
      expect(branchRuns).toBe(1);
      m.set('unrelated', 99);
      flush();
      expect(branchRuns).toBe(1);
      m.set('target', 42);
      flush();
      expect(branchRuns).toBe(2);
      expect(outMemo).toEqual([undefined, 42]);

      deleteSpy.mockClear();
      dMemo();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'target')).toHaveLength(1);
    } finally {
      deleteSpy.mockRestore();
    }
  });

  it('does not tear down or double-mount memos on same-key collisions and handles delete/clear and special keys', () => {
    const deleteSpy = vi.spyOn(Map.prototype, 'delete');
    try {
      const m = new ReactiveMap<unknown, number>([
        ['k', 1],
        [Number.NaN, 10],
        [0, 20],
      ]);
      const objKey = { id: 1 };
      m.set(objKey, 30);

      // 1. Single memo reading has(k) and get(k) mounts once with zero cleanups.
      let singleMounts = 0;
      let singleCleanups = 0;
      const singleOut: (number | null)[] = [];
      const dSingle = createRoot((d) => {
        const read = memo(() => {
          singleMounts++;
          onCleanup(() => singleCleanups++);
          return m.has('k') ? m.get('k')! : null;
        });
        renderEffect(() => {
          singleOut.push(read());
        });
        return d;
      });
      expect(singleMounts).toBe(1);
      expect(singleCleanups).toBe(0);
      expect(singleOut).toEqual([1]);

      // 2. Parent memo + inner renderEffect reading the same key mounts once with zero cleanups.
      let parentMounts = 0;
      let parentCleanups = 0;
      const childOut: (number | undefined)[] = [];
      const dParent = createRoot((d) => {
        const read = memo(() => {
          parentMounts++;
          onCleanup(() => parentCleanups++);
          const v = m.get('p');
          renderEffect(() => {
            childOut.push(m.get('p'));
          });
          return v;
        });
        renderEffect(() => {
          read();
        });
        return d;
      });
      expect(parentMounts).toBe(1);
      expect(parentCleanups).toBe(0);
      expect(childOut).toEqual([undefined]);

      // 3. Two sibling memos reading the same key mount once each with zero cleanups.
      let sib1Mounts = 0;
      let sib2Mounts = 0;
      let sibCleanups = 0;
      const dSibs = createRoot((d) => {
        const r1 = memo(() => {
          sib1Mounts++;
          onCleanup(() => sibCleanups++);
          return m.get('s');
        });
        const r2 = memo(() => {
          sib2Mounts++;
          onCleanup(() => sibCleanups++);
          return m.get('s');
        });
        renderEffect(() => {
          r1();
          r2();
        });
        return d;
      });
      expect(sib1Mounts).toBe(1);
      expect(sib2Mounts).toBe(1);
      expect(sibCleanups).toBe(0);

      // Writes update all collided readers, and disposal prunes cleanly.
      m.set('k', 2);
      m.set('p', 3);
      m.set('s', 4);
      flush();
      expect(singleOut).toEqual([1, 2]);
      expect(parentMounts).toBe(2);
      expect(childOut[childOut.length - 1]).toBe(3);
      expect(sib1Mounts).toBe(2);
      expect(sib2Mounts).toBe(2);

      deleteSpy.mockClear();
      dSingle();
      dParent();
      dSibs();
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'k')).toHaveLength(1);
      expect(deleteSpy.mock.calls.filter(([k]) => k === 'p')).toHaveLength(1);
      expect(deleteSpy.mock.calls.filter(([k]) => k === 's')).toHaveLength(1);

      // 4. Unrelated writes before first subscription do not tear down an eager memo,
      // and unwatched computeds preserve per-key memoization across unrelated writes.
      let eagerMounts = 0;
      let eagerCleanups = 0;
      const dEager = createRoot((d) => {
        const read = memo(() => {
          eagerMounts++;
          onCleanup(() => eagerCleanups++);
          const v = m.get('watched');
          m.set('other', 99);
          return v;
        });
        m.set('other2', 100);
        renderEffect(() => {
          read();
        });
        return d;
      });
      expect(eagerMounts).toBe(1);
      expect(eagerCleanups).toBe(0);
      dEager();

      let unwatchedRuns = 0;
      const unwatchedBox = computed(() => {
        unwatchedRuns++;
        return { v: m.get('watched') };
      });
      const firstBox = unwatchedBox.get();
      m.set('unrelated_key', 123);
      expect(unwatchedBox.get()).toBe(firstBox);
      expect(unwatchedRuns).toBe(1);
      m.set('watched', 77);
      expect(unwatchedBox.get()).toEqual({ v: 77 });
      expect(unwatchedRuns).toBe(2);

      // 5. Disposing one of two collided siblings before any write keeps the survivor active and prunes on last dispose.
      let survRuns = 0;
      const dColl1 = createRoot((d) => {
        const r = memo(() => m.get('coll'));
        renderEffect(() => void r());
        return d;
      });
      let dColl2!: () => void;
      const dWrap = createRoot((d) => {
        const r1 = memo(() => m.get('coll2'));
        const r2 = memo(() => {
          survRuns++;
          return m.get('coll2');
        });
        const dA = createRoot((da) => {
          renderEffect(() => void r1());
          return da;
        });
        dColl2 = createRoot((db) => {
          renderEffect(() => void r2());
          return db;
        });
        deleteSpy.mockClear();
        dA();
        expect(deleteSpy.mock.calls.filter(([k]) => k === 'coll2')).toHaveLength(0);
        m.set('coll2', 55);
        flush();
        expect(survRuns).toBe(2);
        deleteSpy.mockClear();
        dColl2();
        expect(deleteSpy.mock.calls.filter(([k]) => k === 'coll2')).toHaveLength(1);
        return d;
      });
      dColl1();
      dWrap();

      // 6. Special keys (NaN, -0/+0, object) notify independently and prune cleanly.
      const seenNan: (number | undefined)[] = [];
      const seenZero: (number | undefined)[] = [];
      const seenObj: (number | undefined)[] = [];
      const dSpec = createRoot((d) => {
        effect(() => void seenNan.push(m.get(Number.NaN)));
        effect(() => void seenZero.push(m.get(-0)));
        effect(() => void seenObj.push(m.get(objKey)));
        return d;
      });
      flush();
      expect(seenNan).toEqual([10]);
      expect(seenZero).toEqual([20]);
      expect(seenObj).toEqual([30]);
      m.set(Number.NaN, 11);
      flush();
      expect(seenNan).toEqual([10, 11]);
      expect(seenZero).toEqual([20]);
      expect(seenObj).toEqual([30]);
      m.set(0, 21);
      flush();
      expect(seenZero).toEqual([20, 21]);
      expect(seenObj).toEqual([30]);
      m.set(objKey, 31);
      flush();
      expect(seenObj).toEqual([30, 31]);
      deleteSpy.mockClear();
      dSpec();
      expect(deleteSpy.mock.calls.filter(([k]) => Number.isNaN(k as number))).toHaveLength(1);
      expect(deleteSpy.mock.calls.filter(([k]) => k === 0)).toHaveLength(1);
      expect(deleteSpy.mock.calls.filter(([k]) => k === objKey)).toHaveLength(1);

      // 7. delete() and clear() invalidate unwatched computeds on ReactiveMap and ReactiveSet.
      const looseMap = computed(() => m.get('k'));
      const rs = new ReactiveSet<string>(['u', 'v']);
      const looseSetU = computed(() => rs.has('u'));
      const looseSetV = computed(() => rs.has('v'));
      expect(looseMap.get()).toBe(2);
      expect(looseSetU.get()).toBe(true);
      expect(looseSetV.get()).toBe(true);
      m.delete('k');
      expect(looseMap.get()).toBeUndefined();
      m.set('k', 9);
      expect(looseMap.get()).toBe(9);
      m.clear();
      expect(looseMap.get()).toBeUndefined();
      rs.delete('u');
      expect(looseSetU.get()).toBe(false);
      expect(looseSetV.get()).toBe(true);
      rs.clear();
      expect(looseSetV.get()).toBe(false);
    } finally {
      deleteSpy.mockRestore();
    }
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

  it('skips scheduling rAF on no-op target updates and duplicate in-flight targets, and handles zero dynamic duration', async () => {
    const rafSpy = vi.spyOn(globalThis, 'requestAnimationFrame');
    const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame');
    try {
      const t = new Tween(5, { duration: 50 });
      rafSpy.mockClear();
      await t.set(5);
      expect(rafSpy).not.toHaveBeenCalled();

      // Duplicate in-flight set with no options reuses the running promise without restarting.
      const p1 = t.set(20);
      const p2 = t.set(20);
      expect(p2).toBe(p1);
      // Explicit options override (duration: 0) snaps immediately and cancels the pending frame.
      await t.set(20, { duration: 0 });
      expect(t.current).toBe(20);
      expect(cancelSpy).toHaveBeenCalled();
      rafSpy.mockClear();
      await t.set(20);
      expect(rafSpy).not.toHaveBeenCalled();

      // Dynamic duration returning 0 snaps cleanly without NaN.
      const zeroDur = new Tween(1, { duration: () => 0 });
      await zeroDur.set(9);
      expect(zeroDur.current).toBe(9);
    } finally {
      rafSpy.mockRestore();
      cancelSpy.mockRestore();
    }
  });

  it('aborts active and chained delayed tasks on owner cleanup, and Tween.of skips initial mount rAF', () => {
    const rafQueue = new Map<number, FrameRequestCallback>();
    let nextRaf = 1;
    const origRaf = globalThis.requestAnimationFrame;
    const origCancel = globalThis.cancelAnimationFrame;
    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      const id = nextRaf++;
      rafQueue.set(id, cb);
      return id;
    }) as typeof requestAnimationFrame;
    globalThis.cancelAnimationFrame = ((id: number) => {
      rafQueue.delete(id);
    }) as typeof cancelAnimationFrame;

    try {
      const src = signal(10);
      const reads = vi.fn(() => src.get());
      let tw!: Tween<number>;
      const dispose = createRoot((d) => {
        tw = Tween.of(reads, { duration: 100 });
        return d;
      });
      // Initial mount reads once and schedules zero frames.
      expect(reads).toHaveBeenCalledTimes(1);
      expect(tw.current).toBe(10);
      expect(rafQueue.size).toBe(0);

      // Start an animation, then queue a delayed animation on top, then dispose owner.
      src.set(50);
      flush();
      expect(rafQueue.size).toBe(1);
      void tw.set(100, { delay: 50, duration: 100 });
      expect(rafQueue.size).toBe(1);

      dispose();
      // Both tasks are aborted and the pending rAF is cancelled.
      expect(rafQueue.size).toBe(0);
    } finally {
      globalThis.requestAnimationFrame = origRaf;
      globalThis.cancelAnimationFrame = origCancel;
    }
  });

  it('handles zero dynamic duration at elapsed 0, re-entrant Tween.of init, mid-frame disposal, and nested snap without extra rAF', async () => {
    const rafQueue = new Map<number, FrameRequestCallback>();
    let nextRaf = 1;
    let fakeNow = 1000;
    const origRaf = globalThis.requestAnimationFrame;
    const origCancel = globalThis.cancelAnimationFrame;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => fakeNow);
    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      const id = nextRaf++;
      rafQueue.set(id, cb);
      return id;
    }) as typeof requestAnimationFrame;
    globalThis.cancelAnimationFrame = ((id: number) => {
      rafQueue.delete(id);
    }) as typeof cancelAnimationFrame;
    const step = (ms: number) => {
      fakeNow += ms;
      const batch = [...rafQueue.values()];
      rafQueue.clear();
      for (const cb of batch) cb(fakeNow);
    };

    try {
      // 1. Dynamic duration returning 0 at elapsed === 0 skips interpolator and does not leave a frame queued.
      const interp = vi.fn(() => () => 0);
      const other = new Tween(0, { duration: 100 });
      const zeroDur = new Tween<number>(1, {
        interpolate: interp,
        duration: () => {
          // Spawning and immediately snapping another tween mid-tick must not leave an extra rAF queued.
          void other.set(5);
          void other.set(5, { duration: 0 });
          return 0;
        },
      });
      const done = zeroDur.set(9);
      step(0);
      await done;
      expect(zeroDur.current).toBe(9);
      expect(interp).not.toHaveBeenCalled();
      expect(rafQueue.size).toBe(0);
      // Identical set after natural completion (standard interpolator) schedules no frame.
      const std = new Tween(1, { duration: 50 });
      const stdDone = std.set(9);
      step(50);
      await stdDone;
      expect(std.current).toBe(9);
      expect(rafQueue.size).toBe(0);
      await std.set(9);
      expect(rafQueue.size).toBe(0);

      // 2. Easing callback synchronously disposing owner must not invoke interpolator fn or write after cleanup.
      const interpStep = vi.fn((t: number) => t * 100);
      let tw!: Tween<number>;
      let dispose!: () => void;
      dispose = createRoot((d) => {
        tw = new Tween(0, {
          duration: 100,
          interpolate: () => interpStep,
          easing: (t) => {
            dispose();
            return t;
          },
        });
        return d;
      });
      void tw.set(100);
      step(50);
      expect(interpStep).not.toHaveBeenCalled();
      expect(tw.current).toBe(0);
      expect(rafQueue.size).toBe(0);

      // 3. Tween.of survives re-entrant flush during initial fn() evaluation.
      const s = signal(1);
      let bumped = false;
      let reentrant!: Tween<number>;
      const dispose2 = createRoot((d) => {
        reentrant = Tween.of(
          () => {
            const v = s.get();
            if (!bumped) {
              bumped = true;
              s.set(2);
              flush();
            }
            return v;
          },
          { duration: 50 },
        );
        return d;
      });
      expect(reentrant.current).toBe(2);
      expect(rafQueue.size).toBe(0);
      dispose2();

      // 4. Tween.of does not start a fresh animation if fn() disposes its own owner.
      const src3 = signal(1);
      let tw3!: Tween<number>;
      let dispose3!: () => void;
      dispose3 = createRoot((d) => {
        tw3 = Tween.of(
          () => {
            const v = src3.get();
            if (v === 2) dispose3();
            return v;
          },
          { duration: 50 },
        );
        return d;
      });
      src3.set(2);
      flush();
      expect(rafQueue.size).toBe(0);
      expect(tw3.current).toBe(1);

      // 5. Throwing duration callback removes broken task, lets sibling tweens advance, and stops scheduling.
      const throwing = new Tween(0, {
        duration: () => {
          throw new Error('boom');
        },
      });
      const sibling = new Tween(0, { duration: 100 });
      void throwing.set(10);
      void sibling.set(100);
      expect(() => step(50)).toThrow('boom');
      expect(sibling.current).toBe(50);
      step(50);
      expect(sibling.current).toBe(100);
      expect(rafQueue.size).toBe(0);

      // 6. NaN duration snaps cleanly without an infinite rAF loop.
      const nanStatic = new Tween(0, { duration: Number.NaN });
      await nanStatic.set(7);
      expect(nanStatic.current).toBe(7);
      expect(rafQueue.size).toBe(0);
      const nanDyn = new Tween(0, { duration: () => Number.NaN });
      const nanDone = nanDyn.set(8);
      step(0);
      await nanDone;
      expect(nanDyn.current).toBe(8);
      expect(rafQueue.size).toBe(0);

      // 7. Custom interpolator still runs even when target matches current.
      const custom = new Tween(0, { duration: 100 });
      const bump = custom.set(0, { interpolate: () => (t) => (t < 1 ? 42 : 0) });
      step(50);
      expect(custom.current).toBe(42);
      step(50);
      await bump;
      expect(custom.current).toBe(0);
      expect(rafQueue.size).toBe(0);
    } finally {
      nowSpy.mockRestore();
      globalThis.requestAnimationFrame = origRaf;
      globalThis.cancelAnimationFrame = origCancel;
    }
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
