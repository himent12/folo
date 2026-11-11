import { describe, expect, it, vi } from 'vitest';
import {
  computed,
  createContext,
  createRoot,
  effect,
  flush,
  getContext,
  memo,
  onCleanup,
  onMount,
  renderEffect,
  setContext,
  signal,
  tick,
  untrack,
} from './core';
import { createComponent } from './component';

describe('signal', () => {
  it('reads, writes, updates with a function, and skips equal writes', () => {
    const s = signal(1);
    expect(s.get()).toBe(1);
    s.set(2);
    expect(s.peek()).toBe(2);
    s.set((v) => v + 1);
    expect(s.get()).toBe(3);
    const runs = vi.fn();
    createRoot(() => effect(() => runs(s.get())));
    flush();
    s.set(3);
    flush();
    expect(runs).toHaveBeenCalledTimes(1);
  });

  it('equals: false notifies on every write', () => {
    const s = signal({ n: 1 }, { equals: false });
    const runs = vi.fn();
    createRoot(() => effect(() => runs(s.get())));
    flush();
    s.set(s.peek());
    flush();
    expect(runs).toHaveBeenCalledTimes(2);
  });
});

describe('computed', () => {
  it('is lazy and memoized', () => {
    const s = signal(1);
    const fn = vi.fn(() => s.get() * 2);
    const c = computed(fn);
    expect(fn).not.toHaveBeenCalled();
    expect(c.get()).toBe(2);
    expect(c.get()).toBe(2);
    expect(fn).toHaveBeenCalledTimes(1);
    s.set(2);
    expect(fn).toHaveBeenCalledTimes(1);
    expect(c.get()).toBe(4);
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('is glitch-free: a diamond recomputes once and an effect sees consistent values', () => {
    const a = signal(1);
    const b = computed(() => a.get() + 1);
    const c = computed(() => a.get() * 10);
    const d = vi.fn(() => `${b.get()}/${c.get()}`);
    const dc = computed(d);
    const seen: string[] = [];
    createRoot(() => effect(() => void seen.push(dc.get())));
    flush();
    a.set(2);
    a.set(3);
    flush();
    expect(seen).toEqual(['2/10', '4/30']);
    expect(d).toHaveBeenCalledTimes(2);
  });

  it('stops an effect re-running when the computed value did not change', () => {
    const s = signal(1);
    const parity = computed(() => s.get() % 2);
    const runs = vi.fn();
    createRoot(() => effect(() => runs(parity.get())));
    flush();
    s.set(3);
    flush();
    expect(runs).toHaveBeenCalledTimes(1);
    s.set(4);
    flush();
    expect(runs).toHaveBeenCalledTimes(2);
  });

  it('drops dependencies it no longer reads', () => {
    const which = signal(true);
    const a = signal(1);
    const b = signal(2);
    const fn = vi.fn(() => (which.get() ? a.get() : b.get()));
    const c = computed(fn);
    createRoot(() => effect(() => void c.get()));
    flush();
    which.set(false);
    flush();
    const n = fn.mock.calls.length;
    a.set(10);
    flush();
    expect(fn.mock.calls.length).toBe(n);
    b.set(20);
    flush();
    expect(c.peek()).toBe(20);
  });

  it('rethrows an error until a source changes', () => {
    const s = signal(0);
    const c = computed(() => {
      if (s.get() === 0) throw new Error('zero');
      return s.get();
    });
    expect(() => c.get()).toThrow('zero');
    expect(() => c.get()).toThrow('zero');
    s.set(1);
    expect(c.get()).toBe(1);
  });

  it('detects cycles', () => {
    const c: { get(): number } = computed((): number => c.get() + 1);
    expect(() => c.get()).toThrow();
  });
});

describe('effects', () => {
  it('run after a microtask or flush(), and again on change', async () => {
    const s = signal(1);
    const seen: number[] = [];
    createRoot(() => effect(() => void seen.push(s.get())));
    expect(seen).toEqual([]);
    await Promise.resolve();
    expect(seen).toEqual([1]);
    s.set(2);
    await tick();
    expect(seen).toEqual([1, 2]);
  });

  it('runs render effects before user effects in a flush', () => {
    const s = signal(0);
    const order: string[] = [];
    createRoot(() => {
      effect(() => void order.push(`user ${s.get()}`));
      renderEffect(() => void order.push(`render ${s.get()}`));
    });
    flush();
    s.set(1);
    flush();
    expect(order).toEqual(['render 0', 'user 0', 'render 1', 'user 1']);
  });

  it('runs a cleanup before the next run and on dispose, children first', () => {
    const s = signal(0);
    const log: string[] = [];
    const dispose = createRoot((d) => {
      effect(() => {
        const v = s.get();
        effect(() => () => log.push(`child ${v} cleanup`));
        return () => log.push(`cleanup ${v}`);
      });
      return d;
    });
    flush();
    s.set(1);
    flush();
    expect(log).toEqual(['child 0 cleanup', 'cleanup 0']);
    dispose();
    expect(log).toEqual(['child 0 cleanup', 'cleanup 0', 'child 1 cleanup', 'cleanup 1']);
    s.set(2);
    flush();
    expect(log.length).toBe(4);
  });

  it('runs parents before children; a child disposed by its parent does not run', () => {
    const s = signal(0);
    const log: string[] = [];
    createRoot(() =>
      effect(() => {
        const v = s.get();
        log.push(`parent ${v}`);
        effect(() => void log.push(`child ${v} sees ${s.get()}`));
      }),
    );
    flush();
    s.set(1);
    flush();
    expect(log).toEqual(['parent 0', 'child 0 sees 0', 'parent 1', 'child 1 sees 1']);
  });

  it('settles effects that write what other effects read', () => {
    const a = signal(1);
    const b = signal(0);
    const seen: number[] = [];
    createRoot(() => {
      effect(() => b.set(a.get() * 2));
      effect(() => void seen.push(b.get()));
    });
    flush();
    a.set(2);
    flush();
    expect(seen.at(-1)).toBe(4);
  });

  it('throws on an infinite update loop', () => {
    const s = signal(0);
    createRoot(() => effect(() => s.set(s.get() + 1)));
    expect(() => flush()).toThrow(/1000/);
  });

  it('untrack reads without subscribing', () => {
    const a = signal(1);
    const b = signal(1);
    const runs = vi.fn();
    createRoot(() => effect(() => runs(a.get() + untrack(() => b.get()))));
    flush();
    b.set(2);
    flush();
    expect(runs).toHaveBeenCalledTimes(1);
    a.set(2);
    flush();
    expect(runs).toHaveBeenLastCalledWith(4);
  });

  it('effect.pre runs synchronously, before render effects of the same flush', () => {
    const s = signal(0);
    const log: string[] = [];
    createRoot(() => {
      effect.pre(() => void log.push(`pre ${s.get()}`));
      renderEffect(() => void log.push(`render ${s.get()}`));
    });
    expect(log).toEqual(['pre 0', 'render 0']);
    s.set(1);
    flush();
    expect(log).toEqual(['pre 0', 'render 0', 'pre 1', 'render 1']);
  });

  it('onMount runs once, untracked; its cleanup on dispose', () => {
    const s = signal(0);
    const log: string[] = [];
    const dispose = createRoot((d) => {
      onMount(() => {
        log.push(`mount ${s.get()}`);
        return () => log.push('unmount');
      });
      return d;
    });
    flush();
    s.set(1);
    flush();
    dispose();
    expect(log).toEqual(['mount 0', 'unmount']);
  });

  it('onCleanup of a root runs on dispose', () => {
    const fn = vi.fn();
    createRoot((d) => {
      onCleanup(fn);
      d();
    });
    expect(fn).toHaveBeenCalledOnce();
  });
});

describe('context', () => {
  it('is scoped to the component that sets it and what it renders', () => {
    const KEY = Symbol();
    let inChild: unknown;
    let inSibling: unknown;
    const Child = () => {
      inChild = getContext(KEY);
      return null;
    };
    const Parent = () => {
      setContext(KEY, 'parent');
      createComponent(Child, {});
      return null;
    };
    const Sibling = () => {
      inSibling = getContext(KEY);
      return null;
    };
    createRoot(() => {
      createComponent(Parent, {});
      createComponent(Sibling, {});
    });
    expect(inChild).toBe('parent');
    expect(inSibling).toBeUndefined();
  });

  it('createContext has a default', () => {
    const Theme = createContext('dark');
    let seen = '';
    createRoot(() => {
      seen = Theme.get();
      createComponent(() => {
        Theme.set('light');
        createComponent(() => {
          seen += ' ' + Theme.get();
          return null;
        }, {});
        return null;
      }, {});
    });
    expect(seen).toBe('dark light');
  });
});

describe('memo', () => {
  it('owns what it creates: disposed when it re-runs, kept while read again', () => {
    const s = signal(0);
    const cleanups: number[] = [];
    let creations = 0;
    const read = createRoot(() =>
      memo(() => {
        const v = s.get();
        creations++;
        onCleanup(() => cleanups.push(v));
        return v;
      }),
    );
    expect(read()).toBe(0);
    expect(read()).toBe(0);
    expect(creations).toBe(1);
    s.set(1);
    expect(read()).toBe(1);
    expect(cleanups).toEqual([0]);
  });
});

describe('steady state', () => {
  it('re-running with the same dependencies keeps the same links (no allocation)', () => {
    const a = signal(1);
    const b = signal(2);
    let node: unknown;
    createRoot(() =>
      effect(() => {
        a.get();
        b.get();
      }),
    );
    flush();
    // The effect is the only subscriber of `a`: its link object stays the same across runs.
    node = (a as unknown as { _targets: unknown })._targets;
    for (let i = 0; i < 5; i++) {
      a.set(a.peek() + 1);
      flush();
    }
    expect((a as unknown as { _targets: unknown })._targets).toBe(node);
  });
});
