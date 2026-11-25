import { describe, expect, it, vi } from 'vitest';
import { createRoot, deep, effect, flush, signal } from 'fw';
import { bindable, bound, defaults, mergeProps, propMemo, PROXY, splitProps } from './component';

describe('defaults & splitProps static fast path', () => {
  it('returns plain non-PROXY objects with plain data descriptors for static props', () => {
    interface StaticProps {
      variant?: string;
      size?: string;
      count?: number;
      flag?: boolean;
      empty?: string;
      nil?: string | null;
      explicitUndef?: string;
      class?: string;
    }
    const raw: StaticProps = { variant: 'primary', count: 0, flag: false, empty: '', nil: null, explicitUndef: undefined, class: 'btn' };
    const p = defaults(raw, { variant: 'default', size: 'md', count: 5, flag: true, empty: 'x', nil: 'y', explicitUndef: 'fb' });
    expect((p as Record<PropertyKey, unknown>)[PROXY]).toBeUndefined();
    expect(p).toEqual({
      variant: 'primary',
      count: 0,
      flag: false,
      empty: '',
      nil: null,
      explicitUndef: 'fb',
      class: 'btn',
      size: 'md',
    });
    expect(Object.keys(p)).toEqual(['variant', 'count', 'flag', 'empty', 'nil', 'explicitUndef', 'class', 'size']);
    expect(Object.getOwnPropertyDescriptor(p, 'variant')?.get).toBeUndefined();
    expect(Object.getOwnPropertyDescriptor(p, 'size')?.get).toBeUndefined();

    const [local, rest] = splitProps(p, ['variant', 'size', 'missing' as keyof typeof p]);
    expect((rest as Record<PropertyKey, unknown>)[PROXY]).toBeUndefined();
    expect(local).toEqual({ variant: 'primary', size: 'md', missing: undefined });
    expect('missing' in local).toBe(true);
    expect(Object.getOwnPropertyDescriptor(local, 'variant')?.get).toBeUndefined();
    expect(Object.getOwnPropertyDescriptor(rest, 'class')?.get).toBeUndefined();
    expect(rest).toEqual({ count: 0, flag: false, empty: '', nil: null, explicitUndef: 'fb', class: 'btn' });
  });

  it('preserves reactive getters, propMemo sharing, and bind: setters without wrapping in a Proxy', () => {
    const v = signal<string | undefined>(undefined);
    const boundSig = signal<number | undefined>(undefined);
    const memoFn = vi.fn(() => `${v.get() ?? 'none'}!`);
    const props = {
      staticKey: 'ok',
      get label() {
        return v.get();
      },
      get derived() {
        return propMemo(this, 'derived', memoFn);
      },
    };
    const boundNum = bound('num', boundSig);
    const getterReads = vi.fn(() => 'lazy-fallback');
    const fallbacks = {
      label: 'default-label',
      num: 10,
      get unused() {
        return getterReads();
      },
      get used() {
        return getterReads();
      },
    };
    // Creating defaults + splitProps must not eagerly evaluate any getters.
    const rawWithUnused = Object.defineProperties(
      { staticKey: 'ok', unused: 'present' },
      {
        label: Object.getOwnPropertyDescriptor(props, 'label')!,
        derived: Object.getOwnPropertyDescriptor(props, 'derived')!,
        num: Object.getOwnPropertyDescriptor(boundNum, 'num')!,
      },
    ) as { staticKey: string; unused?: string; used?: string; label?: string; derived: string; num?: number };

    const p = defaults(rawWithUnused, fallbacks);
    expect(memoFn).not.toHaveBeenCalled();
    expect(getterReads).not.toHaveBeenCalled();
    expect((p as Record<PropertyKey, unknown>)[PROXY]).toBeUndefined();
    expect(p.unused).toBe('present');
    expect(getterReads).not.toHaveBeenCalled();
    expect(p.used).toBe('lazy-fallback');
    expect(getterReads).toHaveBeenCalledTimes(1);

    const [mine, rest] = splitProps(p, ['label', 'num']);
    expect((rest as Record<PropertyKey, unknown>)[PROXY]).toBeUndefined();
    expect(mine.label).toBe('default-label');
    expect(rest.derived).toBe('none!');
    expect(rawWithUnused.derived).toBe('none!');
    // propMemo shares its cache on the original props holder (only 1 evaluation).
    expect(memoFn).toHaveBeenCalledTimes(1);
    // Primed MEMOS symbol on source is never leaked by defaults or splitProps.
    const primedDef = defaults(rawWithUnused, {});
    const [, primedRest] = splitProps(rawWithUnused, ['staticKey']);
    expect(Object.getOwnPropertySymbols(rawWithUnused)).toHaveLength(1);
    expect(Object.getOwnPropertySymbols(primedDef)).toEqual([]);
    expect(Object.getOwnPropertySymbols(primedRest)).toEqual([]);

    // Bound prop still reads fallback and writes through to parent signal.
    const b = bindable(mine, 'num', 99);
    expect(b.get()).toBe(10);
    b.set(42);
    expect(boundSig.get()).toBe(42);
    expect(mine.num).toBe(42);

    // A setter on `fallbacks` is NOT forwarded onto defaults output (bindable stays local).
    const fbSig = signal(7);
    const withBoundFb = defaults({} as { x?: number }, bound('x', fbSig) as { x: number });
    const localBind = bindable(withBoundFb, 'x', 0);
    expect(localBind.get()).toBe(7);
    localBind.set(99);
    expect(localBind.get()).toBe(99);
    expect(fbSig.get()).toBe(7);

    // Multi-list splitProps and Object.create(null) take the static path cleanly.
    const nullProto = Object.assign(Object.create(null), { a: 1, b: 2, c: 3 }) as { a: number; b: number; c: number };
    const [pA, pB, pC] = splitProps(nullProto, ['a'], ['b']);
    expect(pA).toEqual({ a: 1 });
    expect(pB).toEqual({ b: 2 });
    expect(pC).toEqual({ c: 3 });
    expect((pC as Record<PropertyKey, unknown>)[PROXY]).toBeUndefined();

    // Reactive getter transitions undefined -> value -> undefined cleanly.
    const seen: string[] = [];
    createRoot(() => effect(() => void seen.push(mine.label)));
    flush();
    v.set('hi');
    flush();
    v.set(undefined);
    flush();
    expect(seen).toEqual(['default-label', 'hi', 'default-label']);
  });

  it('keeps Proxy path for dynamic mergeProps, deep() proxies, and reactive fallbacks', () => {
    const dyn = signal<Record<string, unknown>>({ a: 1, b: 2 });
    const merged = mergeProps(() => dyn.get());
    const p = defaults(merged, { c: 3 });
    const [aPart, rest] = splitProps(p, ['a']);
    const seen: string[] = [];
    createRoot(() => effect(() => void seen.push(`${aPart.a}:${Object.keys(rest).sort().join(',')}`)));
    flush();
    dyn.set({ a: 9, d: 4 });
    flush();
    expect(seen).toEqual(['1:b,c', '9:c,d']);

    // deep() proxy stays reactive through defaults and splitProps (both as props and as fallbacks).
    const dp = deep<{ x?: number; y: number }>({ y: 1 });
    const dpDef = defaults(dp, { x: 5 });
    const [xPart, yRest] = splitProps(dpDef, ['x']);
    const deepSeen: string[] = [];
    createRoot(() => effect(() => void deepSeen.push(`${xPart.x}:${yRest.y}`)));
    flush();
    dp.x = 7;
    dp.y = 2;
    flush();
    expect(deepSeen).toEqual(['5:1', '7:2']);

    const fbDeep = deep({ a: 10 });
    const withDeepFb = defaults({ a: undefined as number | undefined }, fbDeep);
    const fbSeen: number[] = [];
    createRoot(() => effect(() => void fbSeen.push(withDeepFb.a)));
    flush();
    fbDeep.a = 20;
    flush();
    expect(fbSeen).toEqual([10, 20]);
  });
});
