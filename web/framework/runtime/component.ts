// Components and props. A component is a plain function `(props) => JSX.Element`, run once. Its props object has
// getters for dynamic values (the compiler emits them), so reading `props.x` where it is used tracks it; never
// destructure props at the top of a component (that reads them once). These helpers keep the getters intact.

import { computed, runComponent, signal, untrack, type Signal } from './core';

/** A component: called once with its props; returns what it renders. */
export type Component<P = Record<string, never>> = (props: P) => JSX.Element;

/** The props type of a component. */
export type ComponentProps<C> = C extends (props: infer P) => unknown ? P : never;

/** Internal: calls a component untracked, in an owner of its own (its context scope). */
export function createComponent<P>(Comp: (props: P) => JSX.Element, props: P): JSX.Element {
  return runComponent(Comp, props);
}

type AnyProps = Record<PropertyKey, unknown>;
type Source = AnyProps | (() => AnyProps | null | undefined) | null | undefined | false;

/** Internal marker on reactive props proxies (`mergeProps`, `splitProps`, `defaults`). */
export const PROXY = Symbol('fw.props');
const DEEP = Symbol.for('fw.deep');

function isProxy(o: AnyProps): boolean {
  if (o[PROXY] || o[DEEP]) return true;
  const proto = Object.getPrototypeOf(o);
  return proto !== Object.prototype && proto !== null;
}

function setDataProp(target: AnyProps, key: PropertyKey, value: unknown): void {
  if (key !== '__proto__') target[key as string] = value;
  else Object.defineProperty(target, key, { value, writable: true, enumerable: true, configurable: true });
}

function resolve(s: Source): AnyProps | null | undefined | false {
  return typeof s === 'function' ? s() : s;
}

/** The descriptor `obj[key]` reads through (walking the prototype chain is not needed: props are own properties). */
function descriptor(obj: object, key: PropertyKey): PropertyDescriptor | undefined {
  return Reflect.getOwnPropertyDescriptor(obj, key);
}

/** A live descriptor for `src[key]`: a getter, and a setter only if `src` has one (a bound prop). */
function live(src: AnyProps, key: PropertyKey, d = descriptor(src, key)): PropertyDescriptor {
  return {
    configurable: true,
    enumerable: true,
    get: () => src[key as string],
    set: d?.set ? (v: unknown) => (src[key as string] = v) : undefined,
  };
}

type Merged<T extends unknown[]> = T extends [infer A, ...infer R] ? Omit<NonFunction<A>, keyof Merged<R>> & Merged<R> : unknown;
type NonFunction<A> = A extends () => infer O ? NonNullable<O> : NonNullable<A>;

/**
 * Merges props objects; a later one wins for every key it has (even `undefined`, as with Svelte's spread). Getters and
 * setters are kept, so the result stays reactive and `$bindable` writes still reach the parent. A function source is
 * a reactive spread (`{...obj}` of a changing object): then the result is a proxy that re-reads it.
 */
export function mergeProps<T extends Source[]>(...sources: T): Merged<T> {
  let dynamic = false;
  for (const s of sources) if (typeof s === 'function' || (s && (s as AnyProps)[PROXY])) dynamic = true;
  if (!dynamic) {
    const out: AnyProps = {};
    for (const s of sources) {
      if (!s) continue;
      for (const key of Reflect.ownKeys(s)) {
        const d = descriptor(s, key)!;
        Object.defineProperty(out, key, { ...d, configurable: true, enumerable: true });
      }
    }
    return out as Merged<T>;
  }
  const find = (key: PropertyKey): AnyProps | undefined => {
    for (let i = sources.length - 1; i >= 0; i--) {
      const s = resolve(sources[i]);
      if (s && key in s) return s;
    }
    return undefined;
  };
  return new Proxy({} as AnyProps, {
    get(_, key) {
      if (key === PROXY) return true;
      return find(key)?.[key as string];
    },
    set(_, key, value) {
      const s = find(key);
      if (s) s[key as string] = value;
      return true;
    },
    has(_, key) {
      return find(key) !== undefined;
    },
    ownKeys() {
      const keys = new Set<PropertyKey>();
      for (const s of sources) {
        const o = resolve(s);
        if (o) for (const k of Reflect.ownKeys(o)) keys.add(k);
      }
      return [...keys] as (string | symbol)[];
    },
    getOwnPropertyDescriptor(_, key) {
      const s = find(key);
      return s ? live(s, key) : undefined;
    },
  }) as Merged<T>;
}

/**
 * Splits props into objects with the listed keys and one with the rest (port of `let { a, b, ...rest } = $props()`).
 * Every part stays live: reads go through to `props`, and so do writes (`$bindable`).
 */
export function splitProps<T extends object, K extends readonly (keyof T)[]>(props: T, keys: K): [Pick<T, K[number]>, Omit<T, K[number]>];
export function splitProps<T extends object, K1 extends readonly (keyof T)[], K2 extends readonly (keyof T)[]>(
  props: T,
  keys1: K1,
  keys2: K2,
): [Pick<T, K1[number]>, Pick<T, K2[number]>, Omit<T, K1[number] | K2[number]>];
export function splitProps(props: AnyProps, ...lists: readonly PropertyKey[][]): AnyProps[] {
  const excluded = new Set<PropertyKey>(lists.length === 1 ? lists[0] : lists.flat());
  if (!isProxy(props)) {
    const parts: AnyProps[] = lists.map((keys) => {
      const part: AnyProps = {};
      for (const key of keys) {
        const d = descriptor(props, key);
        if (d === undefined) setDataProp(part, key, undefined);
        else if (d.get === undefined && d.set === undefined) setDataProp(part, key, d.value);
        else Object.defineProperty(part, key, live(props, key, d));
      }
      return part;
    });
    const rest: AnyProps = {};
    for (const key of Reflect.ownKeys(props)) {
      if (key === MEMOS || excluded.has(key)) continue;
      const d = descriptor(props, key)!;
      if (d.get === undefined && d.set === undefined) setDataProp(rest, key, d.value);
      else Object.defineProperty(rest, key, live(props, key, d));
    }
    parts.push(rest);
    return parts;
  }
  const parts: AnyProps[] = lists.map((keys) => {
    const part: AnyProps = {};
    for (const key of keys) Object.defineProperty(part, key, live(props, key));
    return part;
  });
  parts.push(
    new Proxy({} as AnyProps, {
      get: (_, key) => (key === PROXY ? true : excluded.has(key) ? undefined : props[key as string]),
      set: (_, key, value) => ((props[key as string] = value), true),
      has: (_, key) => !excluded.has(key) && key in props,
      ownKeys: () => Reflect.ownKeys(props).filter((k) => !excluded.has(k)) as (string | symbol)[],
      getOwnPropertyDescriptor: (_, key) => (excluded.has(key) || !(key in props) ? undefined : live(props, key)),
    }),
  );
  return parts;
}

type WithDefaults<T, D> = Omit<T, keyof D> & { [K in keyof D & keyof T]-?: Exclude<T[K], undefined> };

/**
 * Props with fallbacks: `props.x` is `defaults.x` while the parent passes `undefined` (or nothing), as with
 * `let { x = 1 } = $props()`. Other keys pass through; writes go to the parent.
 */
export function defaults<T extends object, const D extends Partial<T>>(props: T, fallbacks: D): WithDefaults<T, D> {
  const p = props as AnyProps;
  const d = fallbacks as AnyProps;
  if (!isProxy(p) && !isProxy(d)) {
    const out: AnyProps = {};
    for (const key of Reflect.ownKeys(p)) {
      if (key === MEMOS) continue;
      const dp = descriptor(p, key)!;
      const hasFallback = Object.prototype.hasOwnProperty.call(d, key);
      if (dp.get === undefined && dp.set === undefined) {
        if (dp.value !== undefined || !hasFallback) setDataProp(out, key, dp.value);
        else {
          const dd = descriptor(d, key)!;
          if (dd.get === undefined && dd.set === undefined) setDataProp(out, key, dd.value);
          else Object.defineProperty(out, key, { configurable: true, enumerable: true, get: () => d[key as string] });
        }
      } else if (hasFallback) {
        Object.defineProperty(out, key, {
          configurable: true,
          enumerable: true,
          get: () => {
            const v = p[key as string];
            return v === undefined ? d[key as string] : v;
          },
          set: dp.set ? (v: unknown) => (p[key as string] = v) : undefined,
        });
      } else Object.defineProperty(out, key, live(p, key, dp));
    }
    for (const key of Reflect.ownKeys(d)) {
      if (key === MEMOS || Object.prototype.hasOwnProperty.call(p, key)) continue;
      const dd = descriptor(d, key)!;
      if (dd.get === undefined && dd.set === undefined) setDataProp(out, key, dd.value);
      else Object.defineProperty(out, key, { configurable: true, enumerable: true, get: () => d[key as string] });
    }
    return out as WithDefaults<T, D>;
  }
  const read = (key: PropertyKey) => {
    const v = p[key as string];
    return v === undefined && key in d ? d[key as string] : v;
  };
  return new Proxy({} as AnyProps, {
    get: (_, key) => (key === PROXY ? true : read(key)),
    set: (_, key, value) => ((p[key as string] = value), true),
    has: (_, key) => key in p || key in d,
    ownKeys: () => [...new Set([...Reflect.ownKeys(p), ...Reflect.ownKeys(d)])] as (string | symbol)[],
    getOwnPropertyDescriptor: (_, key) => {
      if (!(key in p) && !(key in d)) return undefined;
      return { ...live(p, key), get: () => read(key) };
    },
  }) as WithDefaults<T, D>;
}

/** Whether `props` has a setter for `key` (the parent bound it: `bind:key={...}`). */
function isBound(props: object, key: PropertyKey): boolean {
  const d = Reflect.getOwnPropertyDescriptor(props, key);
  return !!d?.set;
}

/**
 * A `$bindable` prop as a signal. Bound by the parent (`bind:value={sig}`): reads and writes go to the parent's
 * signal. Not bound: writes are local, until the parent's value changes (Svelte's semantics: any change, even back to
 * the value the write was made over). `fallback` stands in while the value is `undefined`.
 */
export function bindable<T extends object, K extends keyof T>(props: T, key: K, fallback?: Exclude<T[K], undefined>): Signal<Exclude<T[K], undefined>>;
export function bindable<T extends object, K extends keyof T>(props: T, key: K): Signal<T[K]>;
export function bindable(props: AnyProps, key: string, fallback?: unknown): Signal<unknown> {
  const read = () => {
    const v = props[key];
    return v === undefined ? fallback : v;
  };
  if (isBound(props, key)) {
    return {
      get: read,
      peek: () => untrack(read),
      set(v) {
        props[key] = typeof v === 'function' ? (v as (p: unknown) => unknown)(untrack(read)) : v;
      },
    };
  }
  // As Svelte's prop: a local write holds until the parent's value changes, whatever it changes to (A → B → A too).
  // The parent's value is boxed anew each time what it reads changes, so a write holds while the box it was made over
  // is the current one.
  const parent = computed(() => ({ v: props[key] }));
  const local = signal<{ v: unknown; over: { v: unknown } } | null>(null, { equals: false });
  const get = () => {
    const p = parent.get();
    const l = local.get();
    if (l !== null && l.over === p) return l.v;
    return p.v === undefined ? fallback : p.v;
  };
  return {
    get,
    peek: () => untrack(get),
    set(v) {
      const next = typeof v === 'function' ? (v as (p: unknown) => unknown)(untrack(get)) : v;
      local.set({ v: next, over: parent.peek() });
    },
  };
}

const MEMOS = Symbol('fw.memos');

/**
 * Internal: a component prop memoized as Svelte 5 memoizes non-trivial prop expressions (`$.derived`): the getter
 * reads a lazy computed, so the child sees a new value (and its effects re-run) only when the value changed.
 */
export function propMemo(holder: object, key: string, fn: () => unknown): unknown {
  let memos = (holder as Record<symbol, Record<string, ReturnType<typeof computed>> | undefined>)[MEMOS];
  if (memos === undefined) Object.defineProperty(holder, MEMOS, { value: (memos = {}) });
  return (memos[key] ??= computed(fn)).get();
}

/** Internal: a memoized spread source (`{...attrs()}` on a component), read lazily. */
export function derive(fn: () => unknown): () => unknown {
  const c = computed(fn);
  return () => c.get();
}

/** Internal: a `bind:name={sig}` on a component, as a getter/setter pair for `mergeProps`. */
export function bound(name: string, sig: { get(): unknown; set(v: unknown): void }): AnyProps {
  const out: AnyProps = {};
  Object.defineProperty(out, name, { configurable: true, enumerable: true, get: () => sig.get(), set: (v) => sig.set(v) });
  return out;
}

/** A `{get, set}` view of `obj[key]` (for `bind:value={field(form, 'name')}` on a `deep` object or a props object). */
export function field<T extends object, K extends keyof T>(obj: T, key: K): Signal<T[K]> {
  return {
    get: () => obj[key],
    peek: () => untrack(() => obj[key]),
    set: (v) => {
      obj[key] = typeof v === 'function' ? (v as (p: T[K]) => T[K])(untrack(() => obj[key])) : v;
    },
  };
}

let uid = 1;
/** A unique id per call (port of `$props.id()`; the same `c1`, `c2`, ... sequence Svelte hands out). */
export function createUniqueId(): string {
  return `c${uid++}`;
}
