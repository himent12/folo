// Reactive data structures: `deep` (Svelte's `$state` proxy for objects and arrays mutated in place), `ReactiveMap` /
// `ReactiveSet` (SvelteMap / SvelteSet) and `MediaQuery` (svelte/reactivity).

import { isTracking, signal, untrack, watchedSignal, type Signal } from './core';

const STATE = Symbol('fw.deep');

/** A change counter: `read()` tracks it, `bump()` notifies. */
function counter(): Signal<number> {
  return signal(0);
}
const bump = (s: Signal<number>) => s.set((n) => n + 1);

function isPlain(v: unknown): v is object {
  if (typeof v !== 'object' || v === null) return false;
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === Array.prototype || proto === null;
}

/**
 * A deeply reactive proxy of a plain object or array (port of `$state({...})` / `$state([...])`): reading a property
 * tracks it, writing it (or `push`, `splice`, `delete`, ...) updates what read it. Nested plain objects and arrays are
 * proxied when read. Other values are returned as they are. Use it where the app mutates state in place; prefer a
 * `signal` holding immutable values elsewhere.
 */
export function deep<T>(value: T): T {
  if (!isPlain(value) || STATE in (value as object)) return value;
  const target = value as Record<PropertyKey, unknown>;
  const keys = new Map<PropertyKey, Signal<number>>();
  const version = counter();
  const isArray = Array.isArray(target);
  const key = (prop: PropertyKey) => {
    let s = keys.get(prop);
    if (s === undefined) keys.set(prop, (s = counter()));
    return s;
  };
  const proxy: Record<PropertyKey, unknown> = new Proxy(target, {
    get(t, prop, receiver) {
      if (prop === STATE) return target;
      // Own and missing keys track (an addition notifies); inherited ones (array methods) do not.
      if (isTracking() && (Object.prototype.hasOwnProperty.call(t, prop) || !(prop in t))) key(prop).get();
      const v = Reflect.get(t, prop, receiver);
      if (isPlain(v) && !(STATE in v)) {
        const d = Object.getOwnPropertyDescriptor(t, prop);
        if (d?.writable) {
          const p = deep(v);
          t[prop] = p;
          return p;
        }
      }
      return v;
    },
    has(t, prop) {
      if (prop === STATE) return true;
      if (isTracking()) key(prop).get();
      return prop in t;
    },
    set(t, prop, v) {
      const had = Object.prototype.hasOwnProperty.call(t, prop);
      const old = t[prop];
      if (isArray && prop === 'length') {
        const oldLen = (t as unknown as unknown[]).length;
        (t as unknown as unknown[]).length = v as number;
        if (oldLen !== v) {
          for (let i = v as number; i < oldLen; i++) {
            const s = keys.get(String(i));
            if (s) bump(s);
          }
          const s = keys.get('length');
          if (s) bump(s);
          bump(version);
        }
        return true;
      }
      const next = deep(v);
      t[prop] = next;
      if (!had || !Object.is(old, next)) {
        const s = keys.get(prop);
        if (s) bump(s);
        if (!had) {
          bump(version);
          if (isArray) {
            const s2 = keys.get('length');
            if (s2) bump(s2);
          }
        }
      }
      return true;
    },
    deleteProperty(t, prop) {
      if (!(prop in t)) return true;
      delete t[prop];
      const s = keys.get(prop);
      if (s) bump(s);
      bump(version);
      return true;
    },
    ownKeys(t) {
      version.get();
      return Reflect.ownKeys(t);
    },
    getOwnPropertyDescriptor(t, prop) {
      if (isTracking()) key(prop).get();
      return Reflect.getOwnPropertyDescriptor(t, prop);
    },
  });
  return proxy as T;
}

/** The plain object or array behind a `deep` proxy (port of `$state.snapshot`, shallow on the proxy itself). */
export function snapshot<T>(value: T): T {
  if (typeof value !== 'object' || value === null) return value;
  const raw = (value as Record<symbol, unknown>)[STATE];
  const src = (raw ?? value) as object;
  if (!isPlain(src)) return value;
  if (Array.isArray(src)) return src.map((v) => snapshot(v)) as T;
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(src)) out[k] = snapshot((src as Record<string, unknown>)[k]);
  return out as T;
}

/** A Map whose reads track and whose writes update readers (port of `SvelteMap`). */
export class ReactiveMap<K, V> extends Map<K, V> {
  #keys = new Map<K, Signal<number>>();
  #version = counter();
  constructor(entries?: Iterable<readonly [K, V]> | null) {
    super();
    if (entries) for (const [k, v] of entries) super.set(k, v);
  }
  #key(k: K): Signal<number> {
    let s = this.#keys.get(k);
    if (s === undefined) this.#keys.set(k, (s = counter()));
    return s;
  }
  override get(k: K): V | undefined {
    if (isTracking()) this.#key(k).get();
    return super.get(k);
  }
  override has(k: K): boolean {
    if (isTracking()) this.#key(k).get();
    return super.has(k);
  }
  override set(k: K, v: V): this {
    const had = super.has(k);
    const old = super.get(k);
    super.set(k, v);
    if (!had || !Object.is(old, v)) {
      const s = this.#keys.get(k);
      if (s) bump(s);
      bump(this.#version);
    }
    return this;
  }
  override delete(k: K): boolean {
    const had = super.delete(k);
    if (had) {
      const s = this.#keys.get(k);
      if (s) bump(s);
      bump(this.#version);
    }
    return had;
  }
  override clear(): void {
    if (super.size === 0) return;
    super.clear();
    for (const s of this.#keys.values()) bump(s);
    bump(this.#version);
  }
  override get size(): number {
    this.#version.get();
    return super.size;
  }
  override forEach(fn: (v: V, k: K, m: Map<K, V>) => void, thisArg?: unknown): void {
    this.#version.get();
    super.forEach(fn, thisArg);
  }
  override keys(): MapIterator<K> {
    this.#version.get();
    return super.keys();
  }
  override values(): MapIterator<V> {
    this.#version.get();
    return super.values();
  }
  override entries(): MapIterator<[K, V]> {
    this.#version.get();
    return super.entries();
  }
  override [Symbol.iterator](): MapIterator<[K, V]> {
    return this.entries();
  }
}

/** A Set whose reads track and whose writes update readers (port of `SvelteSet`). */
export class ReactiveSet<T> extends Set<T> {
  #keys = new Map<T, Signal<number>>();
  #version = counter();
  constructor(values?: Iterable<T> | null) {
    super();
    if (values) for (const v of values) super.add(v);
  }
  override has(v: T): boolean {
    if (isTracking()) {
      let s = this.#keys.get(v);
      if (s === undefined) this.#keys.set(v, (s = counter()));
      s.get();
    }
    return super.has(v);
  }
  override add(v: T): this {
    if (!super.has(v)) {
      super.add(v);
      const s = this.#keys.get(v);
      if (s) bump(s);
      bump(this.#version);
    }
    return this;
  }
  override delete(v: T): boolean {
    const had = super.delete(v);
    if (had) {
      const s = this.#keys.get(v);
      if (s) bump(s);
      bump(this.#version);
    }
    return had;
  }
  override clear(): void {
    if (super.size === 0) return;
    super.clear();
    for (const s of this.#keys.values()) bump(s);
    bump(this.#version);
  }
  override get size(): number {
    this.#version.get();
    return super.size;
  }
  override forEach(fn: (v: T, k: T, s: Set<T>) => void, thisArg?: unknown): void {
    this.#version.get();
    super.forEach(fn, thisArg);
  }
  override keys(): SetIterator<T> {
    this.#version.get();
    return super.keys();
  }
  override values(): SetIterator<T> {
    this.#version.get();
    return super.values();
  }
  override entries(): SetIterator<[T, T]> {
    this.#version.get();
    return super.entries();
  }
  override [Symbol.iterator](): SetIterator<T> {
    return this.values();
  }
}

const KEYWORDS = new Set(['all', 'print', 'screen', 'and', 'or', 'not', 'only']);

/**
 * A media query's live result (port of svelte/reactivity `MediaQuery`): `current` is whether it matches now, and
 * reading it in an effect or computed re-runs that on a change. As in Svelte, each instance has its own
 * `matchMedia` list (so a test can stub `matchMedia`), listened to only while something tracks it.
 */
export class MediaQuery {
  #mql: MediaQueryList | null;
  #subscribe: () => void;
  /** `fallback` is accepted for parity with Svelte (it only matters for server rendering). */
  constructor(query: string, _fallback?: boolean) {
    const text = /\(.+\)/.test(query) || query.split(/[\s,]+/).some((k) => KEYWORDS.has(k.trim())) ? query : `(${query})`;
    const mql = (this.#mql = typeof matchMedia === 'function' ? matchMedia(text) : null);
    this.#subscribe = createSubscriber((update) => {
      mql?.addEventListener('change', update);
      return () => mql?.removeEventListener('change', update);
    });
  }
  get current(): boolean {
    this.#subscribe();
    return this.#mql?.matches ?? false;
  }
  get(): boolean {
    return this.current;
  }
  peek(): boolean {
    return this.#mql?.matches ?? false;
  }
}

/** `new MediaQuery(query)` as a function. */
export function mediaQuery(query: string): MediaQuery {
  return new MediaQuery(query);
}

/**
 * Port of svelte/reactivity `createSubscriber`: `subscribe()` read by an effect, or by a computed an effect reads,
 * starts `start(update)` once for all of them and stops it when the last one goes; `update()` re-runs what called
 * `subscribe()`. Read anywhere else (untracked, or in a computed nothing watches) it starts nothing.
 */
export function createSubscriber(start: (update: () => void) => (() => void) | void): () => void {
  let stop: (() => void) | void;
  const version: Signal<number> = watchedSignal(0, (on) => {
    if (on) stop = untrack(() => start(() => bump(version)));
    else {
      const s = stop;
      stop = undefined;
      s?.();
    }
  });
  return () => {
    if (isTracking()) version.get();
  };
}
