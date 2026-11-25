// Component context: scoped to the owner tree (`setContext`, `getContext`, `hasContext`, `createContext`).

import { getOwner } from './core';

/** Sets a context value for the current component and everything it renders (port of `setContext`). */
export function setContext<T>(key: unknown, value: T): T {
  const o = getOwner();
  if (o === null) throw new Error('fw: setContext outside a component');
  (o._ctx ??= new Map()).set(key, value);
  return value;
}

/** The nearest context value for `key` above the current owner (port of `getContext`). Call it at component init. */
export function getContext<T>(key: unknown): T | undefined {
  for (let o = getOwner(); o !== null; o = o._parent) {
    if (o._ctx !== null && o._ctx.has(key)) return o._ctx.get(key) as T;
  }
  return undefined;
}

/** Whether a context value for `key` is set above the current owner. */
export function hasContext(key: unknown): boolean {
  for (let o = getOwner(); o !== null; o = o._parent) if (o._ctx !== null && o._ctx.has(key)) return true;
  return false;
}

export interface Context<T> {
  readonly key: symbol;
  readonly defaultValue: T;
  /** Sets this context for the current component's subtree. */
  set(value: T): T;
  /** The nearest value, else the default. */
  get(): T;
}

/** A typed context (port of Svelte's `createContext`): `const Theme = createContext('dark'); Theme.set(x); Theme.get()`. */
export function createContext<T>(defaultValue: T): Context<T>;
export function createContext<T>(): Context<T | undefined>;
export function createContext<T>(defaultValue?: T): Context<T | undefined> {
  const key = Symbol('context');
  return {
    key,
    defaultValue,
    set: (v) => setContext(key, v),
    get: () => (hasContext(key) ? getContext<T>(key) : defaultValue),
  };
}
