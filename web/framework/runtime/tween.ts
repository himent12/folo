// Port of svelte/motion `Tween`: `current` eases toward `target` over `duration` ms, frame by frame.

import { renderEffect, signal, type Signal } from './core';
import { linear, type EasingFunction } from './transition';

export interface TweenOptions<T> {
  delay?: number;
  duration?: number | ((from: T, to: T) => number);
  easing?: EasingFunction;
  interpolate?: (a: T, b: T) => (t: number) => T;
}

interface Task {
  c: (now: number) => boolean;
  f: () => void;
}

const tasks = new Set<Task>();
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const tick = (fn: () => void) => (typeof requestAnimationFrame === 'function' ? requestAnimationFrame(fn) : setTimeout(fn, 16));

function runTasks(): void {
  const t = now();
  for (const task of tasks) {
    if (!task.c(t)) {
      tasks.delete(task);
      task.f();
    }
  }
  if (tasks.size !== 0) tick(runTasks);
}

function loop(c: (now: number) => boolean): { promise: Promise<void>; abort(): void } {
  let task: Task;
  if (tasks.size === 0) tick(runTasks);
  return {
    promise: new Promise((f) => tasks.add((task = { c, f }))),
    abort: () => tasks.delete(task),
  };
}

function isDate(v: unknown): v is Date {
  return Object.prototype.toString.call(v) === '[object Date]';
}

function interpolator<T>(a: T, b: T): (t: number) => T {
  // eslint-disable-next-line no-self-compare
  if (a === b || a !== a) return () => a;
  const type = typeof a;
  if (type !== typeof b || Array.isArray(a) !== Array.isArray(b)) throw new Error('Cannot interpolate values of different type');
  if (Array.isArray(a)) {
    const arr = (b as unknown[]).map((bi, i) => interpolator((a as unknown[])[i], bi));
    return ((t: number) => arr.map((fn) => fn(t))) as (t: number) => T;
  }
  if (type === 'object') {
    if (!a || !b) throw new Error('Object cannot be null');
    if (isDate(a) && isDate(b)) {
      const an = a.getTime();
      const delta = b.getTime() - an;
      return ((t: number) => new Date(an + t * delta)) as (t: number) => T;
    }
    const keys = Object.keys(b as object);
    const fns: Record<string, (t: number) => unknown> = {};
    for (const k of keys) fns[k] = interpolator((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]);
    return ((t: number) => {
      const out: Record<string, unknown> = {};
      for (const k of keys) out[k] = fns[k]!(t);
      return out;
    }) as (t: number) => T;
  }
  if (type === 'number') {
    const delta = (b as number) - (a as number);
    return ((t: number) => (a as number) + t * delta) as (t: number) => T;
  }
  return () => b;
}

/** A value that eases toward its target (port of svelte/motion `Tween`). */
export class Tween<T> {
  #current: Signal<T>;
  #target: Signal<T>;
  #defaults: TweenOptions<T>;
  #task: { promise: Promise<void>; abort(): void } | null = null;

  constructor(value: T, options: TweenOptions<T> = {}) {
    this.#current = signal(value);
    this.#target = signal(value);
    this.#defaults = options;
  }

  /** A tween that follows `fn()` (port of `Tween.of`). */
  static of<U>(fn: () => U, options?: TweenOptions<U>): Tween<U> {
    const tween = new Tween(fn(), options);
    renderEffect(() => {
      void tween.set(fn());
    });
    return tween;
  }

  /** Eases to `value`; resolves when there. */
  set(value: T, options?: TweenOptions<T>): Promise<void> {
    this.#target.set(value);
    const o = { ...this.#defaults, ...options };
    const delay = o.delay ?? 0;
    let duration = o.duration ?? 400;
    const easing = o.easing ?? linear;
    const interpolate = o.interpolate ?? interpolator;
    if (duration === 0) {
      this.#task?.abort();
      this.#current.set(value);
      return Promise.resolve();
    }
    const start = now() + delay;
    let fn: (t: number) => T;
    let started = false;
    let previous = this.#task;
    this.#task = loop((t) => {
      if (t < start) return true;
      if (!started) {
        started = true;
        const prev = this.#current.peek();
        fn = interpolate(prev, value);
        if (typeof duration === 'function') duration = duration(prev, value);
        previous?.abort();
        previous = null;
      }
      const elapsed = t - start;
      if (elapsed > (duration as number)) {
        this.#current.set(value);
        return false;
      }
      this.#current.set(fn(easing(elapsed / (duration as number))));
      return true;
    });
    return this.#task.promise;
  }

  get current(): T {
    return this.#current.get();
  }
  get target(): T {
    return this.#target.get();
  }
  set target(v: T) {
    void this.set(v);
  }
}
