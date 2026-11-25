// Port of svelte/motion `Tween`: `current` eases toward `target` over `duration` ms, frame by frame.

import { onCleanup, renderEffect, signal, type Signal } from './core';
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

interface TaskHandle {
  promise: Promise<void>;
  abort(): void;
}

const tasks = new Set<Task>();
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
let scheduled: number | ReturnType<typeof setTimeout> | null = null;
let scheduledWithRaf = false;
let running = false;

function schedule(): void {
  if (scheduled !== null || running) return;
  if (typeof requestAnimationFrame === 'function') {
    scheduledWithRaf = true;
    scheduled = requestAnimationFrame(runTasks);
  } else {
    scheduledWithRaf = false;
    scheduled = setTimeout(runTasks, 16);
  }
}

function cancelScheduled(): void {
  if (scheduled === null) return;
  if (scheduledWithRaf) {
    if (typeof cancelAnimationFrame !== 'function') return;
    const id = scheduled as number;
    scheduled = null;
    cancelAnimationFrame(id);
  } else {
    const id = scheduled;
    scheduled = null;
    clearTimeout(id);
  }
}

function runTasks(): void {
  scheduled = null;
  running = true;
  let error: unknown;
  let hasError = false;
  try {
    const t = now();
    for (const task of tasks) {
      try {
        if (!task.c(t) && tasks.delete(task)) {
          task.f();
        }
      } catch (e) {
        tasks.delete(task);
        if (!hasError) {
          hasError = true;
          error = e;
        }
      }
    }
  } finally {
    running = false;
    if (tasks.size !== 0) schedule();
  }
  if (hasError) throw error;
}

function loop(c: (now: number) => boolean): TaskHandle {
  let task: Task;
  const promise = new Promise<void>((f) => tasks.add((task = { c, f })));
  schedule();
  return {
    promise,
    abort() {
      if (tasks.delete(task) && tasks.size === 0) cancelScheduled();
    },
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
  #task: TaskHandle | null = null;

  constructor(value: T, options: TweenOptions<T> = {}) {
    this.#current = signal(value);
    this.#target = signal(value);
    this.#defaults = options;
    onCleanup(() => this.#abort());
  }

  #abort(): void {
    const task = this.#task;
    this.#task = null;
    task?.abort();
  }

  /** A tween that follows `fn()` (port of `Tween.of`). */
  static of<U>(fn: () => U, options?: TweenOptions<U>): Tween<U> {
    let init!: U;
    let initSeq = 0;
    let disposed = false;
    let tween: Tween<U> | undefined;
    onCleanup(() => {
      disposed = true;
    });
    renderEffect(() => {
      const seq = ++initSeq;
      const v = fn();
      if (tween === undefined) {
        if (seq === initSeq) init = v;
      } else if (!disposed) {
        void tween.set(v);
      }
    });
    tween = new Tween(init, options);
    return tween;
  }

  /** Eases to `value`; resolves when there. */
  set(value: T, options?: TweenOptions<T>): Promise<void> {
    const prevTarget = this.#target.peek();
    this.#target.set(value);
    const hasCustomInterpolate = (options?.interpolate ?? this.#defaults.interpolate) !== undefined;
    // Skip scheduling a frame if already settled at `value` (with built-in interpolation),
    // or already easing to `value` with no new options.
    if (this.#task === null && !hasCustomInterpolate && Object.is(this.#current.peek(), value)) return Promise.resolve();
    if (this.#task !== null && options === undefined && Object.is(prevTarget, value)) return this.#task.promise;
    const o = options ? { ...this.#defaults, ...options } : this.#defaults;
    const delay = o.delay ?? 0;
    let duration = o.duration ?? 400;
    const easing = o.easing ?? linear;
    const interpolate = o.interpolate ?? interpolator;
    if (typeof duration === 'number' && !(duration > 0)) {
      this.#abort();
      this.#current.set(value);
      return Promise.resolve();
    }
    const start = now() + delay;
    let fn: (t: number) => T;
    let started = false;
    let aborted = false;
    let activePrev = this.#task;
    const raw = loop((t) => {
      try {
        if (aborted) return false;
        if (t < start) return true;
        if (!started) {
          started = true;
          const prev = this.#current.peek();
          if (typeof duration === 'function') duration = duration(prev, value);
          if (aborted) return false;
          const p = activePrev;
          activePrev = null;
          p?.abort();
          if (aborted) return false;
          if (!(t - start < (duration as number))) {
            this.#current.set(value);
            if (this.#task === handle) this.#task = null;
            return false;
          }
          fn = interpolate(prev, value);
          if (aborted) return false;
        }
        const elapsed = t - start;
        if (!(elapsed < (duration as number))) {
          this.#current.set(value);
          if (this.#task === handle) this.#task = null;
          return false;
        }
        const eased = easing(elapsed / (duration as number));
        if (aborted) return false;
        const next = fn(eased);
        if (aborted) return false;
        this.#current.set(next);
        return true;
      } catch (e) {
        if (this.#task === handle) this.#task = null;
        handle.abort();
        throw e;
      }
    });
    const handle: TaskHandle = {
      promise: raw.promise,
      abort() {
        aborted = true;
        raw.abort();
        const p = activePrev;
        activePrev = null;
        p?.abort();
      },
    };
    this.#task = handle;
    return handle.promise;
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
