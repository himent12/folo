// The reactive core: signals, lazy computeds, effects, the owner tree and the scheduler.
//
// Push-pull, glitch-free, in the style of Preact Signals / alien-signals: a write bumps the signal's version and pushes
// a "maybe stale" mark down the graph; nothing recomputes until it is read (computeds) or flushed (effects), and then
// each node pulls: it re-runs only if one of its sources' versions actually moved. Dependencies are doubly linked
// `Link` nodes shared between a source's subscriber list and a target's source list; a re-run reuses the links it
// read last time (no allocation in steady state) and unlinks the ones it stopped reading.
//
// Short internal names keep the runtime small (property names survive minification):
//   Link:   _s source, _ps/_ns prev/next in the target's source list, _t target, _pt/_nt prev/next in the source's
//           subscriber list, _v the source version seen, _rb the link to restore on the source after an evaluation.
//   Source: _value, _version, _node (the link to the target being evaluated), _targets (head of subscriber list).
//   Owner:  _parent, _prev/_next siblings, _first/_last children, _cleanups, _ctx (context map).

/* eslint-disable @typescript-eslint/no-explicit-any */

import { sortJobs } from './flushOrder';
import {
  cleanupSources,
  linkDependency,
  needsToRecompute,
  prepareSources,
  retainSources,
  subscribeLink,
  TRACKING,
  unsubscribeLink,
  type Link,
  type Source,
  type Target,
} from './links';

const NOTIFIED = 1;
const OUTDATED = 2;
const RUNNING = 4;
const DISPOSED = 8;
const HAS_ERROR = 16;
/** A user effect (`effect`, `onMount`): runs after the render effects of the same flush; a returned function is its cleanup. */
const USER = 64;
/** An effect that has not run yet. */
const FRESH = 128;
/** An owner linked into its parent's child list (disposed with it). */
const ATTACHED = 256;
/**
 * A computed that owns a subtree (a memo's branch, a list's rows). When what it reads changes it is re-evaluated in the
 * render phase at its own place (its creation order, which is document order), as Svelte runs a block before what
 * comes after it, not when the effect that reads it gets to it.
 */
const OWNING = 512;

export type Equals<T> = false | ((a: T, b: T) => boolean);
export interface SignalOptions<T> {
  /** When the new value counts as unchanged (no update). Default `Object.is`; `false` notifies on every write. */
  equals?: Equals<T>;
}

/** A readable reactive value: `get()` tracks, `peek()` does not. */
export interface ReadSignal<T> {
  get(): T;
  peek(): T;
}
/** A writable reactive value. `set` takes a value or an updater `(prev) => next` (store a function as `set(() => fn)`). */
export interface Signal<T> extends ReadSignal<T> {
  set(v: T | ((prev: T) => T)): void;
}

/** The node being evaluated (a computed or an effect): what reads subscribe. */
let listener: Target | undefined;
/** The owner new effects, roots, components and cleanups belong to. */
let owner: Owner | null = null;
/** Bumped on every write; lets an untracked computed skip a check when nothing at all changed. */
let globalVersion = 0;
let nextId = 0;

// ---------------------------------------------------------------------------------------------------------------------
// Dependency links

function addDependency(source: Source): Link | undefined {
  return listener === undefined ? undefined : linkDependency(source, listener);
}

/**
 * Internal (text runs): a position in the running evaluation's reads: the last source read so far, to pass to
 * `dropSourcesThrough`.
 */
export function sourceMark(): unknown {
  // Every read moves its link to the tail: an unread tail (left from the previous run) means nothing was read yet.
  const tail = listener?._sources;
  return tail !== undefined && tail._v !== -1 ? tail : undefined;
}

/**
 * Internal (text runs): the running evaluation stops depending on what it read up to `mark` (from `sourceMark`); only
 * the reads after it stay. A source read both before and after the mark is dropped too (its link is shared).
 */
export function dropSourcesThrough(mark: unknown): void {
  for (let node = mark as Link | undefined; node !== undefined; node = node._ps) node._v = -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// Signals

const is = Object.is;

class SignalNode<T> implements Source, Signal<T> {
  _value: T;
  _version = 0;
  _node: Link | undefined = undefined;
  _targets: Link | undefined = undefined;
  _equals: Equals<T>;
  constructor(value: T, equals: Equals<T> | undefined) {
    this._value = value;
    this._equals = equals === undefined ? is : equals;
  }
  _refresh(): boolean {
    return true;
  }
  _subscribe(node: Link): void {
    subscribeLink(this, node);
  }
  _unsubscribe(node: Link): void {
    unsubscribeLink(this, node);
  }
  get(): T {
    const node = addDependency(this);
    if (node !== undefined) node._v = this._version;
    return this._value;
  }
  peek(): T {
    return this._value;
  }
  set(v: T | ((prev: T) => T)): void {
    const value = typeof v === 'function' ? (v as (p: T) => T)(this._value) : v;
    if (this._equals !== false && this._equals(this._value, value)) return;
    this._value = value;
    this._version++;
    globalVersion++;
    for (let node = this._targets; node !== undefined; node = node._nt) node._t._notify();
  }
}

/** A signal that calls `watch(true)` when it gains its first subscriber and `watch(false)` when it loses the last. */
class WatchedSignalNode<T> extends SignalNode<T> {
  _watch: (on: boolean) => void;
  _check: (() => boolean) | undefined;
  constructor(value: T, watch: (on: boolean) => void, check?: () => boolean) {
    super(value, undefined);
    this._watch = watch;
    this._check = check;
  }
  override _refresh(): boolean {
    return this._targets !== undefined || this._check === undefined || this._check();
  }
  override _subscribe(node: Link): void {
    const first = this._targets === undefined;
    subscribeLink(this, node);
    if (first && this._targets !== undefined) this._watch(true);
  }
  override _unsubscribe(node: Link): void {
    if (this._targets === undefined) return;
    unsubscribeLink(this, node);
    if (this._targets === undefined) this._watch(false);
  }
}

/**
 * Internal: a signal that knows whether anything subscribes to it (an effect, or a computed an effect reads): `watch`
 * runs with `true` on the first subscriber and `false` when the last one goes. A computed read outside any effect does
 * not subscribe, so it starts nothing (and so leaks nothing). When `check` is given, an unsubscribed pull calls it and
 * treats `false` as stale so unwatched computeds still recompute when the backing store changed. Note: `ComputedNode`
 * short-circuits pulls while `globalVersion` is unchanged, so callers using `check` must also write a signal (such as a
 * collection version counter) on every backing-store mutation.
 */
export function watchedSignal<T>(value: T, watch: (on: boolean) => void, check?: () => boolean): Signal<T> {
  return new WatchedSignalNode(value, watch, check);
}

/** A writable reactive value (port of `$state` for primitives and `$state.raw`). */
export function signal<T>(value: T, options?: SignalOptions<T>): Signal<T>;
export function signal<T = undefined>(): Signal<T | undefined>;
export function signal<T>(value?: T, options?: SignalOptions<T>): Signal<T> {
  return new SignalNode(value as T, options?.equals);
}

/** Alias of `signal`, to read like the `$state` it replaces. */
export const state: typeof signal = signal;

// ---------------------------------------------------------------------------------------------------------------------
// Computeds

class ComputedNode<T> implements Source, Target, ReadSignal<T> {
  _value: T = undefined as T;
  _version = 0;
  _globalVersion = globalVersion - 1;
  _flags = OUTDATED;
  _id = nextId++;
  /** An owning computed's place: the owner it was created in (for tree-order flushing). */
  _parent: Owner | null = null;
  _depth = 0;
  _node: Link | undefined = undefined;
  _targets: Link | undefined = undefined;
  _sources: Link | undefined = undefined;
  _fn: () => T;
  _equals: Equals<T>;
  constructor(fn: () => T, equals: Equals<T> | undefined) {
    this._fn = fn;
    this._equals = equals === undefined ? is : equals;
  }
  _refresh(): boolean {
    this._flags &= ~NOTIFIED;
    if (this._flags & RUNNING) return false;
    // Subscribed and not marked: every source change would have reached it, so it is current.
    if ((this._flags & (OUTDATED | TRACKING)) === TRACKING) return true;
    this._flags &= ~OUTDATED;
    if (this._globalVersion === globalVersion) return true;
    this._globalVersion = globalVersion;
    this._flags |= RUNNING;
    if (this._version > 0 && !needsToRecompute(this)) {
      this._flags &= ~RUNNING;
      return true;
    }
    const prevListener = listener;
    const prevOwner = owner;
    try {
      prepareSources(this);
      listener = this;
      // A computed owns nothing: effects must not be created while deriving a value.
      owner = null;
      const value = this._fn();
      if (this._flags & HAS_ERROR || this._version === 0 || this._equals === false || !this._equals(this._value, value)) {
        this._value = value;
        this._flags &= ~HAS_ERROR;
        this._version++;
      }
    } catch (err) {
      this._value = err as T;
      this._flags |= HAS_ERROR;
      this._version++;
    }
    listener = prevListener;
    owner = prevOwner;
    cleanupSources(this);
    this._flags &= ~RUNNING;
    return true;
  }
  _subscribe(node: Link): void {
    if (this._targets === undefined) {
      this._flags |= OUTDATED | TRACKING;
      // First subscriber: subscribe to the sources too (a computed nobody watches holds no subscriptions, so it can be
      // garbage collected with whatever created it).
      for (let n = this._sources; n !== undefined; n = n._ns) n._s._subscribe(n);
    }
    subscribeLink(this, node);
  }
  _unsubscribe(node: Link): void {
    if (this._targets === undefined) return;
    unsubscribeLink(this, node);
    if (this._targets === undefined) {
      // Nobody reads it any more (its branch went): a queued re-evaluation is dropped (see `_update`).
      if (this._flags & OWNING) this._flags = (this._flags & ~NOTIFIED) | OUTDATED;
      this._flags &= ~TRACKING;
      for (let n = this._sources; n !== undefined; n = n._ns) n._s._unsubscribe(n);
    }
  }
  _notify(): void {
    if (!(this._flags & NOTIFIED)) {
      if (this._flags & OWNING) {
        renderQueue.push(this);
        schedule();
      }
      this._flags |= OUTDATED | NOTIFIED;
      for (let node = this._targets; node !== undefined; node = node._nt) node._t._notify();
    }
  }
  get(): T {
    if (this._flags & RUNNING) throw new Error('fw: cycle detected in a computed');
    const node = addDependency(this);
    this._refresh();
    if (node !== undefined) node._v = this._version;
    if (this._flags & HAS_ERROR) throw this._value;
    return this._value;
  }
  peek(): T {
    const prev = listener;
    listener = undefined;
    try {
      return this.get();
    } finally {
      listener = prev;
    }
  }
  /** An owning computed's turn in the render phase (see OWNING): brought up to date if it is still read. */
  _update(): void {
    if ((this._flags & (NOTIFIED | TRACKING)) !== (NOTIFIED | TRACKING)) return;
    const prev = listener;
    listener = undefined;
    try {
      this._refresh();
    } finally {
      listener = prev;
    }
  }
}

/** A lazy, memoized derived value (port of `$derived` / `$derived.by`). Recomputes on read, only if a source changed. */
export function computed<T>(fn: () => T, options?: SignalOptions<T>): ReadSignal<T> {
  return new ComputedNode(fn, options?.equals);
}

// ---------------------------------------------------------------------------------------------------------------------
// Owners

export type Cleanup = () => void;

/** A node of the owner tree: what effects, components and roots belong to, and are disposed with. */
export class Owner {
  _parent: Owner | null = null;
  _depth = 0;
  _prev: Owner | null = null;
  _next: Owner | null = null;
  _first: Owner | null = null;
  _last: Owner | null = null;
  _cleanups: Cleanup | Cleanup[] | null = null;
  _ctx: Map<unknown, unknown> | null = null;
  _flags = 0;
  /**
   * Its place among its siblings (creation order; a branch's root takes its memo's place): with its ancestors', its
   * position in the tree, which is the order effects run in.
   */
  _id = nextId++;
}

/** Internal: `c` owns what it creates (a list's rows): it is re-evaluated at its own place in the render phase. */
export function owning(c: ReadSignal<unknown>): void {
  const n = c as ComputedNode<unknown>;
  n._flags |= OWNING;
  n._parent = owner;
  n._depth = owner === null ? 0 : owner._depth + 1;
}

function attach(parent: Owner | null, child: Owner): void {
  child._parent = parent;
  child._depth = parent === null ? 0 : parent._depth + 1;
  if (parent === null) return;
  if (parent._flags & DISPOSED) {
    child._flags |= DISPOSED;
    return;
  }
  child._flags |= ATTACHED;
  child._prev = parent._last;
  if (parent._last !== null) parent._last._next = child;
  else parent._first = child;
  parent._last = child;
}

function detach(child: Owner): void {
  const parent = child._parent;
  if (!(child._flags & ATTACHED) || parent === null) return;
  child._flags &= ~ATTACHED;
  if (child._prev !== null) child._prev._next = child._next;
  else parent._first = child._next;
  if (child._next !== null) child._next._prev = child._prev;
  else parent._last = child._prev;
  child._prev = child._next = null;
}

function addCleanup(node: Owner, fn: Cleanup): void {
  const c = node._cleanups;
  if (c === null) node._cleanups = fn;
  else if (typeof c === 'function') node._cleanups = [c, fn];
  else c.push(fn);
}

/** Disposes `node`'s children (in order), then runs its cleanups (in the order registered). */
function cleanOwner(node: Owner): void {
  const prevListener = listener;
  const prevOwner = owner;
  listener = undefined;
  owner = null;
  let error: unknown;
  try {
    let child = node._first;
    node._first = node._last = null;
    while (child !== null) {
      const next = child._next;
      child._prev = child._next = null;
      child._flags &= ~ATTACHED;
      try {
        disposeTree(child);
      } catch (e) {
        error ??= e;
      }
      child = next;
    }
    const c = node._cleanups;
    if (c !== null) {
      node._cleanups = null;
      try {
        if (typeof c === 'function') c();
        else
          for (const fn of c) {
            try {
              fn();
            } catch (e) {
              error ??= e;
            }
          }
      } catch (e) {
        error ??= e;
      }
    }
  } finally {
    listener = prevListener;
    owner = prevOwner;
  }
  if (error !== undefined) throw error;
}

function disposeTree(node: Owner): void {
  if (node._flags & DISPOSED) return;
  node._flags |= DISPOSED;
  try {
    cleanOwner(node);
  } finally {
    // A currently running effect still has `_sources` pointing at its tail with active `_node`/`_rb` save slots;
    // `EffectNode._run`'s `finally` will restore those links via `cleanupSources` and unsubscribe every source.
    if (node instanceof EffectNode && !(node._flags & RUNNING)) {
      for (let n = node._sources; n !== undefined; n = n._ns) n._s._unsubscribe(n);
      node._sources = undefined;
    }
  }
}

/** Disposes an owner and everything under it, and unlinks it from its parent. */
export function dispose(node: Owner): void {
  detach(node);
  disposeTree(node);
}

/** The current owner (to restore later with `runWithOwner`). */
export function getOwner(): Owner | null {
  return owner;
}

/** Runs `fn` with `o` as the owner (and no tracking), e.g. to create effects from an event handler or a timer. */
export function runWithOwner<T>(o: Owner | null, fn: () => T): T {
  const prevOwner = owner;
  const prevListener = listener;
  owner = o;
  listener = undefined;
  try {
    return fn();
  } finally {
    owner = prevOwner;
    listener = prevListener;
  }
}

/**
 * Runs `fn` in a new root owner and returns its result; `dispose` tears everything created inside down (port of
 * `$effect.root`). A root is not disposed with the current owner (it only inherits its context), unless `parent` is
 * given: then it belongs to `parent` and goes with it.
 */
export function createRoot<T>(fn: (dispose: () => void) => T, parent?: Owner | null): T {
  const root = new Owner();
  if (parent === undefined) {
    root._parent = owner;
    root._depth = owner === null ? 0 : owner._depth + 1;
  } else attach(parent, root);
  try {
    return runWithOwner(root, () => fn(() => dispose(root)));
  } catch (e) {
    try {
      dispose(root);
    } catch {
      // Keep the root initializer error.
    }
    throw e;
  }
}

/** Registers `fn` to run when the current owner (component, effect run, root) is disposed or re-runs. */
export function onCleanup(fn: Cleanup): void {
  if (owner === null) return;
  if (owner._flags & DISPOSED) runWithOwner(null, fn);
  else addCleanup(owner, fn);
}

/**
 * A function that only reads (a memo, a list's rows): calling it never creates DOM, so `insert` may call it inside the
 * effect that consumes it. Unmarked functions (snippets, render props) are called in an owner of their own.
 */
export type Reader<T> = (() => T) & { $$r: 1 };

/** Internal: marks `fn` as a reader. */
export function reader<T>(fn: () => T): Reader<T> {
  (fn as Reader<T>).$$r = 1;
  return fn as Reader<T>;
}

/** Whether `fn` is a reader (see `Reader`). */
export function isReader(fn: unknown): fn is Reader<unknown> {
  return typeof fn === 'function' && (fn as Reader<unknown>).$$r === 1;
}

/**
 * Internal: an owning memo. Like `computed`, but what `fn` creates (DOM, effects, components) belongs to the memo:
 * it is disposed when `fn` re-runs and when the current owner goes away. Branches (`Show`, conditionals, `Key`) are
 * memos, so reading one from several places, or again, never re-creates its content. A snippet `fn` returns is
 * wrapped in a memo of its own, so a branch's content is created once and read thereafter.
 */
export function memo<T>(fn: () => T, options?: SignalOptions<T>): Reader<T> {
  const parent = owner;
  let root: Owner | null = null;
  const c = new ComputedNode<T>(() => {
    if (root !== null) {
      const prev = root;
      root = null;
      try {
        dispose(prev);
      } catch (e) {
        // `fn()` never ran, so `cleanupSources(c)` would otherwise unsubscribe every dependency and strand `c` in HAS_ERROR.
        retainSources(c);
        throw e;
      }
    }
    const r = new Owner();
    // The branch stands where the memo was written, not where it was (re)created.
    r._id = c._id;
    const dead = parent !== null && (parent._flags & DISPOSED) !== 0;
    if (dead) r._flags |= DISPOSED;
    else {
      root = r;
      attach(parent, r);
    }
    owner = r;
    try {
      const v = fn();
      return (typeof v === 'function' && !isReader(v) ? memo(v as () => unknown) : v) as T;
    } catch (e) {
      if (!dead) {
        try {
          dispose(r);
        } catch {
          // Keep the original branch error.
        }
        root = null;
      }
      throw e;
    }
  }, options?.equals);
  c._flags |= OWNING;
  c._parent = parent;
  c._depth = parent === null ? 0 : parent._depth + 1;
  // Created now, where it is written, as Svelte creates a block: what it renders comes before what follows it in the
  // document (component order, `createUniqueId` numbering). Untracked: whoever reads it later subscribes then. An
  // error is kept for that first read.
  const prev = listener;
  listener = undefined;
  try {
    c._refresh();
  } finally {
    listener = prev;
  }
  return reader(() => c.get());
}

/** Internal: creates an attached child root owner (disposed with `parent` via `dispose(root)`). */
export function ownedRoot(parent: Owner | null): Owner {
  const root = new Owner();
  attach(parent, root);
  return root;
}

/** Internal: swaps the active owner for a zero-closure `try/finally` scope (caller must already be untracked). */
export function swapOwner(next: Owner | null): Owner | null {
  const prev = owner;
  owner = next;
  return prev;
}

/** Internal: a component's own owner (context scope), attached to the current owner. */
export function componentOwner(): Owner {
  const o = new Owner();
  attach(owner, o);
  return o;
}

// ---------------------------------------------------------------------------------------------------------------------
// Context (extracted to ./context.ts, re-exported here for callers importing from ./core)

export { createContext, getContext, hasContext, setContext, type Context } from './context';

// ---------------------------------------------------------------------------------------------------------------------
// Effects and the scheduler

/** What the render phase runs, in creation order: DOM bindings, and owning computeds (branches, lists). */
interface Job {
  readonly _id: number;
  readonly _parent: Owner | null;
  readonly _depth: number;
  _flags: number;
  _update(): void;
}

let renderQueue: Job[] = [];
let renderSpare: Job[] = [];
let userQueue: EffectNode[] = [];
let userSpare: EffectNode[] = [];
let scheduled = false;
let flushing = false;
let batchDepth = 0;
let batchFlush = false;
const MAX_SPARE_QUEUE = 256;

class EffectNode extends Owner implements Target {
  _sources: Link | undefined = undefined;
  _fn: (prev: any) => any;
  _value: unknown;
  constructor(fn: (prev: any) => any, value: unknown, flags: number) {
    super();
    this._fn = fn;
    this._value = value;
    this._flags = flags | TRACKING | FRESH;
  }
  _notify(): void {
    if (!(this._flags & NOTIFIED)) {
      this._flags |= NOTIFIED;
      (this._flags & USER ? userQueue : renderQueue).push(this);
      schedule();
    }
  }
  _update(): void {
    if (this._flags & DISPOSED) return;
    const f = this._flags;
    this._flags = f & ~NOTIFIED;
    if (f & FRESH || needsToRecompute(this)) this._run();
  }
  _run(): void {
    cleanOwner(this);
    if (this._flags & DISPOSED) return;
    const prevListener = listener;
    const prevOwner = owner;
    prepareSources(this);
    listener = this;
    owner = this;
    this._flags = (this._flags | RUNNING) & ~FRESH;
    let threw = false;
    try {
      if (this._flags & USER) {
        const r = this._fn(undefined);
        if (typeof r === 'function') {
          if (this._flags & DISPOSED) runWithOwner(null, r);
          else addCleanup(this, r);
        }
      } else this._value = this._fn(this._value);
    } catch (e) {
      threw = true;
      throw e;
    } finally {
      listener = prevListener;
      owner = prevOwner;
      cleanupSources(this);
      this._flags &= ~RUNNING;
      if (this._flags & DISPOSED) {
        try {
          cleanOwner(this);
        } catch (e) {
          if (!threw) throw e;
        } finally {
          for (let n = this._sources; n !== undefined; n = n._ns) n._s._unsubscribe(n);
          this._sources = undefined;
        }
      }
    }
  }
}

function flushMicrotask(): void {
  scheduled = false;
  flush();
}

function schedule(): void {
  if (!scheduled) {
    scheduled = true;
    queueMicrotask(flushMicrotask);
  }
}

/**
 * Runs `q` in tree order, as Svelte walks its effect tree (parents before children, siblings in document order: a
 * branch created late runs at its memo's place); returns the first error.
 */
function run(q: Job[], error: unknown): unknown {
  sortJobs(q);
  for (let i = 0; i < q.length; i++) {
    try {
      q[i]!._update();
    } catch (e) {
      error ??= e;
    }
  }
  return error;
}

/**
 * Runs every pending effect now, in cycles: the DOM bindings (render effects), then user effects; repeats while
 * effects write (port of `flushSync`). Writes otherwise flush on their own in a microtask. Effects run parent before child,
 * siblings in creation order.
 */
export function flush(): void {
  if (flushing) return;
  if (batchDepth > 0) {
    batchFlush = true;
    return;
  }
  flushing = true;
  let error: unknown;
  let rounds = 0;
  try {
    while (renderQueue.length > 0 || userQueue.length > 0) {
      if (++rounds > 1000) {
        // Drop what is pending, or every later flush would loop again.
        for (const e of renderQueue) e._flags &= ~NOTIFIED;
        for (const e of userQueue) e._flags &= ~NOTIFIED;
        renderQueue = [];
        renderSpare = [];
        userQueue = [];
        userSpare = [];
        throw new Error('fw: effects kept writing what they read (more than 1000 rounds in one flush)');
      }
      // One cycle, as Svelte's flush: the render jobs queued so far, then the user effects queued so far (those of
      // components the render jobs just created included); what they queue in turn is the next cycle. A page that goes
      // and the page that comes so settle in one cycle before anything rendered earlier re-runs over the gap between
      // them (a count one page's cleanup lowers and the next page's effect raises).
      if (renderQueue.length > 0) {
        const q = renderQueue;
        renderQueue = renderSpare;
        try {
          error = run(q, error);
        } finally {
          if (q.length <= MAX_SPARE_QUEUE) {
            q.length = 0;
            renderSpare = q;
          } else {
            renderSpare = [];
          }
        }
      }
      if (userQueue.length > 0) {
        const u = userQueue;
        userQueue = userSpare;
        try {
          error = run(u, error);
        } finally {
          if (u.length <= MAX_SPARE_QUEUE) {
            u.length = 0;
            userSpare = u;
          } else {
            userSpare = [];
          }
        }
      }
    }
  } finally {
    flushing = false;
  }
  if (error !== undefined) throw error;
}

/** Resolves after pending updates have been applied to the DOM (port of `tick`). */
export async function tick(): Promise<void> {
  await Promise.resolve();
  flush();
}

/**
 * Groups writes. Any `flush()` called inside `fn` (including nested batches) is deferred until the outermost batch
 * finishes so observers never see torn intermediate state; computeds read inside `fn` still pull every write made so far.
 */
export function batch<T>(fn: () => T): T {
  batchDepth++;
  let threw = false;
  try {
    return fn();
  } catch (e) {
    threw = true;
    throw e;
  } finally {
    batchDepth--;
    if (batchDepth === 0 && batchFlush) {
      batchFlush = false;
      if (!threw) flush();
    }
  }
}

/** Runs `fn` without tracking what it reads (port of `untrack`). */
export function untrack<T>(fn: () => T): T {
  if (listener === undefined) return fn();
  const prev = listener;
  listener = undefined;
  try {
    return fn();
  } finally {
    listener = prev;
  }
}

/** Whether code runs inside an effect or computed that tracks reads. */
export function isTracking(): boolean {
  return listener !== undefined;
}

/**
 * Internal: a DOM binding. Runs now, synchronously, then again in the render phase of a flush whenever what it read
 * changed. `fn` gets the value it returned last time (initially `init`).
 */
export function renderEffect<T>(fn: (prev: T) => T, init?: T): void {
  if (owner !== null && (owner._flags & DISPOSED) !== 0) return;
  const e = new EffectNode(fn, init, 0);
  attach(owner, e);
  e._run();
}

/**
 * A side effect (port of `$effect`): runs after the component is in the DOM, then whenever what it read changes, after
 * the DOM bindings of that flush. A returned function runs before the next run and when the owner goes away.
 *
 * Called in a component's body (not inside another effect), it is created once the component has rendered, as Svelte
 * defers a component's top-level `$effect` to its mount: the effects of what it renders (child components, actions)
 * come first, and so run first.
 */
export function effect(fn: () => void | Cleanup): void {
  if (deferredOwner !== undefined && owner === deferredOwner && listener === undefined) {
    if (deferredFns === null) deferredFns = [fn];
    else deferredFns.push(fn);
  } else effectNow(fn);
}

/** Internal: a user effect created now, never deferred (actions and bindings, which Svelte creates in place). */
export function effectNow(fn: () => void | Cleanup): void {
  if (owner !== null && (owner._flags & DISPOSED) !== 0) return;
  const e = new EffectNode(fn, undefined, USER);
  attach(owner, e);
  userQueue.push(e);
  e._flags |= NOTIFIED;
  schedule();
}

/** The component being created (`undefined` when inactive) and the effects its body asked for (lazily allocated). */
let deferredOwner: Owner | null | undefined = undefined;
let deferredFns: (() => void | Cleanup)[] | null = null;

/**
 * Internal: runs a component's function (in its owner) and then creates the effects its body asked for, after what it
 * rendered (Svelte's `pop()`).
 */
export function deferEffects<T>(fn: () => T): T {
  const prevOwner = deferredOwner;
  const prevFns = deferredFns;
  deferredOwner = owner;
  deferredFns = null;
  let out: T;
  let fns: (() => void | Cleanup)[] | null;
  try {
    out = fn();
    fns = deferredFns as (() => void | Cleanup)[] | null;
  } finally {
    deferredOwner = prevOwner;
    deferredFns = prevFns;
  }
  if (fns !== null) for (let i = 0; i < fns.length; i++) effectNow(fns[i]!);
  return out;
}

/** Internal: creates a component's owner, runs `Comp(props)` untracked with deferred effects, with zero wrapper closures. */
export function runComponent<P>(Comp: (props: P) => JSX.Element, props: P): JSX.Element {
  const o = new Owner();
  attach(owner, o);
  const prevOwner = owner;
  const prevListener = listener;
  const prevDeferredOwner = deferredOwner;
  const prevDeferredFns = deferredFns;
  owner = o;
  listener = undefined;
  deferredOwner = o;
  deferredFns = null;
  try {
    let out: JSX.Element;
    let fns: (() => void | Cleanup)[] | null;
    try {
      out = Comp(props);
      fns = deferredFns as (() => void | Cleanup)[] | null;
    } finally {
      deferredOwner = prevDeferredOwner;
      deferredFns = prevDeferredFns;
    }
    if (fns !== null) for (let i = 0; i < fns.length; i++) effectNow(fns[i]!);
    return out;
  } catch (e) {
    try {
      dispose(o);
    } catch {
      // Keep the component's own error.
    }
    throw e;
  } finally {
    owner = prevOwner;
    listener = prevListener;
  }
}

/**
 * Like `effect`, but runs before the DOM bindings, and the first time synchronously (port of `$effect.pre`).
 * A returned function is its cleanup.
 */
effect.pre = function pre(fn: () => void | Cleanup): void {
  renderEffect<unknown>(() => {
    const r = fn();
    if (typeof r === 'function') onCleanup(r);
    return undefined;
  });
};

/** Runs `fn` once after the component is in the DOM, untracked; a returned function runs on teardown (port of `onMount`). */
export function onMount(fn: () => void | Cleanup): void {
  effect(() => untrack(fn));
}

/** Runs `fn` when the current component is torn down (port of `onDestroy`). */
export const onDestroy = onCleanup;
