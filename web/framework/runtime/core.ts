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

const NOTIFIED = 1;
const OUTDATED = 2;
const RUNNING = 4;
const DISPOSED = 8;
const HAS_ERROR = 16;
/** A computed with subscribers (it is linked into its sources), or any effect. */
const TRACKING = 32;
/** A user effect (`effect`, `onMount`): runs after the render effects of the same flush; a returned function is its cleanup. */
const USER = 64;
/** An effect that has not run yet. */
const FRESH = 128;
/** An owner linked into its parent's child list (disposed with it). */
const ATTACHED = 256;

interface Link {
  _s: Source;
  _ps: Link | undefined;
  _ns: Link | undefined;
  _t: Target;
  _pt: Link | undefined;
  _nt: Link | undefined;
  _v: number;
  _rb: Link | undefined;
}

interface Source {
  _value: unknown;
  _version: number;
  _node: Link | undefined;
  _targets: Link | undefined;
  _refresh(): boolean;
  _subscribe(link: Link): void;
  _unsubscribe(link: Link): void;
}

interface Target {
  _sources: Link | undefined;
  _flags: number;
  _notify(): void;
}

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
  const target = listener;
  if (target === undefined) return undefined;
  let node = source._node;
  if (node === undefined || node._t !== target) {
    // A new dependency: append it to the target's source list (its tail is `_sources` while evaluating).
    node = { _v: 0, _s: source, _ps: target._sources, _ns: undefined, _t: target, _pt: undefined, _nt: undefined, _rb: node };
    if (target._sources !== undefined) target._sources._ns = node;
    target._sources = node;
    source._node = node;
    if (target._flags & TRACKING) source._subscribe(node);
    return node;
  }
  if (node._v === -1) {
    // Read again in this evaluation: reuse the link, moved to the tail so the list stays in read order.
    node._v = 0;
    if (node._ns !== undefined) {
      node._ns._ps = node._ps;
      if (node._ps !== undefined) node._ps._ns = node._ns;
      node._ps = target._sources;
      node._ns = undefined;
      target._sources!._ns = node;
      target._sources = node;
    }
    return node;
  }
  return undefined;
}

/** Before an evaluation: mark every current link unread (-1) and point `_sources` at the tail. */
function prepareSources(target: Target): void {
  for (let node = target._sources; node !== undefined; node = node._ns) {
    const rb = node._s._node;
    if (rb !== undefined) node._rb = rb;
    node._s._node = node;
    node._v = -1;
    if (node._ns === undefined) {
      target._sources = node;
      break;
    }
  }
}

/** After an evaluation: drop links that were not read, restore the sources' outer links, point `_sources` at the head. */
function cleanupSources(target: Target): void {
  let node = target._sources;
  let head: Link | undefined;
  while (node !== undefined) {
    const prev = node._ps;
    if (node._v === -1) {
      node._s._unsubscribe(node);
      if (prev !== undefined) prev._ns = node._ns;
      if (node._ns !== undefined) node._ns._ps = prev;
    } else head = node;
    node._s._node = node._rb;
    if (node._rb !== undefined) node._rb = undefined;
    node = prev;
  }
  target._sources = head;
}

/** Whether any source of `target` changed since it last ran (pulls computeds up to date on the way). */
function needsToRecompute(target: Target): boolean {
  for (let node = target._sources; node !== undefined; node = node._ns) {
    if (node._s._version !== node._v || !node._s._refresh() || node._s._version !== node._v) return true;
  }
  return false;
}

function subscribeLink(source: Source, node: Link): void {
  if (source._targets !== node && node._pt === undefined) {
    node._nt = source._targets;
    if (source._targets !== undefined) source._targets._pt = node;
    source._targets = node;
  }
}

function unsubscribeLink(source: Source, node: Link): void {
  if (source._targets === undefined) return;
  const prev = node._pt;
  const next = node._nt;
  if (prev !== undefined) {
    prev._nt = next;
    node._pt = undefined;
  }
  if (next !== undefined) {
    next._pt = prev;
    node._nt = undefined;
  }
  if (node === source._targets) source._targets = next;
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
      this._flags &= ~TRACKING;
      for (let n = this._sources; n !== undefined; n = n._ns) n._s._unsubscribe(n);
    }
  }
  _notify(): void {
    if (!(this._flags & NOTIFIED)) {
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
  _prev: Owner | null = null;
  _next: Owner | null = null;
  _first: Owner | null = null;
  _last: Owner | null = null;
  _cleanups: Cleanup | Cleanup[] | null = null;
  _ctx: Map<unknown, unknown> | null = null;
  _flags = 0;
}

function attach(parent: Owner | null, child: Owner): void {
  child._parent = parent;
  if (parent === null) return;
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
  let child = node._first;
  node._first = node._last = null;
  let error: unknown;
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
    const prevListener = listener;
    listener = undefined;
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
    } finally {
      listener = prevListener;
    }
  }
  if (error !== undefined) throw error;
}

function disposeTree(node: Owner): void {
  if (node._flags & DISPOSED) return;
  node._flags |= DISPOSED;
  try {
    cleanOwner(node);
  } finally {
    if (node instanceof EffectNode) {
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
  if (parent === undefined) root._parent = owner;
  else attach(parent, root);
  return runWithOwner(root, () => fn(() => dispose(root)));
}

/** Registers `fn` to run when the current owner (component, effect run, root) is disposed or re-runs. */
export function onCleanup(fn: Cleanup): void {
  if (owner !== null) addCleanup(owner, fn);
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
    if (root !== null) dispose(root);
    const r = (root = new Owner());
    attach(parent, r);
    owner = r;
    const v = fn();
    return (typeof v === 'function' && !isReader(v) ? memo(v as () => unknown) : v) as T;
  }, options?.equals);
  return reader(() => c.get());
}

/** Internal: a component's own owner (context scope), attached to the current owner. */
export function componentOwner(): Owner {
  const o = new Owner();
  attach(owner, o);
  return o;
}

// ---------------------------------------------------------------------------------------------------------------------
// Context

/** Sets a context value for the current component and everything it renders (port of `setContext`). */
export function setContext<T>(key: unknown, value: T): T {
  if (owner === null) throw new Error('fw: setContext outside a component');
  (owner._ctx ??= new Map()).set(key, value);
  return value;
}

/** The nearest context value for `key` above the current owner (port of `getContext`). Call it at component init. */
export function getContext<T>(key: unknown): T | undefined {
  for (let o = owner; o !== null; o = o._parent) {
    if (o._ctx !== null && o._ctx.has(key)) return o._ctx.get(key) as T;
  }
  return undefined;
}

/** Whether a context value for `key` is set above the current owner. */
export function hasContext(key: unknown): boolean {
  for (let o = owner; o !== null; o = o._parent) if (o._ctx !== null && o._ctx.has(key)) return true;
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

// ---------------------------------------------------------------------------------------------------------------------
// Effects and the scheduler

let renderQueue: EffectNode[] = [];
let userQueue: EffectNode[] = [];
let scheduled = false;
let flushing = false;

class EffectNode extends Owner implements Target {
  _sources: Link | undefined = undefined;
  _fn: (prev: any) => any;
  _value: unknown;
  _id = nextId++;
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
    const f = this._flags;
    if (f & DISPOSED) return;
    this._flags = f & ~NOTIFIED;
    if (f & FRESH || needsToRecompute(this)) this._run();
  }
  _run(): void {
    cleanOwner(this);
    const prevListener = listener;
    const prevOwner = owner;
    prepareSources(this);
    listener = this;
    owner = this;
    this._flags = (this._flags | RUNNING) & ~FRESH;
    try {
      if (this._flags & USER) {
        const r = this._fn(undefined);
        if (typeof r === 'function') addCleanup(this, r);
      } else this._value = this._fn(this._value);
    } finally {
      listener = prevListener;
      owner = prevOwner;
      cleanupSources(this);
      this._flags &= ~RUNNING;
    }
  }
}

function schedule(): void {
  if (!scheduled) {
    scheduled = true;
    queueMicrotask(() => {
      scheduled = false;
      flush();
    });
  }
}

const byId = (a: EffectNode, b: EffectNode) => a._id - b._id;

/**
 * Runs every pending effect now: first the DOM bindings (render effects), then user effects; repeats while effects
 * write (port of `flushSync`). Writes otherwise flush on their own in a microtask. Effects run parent before child,
 * siblings in creation order.
 */
export function flush(): void {
  if (flushing) return;
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
        userQueue = [];
        throw new Error('fw: effects kept writing what they read (more than 1000 rounds in one flush)');
      }
      let q: EffectNode[];
      if (renderQueue.length > 0) {
        q = renderQueue;
        renderQueue = [];
      } else {
        q = userQueue;
        userQueue = [];
      }
      for (let i = 1; i < q.length; i++) {
        if (q[i - 1]!._id > q[i]!._id) {
          q.sort(byId);
          break;
        }
      }
      for (let i = 0; i < q.length; i++) {
        try {
          q[i]!._update();
        } catch (e) {
          error ??= e;
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
 * Groups writes. Effects already wait for a microtask, so this only matters for readability; computeds read inside
 * `fn` see every write made so far (they are pulled, never stale).
 */
export function batch<T>(fn: () => T): T {
  return fn();
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
  const e = new EffectNode(fn, init, 0);
  attach(owner, e);
  e._run();
}

/**
 * A side effect (port of `$effect`): runs after the component is in the DOM, then whenever what it read changes, after
 * the DOM bindings of that flush. A returned function runs before the next run and when the owner goes away.
 */
export function effect(fn: () => void | Cleanup): void {
  const e = new EffectNode(fn, undefined, USER);
  attach(owner, e);
  userQueue.push(e);
  e._flags |= NOTIFIED;
  schedule();
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
