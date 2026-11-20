// What the fw app's data layer (./resource) shares between the components that read the same data, without changing
// what any of them shows or when:
//
// - Reads issued together are one request. A screen mounts several components that each ask for the same list (the
//   system, the models, the datasets); a reconnect refetches everything at once. Inside `together(fn)` (a Resource's
//   fetcher, run synchronously) a read on the shared Api (`shareApi`) joins an identical read issued earlier in the same
//   turn (before the next microtask), so they get the same answer at the same time, which is what separate requests
//   issued at that moment would have returned. A read issued later is a request of its own, as before, but for one
//   case: a Resource's first read, as it mounts (`together(fn, true)`), joins an identical read on that Api that is
//   still in flight (and not given up), unless a write was sent on that Api since that read was. The parts of a screen
//   mount a few milliseconds apart (the shell's run list and Overview's, Overview's system and its machine strip's:
//   issue #309, item 2), each with a request of its own for what the other was about to be told. An answer is never
//   taken again: polls, refreshes and a Resource mounted once a read has answered always ask.
// - Polls that tick together are one timer (`pollTogether`): pollers started in the same turn with the same period
//   share an interval, so their ticks run in one task and their identical reads are one request (above). Each poller
//   keeps the phase it started with, as a timer of its own would. While the tab is hidden the shared timer stops (the
//   pollers skip hidden ticks anyway) and comes back on the same phase.
// - An answer equal to what is shown keeps what is shown (`reuse`): an unchanged poll writes nothing, and a changed list
//   keeps its unchanged items, so keyed rows and memos over them see no change.

import type { Api } from './client';

// ---------------------------------------------------------------------------------------------------------------- turns

/**
 * Things that can be joined only in the turn they were made in: cleared at the next microtask. Reads are keyed by their
 * shared Api's number and the call (`3|runs()`), so two Apis never join each other's.
 */
const joinable = new Map<string, Shared>();
let apis = 0;
const groups = new Map<string, PollGroup>();
let clearing = false;

function endOfTurn() {
  if (clearing) return;
  clearing = true;
  queueMicrotask(() => {
    clearing = false;
    joinable.clear();
    for (const [k, g] of groups) if (g.open) (g.open = false), groups.delete(k);
  });
}

// ------------------------------------------------------------------------------------------------------- shared reads

/** The Api methods that only read (GET): the ones a request may be shared for. */
const READS = new Set<string>([
  'system',
  'throughput',
  'datasets',
  'dataset',
  'datasetImages',
  'datasetIssues',
  'models',
  'model',
  'modelEval',
  'runs',
  'run',
  'runCli',
  'runMetrics',
  'runLogs',
  'runSamples',
  'cloudProviders',
  'cloudConnections',
  'cloudInstances',
  'labelAgents',
  'labelJobs',
  'labelJob',
  'labelImages',
  'labelLogs',
  'benchCatalog',
  'benches',
  'bench',
  'benchEntries',
  'benchEntry',
  'benchResult',
  'benchEntryImages',
  'benchLoops',
  'benchLoop',
  'benchImages',
]);

interface Shared {
  promise: Promise<unknown>;
  ctrl: AbortController;
  /** Callers still waiting: the request is aborted once every one of them has given up. */
  waiting: number;
  /** Callers so far: the first gets the answer itself, the others a copy (no caller sees another's edits). */
  callers: number;
}

/** The read keys a `together` scope saw, in order; null outside one. */
let scope: string[] | null = null;
/** The scope is a Resource's first read (it may join a read in flight: the top of this file). */
let mounting = false;

/**
 * Runs `fn` with reads on a shared Api grouped (see the top of this file) and returns what it returned, with the keys
 * of the reads it issued synchronously (`runs()`, `run("run_7d1e")`), joined: what identifies the data it loads.
 */
export function together<T>(fn: () => T, mount = false): { value: T; key: string } {
  const prev = scope;
  const prevMounting = mounting;
  const keys: string[] = (scope = []);
  mounting = mount;
  try {
    return { value: fn(), key: keys.join(' ') };
  } finally {
    scope = prev;
    mounting = prevMounting;
  }
}

const abortError = () => new DOMException('The operation was aborted.', 'AbortError');

function join<T>(entry: Shared, signal: AbortSignal | undefined): Promise<T> {
  const first = entry.callers++ === 0;
  entry.waiting++;
  const answer = entry.promise.then((v) => (first ? v : copy(v)) as T);
  if (!signal) return answer;
  if (signal.aborted) {
    if (--entry.waiting === 0) entry.ctrl.abort();
    return Promise.reject(abortError());
  }
  return new Promise<T>((resolve, reject) => {
    const gaveUp = () => {
      reject(abortError());
      if (--entry.waiting === 0) entry.ctrl.abort();
    };
    signal.addEventListener('abort', gaveUp, { once: true });
    answer.then(
      (v) => {
        signal.removeEventListener('abort', gaveUp);
        resolve(v);
      },
      (e) => {
        signal.removeEventListener('abort', gaveUp);
        reject(e);
      },
    );
  });
}

/** A JSON-like answer copied (plain objects and arrays); anything else is given as it is. */
function copy<T>(v: T): T {
  if (Array.isArray(v)) return v.map(copy) as T;
  if (v !== null && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype) {
    const out: Record<string, unknown> = {};
    for (const k in v) out[k] = copy((v as Record<string, unknown>)[k]);
    return out as T;
  }
  return v;
}

const shared = new WeakMap<object, Api>();
const unshared = new WeakMap<object, Api>();

/** `api` with its reads grouped inside `together` (outside one it is `api` itself, call for call). */
export function shareApi(api: Api): Api {
  let s = shared.get(api);
  if (s) return s;
  const fns = new Map<PropertyKey, unknown>();
  /**
   * This Api's reads still in flight, for a mounting Resource (the top of this file). Emptied when a write is sent and
   * again when it answers: a read sent while it was out may hold what it changes.
   */
  const inFlight = new Map<string, Shared>();
  const id = `${++apis}|`;
  const forget = () => {
    inFlight.clear();
    for (const k of joinable.keys()) if (k.startsWith(id)) joinable.delete(k);
  };
  s = new Proxy(api, {
    get(target, prop) {
      const v = Reflect.get(target, prop, target);
      if (typeof v !== 'function') return v;
      let f = fns.get(prop);
      if (f === undefined || (f as { $of?: unknown }).$of !== v) {
        const method = v as (...a: unknown[]) => unknown;
        const name = String(prop);
        const read = READS.has(name);
        const wrapped = (...args: unknown[]) => {
          if (!read) {
            // A request that is no read may change what any read answers (one that is not a request, a URL or a
            // subscription, changes nothing).
            const out = method.apply(target, args);
            if (out instanceof Promise) {
              forget();
              out.then(forget, forget);
            }
            return out;
          }
          if (scope === null) return method.apply(target, args);
          const signalAt = args.findIndex((a) => a instanceof AbortSignal);
          const signal = signalAt >= 0 ? (args[signalAt] as AbortSignal) : undefined;
          const key = `${name}(${args.map((a, i) => (i === signalAt ? '' : a === undefined ? '-' : JSON.stringify(a))).join(',')})`;
          scope.push(key);
          // A read every caller gave up on (aborted) is joined by nobody: the next caller asks again.
          let entry = joinable.get(id + key);
          if (entry?.ctrl.signal.aborted) entry = undefined;
          // A mounting Resource joins an identical read still out (not one every caller gave up on), never an answer.
          const out = mounting && !entry ? inFlight.get(key) : undefined;
          if (out && !out.ctrl.signal.aborted) entry = out;
          if (!entry) {
            const ctrl = new AbortController();
            const call = [...args];
            // The request's own signal, aborted once every caller has given up (a caller without one never does).
            if (signalAt >= 0) call[signalAt] = ctrl.signal;
            let promise: Promise<unknown>;
            try {
              promise = Promise.resolve(method.apply(target, call));
            } catch (e) {
              promise = Promise.reject(e);
            }
            // Nobody may be left to see a failure (every caller gave up): never an unhandled rejection.
            promise.catch(() => {});
            const made: Shared = (entry = { promise, ctrl, waiting: 0, callers: 0 });
            joinable.set(id + key, entry);
            endOfTurn();
            // Kept for a mounting Resource while it is out.
            inFlight.set(key, made);
            const done = () => void (inFlight.get(key) === made && inFlight.delete(key));
            promise.then(done, done);
          }
          return join(entry, signal);
        };
        Object.assign(wrapped, { $of: v });
        // What the method carries besides being called (a test spy's `.mock`, `mockResolvedValue`...) reads through to
        // it, so `vi.mocked(hub.api.x).mock.calls` sees the calls made through the shared Api.
        f = new Proxy(wrapped, { get: (t, p) => (p in t ? Reflect.get(t, p) : Reflect.get(method, p, method)) });
        fns.set(prop, f);
      }
      return f;
    },
  });
  shared.set(api, s);
  unshared.set(s, api);
  return s;
}

/** The Api a shared one wraps (or `api` itself). */
export function unshare(api: Api): Api {
  return unshared.get(api) ?? api;
}

// ------------------------------------------------------------------------------------------------------------- polls

interface PollGroup {
  every: number;
  /** When the group's first tick was due less one period: its phase. */
  start: number;
  ticks: Set<() => void>;
  timer: ReturnType<typeof setInterval> | ReturnType<typeof setTimeout> | undefined;
  /** Still taking pollers (the turn it was made in). */
  open: boolean;
}

let hiddenWatch = false;
const running = new Set<PollGroup>();

const hidden = () => typeof document !== 'undefined' && document.visibilityState === 'hidden';

/** Starts the group's timer on its phase: the next tick where an interval started at `start` would have it. */
function arm(g: PollGroup) {
  if (hidden()) return;
  const fire = () => {
    for (const t of [...g.ticks]) t();
  };
  const elapsed = Date.now() - g.start;
  if (elapsed <= 0) {
    g.timer = setInterval(fire, g.every);
    return;
  }
  g.timer = setTimeout(() => {
    g.start = Date.now();
    g.timer = setInterval(fire, g.every);
    fire();
  }, g.every - (elapsed % g.every));
}

function disarm(g: PollGroup) {
  clearTimeout(g.timer as ReturnType<typeof setTimeout>);
  clearInterval(g.timer as ReturnType<typeof setInterval>);
  g.timer = undefined;
}

function watchVisibility() {
  if (hiddenWatch || typeof document === 'undefined') return;
  hiddenWatch = true;
  document.addEventListener('visibilitychange', () => {
    for (const g of running) {
      disarm(g);
      arm(g);
    }
  });
}

/**
 * Calls `tick` every `every` ms, as `setInterval(tick, every)` would, until the returned function is called. Pollers
 * started in the same turn with the same period share one timer (their ticks run in one task, in the order they
 * started). Ticks are skipped while the tab is hidden: callers must not need them (a Resource's poll does nothing then).
 */
export function pollTogether(every: number, tick: () => void): () => void {
  const id = String(every);
  let g = groups.get(id);
  if (!g?.open) {
    g = { every, start: Date.now(), ticks: new Set(), timer: undefined, open: true };
    groups.set(id, g);
    endOfTurn();
    running.add(g);
    watchVisibility();
    arm(g);
  }
  const group = g;
  // A function of its own per poller: the same callback given twice still ticks twice.
  const mine = () => tick();
  group.ticks.add(mine);
  return () => {
    group.ticks.delete(mine);
    if (group.ticks.size === 0 && running.delete(group)) disarm(group);
  };
}

// ------------------------------------------------------------------------------------------------------------- reuse

/**
 * `next` with every part equal to `prev`'s replaced by `prev`'s (plain objects and arrays compared by value, anything
 * else by identity): `prev` itself when nothing changed. Neither is modified.
 */
export function reuse<T>(prev: T, next: T): T {
  return same(prev, next) as T;
}

function plain(v: unknown): v is Record<string, unknown> {
  if (v === null || typeof v !== 'object') return false;
  const p = Object.getPrototypeOf(v);
  return p === Object.prototype || p === null;
}

function same(prev: unknown, next: unknown): unknown {
  if (Object.is(prev, next)) return prev;
  if (Array.isArray(prev) && Array.isArray(next)) {
    let changed = prev.length !== next.length;
    const out = new Array(next.length);
    for (let i = 0; i < next.length; i++) {
      const v = i < prev.length ? same(prev[i], next[i]) : next[i];
      if (v !== prev[i]) changed = true;
      out[i] = v;
    }
    return changed ? out : prev;
  }
  if (plain(prev) && plain(next)) {
    let changed = false;
    let n = 0;
    const out: Record<string, unknown> = {};
    for (const k in next) {
      n++;
      const v = Object.prototype.hasOwnProperty.call(prev, k) ? same(prev[k], next[k]) : next[k];
      if (v !== prev[k] || !(k in prev)) changed = true;
      out[k] = v;
    }
    if (!changed) {
      let m = 0;
      for (const _ in prev) m++;
      if (m !== n) changed = true;
    }
    if (!changed) return prev;
    // Same key order as `next` (what a page renders in key order sees the answer's).
    return Object.getPrototypeOf(next) === null ? Object.assign(Object.create(null), out) : out;
  }
  return next;
}
