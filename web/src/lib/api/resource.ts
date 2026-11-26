// Resource: data loaded for what is on screen, sharing what it can with the other Resources on
// screen (./share) without changing what any of them shows or when:
// - its fetcher runs inside `together`: an identical read another Resource issued in the same turn (a screen mounting,
//   a reconnect) is the same request;
// - its poll timer is shared with the pollers started in the same turn with the same period (one task per tick, and
//   their identical reads one request), stops when the last one goes, and stops while the tab is hidden;
// - an answer equal to the data shown keeps the data shown (`reuse`): an unchanged poll writes nothing, so nothing that
//   reads `data` recomputes, and a changed list keeps its unchanged items (keyed rows stay as they are);
// - with `startFrom`, its first fetch takes the answer of another Resource reading the same thing, a few ms apart.
//
// Reactive async data for components. Create inside a component's body: the fetcher re-runs whenever a reactive value
// it reads synchronously changes (route params, filters) and when the API hub reconnects. A reconnect (`hub.epoch`: the
// server in use may be another one, or the mock) drops what the last server sent: until the new one answers there is
// no data, and if it fails, its error. When the server answers
// again after an outage (`hub.back`), a Resource that failed retries by itself and one holding data refreshes quietly.

import { batch, effect, onCleanup, signal, untrack } from 'fw';
import { hub } from './index';
import { abortError, copy, pollTogether, reuse, together } from './share';

/** Global monotonic sequence counter across all Resources so cross-Resource freshness checks have one total order. */
let nextSeq = 0;

/** Latest fetch sequence number issued by any Resource. */
export function currentResourceSeq(): number {
  return nextSeq;
}

/** How old an answer a Resource may start from (`startFrom`): one poll of the list and the meters that use it. */
export const START_FRESH = 2000;

/** `read`, given up at once when `signal` aborts: what stays on `read` then holds neither the caller nor its data. */
function untilAborted<T>(read: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let slot: { ok: (v: T) => void; no: (e: unknown) => void } | null = {
      ok: (v) => (signal.removeEventListener('abort', left), resolve(v)),
      no: (e) => (signal.removeEventListener('abort', left), reject(e)),
    };
    const left = () => {
      slot = null;
      reject(abortError());
    };
    signal.addEventListener('abort', left, { once: true });
    read.then(
      (v) => slot?.ok(v),
      (e) => slot?.no(e),
    );
  });
}

export interface ResourceOptions<T = unknown> {
  /** Refetch quietly every `poll` ms while mounted (keeps showing the previous data). */
  poll?: number;
  /**
   * A Resource that reads the same thing and may already have asked (the shell's run list, a page's system info). This
   * one's first fetch, instead of a request of its own, takes that Resource's answer: the one it is loading, or one
   * under START_FRESH old whose refreshes have not failed. Only a success is taken: when the read it waits on fails, it
   * asks for itself and reports its own outcome, and with a `poll`, it stops waiting at its first tick.
   */
  startFrom?: () => Resource<T> | null | undefined;
}

export class Resource<T> {
  #data = signal<T | undefined>(undefined);
  #error = signal<unknown>(undefined);
  #loading = signal(true);
  #refreshing = signal(false);
  #stale = signal<unknown>(undefined);
  #updated = signal<number | null>(null);
  #committed = signal(0);

  get data(): T | undefined {
    return this.#data.get();
  }
  set data(v: T | undefined) {
    this.#data.set(v);
  }
  get error(): unknown {
    return this.#error.get();
  }
  set error(v: unknown) {
    this.#error.set(v);
  }
  get loading(): boolean {
    return this.#loading.get();
  }
  set loading(v: boolean) {
    this.#loading.set(v);
  }
  /** True while a background refresh (poll / manual) is in flight with data already shown. */
  get refreshing(): boolean {
    return this.#refreshing.get();
  }
  set refreshing(v: boolean) {
    this.#refreshing.set(v);
  }
  /**
   * The error of the last refresh that failed while data was shown (the data is kept, now stale); cleared by the next
   * success. A page says it once (Overview's stale-data notice) instead of dropping what it has.
   */
  get stale(): unknown {
    return this.#stale.get();
  }
  set stale(v: unknown) {
    this.#stale.set(v);
  }
  /** When the data shown last arrived (ms since the epoch); null before the first success. */
  get updated(): number | null {
    return this.#updated.get();
  }
  set updated(v: number | null) {
    this.#updated.set(v);
  }
  /** Monotonic sequence number of the latest fetch started on this Resource. */
  get requestSeq(): number {
    return this.#seq;
  }
  /** Reactive sequence number of the latest fetch that succeeded and committed data (never bumps on abort or error). */
  get committedSeq(): number {
    return this.#committed.get();
  }

  #fetcher: (signal: AbortSignal) => Promise<T>;
  #ctrl: AbortController | null = null;
  #seq = 0;
  /** The hub epoch of the last fetch (see above). */
  #epoch: number | null = null;
  /** The read this Resource is loading its data with (none during a quiet refresh): what `startFrom` can wait on. */
  #loadingRead: Promise<T> | null = null;
  /** The fetch that is waiting on another Resource's read (`startFrom`), by its controller. */
  #following: AbortController | null = null;
  /** Set by `invalidate()` until a fresh fetch settles, so `#lend()` never hands out pre-mutation data. */
  #invalidated = false;
  /** True once the owning reactive root cleans up, so post-unmount `refresh()` calls never leak live requests. */
  #disposed = false;

  constructor(fetcher: (signal: AbortSignal) => Promise<T>, opts: ResourceOptions<T> = {}) {
    this.#fetcher = fetcher;
    effect(() => {
      if (hub.mode === 'connecting') return;
      const epoch = hub.epoch;
      const first = this.#epoch === null;
      if (epoch !== this.#epoch) {
        this.#epoch = epoch;
        this.#invalidated = false;
        untrack(() => {
          this.data = undefined;
          this.stale = undefined;
          this.updated = null;
        });
      }
      const ctrl = new AbortController();
      // Reactive reads inside the fetcher (before its first await) are tracked by this effect; a synchronous throw
      // settles as a regular fetch error instead of crashing the effect.
      const ask = () => this.#ask(ctrl.signal);
      const from = first ? untrack(() => opts.startFrom?.()) : null;
      const start = from ? untrack(() => from.#lend()) : null;
      if (start && 'data' in start) {
        untrack(() => {
          this.data = copy(start.data);
          this.updated = start.updated;
          this.loading = false;
        });
      } else {
        const promise = start ? this.#follow(start.read, ctrl, ask) : ask();
        untrack(() => void this.#settle(promise, ctrl, false));
      }
      return () => {
        // Invalidate whatever is in flight (including refreshes started since), so nothing lands after cleanup.
        this.#seq = ++nextSeq;
        this.#ctrl?.abort();
        ctrl.abort();
      };
    });
    if (opts.poll) {
      const every = opts.poll;
      effect(() =>
        pollTogether(every, () => {
          // A first fetch still waiting on another Resource's read stops waiting here: that read may be stalled.
          const waiting = () => this.#following !== null && this.#following === this.#ctrl;
          if (hub.mode === 'connecting' || untrack(() => (this.loading && !waiting()) || this.refreshing)) return;
          if (document.visibilityState === 'visible') void this.#poll();
        }),
      );
    }
    // The server is back: what failed meanwhile (an error on screen, data gone stale) or loaded before is refetched, as
    // Retry or a quiet refresh would (a page left in an error state needs no click to recover).
    let back = untrack(() => hub.back);
    effect(() => {
      const now = hub.back;
      if (now === back) return;
      back = now;
      untrack(() => {
        if (hub.mode !== 'connecting' && (this.error !== undefined || this.data !== undefined)) void this.refresh();
      });
    });
    // Synchronous lifetime cleanup, so disposal before the first effect flush still marks `#disposed`.
    onCleanup(() => {
      this.#disposed = true;
      this.#seq = ++nextSeq;
      this.#ctrl?.abort();
    });
  }

  /** Runs the fetcher inside `together`, turning synchronous throws into rejected promises so effects stay healthy. */
  #ask(signal: AbortSignal): Promise<T> {
    try {
      return together(() => this.#fetcher(signal)).value;
    } catch (e) {
      return Promise.reject(e);
    }
  }

  /** Refetch, keeping the current data visible (with no data yet, a Retry: back to loading, then the outcome). */
  refresh(): Promise<void> {
    if (this.#disposed) return Promise.resolve();
    const ctrl = new AbortController();
    return untrack(() => this.#settle(this.#ask(ctrl.signal), ctrl, true));
  }

  /**
   * Marks the current snapshot ineligible for `startFrom` lending until a fresh read succeeds, and starts that read.
   * Keeps `data` and `updated` intact so consumers (and offline freshness timestamps) do not flicker.
   */
  invalidate(): Promise<void> {
    this.#invalidated = true;
    return this.refresh();
  }

  /**
   * A poll: quiet whatever is shown. A known error stays on screen until a poll succeeds, so a page whose server is
   * offline shows the error (and keeps its actions gated) the whole time, never a loading skeleton at every poll.
   */
  #poll(): Promise<void> {
    if (this.#disposed) return Promise.resolve();
    const ctrl = new AbortController();
    return untrack(() => this.#settle(this.#ask(ctrl.signal), ctrl, true, true));
  }

  /** Replace the data locally (optimistic updates). */
  set(v: T) {
    this.data = v;
  }

  /** What another Resource's first fetch can start from (`startFrom`), for the server in use; null when nothing. */
  #lend(): { read: Promise<T> } | { data: T; updated: number } | null {
    if (this.#disposed || this.#epoch !== hub.epoch || this.#invalidated) return null;
    if (this.#loadingRead) return { read: this.#loadingRead };
    const data = this.#data.peek();
    const updated = this.#updated.peek();
    if (data === undefined || updated === null || this.#stale.peek() !== undefined) return null;
    return Date.now() - updated < START_FRESH ? { data, updated } : null;
  }

  /** `read`'s answer, a copy of its own (no Resource sees another's edits); if it fails, this fetch asks for itself. */
  async #follow(read: Promise<T>, ctrl: AbortController, ask: () => Promise<T>): Promise<T> {
    this.#following = ctrl;
    try {
      return copy(await untilAborted(read, ctrl.signal));
    } catch (e) {
      if (ctrl.signal.aborted) throw e;
    } finally {
      if (this.#following === ctrl) this.#following = null;
    }
    return ask();
  }

  async #settle(promise: Promise<T>, ctrl: AbortController, quiet: boolean, poll = false): Promise<void> {
    if (this.#disposed) {
      void promise.catch(() => {});
      ctrl.abort();
      return;
    }
    this.#ctrl?.abort();
    this.#ctrl = ctrl;
    this.#loadingRead = null;
    const seq = (this.#seq = ++nextSeq);
    if (quiet && (this.data !== undefined || (poll && this.error !== undefined))) this.refreshing = true;
    else {
      this.loading = true;
      this.error = undefined;
      this.#loadingRead = promise;
    }
    try {
      const v = await promise;
      if (seq !== this.#seq) return;
      // Commit data, freshness flags, and `committedSeq` in one batch so readers never observe a torn state.
      batch(() => {
        // Equal parts of the answer keep the objects already shown (an unchanged answer writes nothing).
        this.data = reuse(this.#data.peek(), v);
        this.error = undefined;
        this.stale = undefined;
        this.#invalidated = false;
        this.updated = Date.now();
        this.#committed.set(seq);
      });
    } catch (e) {
      if (seq !== this.#seq || (e instanceof DOMException && e.name === 'AbortError')) return;
      if (!quiet || this.data === undefined) this.error = e;
      else this.stale = e;
    } finally {
      if (this.#loadingRead === promise) this.#loadingRead = null;
      if (seq === this.#seq) {
        this.loading = false;
        this.refreshing = false;
      }
    }
  }
}
