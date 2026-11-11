// Resource: data loaded for what is on screen, sharing what it can with the other Resources on
// screen (./share) without changing what any of them shows or when:
// - its fetcher runs inside `together`: an identical read another Resource issued in the same turn (a screen mounting,
//   a reconnect) is the same request;
// - its poll timer is shared with the pollers started in the same turn with the same period (one task per tick, and
//   their identical reads one request), stops when the last one goes, and stops while the tab is hidden;
// - an answer equal to the data shown keeps the data shown (`reuse`): an unchanged poll writes nothing, so nothing that
//   reads `data` recomputes, and a changed list keeps its unchanged items (keyed rows stay as they are).
//
// Reactive async data for components. Create inside a component's body: the fetcher re-runs whenever a reactive value
// it reads synchronously changes (route params, filters) and when the API hub reconnects.

import { effect, signal, untrack } from 'fw';
import { hub } from './index';
import { pollTogether, reuse, together } from './share';

export interface ResourceOptions {
  /** Refetch quietly every `poll` ms while mounted (keeps showing the previous data). */
  poll?: number;
}

export class Resource<T> {
  #data = signal<T | undefined>(undefined);
  #error = signal<unknown>(undefined);
  #loading = signal(true);
  #refreshing = signal(false);
  #stale = signal<unknown>(undefined);
  #updated = signal<number | null>(null);

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

  #fetcher: (signal: AbortSignal) => Promise<T>;
  #ctrl: AbortController | null = null;
  #seq = 0;

  constructor(fetcher: (signal: AbortSignal) => Promise<T>, opts: ResourceOptions = {}) {
    this.#fetcher = fetcher;
    effect(() => {
      if (hub.mode === 'connecting') return;
      void hub.epoch;
      const ctrl = new AbortController();
      // Reactive reads inside the fetcher (before its first await) are tracked by this effect.
      const promise = together(() => fetcher(ctrl.signal)).value;
      untrack(() => void this.#settle(promise, ctrl, false));
      return () => {
        // Invalidate whatever is in flight (including refreshes started since), so nothing lands after cleanup.
        this.#seq++;
        this.#ctrl?.abort();
        ctrl.abort();
      };
    });
    if (opts.poll) {
      const every = opts.poll;
      effect(() =>
        pollTogether(every, () => {
          if (hub.mode === 'connecting' || untrack(() => this.loading || this.refreshing)) return;
          if (document.visibilityState === 'visible') void this.#poll();
        }),
      );
    }
    // Lifetime cleanup, independent of the fetch effect (which returns early while the hub reconnects).
    effect(() => () => {
      this.#seq++;
      this.#ctrl?.abort();
    });
  }

  /** Refetch, keeping the current data visible (with no data yet, a Retry: back to loading, then the outcome). */
  refresh(): Promise<void> {
    const ctrl = new AbortController();
    return untrack(() => this.#settle(together(() => this.#fetcher(ctrl.signal)).value, ctrl, true));
  }

  /**
   * A poll: quiet whatever is shown. A known error stays on screen until a poll succeeds, so a page whose server is
   * offline shows the error (and keeps its actions gated) the whole time, never a loading skeleton at every poll.
   */
  #poll(): Promise<void> {
    const ctrl = new AbortController();
    return untrack(() => this.#settle(together(() => this.#fetcher(ctrl.signal)).value, ctrl, true, true));
  }

  /** Replace the data locally (optimistic updates). */
  set(v: T) {
    this.data = v;
  }

  async #settle(promise: Promise<T>, ctrl: AbortController, quiet: boolean, poll = false): Promise<void> {
    this.#ctrl?.abort();
    this.#ctrl = ctrl;
    const seq = ++this.#seq;
    if (quiet && (this.data !== undefined || (poll && this.error !== undefined))) this.refreshing = true;
    else {
      this.loading = true;
      this.error = undefined;
    }
    try {
      const v = await promise;
      if (seq !== this.#seq) return;
      // Equal parts of the answer keep the objects already shown (an unchanged answer writes nothing).
      this.data = reuse(this.#data.peek(), v);
      this.error = undefined;
      this.stale = undefined;
      this.updated = Date.now();
    } catch (e) {
      if (seq !== this.#seq || (e instanceof DOMException && e.name === 'AbortError')) return;
      if (!quiet || this.data === undefined) this.error = e;
      else this.stale = e;
    } finally {
      if (seq === this.#seq) {
        this.loading = false;
        this.refreshing = false;
      }
    }
  }
}
