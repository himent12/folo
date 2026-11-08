// Reactive async data for components. Create inside a component's script: the fetcher re-runs whenever a reactive
// value it reads synchronously changes (route params, filters) and when the API hub reconnects.

import { untrack } from 'svelte';
import { hub } from './index.svelte';

export interface ResourceOptions {
  /** Refetch quietly every `poll` ms while mounted (keeps showing the previous data). */
  poll?: number;
}

export class Resource<T> {
  data = $state.raw<T | undefined>(undefined);
  error = $state.raw<unknown>(undefined);
  loading = $state(true);
  /** True while a background refresh (poll / manual) is in flight with data already shown. */
  refreshing = $state(false);
  #fetcher: (signal: AbortSignal) => Promise<T>;
  #ctrl: AbortController | null = null;
  #seq = 0;

  constructor(fetcher: (signal: AbortSignal) => Promise<T>, opts: ResourceOptions = {}) {
    this.#fetcher = fetcher;
    $effect(() => {
      if (hub.mode === 'connecting') return;
      void hub.epoch;
      const ctrl = new AbortController();
      // Reactive reads inside the fetcher (before its first await) are tracked by this effect.
      const promise = fetcher(ctrl.signal);
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
      $effect(() => {
        const t = setInterval(() => {
          if (!untrack(() => this.loading || this.refreshing) && document.visibilityState === 'visible') void this.refresh();
        }, every);
        return () => clearInterval(t);
      });
    }
  }

  /** Refetch, keeping the current data visible. */
  refresh(): Promise<void> {
    const ctrl = new AbortController();
    return untrack(() => this.#settle(this.#fetcher(ctrl.signal), ctrl, true));
  }

  /** Replace the data locally (optimistic updates). */
  set(v: T) {
    this.data = v;
  }

  async #settle(promise: Promise<T>, ctrl: AbortController, quiet: boolean): Promise<void> {
    this.#ctrl?.abort();
    this.#ctrl = ctrl;
    const seq = ++this.#seq;
    if (quiet && this.data !== undefined) this.refreshing = true;
    else {
      this.loading = true;
      this.error = undefined;
    }
    try {
      const v = await promise;
      if (seq !== this.#seq) return;
      this.data = v;
      this.error = undefined;
    } catch (e) {
      if (seq !== this.#seq || (e instanceof DOMException && e.name === 'AbortError')) return;
      if (!quiet || this.data === undefined) this.error = e;
    } finally {
      if (seq === this.#seq) {
        this.loading = false;
        this.refreshing = false;
      }
    }
  }
}
