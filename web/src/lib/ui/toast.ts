// fw port of toast.svelte.ts (the same store, on fw signals).
// Toast notifications. `toast.success('Saved')`, `toast.error(err)`; <Toaster /> renders them. The toaster is for
// page-level events; an action taken inside an open dialog reports in the dialog (`ui/feedback.svelte.ts`).

import { signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { sentence } from '$lib/format';
import { isOom } from '$lib/oom';

export type ToastTone = 'success' | 'error' | 'info';

/** An error in a sentence or two: an ApiError's message and hint, out of GPU memory in plain words (as ErrorState). */
export function errorDetail(err: unknown): string {
  const message = err instanceof ApiError && isOom(err) ? 'The GPU ran out of memory.' : err instanceof Error ? err.message : String(err);
  return err instanceof ApiError ? [message, err.hint].filter((x): x is string => !!x).map((x) => sentence(x, true)).join(' ') : message;
}

export interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  /** Line 2: a sentence, or facts (each one a ui/Meta item: "Finished training · best mAP50-95 0.535 at epoch 60"). */
  detail?: string | string[];
  action?: { label: string; run: () => void };
  /** ms before auto-dismiss; 0 keeps it until closed (errors and toasts with an action, by default: WCAG 2.2.1). */
  timeout: number;
}

/** How long a success or a note without an action stays (DESIGN_DIRECTION 9): long enough to read two lines. */
export const AUTO_DISMISS_MS = 6000;

class Toasts {
  #items = signal<Toast[]>([]);
  get items(): Toast[] {
    return this.#items.get();
  }
  set items(v: Toast[]) {
    this.#items.set(v);
  }
  #id = 0;
  #timers = new Map<number, ReturnType<typeof setTimeout>>();
  /** While the pointer or focus is in the stack: no timers run and nothing is evicted (a held toast stays put). */
  #held = false;

  show(t: Omit<Toast, 'id' | 'timeout'> & { timeout?: number }): number {
    const id = ++this.#id;
    // An error, or a toast that offers an action, stays until it is closed: nobody should have to race a timer to read
    // what went wrong or to take the action. The rest (a success, a note) go after 6 s, paused while hovered or focused.
    const toast: Toast = { timeout: t.tone === 'error' || t.action ? 0 : AUTO_DISMISS_MS, ...t, id };
    this.items = this.#held ? [...this.items, toast] : [...this.items.slice(-3), toast];
    if (toast.timeout > 0 && !this.#held) this.#timers.set(id, setTimeout(() => this.dismiss(id), toast.timeout));
    return id;
  }

  success(title: string, detail?: string, action?: Toast['action']) {
    return this.show({ tone: 'success', title, detail, action });
  }

  info(title: string, detail?: string, action?: Toast['action']) {
    return this.show({ tone: 'info', title, detail, action });
  }

  /** Shows an error; ApiErrors contribute their message and hint. */
  error(err: unknown, title = 'Something went wrong') {
    return this.show({ tone: 'error', title, detail: errorDetail(err) });
  }

  dismiss(id: number) {
    clearTimeout(this.#timers.get(id));
    this.#timers.delete(id);
    this.items = this.items.filter((t) => t.id !== id);
  }

  /** Pause auto-dismiss (and eviction) while the pointer or focus is in the stack; new toasts wait too. */
  hold(on: boolean) {
    if (on === this.#held) return;
    this.#held = on;
    if (on) {
      for (const t of this.#timers.values()) clearTimeout(t);
      this.#timers.clear();
    } else {
      this.items = this.items.slice(-4);
      for (const t of this.items) if (t.timeout > 0) this.#timers.set(t.id, setTimeout(() => this.dismiss(t.id), 2500));
    }
  }
}

export const toast = new Toasts();
