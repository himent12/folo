// Toast notifications. `toast.success('Saved')`, `toast.error(err)`; <Toaster /> renders them.

import { ApiError } from '$lib/api/client';
import { sentence } from '$lib/format';
import { isOom } from '$lib/oom';

export type ToastTone = 'success' | 'error' | 'info';

export interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  detail?: string;
  action?: { label: string; run: () => void };
  /** ms before auto-dismiss; 0 keeps it until closed. */
  timeout: number;
}

class Toasts {
  items = $state<Toast[]>([]);
  #id = 0;
  #timers = new Map<number, ReturnType<typeof setTimeout>>();

  show(t: Omit<Toast, 'id' | 'timeout'> & { timeout?: number }): number {
    const id = ++this.#id;
    const toast: Toast = { timeout: t.tone === 'error' ? 8000 : 4000, ...t, id };
    this.items = [...this.items.slice(-3), toast];
    if (toast.timeout > 0) this.#timers.set(id, setTimeout(() => this.dismiss(id), toast.timeout));
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
    // As ErrorState: out of GPU memory in plain words (the driver's report is for the place that shows details).
    const message = err instanceof ApiError && isOom(err) ? 'The GPU ran out of memory.' : err instanceof Error ? err.message : String(err);
    const detail = err instanceof ApiError ? [message, err.hint].filter((x): x is string => !!x).map((x) => sentence(x, true)).join(' ') : message;
    return this.show({ tone: 'error', title, detail });
  }

  dismiss(id: number) {
    clearTimeout(this.#timers.get(id));
    this.#timers.delete(id);
    this.items = this.items.filter((t) => t.id !== id);
  }

  /** Pause auto-dismiss while the pointer is over the stack. */
  hold(on: boolean) {
    if (on) {
      for (const t of this.#timers.values()) clearTimeout(t);
      this.#timers.clear();
    } else {
      for (const t of this.items) if (t.timeout > 0) this.#timers.set(t.id, setTimeout(() => this.dismiss(t.id), 2500));
    }
  }
}

export const toast = new Toasts();
