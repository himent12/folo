// Toast notifications. `toast.success('Saved')`, `toast.error(err)`; <Toaster /> renders them. The toaster is for
// page-level events; an action taken inside an open dialog reports in the dialog (`ui/feedback.svelte.ts`).

import { signal } from 'fw';
import { ApiError } from '$lib/api/client';
import { sentence } from '$lib/format';
import { isOom } from '$lib/oom';

export type ToastTone = 'success' | 'error' | 'info';

/** A server hint written for agents and API clients (an endpoint, an MCP tool): kept with the details, not shown. */
export const forAgents = (hint: string) => /\b(?:GET|POST|PUT|PATCH|DELETE) \/api\/|\(MCP\)|docs\/API\.md/.test(hint);

/** An endpoint after a colon, or in parentheses, at the end of a clause: "stop it first: POST /api/runs/x/stop". */
const AGENT_TAIL = /(?::\s*(?:GET|POST|PUT|PATCH|DELETE) \/api\/\S*|\s*\([^()]*\))\s*$/;

/**
 * What a user can do of a server hint (PR #287 review): the hint less what is written for agents. Its clauses (split at
 * ", " and "; " outside parentheses) lose an endpoint at their end ("label images first: POST /api/…/start" → "label
 * images first"); a clause that still names one ("GET /api/runs lists them") goes. Null when nothing is left.
 */
export function humanHint(hint: string): string | null {
  if (!forAgents(hint)) return hint;
  const parts: { text: string; sep: string }[] = [];
  let depth = 0;
  let from = 0;
  for (let i = 0; i < hint.length; i++) {
    const c = hint[i];
    if (c === '(') depth++;
    else if (c === ')') depth = Math.max(0, depth - 1);
    else if (depth === 0 && (c === ',' || c === ';') && hint[i + 1] === ' ') {
      parts.push({ text: hint.slice(from, i), sep: hint.slice(i, i + 2) });
      from = i + 2;
    }
  }
  parts.push({ text: hint.slice(from), sep: '' });
  const kept = parts
    .map((p) => ({ ...p, text: (forAgents(p.text) ? p.text.replace(AGENT_TAIL, '') : p.text).trim() }))
    .filter((p) => p.text && !forAgents(p.text));
  const out = kept.map((p, i) => p.text + (i < kept.length - 1 ? p.sep : '')).join('');
  return out || null;
}

/**
 * An error in a sentence or two: an ApiError's message and hint, out of GPU memory in plain words (as ErrorState). What
 * of a hint is written for agents ("list_runs (MCP) or GET /api/runs …") is left out, as ErrorState leaves it (stress
 * s2-runs-7; `humanHint`).
 */
export function errorDetail(err: unknown): string {
  const message = err instanceof ApiError && isOom(err) ? 'The GPU ran out of memory.' : err instanceof Error ? err.message : String(err);
  if (!(err instanceof ApiError)) return message;
  const hint = err.hint ? humanHint(err.hint) : null;
  return [message, hint].filter((x): x is string => !!x).map((x) => sentence(x, true)).join(' ');
}

export interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  /** Line 2: a sentence, or facts (each one a ui/Meta item: "Finished training · best mAP50-95 0.535 at epoch 60"). */
  detail?: string | string[];
  action?: { label: string; run: () => void };
  /** ms before auto-dismiss (TOAST_MS by tone and action); 0 keeps it until closed. */
  timeout: number;
  /**
   * The action it reports on ("predict"): a newer toast with the same key takes its place, so an error goes as soon as
   * the next try of the same action succeeds.
   */
  key?: string;
}

/** How long a success or a note without an action stays (DESIGN_DIRECTION 9): long enough to read two lines. */
export const AUTO_DISMISS_MS = 6000;
/**
 * How long each kind stays, paused while the pointer or focus is in the stack (so nobody races a timer to read or to
 * act: WCAG 2.2.1): a note 6 s; one that offers an action twice that; an error longer still, or until the same action
 * (`key`) reports again.
 */
export const TOAST_MS = { note: AUTO_DISMISS_MS, action: 12_000, error: 20_000 };

/** Options of the shorthands: the action the toast reports on (Toast.key). */
export interface ToastOptions {
  key?: string;
}

/** Two toasts that say the same thing: the second takes the first one's place instead of stacking under it. */
const same = (a: Omit<Toast, 'id' | 'timeout'>, b: Toast) =>
  a.tone === b.tone && a.title === b.title && JSON.stringify(a.detail ?? null) === JSON.stringify(b.detail ?? null);

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
    const timeout = t.tone === 'error' ? TOAST_MS.error : t.action ? TOAST_MS.action : TOAST_MS.note;
    const toast: Toast = { timeout, ...t, id };
    // The toast it replaces: the last report of the same action, or one saying the very same thing.
    for (const old of this.items) if ((t.key !== undefined && old.key === t.key) || same(t, old)) this.dismiss(old.id);
    this.items = this.#held ? [...this.items, toast] : [...this.items.slice(-3), toast];
    if (toast.timeout > 0 && !this.#held) this.#timers.set(id, setTimeout(() => this.dismiss(id), toast.timeout));
    return id;
  }

  success(title: string, detail?: string, action?: Toast['action'], opts: ToastOptions = {}) {
    return this.show({ tone: 'success', title, detail, action, ...opts });
  }

  info(title: string, detail?: string, action?: Toast['action'], opts: ToastOptions = {}) {
    return this.show({ tone: 'info', title, detail, action, ...opts });
  }

  /** Shows an error; ApiErrors contribute their message and hint. */
  error(err: unknown, title = 'Something went wrong', opts: ToastOptions = {}) {
    return this.show({ tone: 'error', title, detail: errorDetail(err), ...opts });
  }

  /** Takes back the toast of an action (`key`) that has since gone through, with nothing new to say about it. */
  clear(key: string) {
    for (const t of this.items) if (t.key === key) this.dismiss(t.id);
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
