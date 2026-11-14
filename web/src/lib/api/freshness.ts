// What is current on a screen (DESIGN_DIRECTION §2.8, freshness). Two clocks, never confused: a snapshot's (a list a
// Resource polls; `Resource.stale` and `updated`) and a stream's (the live run's SSE; `LiveRun.lost`). A state that is
// no longer current says "as of" when it was, and stops its live marks; one that still is stays live.

import { effect, signal } from 'fw';
import { ago, eta } from '$lib/format';

/** "2m ago" for a time in ms; "seconds ago" under a minute ("just now" would read as live). */
export function asOf(t: number, now: number): string {
  if (now - t < 45_000) return 'seconds ago';
  return ago(new Date(t).toISOString(), now);
}

/** A remaining-time estimate: "7m left" while current, "~7m left, as of 2m ago" once its source went quiet at `since`. */
export function etaText(seconds: number | null, since: number | null, now: number): string {
  if (seconds === null) return '—';
  return since === null ? `${eta(seconds)} left` : `~${eta(seconds)} left, as of ${asOf(since, now)}`;
}

/** A status that is no longer current: "Last known: Running · 2m ago". */
export function lastKnown(word: string, since: number, now: number): string {
  return `Last known: ${word} · ${asOf(since, now)}`;
}

/**
 * The time now, re-read every `every` ms while `active()` holds (a stale state's "2m ago" keeps counting) and not at
 * all otherwise. Create inside a component's script.
 */
export class Now {
  #current = signal(Date.now());
  get current(): number {
    return this.#current.get();
  }
  set current(v: number) {
    this.#current.set(v);
  }
  constructor(active: () => boolean, every = 15_000) {
    effect(() => {
      if (!active()) return;
      this.current = Date.now();
      const t = setInterval(() => (this.current = Date.now()), every);
      return () => clearInterval(t);
    });
  }
}
