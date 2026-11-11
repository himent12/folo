// A run's times as the lists and its page show them: relative ("9m ago") on a clock that ticks, the exact local date
// and time in the tooltip.

import { effect, signal } from 'fw';

/** "Nov 11, 2026, 14:14:05" in local time: the tooltip of a relative time. */
export function exactTime(iso: string | null | undefined): string | undefined {
  if (!iso) return undefined;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return undefined;
  return d.toLocaleString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

/**
 * The time relative times are read against, ticking every `every` ms while the calling component is mounted, so "1m
 * ago" turns into "2m ago" without a reload (a polled list keeps its rows when nothing else changed).
 */
export function tickingNow(every = 30_000) {
  const now = signal(Date.now());
  effect(() => {
    const t = setInterval(() => now.set(Date.now()), every);
    return () => clearInterval(t);
  });
  return now;
}
