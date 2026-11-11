// The job page's log for run/Logs, which reads an append log (only the lines added since it last read): LiveLabelJob
// keeps the job's log as a list it replaces with each line (merged in time order, capped at the front).

import type { AppendLog } from '$lib/api/appendLog';

/**
 * Brings `log` to `next`: what `next` adds after the last line `log` holds is appended, the lines its cap dropped are
 * trimmed from the front, and anything else (a reload, a line merged in between) replaces the log. `log` keeps its own
 * copy, so `next` is never changed in place.
 */
export function syncLog<T>(log: AppendLog<T>, next: readonly T[]): void {
  const prev = log.items;
  const last = prev.at(-1);
  const k = last === undefined ? -1 : next.lastIndexOf(last);
  const dropped = prev.length - 1 - k;
  if (k >= 0 && dropped >= 0 && next[0] === prev[dropped]) {
    if (dropped) log.trimFront(dropped);
    if (next.length > k + 1) log.push(...next.slice(k + 1));
  } else log.replace([...next]);
}
