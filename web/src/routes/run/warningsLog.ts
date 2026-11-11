// `scheduleWarnings` (./warnings) over a live run's log as it streams: each update reads only the lines added (and
// dropped by the 2000-line cap) since the last, instead of testing every line of the log again (the Svelte page did,
// on every step). The result is the same list, the same array while it does not change.

import type { AppendLog, LogCursor } from '$lib/api/appendLog';
import type { LogLine } from '$lib/api/types';
import { scheduleWarnings } from './warnings';

const isWarning = (l: LogLine) => l.level === 'warn' && scheduleWarnings([l]).length > 0;

export class WarningsLog {
  #cursor: LogCursor | null = null;
  /** The schedule warning lines in the log, oldest first, with their absolute index (the cap trims the front). */
  #kept: { line: LogLine; at: number }[] = [];
  #out: string[] = [];

  /** The schedule warnings in the log now: `scheduleWarnings(log.items)`. */
  sync(log: AppendLog<LogLine>): string[] {
    const r = log.read(this.#cursor);
    this.#cursor = r.cursor;
    let changed = false;
    if (r.reset) {
      changed = this.#kept.length > 0;
      this.#kept = [];
    } else if (r.dropped) {
      let n = 0;
      while (n < this.#kept.length && this.#kept[n]!.at < r.cursor.start) n++;
      if (n) {
        this.#kept.splice(0, n);
        changed = true;
      }
    }
    let at = r.cursor.end - r.added.length;
    for (const line of r.added) {
      if (isWarning(line)) {
        this.#kept.push({ line, at });
        changed = true;
      }
      at++;
    }
    if (changed) {
      const next = scheduleWarnings(this.#kept.map((k) => k.line));
      if (next.length !== this.#out.length || next.some((w, i) => w !== this.#out[i])) this.#out = next;
    }
    return this.#out;
  }
}
