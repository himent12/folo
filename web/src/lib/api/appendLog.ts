// A list that grows at its end almost always (a live run's steps, epochs and log lines), told apart from the rare
// change that is not an append (a history merge, a resume's rewind, the log's cap trimming its oldest lines), so a
// consumer can do its work on the new items only:
//
//   let cursor: LogCursor | null = null;
//   const points: Point[] = [];
//   const series = computed(() => {
//     void live.version;                          // the signal that says the lists changed (at most once a frame)
//     const r = live.stepsLog.read(cursor);
//     if (r.reset) points.length = 0;             // replaced: start over from r.added (all of it)
//     else if (r.dropped) points.splice(0, r.dropped);
//     for (const s of r.added) points.push(toPoint(s));
//     cursor = r.cursor;
//     return { points, n: points.length };        // a new object: the computed's readers see the change
//   });
//
// `items` is the list itself (the same array while it only grows): code that reads it whole keeps working.

/** Where a consumer stands in a log: what it has read so far. */
export interface LogCursor {
  readonly gen: number;
  /** The absolute index of `items[0]` when it read (items trimmed from the front move it). */
  readonly start: number;
  /** The absolute index after the last item it read. */
  readonly end: number;
}

export interface LogRead<T> {
  /** The list was replaced since the cursor (or there was none): `added` is all of it. */
  reset: boolean;
  /** The items after the cursor (all of them on a reset). */
  added: readonly T[];
  /** How many items left the front since the cursor (never on a reset). */
  dropped: number;
  /** The cursor to give the next read. */
  cursor: LogCursor;
}

export class AppendLog<T> {
  #items: T[] = [];
  #gen = 0;
  #start = 0;

  /** The list, oldest first. Appending to it in place (`push`) is an append; anything else goes through the methods. */
  get items(): T[] {
    return this.#items;
  }
  get length(): number {
    return this.#items.length;
  }
  /** Bumps on every change that is not an append or a trim at the front. */
  get gen(): number {
    return this.#gen;
  }

  push(...items: T[]): void {
    this.#items.push(...items);
  }

  /**
   * Replaces the list. A list that starts with the current items (the same objects) is an append of the rest;
   * anything else is a new generation.
   */
  replace(next: T[]): void {
    const cur = this.#items;
    if (next === cur) return;
    let prefix = next.length >= cur.length;
    for (let i = 0; prefix && i < cur.length; i++) if (next[i] !== cur[i]) prefix = false;
    if (prefix) {
      for (let i = cur.length; i < next.length; i++) cur.push(next[i] as T);
      return;
    }
    this.#items = next;
    this.#gen++;
    this.#start = 0;
  }

  /** Drops the `n` oldest items. */
  trimFront(n: number): void {
    if (n <= 0) return;
    this.#items.splice(0, n);
    this.#start += n;
  }

  /** What changed since `cursor` (null: everything, as a reset). */
  read(cursor: LogCursor | null | undefined): LogRead<T> {
    const items = this.#items;
    const start = this.#start;
    const next: LogCursor = { gen: this.#gen, start, end: start + items.length };
    if (!cursor || cursor.gen !== this.#gen || cursor.end < start) return { reset: true, added: items.slice(), dropped: 0, cursor: next };
    return { reset: false, added: items.slice(cursor.end - start), dropped: start - cursor.start, cursor: next };
  }
}
