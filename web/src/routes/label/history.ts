// Undo and redo for the labeling editor: the states one image's draft had before each edit, and the ones an undo
// stepped back from. Edits of one kind in a row (an arrow key held on one box) are one step, as in drawing tools.

/** Steps kept per image; the oldest go first. */
export const HISTORY_MAX = 200;

export class History<T> {
  past: T[] = [];
  future: T[] = [];
  /** The kind of the last edit recorded, while the next one of that kind joins it. */
  #last: string | null = null;

  /** Records `before`, the state an edit replaces. An edit of the same non-null `kind` as the one before joins its step. */
  record(before: T, kind: string | null = null) {
    if (kind === null || kind !== this.#last) {
      this.past.push(before);
      if (this.past.length > HISTORY_MAX) this.past.shift();
    }
    this.future = [];
    this.#last = kind;
  }

  /** The state before the last step, with `current` kept for redo; undefined when there is nothing to undo. */
  undo(current: T): T | undefined {
    const prev = this.past.pop();
    if (prev === undefined) return undefined;
    this.future.push(current);
    this.#last = null;
    return prev;
  }

  /** The state the last undo stepped back from, with `current` kept for undo; undefined when there is none. */
  redo(current: T): T | undefined {
    const next = this.future.pop();
    if (next === undefined) return undefined;
    this.past.push(current);
    this.#last = null;
    return next;
  }

  /** Ends the current run of like edits: the next edit is a step of its own (a key released, a save). */
  seal() {
    this.#last = null;
  }

  clear() {
    this.past = [];
    this.future = [];
    this.#last = null;
  }
}
