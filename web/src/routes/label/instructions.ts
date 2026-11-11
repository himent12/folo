// // The job's instructions for the agent, as the job page's field holds them: saved when the field is left and before a
// run starts. One save runs at a time; a save asked while one runs waits for it and then sends the field's latest
// text, so the latest text wins, and a run started after an edit starts once that text is saved. A change on the server
// shows in the field unless the field holds an edit of its own.

import { signal } from 'fw';
import { isActive } from '$lib/api/liveLabel';
import type { LabelJob } from '$lib/api/types';
import { num } from '$lib/format';
import { INSTRUCTIONS_MAX } from '$lib/label';

export class Instructions {
  #text = signal('');
  #error = signal<string | null>(null);
  /** The field's text. */
  get text(): string {
    return this.#text.get();
  }
  set text(v: string) {
    this.#text.set(v);
  }
  /** Why the text is not saved (too long, or the save failed); null when it is fine. */
  get error(): string | null {
    return this.#error.get();
  }
  set error(v: string | null) {
    this.#error.set(v);
  }
  /** The server's text as last seen or saved: the field shows a newer one unless it holds an edit. */
  #taken = '';
  /** The saves asked so far, one after another. */
  #queue: Promise<boolean> = Promise.resolve(true);

  constructor(
    readonly job: () => LabelJob | undefined,
    /** Saves the instructions (the page applies the answer). */
    readonly send: (jobId: string, instructions: string) => Promise<unknown>,
  ) {}

  /** Takes the server's text (every job the page shows): into the field, unless the field holds an edit. */
  sync(server: string) {
    if (this.#text.peek() === this.#taken) this.text = server;
    this.#taken = server;
  }

  /** Saves the field's text once the saves before it are done; true when it is saved (or nothing needed saving). */
  save(): Promise<boolean> {
    const next = this.#queue.then(() => this.#save());
    this.#queue = next;
    return next;
  }

  async #save(): Promise<boolean> {
    const job = this.job();
    // The text as it is now, not as it was when the save was asked: the latest text wins.
    const text = this.#text.peek();
    if (!job || isActive(job) || text === this.#taken) return true;
    if (text.length > INSTRUCTIONS_MAX) {
      this.error = `At most ${num(INSTRUCTIONS_MAX)} characters (now ${num(text.length)}).`;
      return false;
    }
    try {
      await this.send(job.id, text);
      // What was sent is saved, whatever the field holds by now (an edit typed meanwhile stays an edit).
      this.#taken = text;
      this.error = null;
      return true;
    } catch (e) {
      this.error = `Not saved: ${e instanceof Error ? e.message : String(e)}`;
      return false;
    }
  }
}
