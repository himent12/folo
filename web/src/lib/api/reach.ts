// Whether the server in use answers (`hub.offline`). Fed by what already talks to it: every request (an HTTP answer of
// any status means it is there; a request that reached nothing means it may not be) and every event stream (open, or
// dropped). A failure is only a suspicion: one health probe confirms it before Studio says it is offline, and while it
// is, the probe repeats (PROBE_DELAYS) until the server answers again. Nothing polls while the server answers.

import { signal } from 'fw';

/** What a request or a stream tells about the server: it answered, or it could not be reached. */
export interface Reachability {
  ok(): void;
  failed(): void;
}

/** A health check of the server in use (resolves when it answers). */
export type Probe = (signal: AbortSignal) => Promise<unknown>;

/** Waits between probes while offline: soon at first, then every 5 s. */
export const PROBE_DELAYS = [2000, 3000, 5000];
/** A probe that takes longer than this counts as no answer. */
export const PROBE_TIMEOUT = 2500;

export class Reach implements Reachability {
  #offline = signal(false);
  #back = signal(0);
  #probe: Probe | null = null;
  #gen = 0;
  #probing = false;
  #attempt = 0;
  #timer: ReturnType<typeof setTimeout> | undefined;

  /** True while the server in use does not answer (confirmed by a probe). */
  get offline(): boolean {
    return this.#offline.get();
  }
  /** Bumps each time the server answers again after an outage: what failed or went stale meanwhile refetches. */
  get back(): number {
    return this.#back.get();
  }

  /** The server in use changed: forget any outage; `probe` checks the new one (null: no server, e.g. the mock). */
  reset(probe: Probe | null): void {
    this.#gen++;
    this.#probe = probe;
    this.#probing = false;
    this.#attempt = 0;
    clearTimeout(this.#timer);
    this.#timer = undefined;
    this.#offline.set(false);
  }

  ok(): void {
    if (!this.#offline.peek()) return;
    this.#gen++;
    this.#probing = false;
    this.#attempt = 0;
    clearTimeout(this.#timer);
    this.#timer = undefined;
    this.#offline.set(false);
    this.#back.set(this.#back.peek() + 1);
  }

  failed(): void {
    // Offline already (its probes are on their way), or a probe is out: nothing to add.
    if (!this.#probe || this.#offline.peek() || this.#probing) return;
    void this.#check();
  }

  async #check(): Promise<void> {
    const probe = this.#probe;
    if (!probe) return;
    const gen = this.#gen;
    this.#probing = true;
    try {
      await probe(AbortSignal.timeout(PROBE_TIMEOUT));
      if (gen !== this.#gen) return;
      this.#probing = false;
      this.ok();
    } catch {
      if (gen !== this.#gen) return;
      this.#probing = false;
      this.#offline.set(true);
      const delay = PROBE_DELAYS[Math.min(this.#attempt++, PROBE_DELAYS.length - 1)];
      this.#timer = setTimeout(() => {
        this.#timer = undefined;
        void this.#check();
      }, delay);
    }
  }
}
