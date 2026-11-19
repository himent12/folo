// Whether the server in use answers, and lets Studio in (`hub.offline`, `hub.refused`). Fed by what already talks to
// it: every request (an HTTP answer means it is there; one that reached nothing means it may not be, and a 401 or 403
// that it may refuse Studio now, e.g. a proxy that stopped sending the token) and every event stream (open, or
// dropped). A failure is only a suspicion: one probe (the health, then the token check, as a connect asks them)
// confirms it before Studio says it is offline or refused, and while it is, the probe repeats (PROBE_DELAYS) until the
// server answers and lets Studio in again. Nothing polls while it does.

import { signal } from 'fw';

/** What a request or a stream tells about the server: it answered, or it could not be reached or refused Studio. */
export interface Reachability {
  /** An answer; `sent`: when its request went out (`performance.now()`), so one older than the last probe is moot. */
  ok(sent?: number): void;
  failed(): void;
}

/** A check of the server in use: resolves when it answers at all (any HTTP status, any body), with the reason in words
 * when it answered but refuses Studio (a token wall, as handshake's `refusal` says it); rejects only when nothing
 * answered. */
export type Probe = (signal: AbortSignal) => Promise<unknown>;

/** Waits between probes while offline: soon at first, then every 5 s. */
export const PROBE_DELAYS = [2000, 3000, 5000];
/** A probe that takes longer than this counts as no answer. */
export const PROBE_TIMEOUT = 2500;

export class Reach implements Reachability {
  #offline = signal(false);
  #refused = signal<string | null>(null);
  #back = signal(0);
  #probe: Probe | null = null;
  #gen = 0;
  #probing = false;
  #attempt = 0;
  #timer: ReturnType<typeof setTimeout> | undefined;
  /** When the latest probe went out: an answer to a request sent before it says nothing newer than the probe will. */
  #asked = -Infinity;

  /** True while the server in use does not answer, or refuses Studio (confirmed by a probe). */
  get offline(): boolean {
    return this.#offline.get();
  }
  /** While offline: why, in words, when the server answers but refuses Studio (the probe's reason); else null. */
  get refused(): string | null {
    return this.#refused.get();
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
    this.#asked = -Infinity;
    this.#offline.set(false);
    this.#refused.set(null);
    this.#settle(false);
  }

  ok(sent?: number): void {
    // An answer to a request sent before the latest probe is older news than that probe (a 200 that left before a
    // proxy stopped sending the token): the probe decides. A late 401 is a suspicion of its own (`failed`).
    if (sent !== undefined && sent < this.#asked) return;
    // An answer settles any suspicion: a probe still out is moot, whatever it ends with.
    this.#gen++;
    this.#probing = false;
    this.#settle(true);
    if (!this.#offline.peek()) return;
    this.#attempt = 0;
    clearTimeout(this.#timer);
    this.#timer = undefined;
    this.#offline.set(false);
    this.#refused.set(null);
    this.#back.set(this.#back.peek() + 1);
  }

  /** A request reached nothing, or the server refused it (401, 403: maybe Studio, maybe that one request). */
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
    this.#asked = performance.now();
    // A string: it answered but refuses Studio; null: nothing answered.
    let refused: string | null;
    try {
      const found = await probe(AbortSignal.timeout(PROBE_TIMEOUT));
      if (gen !== this.#gen) return;
      this.#probing = false;
      if (typeof found !== 'string') return this.ok();
      refused = found;
    } catch {
      if (gen !== this.#gen) return;
      this.#probing = false;
      refused = null;
    }
    this.#offline.set(true);
    this.#refused.set(refused);
    this.#settle(false);
    const delay = PROBE_DELAYS[Math.min(this.#attempt++, PROBE_DELAYS.length - 1)];
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      void this.#check();
    }, delay);
  }

  /** Who waits for what the check finds (`suspect`), settled by the next answer, failed probe or reset. */
  #waiters: ((up: boolean) => void)[] = [];

  /**
   * As `failed`, and says what the check found: true when the server answered (a probe, or any request), false when it
   * did not or refuses Studio (offline: its return then ends in `back`), when it is offline already, or when there is
   * nothing to check (the mock). Lets something the browser loads itself (an <img>) retry once when its failure was an
   * outage over by the time the check landed (stress s1-datasets-8).
   */
  suspect(): Promise<boolean> {
    if (!this.#probe || this.#offline.peek()) return Promise.resolve(false);
    const answer = new Promise<boolean>((resolve) => this.#waiters.push(resolve));
    this.failed();
    return answer;
  }

  #settle(up: boolean): void {
    const waiters = this.#waiters;
    this.#waiters = [];
    for (const w of waiters) w(up);
  }
}
