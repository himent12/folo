// Whether the server in use lets Studio in (`hub.refused` while it does not): a fact of its own, beside ./reach's
// whether it answers at all. Only the token check (handshake's `accessCheck`, `HEAD /api/runs`, as a connect asks it)
// changes it: refused when the check answers 401 or 403, let in when it answers 2xx; a check that cannot tell (no
// answer, a timeout, another status) changes nothing. Other answers never change it (a 200 to a request sent earlier,
// an event stream opening, the health): a request's 401 or 403 only asks for a check, one at a time, and one more after
// it when the check out had started before that request went out. Only the latest check started may change the state.
// While refused, the check repeats (PROBE_DELAYS, as a connect's) until Studio is let in again. Nothing checks while
// nothing is refused.

import { signal } from 'fw';
import type { AccessAnswer } from './handshake';
import { PROBE_DELAYS, PROBE_TIMEOUT } from './reach';

/** What a request tells about access: it notes `checks` when it goes out, and calls `refused` with it on a 401 or 403. */
export interface AccessWatch {
  /** How many checks have started: a request answered 401 or 403 passes the count it saw when it went out. */
  readonly checks: number;
  refused(seen: number): void;
}

/** A token check of the server in use (`accessCheck` bound to its address). */
export type Check = (signal: AbortSignal) => Promise<AccessAnswer>;

export class Access implements AccessWatch {
  #refusal = signal<401 | 403 | null>(null);
  #back = signal(0);
  #check: Check | null = null;
  #checks = 0;
  #out = false;
  /** A request refused after the check out had started: one more check once it ends. */
  #again = false;
  #attempt = 0;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #ctrl: AbortController | null = null;

  /** The status the latest check refused Studio with (401, 403), or null: let in, or nothing refused yet. */
  get refusal(): 401 | 403 | null {
    return this.#refusal.get();
  }
  /** Bumps each time Studio is let in again after a refusal: what failed meanwhile refetches. */
  get back(): number {
    return this.#back.get();
  }
  get checks(): number {
    return this.#checks;
  }

  /** The server in use changed: forget any refusal; a check still out is cancelled and says nothing; `check` asks the new one. */
  reset(check: Check | null): void {
    this.#checks++;
    this.#check = check;
    this.#out = false;
    this.#again = false;
    this.#attempt = 0;
    const ctrl = this.#ctrl;
    this.#ctrl = null;
    ctrl?.abort();
    clearTimeout(this.#timer);
    this.#timer = undefined;
    this.#refusal.set(null);
  }

  /** While refused, check again right away (e.g. once the server is reachable again after an outage) instead of waiting out the backoff. */
  recheck(): void {
    if (!this.#check || this.#refusal.peek() === null || this.#out) return;
    clearTimeout(this.#timer);
    this.#timer = undefined;
    this.#attempt = 0;
    void this.#run();
  }

  refused(seen: number): void {
    // Refused already (its checks are on their way): nothing to add.
    if (!this.#check || this.#timer !== undefined) return;
    // The check out started after this request went out: it answers for it too.
    if (this.#out) this.#again ||= seen >= this.#checks;
    else void this.#run();
  }

  async #run(): Promise<void> {
    const check = this.#check;
    if (!check) return;
    const mine = ++this.#checks;
    const ctrl = new AbortController();
    this.#ctrl = ctrl;
    this.#out = true;
    this.#again = false;
    const answer = await check(AbortSignal.any([ctrl.signal, AbortSignal.timeout(PROBE_TIMEOUT)])).catch(() => null);
    if (this.#ctrl === ctrl) this.#ctrl = null;
    if (mine !== this.#checks) return;
    this.#out = false;
    if (answer === 'in') {
      this.#attempt = 0;
      if (this.#refusal.peek() !== null) {
        this.#refusal.set(null);
        this.#back.set(this.#back.peek() + 1);
      }
    } else if (answer !== null) this.#refusal.set(answer);
    if (this.#again) return void this.#run();
    if (this.#refusal.peek() === null) return;
    this.#timer = setTimeout(
      () => {
        this.#timer = undefined;
        void this.#run();
      },
      PROBE_DELAYS[Math.min(this.#attempt++, PROBE_DELAYS.length - 1)],
    );
  }
}
