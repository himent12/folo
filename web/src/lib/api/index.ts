// The hub (the API in use, live or the mock). Two things about how it loads, neither visible:
// - The in-browser mock is loaded only when it is asked for (`?mock=1`): it is a chunk of its own, never fetched or
//   parsed otherwise, and its load starts with this module. Until the first connect settles, `api` is the server's (a
//   placeholder that, like the Svelte hub's initial mock, says `mock`).
// - `api` is the Api as ./share wraps it: reads a Resource issues together with another one are one request. Outside a
//   Resource's fetch it is the Api itself, call for call. Assigning an Api (tests) stores that Api.
//
// Picks the API implementation: the real server, or the in-browser mock with `?mock=1` (and only then: a server that
// is slow or not there is offline, never demo data). Screens read `hub.api` and never care which one it is.

import { computed, signal } from 'fw';
import { prefs } from '$lib/prefs';
import { ApiError, HttpApi, type Api } from './client';
import { isHealth, NOT_FOLO, refusal, TOKEN_WALL, tokenWall } from './handshake';
import { PROBE_DELAYS, Reach } from './reach';
import { shareApi, unshare } from './share';
import type { Scenario } from './mock/scenario';

export type Mode = 'connecting' | 'live' | 'mock';

/** No answer to a connect's first health check within this long: Studio says the server is offline (and still waits
 * for that answer). */
export const SLOW_CONNECT = 2500;
/** A connect's health check that takes longer than this counts as no answer and is asked again (PROBE_DELAYS). */
export const CONNECT_TIMEOUT = 10_000;

const params = typeof location === 'undefined' ? new URLSearchParams() : new URLSearchParams(location.search);
const forcedMock = params.get('mock') === '1';
const scenario = (params.get('scenario') ?? 'normal') as Scenario;

type MockModule = typeof import('./mock');
let mockModule: MockModule | null = null;
let mockLoad: Promise<MockModule> | null = null;
/** The mock's module, loaded once (its own chunk). */
function loadMock(): Promise<MockModule> {
  mockLoad ??= import('./mock').then((m) => (mockModule = m));
  return mockLoad;
}
// Forced: start fetching it while the rest of the app loads.
if (forcedMock && typeof window !== 'undefined') void loadMock().catch(() => {});

/**
 * Before the first connect has settled: the server's Api, saying `mock` as the Svelte hub's initial MockApi does (pages
 * gate on `mode`, which is `connecting` until then).
 */
function placeholder(): Api {
  const http = new HttpApi(prefs.serverUrl);
  return new Proxy(http, {
    get: (t, p) => {
      if (p === 'mock') return true;
      const v = Reflect.get(t, p, t);
      return typeof v === 'function' ? v.bind(t) : v;
    },
  }) as Api;
}

class Hub {
  #mode = signal<Mode>('connecting');
  #reason = signal<'forced' | null>(null);
  /** A connect has had no answer yet, past SLOW_CONNECT or a failed check: offline while it keeps asking. */
  #unreachable = signal(false);
  #refused = signal<string | null>(null);
  #version = signal<string | null>(null);
  #api = signal<Api>(placeholder());
  #shared = computed(() => shareApi(this.#api.get()));
  #epoch = signal(0);
  #reach = new Reach();

  get mode(): Mode {
    return this.#mode.get();
  }
  set mode(v: Mode) {
    this.#mode.set(v);
  }
  /** Why we are on the mock: 'forced' (?mock=1, the one way to it). */
  get reason(): 'forced' | null {
    return this.#reason.get();
  }
  set reason(v: 'forced' | null) {
    this.#reason.set(v);
  }
  get version(): string | null {
    return this.#version.get();
  }
  set version(v: string | null) {
    this.#version.set(v);
  }
  get api(): Api {
    return this.#shared.get();
  }
  set api(v: Api) {
    this.#api.set(unshare(v));
  }
  /**
   * The server in use stopped answering (./reach: a failed request or a dropped stream, confirmed by a health probe,
   * which repeats until it answers again), or has not answered a connect yet (slow or not there; the connect keeps
   * asking), or the demo's `?scenario=error`. Live data on screen is then a last known state, not the current one.
   */
  get offline(): boolean {
    return scenario === 'error' || this.#unreachable.get() || this.#reach.offline;
  }
  /**
   * While connecting: what the address answered when it answered but not as a usable FOLO server (not its health
   * payload, a sign-in wall), in words for the user; null when nothing answered. Offline meanwhile, and still asking.
   */
  get refused(): string | null {
    return this.#refused.get();
  }
  /** Bumps each time the server answers again after an outage: Resources in error or holding data refetch. */
  get back(): number {
    return this.#reach.back;
  }
  /** Bumps on every (re)connect so data views refetch. */
  get epoch(): number {
    return this.#epoch.get();
  }
  set epoch(v: number) {
    this.#epoch.set(v);
  }

  get serverUrl() {
    return prefs.serverUrl || (typeof location === 'undefined' ? '' : location.origin);
  }
  /**
   * Which server the data comes from, as state kept across pages names it (a draft, a session, a history entry):
   * `mock` or the server's address. Known from the start, while connecting too; a reload keeps it.
   */
  get server(): string {
    return forcedMock || this.mode === 'mock' ? 'mock' : this.serverUrl;
  }

  #attempt = 0;
  #ctrl: AbortController | null = null;

  /**
   * (Re)connects. A newer call supersedes an older one still waiting on its health check. Resolves once the user can
   * be told something: connected, or offline (no answer within SLOW_CONNECT, or a failed check). Offline, it keeps
   * asking (PROBE_DELAYS between checks) and goes live with the first answer: a server that is slow or not there
   * never turns into demo data.
   */
  async connect(): Promise<void> {
    const attempt = ++this.#attempt;
    this.#ctrl?.abort();
    const ctrl = new AbortController();
    this.#ctrl = ctrl;
    this.mode = 'connecting';
    this.#unreachable.set(false);
    this.#refused.set(null);
    this.#reach.reset(null);
    if (forcedMock) {
      await this.#useMock('forced', attempt);
      return;
    }
    const http = new HttpApi(prefs.serverUrl, this.#reach);
    await new Promise<void>((settled) => void this.#reachServer(http, attempt, ctrl.signal, settled));
  }

  /**
   * Asks the server for its health until it answers with FOLO's (or a newer connect takes over), then goes live. An
   * answer that is not (another program, a 404 page, a sign-in wall, a server that wants a token) is said (`refused`)
   * and asked again.
   */
  async #reachServer(http: HttpApi, attempt: number, abort: AbortSignal, settled: () => void): Promise<void> {
    const unreachable = (refused?: string | null) => {
      if (attempt !== this.#attempt) return;
      if (refused !== undefined) this.#refused.set(refused);
      this.#unreachable.set(true);
      settled();
    };
    try {
      for (let i = 0; ; i++) {
        const slow = setTimeout(() => unreachable(), SLOW_CONNECT);
        try {
          const check = AbortSignal.any([abort, AbortSignal.timeout(CONNECT_TIMEOUT)]);
          const h: unknown = await http.health(check);
          if (attempt !== this.#attempt) return;
          if (!isHealth(h)) throw new ApiError(200, 'not_folo', NOT_FOLO);
          // A FOLO server that answers its health but nothing else (`--token`) is not one Studio can use.
          if (await tokenWall(http.base, check)) throw new ApiError(401, 'token_wall', TOKEN_WALL);
          if (attempt !== this.#attempt) return;
          this.api = http;
          // Any HTTP answer, whatever its status or body, says the server is there: only a request that reached
          // nothing fails the probe.
          this.#reach.reset((s) =>
            http.health(s).catch((e: unknown) => {
              if (!(e instanceof ApiError) || e.status === 0) throw e;
            }),
          );
          this.version = h.version;
          this.reason = null;
          this.#unreachable.set(false);
          this.#refused.set(null);
          this.mode = 'live';
          this.epoch++;
          return;
        } catch (e) {
          // Nothing answered (the dev server's proxy answers 500 when nothing is behind it), or something that is not
          // a FOLO server's health: ask again shortly.
          unreachable(refusal(e));
        } finally {
          clearTimeout(slow);
        }
        if (attempt !== this.#attempt) return;
        await new Promise((next) => setTimeout(next, PROBE_DELAYS[Math.min(i, PROBE_DELAYS.length - 1)]));
        if (attempt !== this.#attempt) return;
      }
    } finally {
      settled();
    }
  }

  /** Switches to the mock: at once when its module is loaded (as the Svelte hub does), else once it is. */
  async #useMock(reason: 'forced', attempt: number) {
    const m = mockModule ?? (await loadMock());
    if (attempt !== this.#attempt) return;
    if (!(unshare(this.#api.peek()) instanceof m.MockApi)) this.api = new m.MockApi(scenario);
    this.version = '0.1.0';
    this.reason = reason;
    this.mode = 'mock';
    this.epoch++;
  }

      /**
       * Something that talks to the server outside the API failed: an <img> the browser loads itself (`imageUrl` only
       * builds its URL). Checked as a failed request is (./reach: one health probe, none while one is out or while offline),
       * so an outage only images saw still ends in `back` and their tiles load again (stress s1-datasets-8).
       */
      suspect(): void {
        this.#reach.failed();
      }
}

export const hub = new Hub();
export { ApiError } from './client';
export type { Api } from './client';
