// The hub (the API in use, live or the mock). Two things about how it loads, neither visible:
// - The in-browser mock is loaded only when it is used (`?mock=1`, or no server answered): it is a chunk of its own,
//   never fetched or parsed when a server answers. With `?mock=1` its load starts with this module. Until the first
//   connect settles, `api` is the server's (a placeholder that, like the Svelte hub's initial mock, says `mock`).
// - `api` is the Api as ./share wraps it: reads a Resource issues together with another one are one request. Outside a
//   Resource's fetch it is the Api itself, call for call. Assigning an Api (tests) stores that Api.
//
// Picks the API implementation: the real server, or the in-browser mock with `?mock=1` or when the server is
// unreachable. Screens read `hub.api` and never care which one it is.

import { computed, signal } from 'fw';
import { prefs } from '$lib/prefs';
import { HttpApi, type Api } from './client';
import { Reach } from './reach';
import { shareApi, unshare } from './share';
import type { Scenario } from './mock/scenario';

export type Mode = 'connecting' | 'live' | 'mock';

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
  #reason = signal<'forced' | 'unreachable' | null>(null);
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
  /** Why we are on the mock: 'forced' (?mock=1) or 'unreachable'. */
  get reason(): 'forced' | 'unreachable' | null {
    return this.#reason.get();
  }
  set reason(v: 'forced' | 'unreachable' | null) {
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
   * which repeats until it answers again), or the demo's `?scenario=error`. Live data on screen is then a last known
   * state, not the current one.
   */
  get offline(): boolean {
    return scenario === 'error' || this.#reach.offline;
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

  #attempt = 0;
  #ctrl: AbortController | null = null;

  /** (Re)connects. A newer call supersedes an older one still waiting on its health check. */
  async connect(): Promise<void> {
    const attempt = ++this.#attempt;
    this.#ctrl?.abort();
    const ctrl = new AbortController();
    this.#ctrl = ctrl;
    this.mode = 'connecting';
    this.#reach.reset(null);
    if (forcedMock) {
      await this.#useMock('forced', attempt);
      return;
    }
    const http = new HttpApi(prefs.serverUrl, this.#reach);
    try {
      const h = await http.health(AbortSignal.any([ctrl.signal, AbortSignal.timeout(2500)]));
      if (attempt !== this.#attempt) return;
      this.api = http;
      this.#reach.reset((s) => http.health(s));
      this.version = h.version;
      this.reason = null;
      this.mode = 'live';
      this.epoch++;
    } catch {
      if (attempt !== this.#attempt) return;
      await this.#useMock('unreachable', attempt);
    }
  }

  /** Switches to the mock: at once when its module is loaded (as the Svelte hub does), else once it is. */
  async #useMock(reason: 'forced' | 'unreachable', attempt: number) {
    const m = mockModule ?? (await loadMock());
    if (attempt !== this.#attempt) return;
    if (!(unshare(this.#api.peek()) instanceof m.MockApi)) this.api = new m.MockApi(scenario);
    this.version = '0.1.0';
    this.reason = reason;
    this.mode = 'mock';
    this.epoch++;
  }
}

export const hub = new Hub();
export { ApiError } from './client';
export type { Api } from './client';
