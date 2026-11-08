// Picks the API implementation: the real server, or the in-browser mock with `?mock=1` or when the server is
// unreachable. Screens read `hub.api` and never care which one it is.

import { prefs } from '$lib/prefs.svelte';
import { HttpApi, type Api } from './client';
import { MockApi, type Scenario } from './mock';

export type Mode = 'connecting' | 'live' | 'mock';

const params = typeof location === 'undefined' ? new URLSearchParams() : new URLSearchParams(location.search);
const forcedMock = params.get('mock') === '1';
const scenario = (params.get('scenario') ?? 'normal') as Scenario;

class Hub {
  mode = $state<Mode>('connecting');
  /** Why we are on the mock: 'forced' (?mock=1) or 'unreachable'. */
  reason = $state<'forced' | 'unreachable' | null>(null);
  version = $state<string | null>(null);
  api = $state.raw<Api>(new MockApi(scenario));
  /** Bumps on every (re)connect so data views refetch. */
  epoch = $state(0);

  get serverUrl() {
    return prefs.serverUrl || (typeof location === 'undefined' ? '' : location.origin);
  }

  async connect(): Promise<void> {
    this.mode = 'connecting';
    if (forcedMock) {
      this.#useMock('forced');
      return;
    }
    const http = new HttpApi(prefs.serverUrl);
    try {
      const h = await http.health(AbortSignal.timeout(2500));
      this.api = http;
      this.version = h.version;
      this.reason = null;
      this.mode = 'live';
      this.epoch++;
    } catch {
      this.#useMock('unreachable');
    }
  }

  #useMock(reason: 'forced' | 'unreachable') {
    if (!(this.api instanceof MockApi)) this.api = new MockApi(scenario);
    this.version = '0.1.0';
    this.reason = reason;
    this.mode = 'mock';
    this.epoch++;
  }
}

export const hub = new Hub();
export { ApiError } from './client';
export type { Api } from './client';
