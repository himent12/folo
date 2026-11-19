// @vitest-environment happy-dom
// The offline state (shell F1, F6): a failed request or a dropped stream is a suspicion one health probe confirms;
// while offline the probe repeats, and the first answer brings Studio back, with what failed meanwhile refetched.
import { createRoot, flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ConnectionStatus from '$lib/shell/ConnectionStatus';
import Topbar from '$lib/shell/Topbar';
import StatusBadge from '$lib/ui/StatusBadge';
import { render } from '../../../framework/testing';
import { TOKEN_WALL } from './handshake';
import { CONNECT_TIMEOUT, hub, SLOW_CONNECT } from './index';
import { PROBE_DELAYS, Reach } from './reach';
import { Resource } from './resource';
import type { Dataset } from './types';

describe('Reach', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('goes offline only when the probe fails too, and back when one answers', async () => {
    let up = true;
    let probes = 0;
    const reach = new Reach();
    reach.reset(async () => {
      probes++;
      if (!up) throw new Error('down');
    });
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    // A blip: the probe answered, nothing to say.
    expect([reach.offline, probes, reach.back]).toEqual([false, 1, 0]);

    up = false;
    reach.failed();
    reach.failed(); // a second failure while the probe is out adds nothing
    await vi.advanceTimersByTimeAsync(0);
    expect([reach.offline, probes]).toEqual([true, 2]);
    // Offline: further failures wait for the probes, which repeat on their own.
    reach.failed();
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect(probes).toBe(3);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    expect(probes).toBe(4);

    up = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[2]!);
    expect([reach.offline, probes, reach.back]).toEqual([false, 5, 1]);
    // Answering, it does not poll.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(probes).toBe(5);
  });

  it('comes back at once when any request answers, and stops probing', async () => {
    let probes = 0;
    const reach = new Reach();
    reach.reset(async () => {
      probes++;
      throw new Error('down');
    });
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(true);
    reach.ok();
    expect([reach.offline, reach.back]).toEqual([false, 1]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(probes).toBe(1);
  });

  it('drops a suspicion when anything answers while its probe is out (review R1)', async () => {
    let reject!: (e: Error) => void;
    const reach = new Reach();
    reach.reset(() => new Promise((_, no) => (reject = no)));
    reach.failed();
    // A newer request (or the reopened stream) answered: the probe that ends in a timeout afterwards proves nothing.
    reach.ok();
    reject(new Error('timed out'));
    await vi.advanceTimersByTimeAsync(0);
    expect([reach.offline, reach.back]).toEqual([false, 0]);
    // The next failure is a suspicion of its own, with a probe of its own.
    reach.failed();
    reject(new Error('down'));
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(true);
  });

  it('has nothing to probe without a server (the mock), and a new server forgets the outage', async () => {
    const reach = new Reach();
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(false);
    reach.reset(() => Promise.reject(new Error('down')));
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect(reach.offline).toBe(true);
    reach.reset(null);
    expect([reach.offline, reach.back]).toEqual([false, 0]);
  });

  it('says what a suspicion found: up when the probe or any request answers, down when nothing does (stress s1-datasets-8)', async () => {
    let up = true;
    let probes = 0;
    const reach = new Reach();
    expect(await reach.suspect()).toBe(false); // nothing to check (the mock)
    reach.reset(async () => {
      probes++;
      if (!up) throw new Error('down');
    });
    // Two suspicions at once share one probe, and both hear it answered.
    const both = Promise.all([reach.suspect(), reach.suspect()]);
    await vi.advanceTimersByTimeAsync(0);
    expect([await both, probes, reach.back]).toEqual([[true, true], 1, 0]);

    up = false;
    const down = reach.suspect();
    await vi.advanceTimersByTimeAsync(0);
    expect([await down, reach.offline]).toEqual([false, true]);
    // Offline already: no probe of its own, its return ends in `back`.
    expect(await reach.suspect()).toBe(false);
    expect(probes).toBe(2);

    // A probe that hangs: any request that answers meanwhile settles the suspicion as up.
    reach.reset(() => new Promise(() => {}));
    const hung = reach.suspect();
    reach.ok();
    expect(await hung).toBe(true);
  });

  it('says a server that answers but refuses Studio, and an answer older than the probe does not undo it (#323)', async () => {
    let refuses: string | null = 'needs a token';
    let probes = 0;
    const reach = new Reach();
    reach.reset(async () => {
      probes++;
      return refuses ?? undefined;
    });
    // A request that went out before the token stopped coming; its 200 lands after the first 401's probe went out.
    const before = performance.now();
    await vi.advanceTimersByTimeAsync(1);
    reach.failed(); // the 401
    reach.ok(before);
    await vi.advanceTimersByTimeAsync(0);
    expect([reach.offline, reach.refused, probes]).toEqual([true, 'needs a token', 1]);
    // Refused: more 401s and older answers add nothing, and no request is retried here; the probe repeats.
    reach.failed();
    reach.ok(before);
    expect([reach.offline, reach.refused, probes]).toEqual([true, 'needs a token', 1]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([reach.offline, probes]).toEqual([true, 2]);
    // The proxy's session is renewed: the next probe lets Studio in, and what failed meanwhile refetches.
    refuses = null;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    expect([reach.offline, reach.refused, reach.back, probes]).toEqual([false, null, 1, 3]);

    // Refused again, then a request sent after the probe answers: that is news, Studio is let in at once.
    refuses = 'needs a token';
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    expect([reach.offline, reach.refused]).toEqual([true, 'needs a token']);
    reach.ok(performance.now());
    expect([reach.offline, reach.refused, reach.back]).toEqual([false, null, 2]);
    // A new server forgets it.
    refuses = 'needs a token';
    reach.failed();
    await vi.advanceTimersByTimeAsync(0);
    reach.reset(null);
    expect([reach.offline, reach.refused]).toEqual([false, null]);
  });
});

describe('hub.offline with the server', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('stays online when the health probe gets an HTTP error: the server answered (review R2)', async () => {
    let health = 0;
    vi.stubGlobal('fetch', async (url: string) => {
      if (!url.endsWith('/api/health')) throw new TypeError('Failed to fetch');
      // The connect's check answers; the probe's gets a 500, then a 200 that is not JSON.
      if (health++ === 0) return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      return health === 2 ? Response.json({ error: { code: 'internal', message: 'boom' } }, { status: 500 }) : new Response('<html>');
    });
    await hub.connect();
    await expect(hub.api.datasets()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect([health, hub.offline]).toEqual([2, false]);
    await expect(hub.api.datasets()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect([health, hub.offline]).toEqual([3, false]);
  });

  it('turns on when requests stop reaching it, the shell says so, and a failed list refetches by itself when it is back', async () => {
    let up = true;
    let lists = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (!up) throw new TypeError('Failed to fetch');
      if (url.endsWith('/api/health')) return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      // The connect's token check (handshake.tokenWall): no token asked for.
      if (init?.method === 'HEAD') return new Response(null, { status: 200 });
      lists++;
      return Response.json({ datasets: [] });
    });
    await hub.connect();
    expect([hub.mode, hub.offline]).toEqual(['live', false]);
    const shell = document.createElement('div');
    const view = render(
      () => (
        <>
          <ConnectionStatus collapsed={false} />
          <Topbar onsearch={() => {}} />
          <StatusBadge run={{ status: 'running' }} plain />
          <StatusBadge run={{ status: 'completed' }} plain />
        </>
      ),
      shell,
    );
    // The word is in the top bar, whose height is fixed: it never covers the page (v-shell N5). Phones say "Offline".
    const said = () => [shell.querySelector('.conn .label')!.textContent!.trim(), shell.querySelector('.topbar [role=status] .long')?.textContent?.trim() ?? ''];
    const held = () => [...shell.querySelectorAll('.badge')].map((b) => b.classList.contains('held'));
    expect(said()).toEqual(['Connected', '']);
    expect(held()).toEqual([false, false]);

    up = false;
    let r!: Resource<Dataset[]>;
    const stop = createRoot((dispose) => {
      r = new Resource((s) => hub.api.datasets(s));
      return dispose;
    });
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(r.error).toBeDefined();
    expect(hub.offline).toBe(true);
    flush();
    expect(said()).toEqual(['Server offline', "Can't reach folo serve. Retrying…"]);
    // A run's live status is only last known now: neutral and still; a finished one stays as it is.
    expect(held()).toEqual([true, false]);

    up = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect(hub.offline).toBe(false);
    expect(r.error).toBeUndefined();
    expect(r.data).toEqual([]);
    expect(lists).toBe(1);
    expect(said()).toEqual(['Connected', '']);
    expect(held()).toEqual([false, false]);
    stop();
    view.unmount();
  });

  it('says so when the server stops letting Studio in mid-session, in the words of a fresh visit, and comes back by itself (#323)', async () => {
    // `folo serve --token` behind a proxy that sends the header, until it stops: the health stays open, the rest is 401.
    let token = true;
    let health = 0;
    let lists = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) {
        health++;
        return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      }
      if (!token) return Response.json({ error: { code: 'unauthorized', message: 'this server needs a token', hint: null } }, { status: 401 });
      if (init?.method === 'HEAD') return new Response(null, { status: 200 });
      lists++;
      return Response.json({ datasets: [] });
    });
    await hub.connect();
    const shell = document.createElement('div');
    const view = render(
      () => (
        <>
          <ConnectionStatus collapsed={false} />
          <Topbar onsearch={() => {}} />
        </>
      ),
      shell,
    );
    const said = () => [shell.querySelector('.conn .label')!.textContent!.trim(), shell.querySelector('.topbar [role=status] .long')?.textContent?.trim() ?? ''];
    expect(said()).toEqual(['Connected', '']);
    const checks = health;

    token = false;
    let r!: Resource<Dataset[]>;
    const stop = createRoot((dispose) => {
      r = new Resource((s) => hub.api.datasets(s));
      return dispose;
    });
    flush();
    // Several screens' requests refused at once: one probe, one change of state.
    await Promise.allSettled([hub.api.datasets(), hub.api.datasets()]);
    await vi.advanceTimersByTimeAsync(0);
    flush();
    expect(r.error).toBeDefined();
    expect([hub.offline, hub.refused, health - checks]).toEqual([true, TOKEN_WALL, 1]);
    expect(said()).toEqual(['Not connected', `${TOKEN_WALL} Retrying…`]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.refused, health - checks]).toEqual([TOKEN_WALL, 2]);

    // The proxy's session is renewed: the next probe lets Studio in, and the refused list refetches.
    token = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    flush();
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.refused, r.error, r.data, lists]).toEqual([false, null, undefined, [], 1]);
    expect(said()).toEqual(['Connected', '']);
    stop();
    view.unmount();
  });

  it('stays connected when one request is refused but the server still lets Studio in (#323)', async () => {
    let health = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) {
        health++;
        return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
      }
      if (init?.method === 'HEAD') return new Response(null, { status: 200 });
      // A refusal of that request alone (docs/API.md: `403 forbidden_origin` on a state-changing request).
      return Response.json({ error: { code: 'forbidden_origin', message: 'not from this origin', hint: null } }, { status: 403 });
    });
    await hub.connect();
    const checks = health;
    await expect(hub.api.datasets()).rejects.toThrow('not from this origin');
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.refused, health - checks]).toEqual([false, null, 1]);
  });
});

describe('the first connect (v-shell N1, v-models-label N1)', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });
  const healthy = () => Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });
  /** Answers after `ms`, or rejects as fetch does when the request's signal aborts first. */
  const after = (ms: number, init: RequestInit | undefined, answer: () => Response) =>
    new Promise<Response>((ok, no) => {
      const t = setTimeout(() => ok(answer()), ms);
      init?.signal?.addEventListener('abort', () => {
        clearTimeout(t);
        no(new DOMException('aborted', 'AbortError'));
      });
    });

  it('waits out a slow first answer as offline, never as demo data, and goes live when it comes', async () => {
    let health = 0;
    vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
      if (!url.endsWith('/api/health')) return Promise.reject(new TypeError('Failed to fetch'));
      health++;
      return after(SLOW_CONNECT + 4000, init, healthy);
    });
    const shell = document.createElement('div');
    const view = render(() => <ConnectionStatus collapsed={false} />, shell);
    const label = () => shell.querySelector('.conn .label')!.textContent!.trim();
    const connected = hub.connect();
    flush();
    expect([hub.mode, hub.offline, label()]).toEqual(['connecting', false, 'Connecting…']);
    await vi.advanceTimersByTimeAsync(SLOW_CONNECT);
    await connected;
    flush();
    // Slow: offline, still waiting on that same answer (no second request), and nothing of the mock.
    expect([hub.mode, hub.offline, label(), health]).toEqual(['connecting', true, 'Server offline', 1]);
    await vi.advanceTimersByTimeAsync(4000);
    flush();
    expect([hub.mode, hub.offline, label(), health, hub.api.mock]).toEqual(['live', false, 'Connected', 1, false]);
    view.unmount();
  });

  it('asks again while nothing answers (or a dev proxy answers 500), and goes live with the first answer', async () => {
    let health = 0;
    vi.stubGlobal('fetch', async (url: string) => {
      if (!url.endsWith('/api/health')) throw new TypeError('Failed to fetch');
      health++;
      if (health === 1) throw new TypeError('Failed to fetch');
      // Vite's proxy with nothing behind it answers 500 with an empty body: not a FOLO server.
      if (health === 2) return new Response('', { status: 500 });
      return healthy();
    });
    await hub.connect();
    expect([hub.mode, hub.offline, health]).toEqual(['connecting', true, 1]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.mode, hub.offline, health]).toEqual(['connecting', true, 2]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    expect([hub.mode, hub.offline, health]).toEqual(['live', false, 3]);
    // Live, it does not poll.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(health).toBe(3);
  });

  it('gives a check that hangs CONNECT_TIMEOUT, then asks again; a reconnect supersedes the waiting one', async () => {
    let health = 0;
    vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
      if (init?.method === 'HEAD') return Promise.resolve(new Response(null, { status: 200 }));
      health++;
      return health === 1 ? after(10 * CONNECT_TIMEOUT, init, healthy) : Promise.resolve(healthy());
    });
    // AbortSignal.timeout runs on real timers: let it be one the fake clock drives.
    vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms: number) => {
      const c = new AbortController();
      setTimeout(() => c.abort(new DOMException('timed out', 'TimeoutError')), ms);
      return c.signal;
    });
    const connected = hub.connect();
    await vi.advanceTimersByTimeAsync(SLOW_CONNECT);
    await connected;
    expect([hub.offline, health]).toEqual([true, 1]);
    await vi.advanceTimersByTimeAsync(CONNECT_TIMEOUT - SLOW_CONNECT);
    expect([hub.mode, health]).toEqual(['connecting', 1]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.mode, hub.offline, health]).toEqual(['live', false, 2]);
    vi.restoreAllMocks();

    // Down again from the start: Reconnect starts over, and the older connect's next check never goes out.
    health = 0;
    let heads = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (init?.method === 'HEAD') heads++;
      else if (url.endsWith('/api/health')) health++;
      throw new TypeError('Failed to fetch');
    });
    await hub.connect();
    await hub.connect();
    expect([hub.mode, hub.offline, health]).toEqual(['connecting', true, 2]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect(health).toBe(3);
    // The token check goes out with each health check, not after it.
    expect(heads).toBe(3);
  });
});
