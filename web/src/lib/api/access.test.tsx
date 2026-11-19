// @vitest-environment happy-dom
// Whether the server in use still lets Studio in (#323): only the token check changes it, a request's 401 or 403 only
// asks for one, and what decides whether the server answers at all (./reach) is untouched by it. Each review finding of
// round 1 is its exact order of events here.
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ConnectionStatus from '$lib/shell/ConnectionStatus';
import Topbar from '$lib/shell/Topbar';
import { render } from '../../../framework/testing';
import { Access } from './access';
import { REFUSES_ACCESS, TOKEN_WALL, type AccessAnswer } from './handshake';
import { hub } from './index';
import { PROBE_DELAYS } from './reach';

/** A check whose answers the test gives, one per check started, in order. */
function held() {
  const asked: ((a: AccessAnswer) => void)[] = [];
  const access = new Access();
  access.reset(() => new Promise<AccessAnswer>((answer) => asked.push(answer)));
  return { access, asked };
}

describe('Access', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('asks one check for many refusals; refused only when it says so, checked again until let in', async () => {
    const { access, asked } = held();
    const seen = access.checks;
    for (let i = 0; i < 20; i++) access.refused(seen);
    expect([asked.length, access.refusal]).toEqual([1, null]);
    asked[0]!(401);
    await vi.advanceTimersByTimeAsync(0);
    expect(access.refusal).toBe(401);
    // Refused: more refusals add nothing; the check repeats on the connect's cadence.
    access.refused(access.checks);
    expect(asked.length).toBe(1);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect(asked.length).toBe(2);
    // A check that cannot tell changes nothing, and the next one still comes.
    asked[1]!(null);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    expect([asked.length, access.refusal]).toEqual([3, 401]);
    asked[2]!('in');
    await vi.advanceTimersByTimeAsync(0);
    expect([access.refusal, access.back]).toEqual([null, 1]);
    // Let in: nothing checks.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(asked.length).toBe(3);
  });

  it('round 1 #2: a refusal sent after the check out started gets exactly one more check, which decides', async () => {
    const { access, asked } = held();
    // A route's own 401 starts check 1; its HEAD is held.
    access.refused(access.checks);
    // Authorization goes; a request sent now (after check 1 went out) is refused, twice over.
    const sent = access.checks;
    access.refused(sent);
    access.refused(sent);
    expect(asked.length).toBe(1);
    // Check 1's answer was given while Studio was still let in: it says so, and the one more check follows.
    asked[0]!('in');
    await vi.advanceTimersByTimeAsync(0);
    expect([asked.length, access.refusal]).toEqual([2, null]);
    asked[1]!(401);
    await vi.advanceTimersByTimeAsync(0);
    expect([asked.length, access.refusal]).toEqual([2, 401]);
  });

  it('a refusal of a request sent before the check out started adds nothing: that check answers for it', async () => {
    const { access, asked } = held();
    const before = access.checks;
    access.refused(before);
    access.refused(before);
    asked[0]!('in');
    await vi.advanceTimersByTimeAsync(60_000);
    expect([asked.length, access.refusal]).toEqual([1, null]);
  });

  it('round 1 #3: a 403 is a refusal; a check that cannot tell never is, nor lets Studio in', async () => {
    const { access, asked } = held();
    access.refused(access.checks);
    asked[0]!(null);
    await vi.advanceTimersByTimeAsync(60_000);
    expect([asked.length, access.refusal]).toEqual([1, null]);
    access.refused(access.checks);
    asked[1]!(403);
    await vi.advanceTimersByTimeAsync(0);
    expect(access.refusal).toBe(403);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    asked[2]!(null);
    await vi.advanceTimersByTimeAsync(0);
    expect(access.refusal).toBe(403);
  });

  it('only the latest check started may change the state: a new server drops the check out', async () => {
    const { access, asked } = held();
    access.refused(access.checks);
    access.reset(() => new Promise<AccessAnswer>((answer) => asked.push(answer)));
    access.refused(access.checks);
    expect(asked.length).toBe(2);
    asked[0]!(401);
    await vi.advanceTimersByTimeAsync(0);
    expect(access.refusal).toBeNull();
    asked[1]!(403);
    await vi.advanceTimersByTimeAsync(0);
    expect(access.refusal).toBe(403);
    access.reset(null);
    await vi.advanceTimersByTimeAsync(60_000);
    expect([access.refusal, asked.length]).toEqual([null, 2]);
  });
});

/** Settles when `go` is called: a response held back by the test. */
function gate<T>() {
  let go!: (v: T) => void;
  const wait = new Promise<T>((r) => (go = r));
  return { go, wait };
}

describe('hub.refused with the server', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  const HEALTH = { status: 'ok', version: '0.1.0', api: 'v1', mock: false };
  const UNAUTHORIZED = () => Response.json({ error: { code: 'unauthorized', message: 'this server needs a token', hint: null } }, { status: 401 });

  /** The shell's two words: the sidebar's state and the top bar's notice. */
  function shell() {
    const el = document.createElement('div');
    const view = render(
      () => (
        <>
          <ConnectionStatus collapsed={false} />
          <Topbar onsearch={() => {}} />
        </>
      ),
      el,
    );
    const said = () => {
      flush();
      return [el.querySelector('.conn .label')!.textContent!.trim(), el.querySelector('.topbar [role=status] .long')?.textContent?.trim() ?? ''];
    };
    return { said, unmount: () => view.unmount() };
  }

  it('says so when the server stops letting Studio in mid-session, in the words of a fresh visit, and comes back by itself (#323)', async () => {
    let token = true;
    let heads = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      if (init?.method === 'HEAD') heads++;
      if (!token) return UNAUTHORIZED();
      return init?.method === 'HEAD' ? new Response(null, { status: 200 }) : Response.json({ datasets: [] });
    });
    await hub.connect();
    const { said, unmount } = shell();
    expect(said()).toEqual(['Connected', '']);
    const connected = heads;
    const back = hub.back;

    token = false;
    await Promise.allSettled([hub.api.datasets(), hub.api.datasets(), hub.api.system()]);
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.refused, heads - connected]).toEqual([true, TOKEN_WALL, 1]);
    expect(said()).toEqual(['Not connected', `${TOKEN_WALL} Retrying…`]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.refused, heads - connected]).toEqual([TOKEN_WALL, 2]);

    token = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    expect([hub.offline, hub.refused, hub.back - back]).toEqual([false, null, 1]);
    expect(said()).toEqual(['Connected', '']);
    unmount();
  });

  it('round 1 #1: an event stream that opens late, or a 200 to a request sent earlier, does not undo a refusal', async () => {
    let token = true;
    const stream = gate<void>();
    const list = gate<void>();
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      if (url.endsWith('/events')) {
        await stream.wait;
        return new Response(new ReadableStream({ start: () => {} }), { headers: { 'content-type': 'text/event-stream' } });
      }
      if (url.endsWith('/api/datasets') && token) {
        await list.wait;
        return Response.json({ datasets: [] });
      }
      if (!token) return UNAUTHORIZED();
      return init?.method === 'HEAD' ? new Response(null, { status: 200 }) : Response.json({});
    });
    await hub.connect();
    const close = hub.api.runEvents('r1', {});
    const earlier = hub.api.datasets();
    await vi.advanceTimersByTimeAsync(0);

    token = false;
    await expect(hub.api.system()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.refused]).toEqual([true, TOKEN_WALL]);
    stream.go();
    list.go();
    expect(await earlier).toEqual([]);
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.refused]).toEqual([true, TOKEN_WALL]);
    close();
  });

  it('round 1 #3: a proxy that answers 403 past an open health is said as a refusal, not as Connected', async () => {
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      if (init?.method === 'HEAD' && hub.mode !== 'live') return new Response(null, { status: 200 });
      return Response.json({ error: { code: 'forbidden', message: 'proxy session expired', hint: null } }, { status: 403 });
    });
    await hub.connect();
    const { said, unmount } = shell();
    await expect(hub.api.datasets()).rejects.toThrow('proxy session expired');
    await vi.advanceTimersByTimeAsync(0);
    expect(said()).toEqual(['Not connected', `${REFUSES_ACCESS} Retrying…`]);
    unmount();
  });

  it('round 1 #6: a refused export starts the check too', async () => {
    let token = true;
    let heads = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      if (init?.method === 'HEAD') heads++;
      return token ? new Response(null, { status: 200 }) : UNAUTHORIZED();
    });
    await hub.connect();
    const connected = heads;
    token = false;
    await expect(hub.api.f16File('m1')).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect([heads - connected, hub.refused]).toEqual([1, TOKEN_WALL]);
  });

  it('stays connected when one request is refused but the server still lets Studio in', async () => {
    let heads = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      if (init?.method === 'HEAD') return heads++, new Response(null, { status: 200 });
      // A refusal of that request alone (docs/API.md: `403 forbidden_origin` on a state-changing request).
      return Response.json({ error: { code: 'forbidden_origin', message: 'not from this origin', hint: null } }, { status: 403 });
    });
    await hub.connect();
    const connected = heads;
    await expect(hub.api.datasets()).rejects.toThrow('not from this origin');
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.refused, heads - connected]).toEqual([false, null, 1]);
  });

  it('round 1 #4: a server back after a stop is Connected on its first health, however slow its token check', async () => {
    let up = true;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (!up) throw new TypeError('Failed to fetch');
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      // Once connected, the token check never answers (a proxy that holds it past every budget).
      if (init?.method === 'HEAD' && hub.mode === 'live') return new Promise<Response>(() => {});
      return init?.method === 'HEAD' ? new Response(null, { status: 200 }) : Response.json({ datasets: [] });
    });
    await hub.connect();
    up = false;
    await expect(hub.api.datasets()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect(hub.offline).toBe(true);
    up = true;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.offline, hub.refused]).toEqual([false, null]);
  });

  it('round 1 #5: a request held through an outage that answers from the recovered server brings Studio back at once', async () => {
    let up = true;
    const list = gate<void>();
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/datasets')) {
        await list.wait;
        return Response.json({ datasets: [] });
      }
      if (!up) throw new TypeError('Failed to fetch');
      if (url.endsWith('/api/health')) return Response.json(HEALTH);
      return init?.method === 'HEAD' ? new Response(null, { status: 200 }) : Response.json({});
    });
    await hub.connect();
    const older = hub.api.datasets();
    await vi.advanceTimersByTimeAsync(1);
    up = false;
    await expect(hub.api.system()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect(hub.offline).toBe(true);
    up = true;
    list.go();
    expect(await older).toEqual([]);
    expect(hub.offline).toBe(false);
  });
});
