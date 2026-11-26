// @vitest-environment happy-dom
// The first handshake (review studio-2 R1, R4): Studio goes live only on FOLO's health payload. An address that
// answers with something else is not "no answer": the footer, the top bar's notice and Settings › Connection say what
// came back, and Studio keeps asking, never on demo data.
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ConnectionStatus from '$lib/shell/ConnectionStatus';
import OfflineNotice from '$lib/shell/OfflineNotice';
import ConnectionPanel from '../../routes/settings/ConnectionPanel';
import { render } from '../../../framework/testing';
import { ApiError } from './client';
import { isHealth, NOT_FOLO, refusal, REFUSES_ACCESS, TOKEN_WALL } from './handshake';
import { CONNECT_TIMEOUT, hub, TOKEN_GRACE } from './index';
import { PROBE_DELAYS } from './reach';

const healthy = () => Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false });

describe('isHealth and refusal', () => {
  it("takes FOLO's health payload only", () => {
    expect(isHealth({ status: 'ok', version: '0.1.0', api: 'v1', mock: false })).toBe(true);
    // An older server without `api`.
    expect(isHealth({ status: 'ok', version: '0.1.0', mock: true })).toBe(true);
    for (const h of [{}, { service: 'other' }, { status: 'ok' }, { status: 'ok', version: 1, mock: false }, null, 'ok', []]) {
      expect(isHealth(h)).toBe(false);
    }
  });

  it('says what an answer that is not usable was, and nothing for no answer', () => {
    expect(refusal(new TypeError('Failed to fetch'))).toBeNull();
    expect(refusal(new ApiError(0, 'network', 'x'))).toBeNull();
    expect(refusal(new ApiError(500, 'http_500', 'x'))).toBeNull();
    expect(refusal(new ApiError(502, 'http_502', 'x'))).toBeNull();
    expect(refusal(new ApiError(401, 'http_401', 'x'))).toBe('This server asks for sign-in (401).');
    expect(refusal(new ApiError(404, 'bad_response', 'x'))).toBe(NOT_FOLO);
    expect(refusal(new ApiError(500, 'bad_response', 'x'))).toBe(NOT_FOLO);
  });
});

describe('the first handshake with an address that answers', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  type Answer = (init?: RequestInit) => Response | Promise<Response>;

  /**
   * Starts a connect with `answers` for the health checks in turn (the last one repeats), and `runs` for the token check
   * (`HEAD /api/runs`, in turn; nothing answers it by default); what the shell says, and the requests made: token checks
   * cancelled while still waiting for their answer, and requests (either kind) whose signal aborted after they had
   * answered (`abortedAfter`: the browser shows those as cancelled too).
   */
  function start(answers: Answer[], runs: Answer[] = []) {
    let health = 0;
    let wall = 0;
    let pending = 0;
    let cancelled = 0;
    let abortedAfter = 0;
    const afterAnswer = (signal: AbortSignal | null | undefined) =>
      signal?.addEventListener('abort', () => abortedAfter++, { once: true });
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/runs') && init?.method === 'HEAD') {
        wall++;
        if (!runs.length) throw new TypeError('Failed to fetch');
        const answer = runs[Math.min(wall - 1, runs.length - 1)]!;
        pending++;
        try {
          // As a real request: a cancelled one fails at once, whatever was coming.
          return await new Promise<Response>((ok, no) => {
            const signal = init?.signal;
            if (signal?.aborted) return no(signal.reason as Error);
            let out = true;
            const abort = () => {
              cancelled++;
              no(signal!.reason as Error);
            };
            signal?.addEventListener('abort', abort, { once: true });
            const settle = () => {
              out = false;
              signal?.removeEventListener('abort', abort);
              afterAnswer(signal);
            };
            Promise.resolve()
              .then(() => answer(init))
              .then(
                (r) => {
                  if (out) settle();
                  ok(r);
                },
                (e: unknown) => {
                  if (out) settle();
                  no(e);
                },
              );
          });
        } finally {
          pending--;
        }
      }
      if (!url.endsWith('/api/health')) throw new TypeError('Failed to fetch');
      const r = await answers[Math.min(health++, answers.length - 1)]!(init);
      afterAnswer(init?.signal);
      return r;
    });
    const shell = document.createElement('div');
    const view = render(
      () => (
        <>
          <ConnectionStatus collapsed={false} />
          <OfflineNotice />
          <ConnectionPanel />
        </>
      ),
      shell,
    );
    const done = hub.connect();
    const said = () => {
      flush();
      return {
        mode: hub.mode,
        label: shell.querySelector('.conn .label')!.textContent!.trim(),
        footer: shell.querySelector('.conn .sr-only')!.textContent!.trim(),
        notice: shell.querySelector('.notice .long')?.textContent?.trim() ?? '',
        short: shell.querySelector('.notice .short')?.textContent?.trim() ?? '',
        settings: `${shell.querySelector('.s-title')!.textContent!.trim()} | ${shell.querySelector('.s-desc')!.textContent!.trim()}`,
      };
    };
    return {
      said,
      done,
      checks: () => health,
      heads: () => wall,
      pending: () => pending,
      cancelled: () => cancelled,
      abortedAfter: () => abortedAfter,
      unmount: () => view.unmount(),
    };
  }

  /** `start`, once the connect has settled. */
  async function connectTo(answers: Answer[], runs: Answer[] = []) {
    const c = start(answers, runs);
    await c.done;
    flush();
    return c;
  }

  for (const [what, answer] of [
    ['an empty object', () => Response.json({})],
    ['another service', () => Response.json({ service: 'other' })],
    ['a 404 page', () => new Response('<!doctype html><title>404 Not Found</title>', { status: 404, headers: { 'content-type': 'text/html' } })],
  ] as const) {
    it(`does not go live on ${what}, says so, and keeps asking`, async () => {
      const c = await connectTo([answer]);
      const url = hub.serverUrl;
      expect(c.said()).toEqual({
        mode: 'connecting',
        label: 'Not connected',
        footer: `${url}: ${NOT_FOLO} Retrying…`,
        notice: `${NOT_FOLO} Retrying…`,
        short: 'Not connected',
        settings: `Not connected | ${url}: ${NOT_FOLO} Retrying…`,
      });
      expect([hub.offline, hub.reason, c.checks()]).toEqual([true, null, 1]);
      await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
      expect([hub.mode, hub.refused, c.checks()]).toEqual(['connecting', NOT_FOLO, 2]);
      c.unmount();
    });
  }

  it('says a sign-in wall in front of the server is one', async () => {
    const c = await connectTo([() => new Response('', { status: 401 })]);
    const said = c.said();
    expect([said.mode, said.label, said.notice, said.short]).toEqual([
      'connecting',
      'Not connected',
      'This server asks for sign-in (401). Retrying…',
      'Not connected',
    ]);
    expect(said.settings).toContain('This server asks for sign-in (401).');
    c.unmount();
  });

  it('does not go live on a server that answers its health but wants a token, and goes live once it does not', async () => {
    // `folo serve --token`: health is open, everything else is 401 (shell F1: "Connected", then 401 on every screen).
    const wall = () => Response.json({ error: { code: 'unauthorized', message: 'this server needs a token', hint: null } }, { status: 401 });
    const c = await connectTo([healthy], [wall, () => new Response(null, { status: 200 })]);
    const url = hub.serverUrl;
    expect(c.said()).toEqual({
      mode: 'connecting',
      label: 'Not connected',
      footer: `${url}: ${TOKEN_WALL} Retrying…`,
      notice: `${TOKEN_WALL} Retrying…`,
      short: 'Not connected',
      settings: `Not connected | ${url}: ${TOKEN_WALL} Retrying…`,
    });
    expect(hub.offline).toBe(true);
    // Restarted without --token: the next check goes live.
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    const said = c.said();
    expect([said.mode, said.label, hub.refused, hub.offline]).toEqual(['live', 'Connected', null, false]);
    c.unmount();
  });

  it('does not go live when post-health HEAD /api/runs answers 403, says access is refused, and goes live once it answers 200', async () => {
    const forbidden = () => new Response(null, { status: 403 });
    // Early HEAD returns 403 (ignored -> falls back to post-health tokenWall), post-health HEAD returns 403 (refused),
    // then on the next probe cycle HEAD returns 200 and Studio goes live.
    const c = await connectTo([healthy], [forbidden, forbidden, () => new Response(null, { status: 200 })]);
    const said = c.said();
    expect([said.mode, said.label, said.notice, hub.refused, hub.offline]).toEqual([
      'connecting',
      'Not connected',
      `${REFUSES_ACCESS} Retrying…`,
      REFUSES_ACCESS,
      true,
    ]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    const next = c.said();
    expect([next.mode, next.label, hub.refused, hub.offline]).toEqual(['live', 'Connected', null, false]);
    c.unmount();
  });

  it('cancels the retry backoff timer when a newer connect supersedes an unreachable connect', async () => {
    const c = start([() => Promise.reject(new TypeError('Failed to fetch'))]);
    await c.done;
    expect(vi.getTimerCount()).toBe(1);
    const second = hub.connect();
    await second;
    // Only the new connect's retry timer remains scheduled; the superseded one was aborted immediately.
    expect(vi.getTimerCount()).toBe(1);
    c.unmount();
  });

  it('goes live with a real payload after two failures, and forgets what it said', async () => {
    const c = await connectTo([() => new Response('', { status: 401 }), () => Response.json({ service: 'other' }), healthy]);
    expect(hub.refused).toBe('This server asks for sign-in (401).');
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.mode, hub.refused]).toEqual(['connecting', NOT_FOLO]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
    const said = c.said();
    expect([said.mode, said.label, said.notice, hub.refused, hub.offline, hub.version]).toEqual(['live', 'Connected', '', null, false, '0.1.0']);
    c.unmount();
  });

  // #324: what the server can do comes from a health answer of this connection: the handshake's, and after an outage the
  // one the hub asks once the probe finds it back (a server restarted with other features), and no more.
  it('keeps the features its health lists, forgets them in an outage, and asks them again when it is back', async () => {
    const listing = (features: string[]) => () => Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false, features });
    const down = () => Promise.reject(new TypeError('Failed to fetch'));
    const c = await connectTo([listing(['queue_control', 'agent_labeling']), down, listing(['queue_control'])]);
    expect([hub.mode, hub.features, hub.lacks('agent_labeling')]).toEqual(['live', ['queue_control', 'agent_labeling'], false]);
    // A request fails, the probe finds nothing: offline, and what it knew is unknown at once (review of #341, round 2).
    expect(await hub.suspect()).toBe(false);
    expect([hub.offline, hub.features, hub.lacks('agent_labeling')]).toEqual([true, null, false]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.offline, hub.features, hub.lacks('agent_labeling'), c.checks()]).toEqual([false, ['queue_control'], true, 4]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(c.checks()).toBe(4);
    c.unmount();
  });

  it('knows no features from a server whose health lists none, and so lacks none', async () => {
    const c = await connectTo([healthy]);
    expect([hub.mode, hub.features, hub.lacks('agent_labeling')]).toEqual(['live', null, false]);
    c.unmount();
  });

  it('says nothing came back when nothing answers, as before', async () => {
    const c = await connectTo([
      () => {
        throw new TypeError('Failed to fetch');
      },
    ]);
    const said = c.said();
    expect([said.label, said.notice, said.short, hub.refused]).toEqual(['Server offline', "Can't reach folo serve. Retrying…", 'Offline', null]);
    c.unmount();
  });

  // The token check goes out together with the health check (index.ts), so its answer can come before or after the
  // health's, or not at all: what Studio says must be what it said when the check went out after the health.
  const after =
    (ms: number, r: () => Response): Answer =>
    () =>
      new Promise((ok) => setTimeout(() => ok(r()), ms));
  const resetAfter =
    (ms: number): Answer =>
    () =>
      new Promise((_, no) => setTimeout(() => no(new TypeError('Failed to fetch')), ms));
  /** A request that never answers until it is cancelled. */
  const hang: Answer = (init) =>
    new Promise((_, no) => init?.signal?.addEventListener('abort', () => no(init.signal!.reason as Error)));
  const walls = () => Response.json({ error: { code: 'unauthorized', message: 'this server needs a token', hint: null } }, { status: 401 });
  const opens = () => new Response(null, { status: 200 });
  const notFolo = () => Response.json({ service: 'other' });

  // The token check goes out with the health check too (index.ts, #reachServer's table): whatever that early check gets,
  // and whenever, Studio must end where the check after the health alone took it (`main`), and leave nothing in flight.
  // Health answers at HEALTH ms; the early check before it (BEFORE) or after it within the grace (AFTER); the check after
  // the health, when it is asked, answers SEQ ms after it goes out with what the server says (`truth`).
  const HEALTH = 800;
  const BEFORE = 100;
  const AFTER = HEALTH + TOKEN_GRACE / 2;
  const SEQ = 100;
  const status = (code: number, body: string | null = null) => () => new Response(body, { status: code });
  /** What the early check gets, and whether its answer is used (what a FOLO server answers: 401 or 200). */
  const outcomes: [string, (at: number) => Answer, 'wall' | 'open' | null][] = [
    ['401', (at) => after(at, walls), 'wall'],
    ['200', (at) => after(at, opens), 'open'],
    ['403', (at) => after(at, status(403)), null],
    ['404', (at) => after(at, status(404, 'not found')), null],
    ['an empty 500 (the dev proxy)', (at) => after(at, status(500)), null],
    ['a 500 with a body', (at) => after(at, () => Response.json({ error: { code: 'internal', message: 'x', hint: null } }, { status: 500 })), null],
    ['502', (at) => after(at, status(502)), null],
    ['503', (at) => after(at, status(503)), null],
    ['504', (at) => after(at, status(504)), null],
    ['a reset connection', (at) => resetAfter(at), null],
  ];
  const ends = {
    wall: { mode: 'connecting', label: 'Not connected', notice: `${TOKEN_WALL} Retrying…`, refused: TOKEN_WALL, offline: true },
    open: { mode: 'live', label: 'Connected', notice: '', refused: null, offline: false },
  } as const;
  const end = (c: ReturnType<typeof start>) => {
    const s = c.said();
    return { mode: s.mode, label: s.label, notice: s.notice, refused: hub.refused, offline: hub.offline };
  };

  for (const [what, early, used] of outcomes) {
    for (const [when, at] of [
      ['before', BEFORE],
      ['after', AFTER],
    ] as const) {
      for (const truth of used ? [used] : (['wall', 'open'] as const)) {
        it(`early token check: ${what} ${when} the health, the server ${truth === 'wall' ? 'walled' : 'open'}: as the check after the health alone`, async () => {
          const c = start([after(HEALTH, healthy)], [early(at), after(SEQ, truth === 'wall' ? walls : opens)]);
          // Never later than the check after the health alone (HEALTH + SEQ) plus the grace.
          await vi.advanceTimersByTimeAsync(HEALTH + TOKEN_GRACE + SEQ);
          await c.done;
          expect(end(c)).toEqual(ends[truth]);
          // Used: that answer alone. Not used: asked again after the health, as `main` asks it. Nothing that has
          // answered is cancelled afterwards.
          expect([c.heads(), c.pending(), c.abortedAfter()]).toEqual([used ? 1 : 2, 0, 0]);
          c.unmount();
        });
      }
    }
  }

  for (const truth of ['wall', 'open'] as const) {
    it(`early token check that hangs, the server ${truth === 'wall' ? 'walled' : 'open'}: cancelled after the grace, asked again`, async () => {
      const c = start([after(HEALTH, healthy)], [hang, after(SEQ, truth === 'wall' ? walls : opens)]);
      await vi.advanceTimersByTimeAsync(HEALTH + TOKEN_GRACE - 1);
      // The health has answered; the grace is not over: still waiting on the early check, nothing said yet.
      expect([hub.mode, hub.offline, c.heads(), c.pending()]).toEqual(['connecting', false, 1, 1]);
      await vi.advanceTimersByTimeAsync(1);
      expect([c.heads(), c.pending()]).toEqual([2, 1]);
      await vi.advanceTimersByTimeAsync(SEQ);
      await c.done;
      expect(end(c)).toEqual(ends[truth]);
      // The early check was cancelled while out; the check whose answer was used was not cancelled after it.
      expect([c.pending(), c.cancelled(), c.abortedAfter()]).toEqual([0, 1, 0]);
      c.unmount();
    });
  }

  for (const truth of ['wall', 'open'] as const) {
    it(`leaves an early token check that answered alone, the server ${truth === 'wall' ? 'walled' : 'open'}`, async () => {
      // Used (before the health), and not used (a 403 within the grace): neither is cancelled once it has answered.
      for (const early of [after(BEFORE, truth === 'wall' ? walls : opens), after(AFTER, status(403))]) {
        const c = start([after(HEALTH, healthy)], [early, after(SEQ, truth === 'wall' ? walls : opens)]);
        await vi.advanceTimersByTimeAsync(HEALTH + TOKEN_GRACE + SEQ);
        await c.done;
        await vi.advanceTimersByTimeAsync(0);
        expect(end(c)).toEqual(ends[truth]);
        expect([c.pending(), c.cancelled(), c.abortedAfter()]).toEqual([0, 0, 0]);
        c.unmount();
      }
    });
  }

  // The connect's timeout: the check after the health must get the time it had in `main` (what was left of
  // CONNECT_TIMEOUT when the health answered), however long the wait for the early check took. AbortSignal.timeout runs
  // on the fake clock here.
  describe('at the connect timeout', () => {
    beforeEach(() => {
      vi.spyOn(AbortSignal, 'timeout').mockImplementation((ms: number) => {
        const c = new AbortController();
        setTimeout(() => c.abort(new DOMException('The operation timed out.', 'TimeoutError')), ms);
        return c.signal;
      });
    });
    const offline = { mode: 'connecting', label: 'Server offline', notice: "Can't reach folo serve. Retrying…", refused: null, offline: true };
    // [the health answers this long before the timeout, the check after the health answers this long after it goes
    // out]: the grace runs past the timeout; the grace ends before it but takes some of the check's time.
    for (const [before, seq] of [
      [100, 40],
      [300, 150],
    ] as const) {
      for (const truth of ['wall', 'open'] as const) {
        it(`health ${before} ms before the timeout, the early token check stalled, the check after the health in ${seq} ms, the server ${truth === 'wall' ? 'walled' : 'open'}: as the check after the health alone`, async () => {
          const c = start([after(CONNECT_TIMEOUT - before, healthy)], [hang, after(seq, truth === 'wall' ? walls : opens)]);
          await vi.advanceTimersByTimeAsync(CONNECT_TIMEOUT - before + TOKEN_GRACE + seq);
          await c.done;
          expect(end(c)).toEqual(ends[truth]);
          expect([c.heads(), c.pending()]).toEqual([2, 0]);
          c.unmount();
        });
      }
    }
    it('a check after the health that takes longer than it had with `main` still times out', async () => {
      // `main`: the check goes out 100 ms before the timeout and has no answer within them; here too.
      const c = start([after(CONNECT_TIMEOUT - 100, healthy)], [hang, after(150, opens)]);
      await vi.advanceTimersByTimeAsync(CONNECT_TIMEOUT + TOKEN_GRACE + 150);
      await c.done;
      expect(end(c)).toEqual(offline);
      expect([c.heads(), c.pending()]).toEqual([2, 0]);
      c.unmount();
    });
  });

  for (const [what, runs] of [
    ['before', [walls]],
    ['after', [after(300, walls)]],
  ] as const) {
    it(`says an address that is not FOLO is not, when its token check answers 401 ${what} the health`, async () => {
      const c = await connectTo([notFolo], [...runs]);
      await vi.advanceTimersByTimeAsync(300);
      expect(c.said()).toMatchObject({ mode: 'connecting', label: 'Not connected', notice: `${NOT_FOLO} Retrying…` });
      expect([hub.refused, c.pending()]).toEqual([NOT_FOLO, 0]);
      c.unmount();
    });
  }

  for (const [what, health] of [
    ['nothing answers the health check', () => Promise.reject(new TypeError('Failed to fetch'))],
    ['the health answer is not FOLO', notFolo],
  ] as const) {
    it(`cancels a token check that hangs when ${what}, before the next check`, async () => {
      const c = start([health], [hang]);
      await c.done;
      await vi.advanceTimersByTimeAsync(0);
      expect([hub.mode, c.heads(), c.pending()]).toEqual(['connecting', 1, 0]);
      // About five seconds of checks: each one's token check is cancelled when it ends, none left holding a connection.
      await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
      expect([c.checks(), c.heads(), c.pending()]).toEqual([2, 2, 0]);
      await vi.advanceTimersByTimeAsync(PROBE_DELAYS[1]!);
      expect([c.checks(), c.heads(), c.pending()]).toEqual([3, 3, 0]);
      c.unmount();
    });
  }
});
