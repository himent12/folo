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
import { isHealth, NOT_FOLO, refusal, TOKEN_WALL } from './handshake';
import { hub } from './index';
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
    vi.useRealTimers();
  });

  type Answer = (init?: RequestInit) => Response | Promise<Response>;

  /**
   * Starts a connect with `answers` for the health checks in turn (the last one repeats), and `runs` for the token check
   * (`HEAD /api/runs`, in turn; nothing answers it by default); what the shell says, and the requests made.
   */
  function start(answers: Answer[], runs: Answer[] = []) {
    let health = 0;
    let wall = 0;
    let pending = 0;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/runs') && init?.method === 'HEAD') {
        wall++;
        if (!runs.length) throw new TypeError('Failed to fetch');
        pending++;
        try {
          return await runs[Math.min(wall - 1, runs.length - 1)]!(init);
        } finally {
          pending--;
        }
      }
      if (!url.endsWith('/api/health')) throw new TypeError('Failed to fetch');
      return answers[Math.min(health++, answers.length - 1)]!(init);
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
    return { said, done, checks: () => health, heads: () => wall, pending: () => pending, unmount: () => view.unmount() };
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

  for (const [what, early] of [
    ['fails at once (a reset connection)', resetAfter(0)],
    ['fails after the health answered', resetAfter(1200)],
  ] as const) {
    it(`asks the token check again when the early one ${what}, and still finds the wall`, async () => {
      const c = start([after(800, healthy)], [early, walls]);
      await vi.advanceTimersByTimeAsync(1200);
      await c.done;
      expect(c.said()).toMatchObject({ mode: 'connecting', label: 'Not connected', notice: `${TOKEN_WALL} Retrying…` });
      expect([hub.refused, hub.offline, c.heads(), c.pending()]).toEqual([TOKEN_WALL, true, 2, 0]);
      c.unmount();
    });

    it(`asks the token check again when the early one ${what}, and goes live without a wall`, async () => {
      const c = start([after(800, healthy)], [early, opens]);
      await vi.advanceTimersByTimeAsync(1200);
      await c.done;
      expect([c.said().mode, c.said().label, hub.refused, hub.offline, c.heads()]).toEqual(['live', 'Connected', null, false, 2]);
      c.unmount();
    });
  }

  for (const [what, answer, said] of [
    ['the wall', walls, { mode: 'connecting', label: 'Not connected', notice: `${TOKEN_WALL} Retrying…` }],
    ['no wall', opens, { mode: 'live', label: 'Connected', notice: '' }],
  ] as const) {
    it(`waits for a token check that answers after the health, and reads ${what} from it`, async () => {
      const c = start([healthy], [after(300, answer)]);
      await vi.advanceTimersByTimeAsync(299);
      // The health has answered; the token check has not: not live yet.
      expect([hub.mode, hub.refused, c.heads()]).toEqual(['connecting', null, 1]);
      await vi.advanceTimersByTimeAsync(1);
      await c.done;
      expect(c.said()).toMatchObject(said);
      expect(c.heads()).toBe(1);
      c.unmount();
    });
  }

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
