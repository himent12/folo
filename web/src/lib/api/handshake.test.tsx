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
import { isHealth, NOT_FOLO, refusal } from './handshake';
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

  /** Connects with `answers` for the health checks in turn (the last one repeats); what the shell says. */
  async function connectTo(answers: (() => Response | Promise<Response>)[]) {
    let health = 0;
    vi.stubGlobal('fetch', async (url: string) => {
      if (!url.endsWith('/api/health')) throw new TypeError('Failed to fetch');
      return answers[Math.min(health++, answers.length - 1)]!();
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
    await hub.connect();
    flush();
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
    return { said, checks: () => health, unmount: () => view.unmount() };
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
});
