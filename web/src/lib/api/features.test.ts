// @vitest-environment happy-dom
// What the server in use can do (`hub.features`, #324) across an outage and a change of address (review of #341, round
// 2): only a health answer of the connection in use since its last outage says it. A list from before the outage is
// unknown from the moment the server stops answering, and nothing that answers late for the server before (an address
// changed in Settings) changes whether the server in use answers.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { prefs } from '$lib/prefs';
import { hub } from './index';
import { PROBE_DELAYS } from './reach';

const MOCK = ['queue_control'];
const REAL = ['queue_control', 'agent_labeling'];
const health = (features: readonly string[]) => Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock: false, features });

/** A request the test answers when it likes. */
function held() {
  let answer!: (r: Response) => void;
  const r = new Promise<Response>((a) => (answer = a));
  return { r, answer };
}

/**
 * A server per address: `up`, the features its health lists, and `next`, answers to give its next health requests
 * instead (in order); every other request answers 200 (the token check, lists).
 */
function servers() {
  const all = new Map<string, { up: boolean; features: readonly string[]; next: (() => Response | Promise<Response>)[]; healths: number }>();
  const at = (base: string) => {
    if (!all.has(base)) all.set(base, { up: true, features: REAL, next: [], healths: 0 });
    return all.get(base)!;
  };
  const answers = new Map<string, Promise<Response>>();
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    const [base, path] = url.split('/api');
    const s = at(base!);
    if (!s.up) throw new TypeError('Failed to fetch');
    if (path === '/health') {
      s.healths++;
      return (s.next.shift() ?? (() => health(s.features)))();
    }
    if (init?.method === 'HEAD') return new Response(null, { status: 200 });
    return answers.get(url) ?? Response.json({ datasets: [] });
  });
  return { at, answers };
}

describe('hub.features over an outage', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    prefs.serverUrl = '';
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  /** A request fails and the probe confirms it: offline. */
  async function outage() {
    await expect(hub.api.datasets()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(0);
    expect(hub.offline).toBe(true);
  }

  it('finding 1: restarted with labeling, its first health after the outage a 502: unknown, then asked again until known', async () => {
    const net = servers();
    const s = net.at('');
    s.features = MOCK;
    await hub.connect();
    expect([hub.features, hub.lacks('agent_labeling')]).toEqual([MOCK, true]);
    s.up = false;
    await outage();
    // Unknown at once: the command is offered and the form opens, as on main; the page explains.
    expect([hub.features, hub.lacks('agent_labeling')]).toEqual([null, false]);
    // Back as a plain server, noticed by a request; the health asked then answers 502.
    s.up = true;
    s.features = REAL;
    s.next.push(() => Response.json({ error: { code: 'bad_gateway', message: 'x' } }, { status: 502 }));
    const before = s.healths;
    await hub.api.datasets();
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.features, hub.lacks('agent_labeling'), s.healths - before]).toEqual([false, null, false, 1]);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.features, hub.lacks('agent_labeling'), s.healths - before]).toEqual([REAL, false, 2]);
    // Known: nothing more is asked.
    await vi.advanceTimersByTimeAsync(60_000);
    expect(s.healths - before).toBe(2);
  });

  it('finding 1, the other way: restarted without labeling, its health held: unknown until it answers, then lacking', async () => {
    const net = servers();
    const s = net.at('');
    await hub.connect();
    expect([hub.features, hub.lacks('agent_labeling')]).toEqual([REAL, false]);
    s.up = false;
    await outage();
    expect(hub.features).toBeNull();
    s.up = true;
    s.features = MOCK;
    const answer = held();
    s.next.push(() => answer.r);
    await hub.api.datasets();
    await vi.advanceTimersByTimeAsync(0);
    // Connected, and what it can do is not known yet: never the list from before the outage.
    expect([hub.offline, hub.features]).toEqual([false, null]);
    answer.answer(health(MOCK));
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.features, hub.lacks('agent_labeling')]).toEqual([MOCK, true]);
  });

  it('finding 2: a late answer from the address before changes nothing about the one in use', async () => {
    const net = servers();
    await hub.connect();
    const late = held();
    net.answers.set('/api/datasets', late.r);
    const old = hub.api.datasets();
    // Settings › Connection: another address, which answers, then stops.
    prefs.serverUrl = 'http://127.0.0.1:7861';
    const b = net.at('http://127.0.0.1:7861');
    await hub.connect();
    expect([hub.mode, hub.offline, hub.features]).toEqual(['live', false, REAL]);
    b.up = false;
    await outage();
    late.answer(Response.json({ datasets: [] }));
    await old;
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.features]).toEqual([true, null]);
    // The address in use comes back: the probe says so, and the hub asks its features.
    b.up = true;
    b.features = MOCK;
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.offline, hub.features]).toEqual([false, MOCK]);
  });

  // Review of #341, round 3: the probe only says whether the server answers, as on main; what it can do comes from the
  // hub's own ask once it is back.
  it("round 3, finding 1: a probe's late answer after the hub's ask has set the features changes nothing", async () => {
    const net = servers();
    const s = net.at('');
    s.features = MOCK;
    await hub.connect();
    s.up = false;
    await outage();
    // Back as a plain server; the probe's health is held (the `--mock` server's answer, still on its way).
    s.up = true;
    s.features = REAL;
    const probe = held();
    s.next.push(() => probe.r);
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect(hub.offline).toBe(true);
    // An ordinary request reconnects; the hub's ask hears labeling is there.
    await hub.api.datasets();
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.features, hub.lacks('agent_labeling')]).toEqual([false, REAL, false]);
    probe.answer(health(MOCK));
    await vi.advanceTimersByTimeAsync(60_000);
    expect([hub.offline, hub.features, hub.lacks('agent_labeling')]).toEqual([false, REAL, false]);
  });

  it("round 3, finding 2: back when the probe's headers come in, its body still held (as on main)", async () => {
    const net = servers();
    const s = net.at('');
    await hub.connect();
    s.up = false;
    await outage();
    s.up = true;
    let end!: () => void;
    const body = new ReadableStream<Uint8Array>({ start: (c) => void (end = () => (c.enqueue(new TextEncoder().encode('{}')), c.close())) });
    s.next.push(() => new Response(body, { status: 200, headers: { 'content-type': 'application/json' } }));
    await vi.advanceTimersByTimeAsync(PROBE_DELAYS[0]!);
    expect([hub.offline, hub.features]).toEqual([false, REAL]);
    end();
    await vi.advanceTimersByTimeAsync(0);
    expect([hub.offline, hub.features]).toEqual([false, REAL]);
  });
});

describe('hub.serverMock (issue #382: Resume on folo serve --mock)', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    prefs.serverUrl = '';
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("is the connect's health answer's `mock`, and unknown while a connect has not answered", async () => {
    let mock = true;
    vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
      if (url.endsWith('/api/health')) return Response.json({ status: 'ok', version: '0.1.0', api: 'v1', mock, features: MOCK });
      if (init?.method === 'HEAD') return new Response(null, { status: 200 });
      return Response.json({ datasets: [] });
    });
    await hub.connect();
    expect(hub.serverMock).toBe(true);
    mock = false;
    const again = hub.connect();
    expect(hub.serverMock).toBeNull();
    await again;
    expect(hub.serverMock).toBe(false);
  });
});
