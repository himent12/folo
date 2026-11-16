import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LABEL_EVENTS, openEventStream, RUN_EVENTS, sseParser, STREAM_SILENCE_MS } from './sse';
import type { LabelEventMap, RunEventMap } from './types';

/** One request of the fake fetch: the test answers it, sends bytes on it, ends or breaks it, and sees it aborted. */
class Conn {
  static all: Conn[] = [];
  aborted = false;
  #resolve!: (r: Response) => void;
  #reject!: (e: unknown) => void;
  #body: ReadableStreamDefaultController<Uint8Array> | null = null;
  readonly answer = new Promise<Response>((resolve, reject) => {
    this.#resolve = resolve;
    this.#reject = reject;
  });
  constructor(
    readonly url: string,
    readonly init: RequestInit,
  ) {
    Conn.all.push(this);
    init.signal?.addEventListener('abort', () => {
      this.aborted = true;
      const e = new DOMException('aborted', 'AbortError');
      this.#reject(e);
      try {
        this.#body?.error(e);
      } catch {
        // already closed
      }
    });
  }
  open(status = 200, type = 'text/event-stream') {
    const body = new ReadableStream<Uint8Array>({ start: (c) => void (this.#body = c) });
    this.#resolve(new Response(body, { status, headers: { 'content-type': type } }));
  }
  fail() {
    this.#reject(new TypeError('Failed to fetch'));
  }
  bytes(text: string) {
    try {
      this.#body?.enqueue(new TextEncoder().encode(text));
    } catch {
      // aborted: nothing reads it any more
    }
  }
  send(name: string, data: unknown) {
    this.bytes(`event: ${name}\ndata: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`);
  }
  end() {
    this.#body?.close();
  }
}

/** Lets the fetch answers and the stream reads (promises) settle. */
const settle = async () => {
  for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
};

beforeEach(() => {
  Conn.all = [];
  vi.stubGlobal('fetch', (url: string, init: RequestInit) => new Conn(url, init).answer);
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('sseParser', () => {
  it('joins events split across chunks, any line ending, and skips comments', () => {
    const got: string[] = [];
    const feed = sseParser((name, data) => got.push(`${name}=${data}`));
    feed(': keep-alive\n\nevent: st');
    feed('atus\r\ndata: {"a":1}\r');
    feed('\n\r\ndata: x\ndata: y\n\n');
    feed('event: log\nid: 3\nretry: 10\ndata:no-space\n');
    expect(got).toEqual(['status={"a":1}', 'message=x\ny']);
    feed('\n');
    expect(got.at(-1)).toBe('log=no-space');
  });
});

describe('openEventStream', () => {
  it("delivers a run stream's queue and cloud events (servers with queue_control and cloud send them), and closes", async () => {
    const queue = vi.fn();
    const status = vi.fn();
    const cloud = vi.fn();
    const close = openEventStream<RunEventMap>('/api/runs/r/events', RUN_EVENTS, { queue, status, cloud });
    const c = Conn.all[0]!;
    expect(new Headers(c.init.headers).get('accept')).toBe('text/event-stream');
    c.open();
    await settle();
    c.send('queue', { position: 2, waiting: 3 } satisfies RunEventMap['queue']);
    c.bytes(': keep-alive\n\n');
    c.send('status', { status: 'queued', error: null });
    const state = { connection_id: 'cc_1', provider: 'runpod', phase: 'syncing', progress: { done_bytes: 1, total_bytes: 4 } };
    c.send('cloud', state);
    await settle();
    expect(queue).toHaveBeenCalledWith({ position: 2, waiting: 3 });
    expect(status).toHaveBeenCalledOnce();
    expect(cloud).toHaveBeenCalledWith(state);
    close();
    expect(c.aborted).toBe(true);
  });

  it("delivers a labeling job's events by name and ignores names it was not given", async () => {
    const got: string[] = [];
    openEventStream<LabelEventMap>('/api/label/jobs/j/events', LABEL_EVENTS, {
      job: (j) => got.push(`job ${j.id}`),
      image: (i) => got.push(`image ${i.id} ${i.status}`),
      removed: (r) => got.push(`removed ${r.id}`),
      reload: () => got.push('reload'),
      log: (l) => got.push(`log ${l.message}`),
    });
    const c = Conn.all[0]!;
    expect(c.url).toBe('/api/label/jobs/j/events');
    c.open();
    await settle();
    c.send('job', { id: 'lb_1' });
    c.send('image', { id: '7', status: 'labeling' });
    c.send('removed', { id: '7' });
    c.send('reload', {});
    c.send('log', { t: '', level: 'info', message: 'call 1' });
    c.send('later', { id: 'x' });
    await settle();
    expect(got).toEqual(['job lb_1', 'image 7 labeling', 'removed 7', 'reload', 'log call 1']);
  });

  it('skips a payload that is not JSON and keeps listening', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const log = vi.fn();
    openEventStream<LabelEventMap>('/e', LABEL_EVENTS, { log });
    Conn.all[0]!.open();
    await settle();
    Conn.all[0]!.send('log', '{');
    Conn.all[0]!.send('log', { message: 'ok' });
    await settle();
    expect(warn).toHaveBeenCalledOnce();
    expect(log).toHaveBeenCalledWith({ message: 'ok' });
  });

  it('reopens a stream refused, failed or ended, with backoff, and asks for a resync once it is back', async () => {
    const states: string[] = [];
    const reconnect = vi.fn();
    const close = openEventStream<RunEventMap>('/e', RUN_EVENTS, { connection: (s) => states.push(s), reconnect }, { delay: (a) => 100 * (a + 1) });
    // An error status (a deleted run's 404).
    Conn.all[0]!.open(404, 'application/json');
    await settle();
    expect(states).toEqual(['reconnecting']);
    await vi.advanceTimersByTimeAsync(99);
    expect(Conn.all).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(Conn.all).toHaveLength(2);
    // Nothing answers (the server is down): the next waits longer.
    Conn.all[1]!.fail();
    await settle();
    await vi.advanceTimersByTimeAsync(199);
    expect(Conn.all).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    // A page that is not an event stream (a proxy's) is no better.
    Conn.all[2]!.open(200, 'text/html');
    await settle();
    await vi.advanceTimersByTimeAsync(300);
    Conn.all[3]!.open();
    await settle();
    expect(states).toEqual(['reconnecting', 'reconnecting', 'reconnecting', 'open']);
    expect(reconnect).toHaveBeenCalledOnce();
    // The server ends the stream: reopened at the first backoff again.
    Conn.all[3]!.end();
    await settle();
    await vi.advanceTimersByTimeAsync(100);
    expect(Conn.all).toHaveLength(5);
    // Closing stops a pending reopen.
    Conn.all[4]!.open(500, 'application/json');
    await settle();
    close();
    await vi.advanceTimersByTimeAsync(1000);
    expect(Conn.all).toHaveLength(5);
  });

  // Stress d6-connection: a proxy that passed the first events and then held the connection without a byte left the
  // stream (and the run page) "live" for good; EventSource never reported an error.
  it('drops a stream that passes no byte for three keep-alives, reopens it and asks for a resync', async () => {
    const states: string[] = [];
    const reconnect = vi.fn();
    const status = vi.fn();
    openEventStream<RunEventMap>('/e', RUN_EVENTS, { connection: (s) => states.push(s), reconnect, status }, { delay: () => 1000 });
    const c = Conn.all[0]!;
    c.open();
    await settle();
    c.send('status', { status: 'running', error: null });
    await settle();
    // Quiet but alive: the server's keep-alive comments every 15 s keep it open, however long.
    for (let i = 0; i < 8; i++) {
      await vi.advanceTimersByTimeAsync(15_000);
      c.bytes(': keep-alive\n\n');
      await settle();
    }
    expect(c.aborted).toBe(false);
    expect(states).toEqual(['open']);
    // Then nothing at all.
    await vi.advanceTimersByTimeAsync(STREAM_SILENCE_MS - 1);
    expect(c.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(c.aborted).toBe(true);
    expect(states).toEqual(['open', 'reconnecting']);
    await vi.advanceTimersByTimeAsync(1000);
    expect(Conn.all).toHaveLength(2);
    Conn.all[1]!.open();
    await settle();
    expect(states).toEqual(['open', 'reconnecting', 'open']);
    expect(reconnect).toHaveBeenCalledOnce();
    // Bytes the silent connection still lets through are not delivered.
    c.send('status', { status: 'failed', error: 'late' });
    await settle();
    expect(status).toHaveBeenCalledOnce();
  });

  it('gives up on a request that never answers after the same silence', async () => {
    const states: string[] = [];
    openEventStream<RunEventMap>('/e', RUN_EVENTS, { connection: (s) => states.push(s) }, { delay: () => 1000 });
    await vi.advanceTimersByTimeAsync(STREAM_SILENCE_MS);
    await settle();
    expect(Conn.all[0]!.aborted).toBe(true);
    expect(states).toEqual(['reconnecting']);
    await vi.advanceTimersByTimeAsync(1000);
    expect(Conn.all).toHaveLength(2);
  });
});
