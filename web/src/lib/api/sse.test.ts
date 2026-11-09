import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LABEL_EVENTS, openEventStream, RUN_EVENTS } from './sse';
import type { LabelEventMap, RunEventMap } from './types';

/** An EventSource that the test drives: it records listeners and lets the test open, fail and deliver. */
class FakeSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;
  static all: FakeSource[] = [];
  readyState = FakeSource.CONNECTING;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners = new Map<string, ((e: MessageEvent<string>) => void)[]>();
  constructor(readonly url: string) {
    FakeSource.all.push(this);
  }
  addEventListener(name: string, f: (e: MessageEvent<string>) => void) {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), f]);
  }
  close() {
    this.readyState = FakeSource.CLOSED;
  }
  open() {
    this.readyState = FakeSource.OPEN;
    this.onopen?.();
  }
  fail(closed: boolean) {
    this.readyState = closed ? FakeSource.CLOSED : FakeSource.CONNECTING;
    this.onerror?.();
  }
  send(name: string, data: unknown) {
    for (const f of this.listeners.get(name) ?? []) f({ data: typeof data === 'string' ? data : JSON.stringify(data) } as MessageEvent<string>);
  }
}

beforeEach(() => {
  FakeSource.all = [];
  vi.stubGlobal('EventSource', FakeSource);
  vi.useFakeTimers();
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('openEventStream', () => {
  it("delivers a run stream's queue events (servers with queue_control send them)", () => {
    const queue = vi.fn();
    const status = vi.fn();
    openEventStream<RunEventMap>('/api/runs/r/events', RUN_EVENTS, { queue, status });
    const es = FakeSource.all[0]!;
    es.send('queue', { position: 2, waiting: 3 });
    es.send('status', { status: 'queued', error: null });
    expect(queue).toHaveBeenCalledWith({ position: 2, waiting: 3 });
    expect(status).toHaveBeenCalledOnce();
  });

  it("delivers a labeling job's events by name and ignores names it was not given", () => {
    const got: string[] = [];
    openEventStream<LabelEventMap>('/api/label/jobs/j/events', LABEL_EVENTS, {
      job: (j) => got.push(`job ${j.id}`),
      image: (i) => got.push(`image ${i.id} ${i.status}`),
      removed: (r) => got.push(`removed ${r.id}`),
      log: (l) => got.push(`log ${l.message}`),
    });
    const es = FakeSource.all[0]!;
    expect(es.url).toBe('/api/label/jobs/j/events');
    es.send('job', { id: 'lb_1' });
    es.send('image', { id: '7', status: 'labeling' });
    es.send('removed', { id: '7' });
    es.send('log', { t: '', level: 'info', message: 'call 1' });
    es.send('later', { id: 'x' });
    expect(got).toEqual(['job lb_1', 'image 7 labeling', 'removed 7', 'log call 1']);
  });

  it('skips a payload that is not JSON and keeps listening', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const log = vi.fn();
    openEventStream<LabelEventMap>('/e', LABEL_EVENTS, { log });
    FakeSource.all[0]!.send('log', '{');
    FakeSource.all[0]!.send('log', { message: 'ok' });
    expect(warn).toHaveBeenCalledOnce();
    expect(log).toHaveBeenCalledWith({ message: 'ok' });
  });

  it('reopens a closed stream with backoff and asks for a resync once it is back', () => {
    const states: string[] = [];
    const reconnect = vi.fn();
    const close = openEventStream<RunEventMap>('/e', RUN_EVENTS, { connection: (s) => states.push(s), reconnect }, { delay: (a) => 100 * (a + 1) });
    FakeSource.all[0]!.open();
    // A drop EventSource retries by itself: no new source.
    FakeSource.all[0]!.fail(false);
    expect(FakeSource.all).toHaveLength(1);
    // Closed for good (an error status): reopened after the backoff.
    FakeSource.all[0]!.fail(true);
    vi.advanceTimersByTime(99);
    expect(FakeSource.all).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeSource.all).toHaveLength(2);
    FakeSource.all[1]!.open();
    expect(states).toEqual(['open', 'reconnecting', 'reconnecting', 'open']);
    expect(reconnect).toHaveBeenCalledOnce();
    // Closing stops a pending reopen.
    FakeSource.all[1]!.fail(true);
    close();
    vi.advanceTimersByTime(1000);
    expect(FakeSource.all).toHaveLength(2);
  });
});
