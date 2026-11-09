import { afterEach, describe, expect, it, vi } from 'vitest';
import type { RunEventMap } from './types';
import { EVENTS, openEventStream } from './sse';

/** A stand-in EventSource: records listeners, so a test can deliver named events as a server would. */
class FakeSource {
  static CLOSED = 2;
  static last: FakeSource | null = null;
  readyState = 1;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  listeners = new Map<string, (ev: MessageEvent<string>) => void>();
  constructor(readonly url: string) {
    FakeSource.last = this;
  }
  addEventListener(name: string, fn: (ev: MessageEvent<string>) => void) {
    this.listeners.set(name, fn);
  }
  close() {
    this.readyState = FakeSource.CLOSED;
  }
  emit(name: string, data: unknown) {
    this.listeners.get(name)?.({ data: JSON.stringify(data) } as MessageEvent<string>);
  }
}

describe('openEventStream', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('listens to every event of a run stream, queue and cloud included', () => {
    vi.stubGlobal('EventSource', FakeSource);
    const queue = vi.fn();
    const cloud = vi.fn();
    const close = openEventStream('/api/runs/run_1/events', { queue, cloud });
    const es = FakeSource.last!;
    expect([...es.listeners.keys()]).toEqual(['status', 'step', 'epoch', 'log', 'queue', 'cloud']);
    expect(EVENTS).toEqual([...es.listeners.keys()]);
    es.emit('queue', { position: 2, waiting: 3 } satisfies RunEventMap['queue']);
    const state = { connection_id: 'cc_1', provider: 'runpod', phase: 'syncing', progress: { done_bytes: 1, total_bytes: 4 } };
    es.emit('cloud', state);
    expect(queue).toHaveBeenCalledWith({ position: 2, waiting: 3 });
    expect(cloud).toHaveBeenCalledWith(state);
    close();
    expect(es.readyState).toBe(FakeSource.CLOSED);
  });
});
