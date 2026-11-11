// @vitest-environment happy-dom
import { createRoot, flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api } from './client';
import type { Run } from './types';
import { hub } from './index';
import { Resource } from './resource';
import { pollTogether, reuse, shareApi, together, unshare } from './share';

const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

function counting() {
  const calls: string[] = [];
  const api = {
    mock: false,
    runs: (signal?: AbortSignal) => {
      calls.push('runs');
      return new Promise((resolve, reject) => {
        const t = setTimeout(() => resolve([{ id: 'a', n: calls.length }]), 10);
        signal?.addEventListener('abort', () => (clearTimeout(t), reject(new DOMException('aborted', 'AbortError'))));
      });
    },
    run: (id: string) => (calls.push(`run ${id}`), Promise.resolve({ id })),
    stopRun: (id: string) => (calls.push(`stop ${id}`), Promise.resolve({ id })),
  };
  return { api: api as unknown as Api, calls };
}

describe('shareApi', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('makes identical reads issued in one turn one request, each caller with an answer of its own', async () => {
    const { api, calls } = counting();
    const s = shareApi(api);
    const a = together(() => s.runs()).value;
    const b = together(() => s.runs()).value;
    const c = together(() => s.run('x')).value;
    expect(calls).toEqual(['runs', 'run x']);
    await vi.advanceTimersByTimeAsync(10);
    const [ra, rb] = (await Promise.all([a, b, c])) as unknown as [unknown[], unknown[]];
    expect(ra).toEqual(rb);
    expect(ra).not.toBe(rb);
  });

  it('asks again in a later turn, outside `together`, and for writes', async () => {
    const { api, calls } = counting();
    const s = shareApi(api);
    void together(() => s.runs()).value.catch(() => {});
    await Promise.resolve();
    void together(() => s.runs()).value.catch(() => {});
    void s.runs();
    void together(() => s.stopRun('x')).value;
    void together(() => s.stopRun('x')).value;
    expect(calls).toEqual(['runs', 'runs', 'runs', 'stop x', 'stop x']);
  });

  it('names the reads a fetcher made, and unwraps', () => {
    const { api } = counting();
    const s = shareApi(api);
    expect(together(() => (void s.run('r1'), 1)).key).toBe('run("r1")');
    expect(shareApi(api)).toBe(s);
    expect(unshare(s)).toBe(api);
  });

  it("passes what a method carries through (a test spy's .mock)", () => {
    const runs = vi.fn(async () => []);
    const api = shareApi({ runs } as unknown as Api);
    void api.runs();
    expect(vi.mocked(api.runs).mock.calls.length).toBe(1);
  });

  it('aborts the shared request only once every caller has given up', async () => {
    const { api } = counting();
    const s = shareApi(api);
    const one = new AbortController();
    const two = new AbortController();
    const a = together(() => s.runs(one.signal)).value;
    const b = together(() => s.runs(two.signal)).value;
    one.abort();
    await expect(a).rejects.toThrow('aborted');
    await vi.advanceTimersByTimeAsync(10);
    await expect(b).resolves.toEqual([{ id: 'a', n: 1 }]);
  });
});

describe('Resource sharing', () => {
  const saved = hub.api;
  beforeEach(() => {
    vi.useFakeTimers();
    hub.mode = 'mock';
  });
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('two Resources mounted together load with one request, and polling together poll with one', async () => {
    const { api, calls } = counting();
    hub.api = api;
    let a!: Resource<Run[]>;
    let b!: Resource<Run[]>;
    const stop = root(() => {
      a = new Resource((s) => hub.api.runs(s), { poll: 100 });
      b = new Resource((s) => hub.api.runs(s), { poll: 100 });
    });
    flush();
    expect(calls).toEqual(['runs']);
    await vi.advanceTimersByTimeAsync(10);
    expect(a.data).toEqual(b.data);
    await vi.advanceTimersByTimeAsync(100);
    expect(calls).toEqual(['runs', 'runs']);
    stop();
    await vi.advanceTimersByTimeAsync(1000);
    expect(calls).toHaveLength(2);
  });

  it('keeps the data object when an answer is equal to it', async () => {
    let n = 0;
    let r!: Resource<{ id: string; v: number }[]>;
    const stop = root(() => {
      r = new Resource(() => Promise.resolve([{ id: 'a', v: 1 }, { id: 'b', v: n++ < 2 ? 1 : 2 }]));
    });
    flush();
    await vi.advanceTimersByTimeAsync(0);
    const first = r.data!;
    await r.refresh();
    expect(r.data).toBe(first);
    await r.refresh();
    expect(r.data).not.toBe(first);
    expect(r.data![0]).toBe(first[0]);
    expect(r.data![1]).toEqual({ id: 'b', v: 2 });
    stop();
  });
});

describe('pollTogether', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('ticks like setInterval, one timer for pollers started in the same turn, each stopping on its own', async () => {
    const a = vi.fn();
    const b = vi.fn();
    const stopA = pollTogether(50, a);
    const stopB = pollTogether(50, b);
    expect(vi.getTimerCount()).toBe(1);
    vi.advanceTimersByTime(100);
    expect(a).toHaveBeenCalledTimes(2);
    expect(b).toHaveBeenCalledTimes(2);
    stopA();
    vi.advanceTimersByTime(50);
    expect(a).toHaveBeenCalledTimes(2);
    expect(b).toHaveBeenCalledTimes(3);
    stopB();
    expect(vi.getTimerCount()).toBe(0);
    await Promise.resolve();
    // A later turn: a timer of its own, on its own phase.
    const c = vi.fn();
    const stopC = pollTogether(50, c);
    vi.advanceTimersByTime(49);
    expect(c).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(c).toHaveBeenCalledOnce();
    stopC();
  });
});

describe('reuse', () => {
  it('returns the previous value when equal, and keeps equal parts of a changed one', () => {
    const prev = { a: [1, 2, { x: 1 }], b: { c: 'd' }, when: new Date(0) };
    expect(reuse(prev, structuredClone(prev))).not.toBe(prev); // a Date is compared by identity
    const p2 = { a: [1, 2, { x: 1 }], b: { c: 'd' } };
    expect(reuse(p2, structuredClone(p2))).toBe(p2);
    const next = reuse(p2, { a: [1, 2, { x: 2 }], b: { c: 'd' } });
    expect(next).not.toBe(p2);
    expect(next.b).toBe(p2.b);
    expect(next.a[2]).toEqual({ x: 2 });
    expect(reuse(p2, { a: [1, 2, { x: 1 }] } as typeof p2)).not.toBe(p2);
    expect(reuse(p2, { ...structuredClone(p2), e: 1 } as typeof p2)).not.toBe(p2);
  });
});
