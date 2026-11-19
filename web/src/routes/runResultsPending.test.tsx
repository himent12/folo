// @vitest-environment happy-dom
// A short run watched from queued to completed (issue #327): from the instant the header says Completed, a run that
// validated never claims it ended before validation, while its final history and samples are on their way. The events
// come in the order the server sends them (`status running`, the steps, `epoch 1`, `status completed`, 2 ms apart in
// the tester's trace); a run that really ended before its first validation still says so once its history is in.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { RunEventHandlers } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Epoch, Run as RunT, RunMetrics, Step } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Run from './Run';

let target: HTMLElement;
let c: Rendered | undefined;
/** Animation frames, run only when the test says (a browser's next frame; a background tab's, much later). */
let frames = new Map<number, FrameRequestCallback>();
let nextFrame = 0;
const runFrames = () => {
  const due = [...frames.values()];
  frames = new Map();
  for (const f of due) f(performance.now());
};

beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (f: FrameRequestCallback) => (frames.set(++nextFrame, f), nextFrame));
  vi.stubGlobal('cancelAnimationFrame', (id: number) => void frames.delete(id));
});
afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  frames.clear();
  vi.unstubAllGlobals();
});

const NEVER = /ended before its first|not validated/;

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => ((resolve = res), (reject = rej)));
  return { promise, resolve, reject };
}

/** A one-epoch run of four steps, opened while queued; `metrics` answers the history requests made after the open. */
async function watch() {
  const api = new MockApi('normal', 0);
  const done = (await api.runs()).find((r) => r.status === 'completed')!;
  const mh = await api.runMetrics(done.id);
  const steps: Step[] = [1, 2, 3, 4].map((n) => ({ ...mh.steps[n - 1]!, epoch: 1, step: n, steps_per_epoch: 4, epochs: 1 }));
  const epoch: Epoch = { ...mh.epochs[0]!, epoch: 1 };
  let server: RunT = { ...done, status: 'queued', epochs: 1, epoch: 0, step: 0, steps_per_epoch: 0, best: null, model_id: null, started: null, finished: null, queue_position: 1, eta_s: null };
  const first: RunMetrics = { steps: [], epochs: [] };
  let metrics: () => Promise<RunMetrics> = () => Promise.resolve(first);
  let handlers: RunEventHandlers = {};
  api.run = async () => server;
  api.runMetrics = () => metrics();
  api.runLogs = async () => ({ lines: [] });
  api.runEvents = (_id, h) => {
    handlers = h;
    return () => {};
  };
  hub.api = api;
  hub.mode = 'mock';
  router.route = parse(`#/runs/${done.id}`);
  target = document.createElement('div');
  document.body.appendChild(target);
  c = render(() => <Run />, target);
  await vi.waitFor(
    () => {
      runFrames();
      flushSync();
      if (!target.querySelector('.waiting')) throw new Error('not loaded');
    },
    { timeout: 4000, interval: 10 },
  );
  /**
   * The trainer runs: `status running` and the four steps, shown for a while (frames go by); then, in one go, its
   * `epoch 1` (with `validated`) and its end.
   */
  const end = (status: 'completed' | 'failed', validated: boolean) => {
    server = { ...server, status: 'running', queue_position: null, started: '2026-11-19T07:53:25Z', steps_per_epoch: 4 };
    handlers.status?.({ status: 'running', error: null, steps_per_epoch: 4, epochs: 1 });
    for (const s of steps) handlers.step?.(s);
    runFrames();
    flushSync();
    if (validated) handlers.epoch?.(epoch);
    const finished = '2026-11-19T07:53:26Z';
    server =
      status === 'completed'
        ? { ...server, status, step: 4, epoch: 1, best: { epoch: 1, map50_95: epoch.val.map50_95 }, model_id: done.model_id, finished, eta_s: null }
        : { ...server, status, step: 4, epoch: 1, error: 'boom', finished, eta_s: null };
    handlers.status?.({ status, error: status === 'failed' ? 'boom' : null });
    flushSync();
  };
  return {
    steps,
    epoch,
    end,
    /** The history requests from now on answer with this. */
    answer: (m: () => Promise<RunMetrics>) => (metrics = m),
    /** The run's events, sent by the test. */
    stream: () => handlers,
    /** The snapshot from now on. */
    serve: (r: Partial<RunT>) => (server = { ...server, ...r }),
  };
}

const text = () => target.textContent ?? '';
const settle = async (until: () => boolean) =>
  vi.waitFor(
    () => {
      runFrames();
      flushSync();
      if (!until()) throw new Error('not yet');
    },
    { timeout: 4000, interval: 10 },
  );

describe('a run watched to its end while its results arrive (issue #327)', () => {
  it('shows the epoch the stream delivered with the end, never "it ended before its first epoch finished"', async () => {
    const w = await watch();
    // The final history is slow (2.7 s in the tester's trace): the page reads what the stream delivered.
    const final = deferred<RunMetrics>();
    w.answer(() => final.promise);
    w.end('completed', true);
    expect(text()).toContain('Completed');
    expect(text()).not.toMatch(NEVER);
    expect(target.querySelector('#chart-map')?.textContent).toContain('Latest · ep 1');
    expect(target.querySelector('#per-class')?.textContent).toContain('epoch 1');
    final.resolve({ steps: w.steps, epochs: [w.epoch] });
    await settle(() => text().includes('Latest (epoch 1)'));
    expect(text()).not.toMatch(NEVER);
  });

  it('holds the per-epoch panels as loading when the stream missed the epoch, until the final history brings it', async () => {
    const w = await watch();
    const final = deferred<RunMetrics>();
    w.answer(() => final.promise);
    // A reconnect dropped `epoch 1`: only the final history has it.
    w.end('completed', false);
    expect(text()).not.toMatch(NEVER);
    expect(text()).not.toContain('Waiting for the first validation');
    expect(text()).not.toContain('appear after the first validation');
    expect(target.querySelector('#chart-map .cs')).not.toBeNull();
    expect(target.querySelector('#per-class .sk')).not.toBeNull();
    expect(target.querySelector('#samples .cell-sk')).not.toBeNull();
    final.resolve({ steps: w.steps, epochs: [w.epoch] });
    await settle(() => (target.querySelector('#chart-map')?.textContent ?? '').includes('Latest · ep 1'));
    expect(text()).not.toMatch(NEVER);
    expect(target.querySelector('#per-class .sk')).toBeNull();
    expect(target.querySelector('#chart-map .cs')).toBeNull();
  });

  it('holds them as loading when a reconnect read the history before the validation and the run after its end', async () => {
    const w = await watch();
    const h = w.stream();
    w.serve({ status: 'running', queue_position: null, started: '2026-11-19T07:53:25Z', steps_per_epoch: 4 });
    h.status?.({ status: 'running', error: null, steps_per_epoch: 4, epochs: 1 });
    for (const s of w.steps) h.step?.(s);
    // The stream drops; the run validates and completes; its reconnect reads /metrics from before the validation and
    // /run from after the end, and the terminal event is late.
    w.serve({ status: 'completed', step: 4, epoch: 1, best: { epoch: 1, map50_95: w.epoch.val.map50_95 }, finished: '2026-11-19T07:53:26Z', eta_s: null });
    let asked = 0;
    w.answer(() => Promise.resolve({ steps: w.steps, epochs: asked++ ? [w.epoch] : [] }));
    h.reconnect?.();
    await settle(() => text().includes('Completed'));
    expect(text()).not.toMatch(NEVER);
    expect(target.querySelector('#per-class .sk')).not.toBeNull();
    // That history was not the final one: it is asked for again.
    await settle(() => (target.querySelector('#chart-map')?.textContent ?? '').includes('Latest · ep 1'));
    expect(text()).not.toMatch(NEVER);
  });

  it('says it could not load the metrics, with Retry, when that final history fails', async () => {
    const w = await watch();
    w.answer(() => Promise.reject(new Error('Boom.')));
    w.end('completed', false);
    await settle(() => text().includes("Could not load this run's metrics"));
    expect(text()).not.toMatch(NEVER);
    expect([...target.querySelectorAll('button')].filter((b) => b.textContent?.trim() === 'Retry')).toHaveLength(1);
  });

  it('still says a run that failed in its first epoch never validated, once its history says so', async () => {
    const w = await watch();
    w.answer(() => Promise.resolve({ steps: w.steps, epochs: [] }));
    w.end('failed', false);
    await settle(() => (target.querySelector('#samples')?.textContent ?? '').includes('No samples: it ended before its first validation.'));
    expect(target.querySelector('.kpis')?.textContent).toContain('not validated');
    expect(target.querySelector('#per-class')?.textContent).toContain('No per-class results: it ended before its first validation.');
  });
});
