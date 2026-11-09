// @vitest-environment happy-dom
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, LabelEventHandlers } from './client';
import { hub } from './index.svelte';
import { applyChanges, LiveLabelJob, upsertImage } from './liveLabel.svelte';
import { STREAM_GRACE_MS } from './live.svelte';
import { MockApi } from './mock';
import { BATCH_MS } from './mock/label';
import type { LabelImage, LabelImagePage, LabelJob } from './types';

const img = (id: string, status: LabelImage['status'] = 'pending'): LabelImage => ({
  id,
  file: `${id}.jpg`,
  width: 640,
  height: 480,
  status,
  boxes: [],
  source: null,
  error: null,
});
const job = (status: LabelJob['status'], images = 3) =>
  ({ id: 'lb_1', name: 'j', status, counts: { images, pending: images, labeled: 0, failed: 0, boxes: 0 } }) as LabelJob;

function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => (resolve = r));
  return { promise, resolve };
}

describe('upsertImage and applyChanges', () => {
  it('replaces an image in place, or adds a new one at the end', () => {
    const list = [img('1'), img('2')];
    expect(upsertImage(list, img('2', 'labeled')).map((i) => `${i.id} ${i.status}`)).toEqual(['1 pending', '2 labeled']);
    expect(upsertImage(list, img('3')).map((i) => i.id)).toEqual(['1', '2', '3']);
    expect(list.map((i) => i.status)).toEqual(['pending', 'pending']);
  });

  it('applies what the stream said over a loaded list, removals included', () => {
    const changes = new Map<string, LabelImage | null>([
      ['2', img('2', 'labeling')],
      ['1', null],
      ['4', img('4')],
    ]);
    expect(applyChanges([img('1'), img('2'), img('3')], changes).map((i) => `${i.id} ${i.status}`)).toEqual(['2 labeling', '3 pending', '4 pending']);
  });
});

describe('LiveLabelJob ordering', () => {
  let handlers: LabelEventHandlers = {};
  let page = deferred<LabelImagePage>();
  let snapshot: LabelJob = job('idle');
  const loads = vi.fn();
  const saved = hub.api;

  beforeEach(() => {
    vi.useFakeTimers();
    page = deferred<LabelImagePage>();
    snapshot = job('idle');
    loads.mockReset();
    const fake: Partial<Api> = {
      mock: false,
      labelJob: () => Promise.resolve(snapshot),
      labelImages: () => {
        loads();
        return page.promise;
      },
      labelLogs: () => Promise.resolve({ lines: [{ t: '2026-11-09T10:00:00Z', level: 'info', message: 'added 3 images' }] }),
      labelEvents: (_id, h) => {
        handlers = h;
        return () => {};
      },
    };
    hub.api = fake as Api;
    hub.mode = 'live';
  });
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  function mount() {
    let live!: LiveLabelJob;
    const stop = $effect.root(() => {
      live = new LiveLabelJob(() => 'lb_1');
    });
    flushSync();
    return { live, stop };
  }

  it('applies image and job events heard during the load over its answer', async () => {
    const { live, stop } = mount();
    expect(live.loading).toBe(true);
    handlers.image?.(img('2', 'labeling'));
    handlers.removed?.({ id: '3' });
    handlers.job?.(job('running'));
    page.resolve({ total: 3, images: [img('1'), img('2'), img('3')] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.loading).toBe(false);
    expect(live.images.map((i) => `${i.id} ${i.status}`)).toEqual(['1 pending', '2 labeling']);
    // The snapshot was requested before the `job` event: the event is newer.
    expect(live.job?.status).toBe('running');
    expect(live.logs.map((l) => l.message)).toEqual(['added 3 images']);
    stop();
  });

  it('loads every page of a large job', async () => {
    const all = Array.from({ length: 2500 }, (_, i) => img(String(i + 1)));
    hub.api = {
      ...hub.api,
      labelImages: (_id: string, q: { offset?: number; limit?: number } = {}) =>
        Promise.resolve({ total: all.length, images: all.slice(q.offset ?? 0, (q.offset ?? 0) + (q.limit ?? 100)) }),
    } as Api;
    const { live, stop } = mount();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.images).toHaveLength(2500);
    expect(live.images.at(-1)?.id).toBe('2500');
    stop();
  });

  it('loads again when the stream opens and after a reconnect, and calls a stream lost only after the grace', async () => {
    const { live, stop } = mount();
    page.resolve({ total: 1, images: [img('1')] });
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(1);
    handlers.connection?.('open');
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(2);
    handlers.connection?.('reconnecting');
    await vi.advanceTimersByTimeAsync(STREAM_GRACE_MS - 1);
    expect(live.lost).toBeNull();
    await vi.advanceTimersByTimeAsync(1);
    expect(live.lost).not.toBeNull();
    handlers.connection?.('open');
    handlers.reconnect?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(3);
    expect(live.lost).toBeNull();
    stop();
  });

  it("keeps a job event that arrived while an action was answered, over the action's older answer", async () => {
    const { live, stop } = mount();
    page.resolve({ total: 0, images: [] });
    await vi.advanceTimersByTimeAsync(0);
    const answer = deferred<LabelJob>();
    const acting = live.act(() => answer.promise);
    handlers.job?.(job('stopping'));
    answer.resolve(job('running'));
    await acting;
    expect(live.job?.status).toBe('stopping');
    await live.act(() => Promise.resolve(job('idle')));
    expect(live.job?.status).toBe('idle');
    stop();
  });
});

describe('LiveLabelJob with the mock', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
    vi.useRealTimers();
  });

  it('follows a running job as the agent labels its images', async () => {
    vi.useFakeTimers({ now: Date.parse('2026-11-09T12:00:00Z') });
    hub.api = new MockApi('normal', 0);
    hub.mode = 'mock';
    let live!: LiveLabelJob;
    const stop = $effect.root(() => {
      live = new LiveLabelJob(() => 'lb_3f9a');
    });
    flushSync();
    await vi.advanceTimersByTimeAsync(100);
    expect(live.images).toHaveLength(96);
    const done = live.job!.run!.done + live.job!.run!.failed;
    await vi.advanceTimersByTimeAsync(3 * BATCH_MS);
    expect(live.job!.run!.done + live.job!.run!.failed).toBe(done + 12);
    const labeled = live.images.filter((i) => i.status === 'labeled' || i.status === 'failed').length;
    expect(labeled).toBe(done + 12);
    expect(live.images.filter((i) => i.status === 'labeling')).toHaveLength(12);
    expect(live.logs.some((l) => l.message.startsWith('call '))).toBe(true);
    stop();
  });
});
