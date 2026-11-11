// @vitest-environment happy-dom
import { createRoot, flush as flushSync } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Api, LabelEventHandlers } from './client';
import { hub } from './index';
import { applyChanges, LiveLabelJob, upsertImage } from './liveLabel';
import { STREAM_GRACE_MS } from './live';
import { MockApi } from './mock';
import { BATCH_MS } from './mock/label';
import type { LabelImage, LabelImagePage, LabelJob } from './types';

/** Port of `root(fn)`: runs `fn` in a root of its own and returns its teardown. */
const root = (fn: () => void) => createRoot((dispose) => (fn(), dispose));

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

  it('keeps the order: changed images in place, new ones at the end in the order given', () => {
    const out = applyChanges([img('1'), img('2')], [['5', img('5')], ['2', img('2', 'labeled')], ['4', img('4')], ['5', img('5', 'labeling')]]);
    expect(out.map((i) => `${i.id} ${i.status}`)).toEqual(['1 pending', '2 labeled', '5 labeling', '4 pending']);
  });

  it('merges a bulk answer in one pass, not a search of the list per image', () => {
    // Counts every read of an image's id: a search per image would read about 10,000 × 10,000 / 2 of them.
    let reads = 0;
    const counted = (i: LabelImage) => new Proxy(i, { get: (t, k) => (k === 'id' && reads++, Reflect.get(t, k)) });
    const list = Array.from({ length: 10_000 }, (_, i) => counted(img(String(i))));
    const answer = Array.from({ length: 10_000 }, (_, i) => counted(img(String(i + 5_000), 'labeled')));
    const out = applyChanges(list, answer.map((i) => [i.id, i] as const));
    expect(out).toHaveLength(15_000);
    expect(out[4_999]?.status).toBe('pending');
    expect(out[5_000]?.status).toBe('labeled');
    expect(out.at(-1)?.id).toBe('14999');
    expect(reads).toBeLessThan(100_000);
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
    const stop = root(() => {
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

  it('keeps its log as an append log: a later line is appended in place, one out of order replaces the list', async () => {
    const { live, stop } = mount();
    page.resolve({ total: 0, images: [] });
    await vi.advanceTimersByTimeAsync(0);
    const r0 = live.logsLog.read(null);
    expect(r0.added.map((l) => l.message)).toEqual(['added 3 images']);
    handlers.log?.({ t: '2026-11-09T10:00:05Z', level: 'info', message: 'b' });
    const r1 = live.logsLog.read(r0.cursor);
    expect(r1.reset).toBe(false);
    expect(r1.added.map((l) => l.message)).toEqual(['b']);
    // Earlier than the last line: merged in time order, as before (a new list).
    handlers.log?.({ t: '2026-11-09T10:00:01Z', level: 'info', message: 'a' });
    const r2 = live.logsLog.read(r1.cursor);
    expect(r2.reset).toBe(true);
    expect(live.logs.map((l) => l.message)).toEqual(['added 3 images', 'a', 'b']);
    stop();
  });

  it('loads the images again on reload, replacing its list', async () => {
    const { live, stop } = mount();
    page.resolve({ total: 1, images: [img('1')] });
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(1);
    page = deferred<LabelImagePage>();
    handlers.reload?.({});
    page.resolve({ total: 3, images: [img('1', 'labeled'), img('2'), img('3')] });
    await vi.advanceTimersByTimeAsync(0);
    expect(loads).toHaveBeenCalledTimes(2);
    expect(live.images.map((i) => `${i.id} ${i.status}`)).toEqual(['1 labeled', '2 pending', '3 pending']);
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

  it('loads again when an image was removed while it loaded page after page, so no image is skipped', async () => {
    // 1,001 images: after the first page one of them is removed, the last one moves to offset 999 and the second
    // page (from 1,000) comes back empty. The removal cannot bring it back: the images load again.
    let server = Array.from({ length: 1001 }, (_, i) => img(String(i + 1)));
    let calls = 0;
    hub.api = {
      ...hub.api,
      labelImages: async (_id: string, q: { offset?: number; limit?: number } = {}) => {
        await Promise.resolve();
        calls++;
        const page = { total: server.length, images: server.slice(q.offset ?? 0, (q.offset ?? 0) + (q.limit ?? 100)) };
        if (calls === 1) {
          server = server.filter((i) => i.id !== '5');
          handlers.removed?.({ id: '5' });
        }
        return page;
      },
    } as Api;
    const { live, stop } = mount();
    await vi.advanceTimersByTimeAsync(0);
    expect(live.images).toHaveLength(1000);
    expect(live.images.some((i) => i.id === '5')).toBe(false);
    expect(live.images.at(-1)?.id).toBe('1001');
    stop();
  });

  it("shows an action's answer, except for an image the stream told of after the action began", async () => {
    const { live, stop } = mount();
    page.resolve({ total: 3, images: [img('1'), img('2'), img('3')] });
    await vi.advanceTimersByTimeAsync(0);
    // An upload in several requests: image 2 was removed while the later ones were sent.
    let mark = live.mark();
    handlers.removed?.({ id: '2' });
    live.put([img('2'), img('4')], mark);
    expect(live.images.map((i) => i.id)).toEqual(['1', '3', '4']);
    // A slow save: the stream has a newer state of image 1 by the time its answer comes.
    mark = live.mark();
    handlers.image?.(img('1', 'labeling'));
    live.put([img('1', 'labeled')], mark);
    expect(live.images[0]?.status).toBe('labeling');
    // Nothing heard since: the answer shows.
    mark = live.mark();
    live.put([img('3', 'labeled')], mark);
    expect(live.images.map((i) => `${i.id} ${i.status}`)).toEqual(['1 labeling', '3 labeled', '4 pending']);
    stop();
  });

  it('applies an answer and a removal over a load that was in flight, which may predate them', async () => {
    const { live, stop } = mount();
    page.resolve({ total: 2, images: [img('1'), img('2')] });
    await vi.advanceTimersByTimeAsync(0);
    page = deferred<LabelImagePage>();
    handlers.reload?.({});
    live.put([img('1', 'labeled')], live.mark());
    live.drop('2');
    page.resolve({ total: 2, images: [img('1'), img('2')] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.images.map((i) => `${i.id} ${i.status}`)).toEqual(['1 labeled']);
    stop();
  });

  it('does not let an answer bring back what a list asked for since the action began no longer has', async () => {
    const { live, stop } = mount();
    page.resolve({ total: 1, images: [img('1')] });
    await vi.advanceTimersByTimeAsync(0);
    // An upload in several requests begins; its first request adds image 2. The stream drops, someone else removes
    // image 2, the stream comes back and the list is loaded again (without image 2).
    const mark = live.mark();
    page = deferred<LabelImagePage>();
    handlers.reload?.({});
    page.resolve({ total: 1, images: [img('1')] });
    await vi.advanceTimersByTimeAsync(0);
    // The upload's answer, with image 2 in it, comes last: the list is newer.
    live.put([img('2')], mark);
    expect(live.images.map((i) => i.id)).toEqual(['1']);

    // The same while the newer list is still on its way: the answer shows until the list replaces it.
    const again = live.mark();
    page = deferred<LabelImagePage>();
    handlers.reload?.({});
    live.put([img('3')], again);
    expect(live.images.map((i) => i.id)).toEqual(['1', '3']);
    page.resolve({ total: 1, images: [img('1')] });
    await vi.advanceTimersByTimeAsync(0);
    expect(live.images.map((i) => i.id)).toEqual(['1']);
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
    const stop = root(() => {
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
