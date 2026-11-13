import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Api } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { ImagePager } from './pager';

describe('ImagePager', () => {
  const saved = hub.api;
  afterEach(() => {
    hub.api = saved;
  });

  it('surfaces a failed later page and loads it again on retry', async () => {
    const mock = new MockApi('normal', 0);
    let failOnce = true;
    const api: Partial<Api> = {
      datasetImages: (id, q, s) => {
        if (q.offset === 100 && failOnce) {
          failOnce = false;
          return Promise.reject(new Error('flaky'));
        }
        void s;
        return mock.datasetImages(id, q);
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    await pager.at(0);
    expect(pager.total).toBe(2400);
    await pager.at(150);
    expect(pager.error).toBeTruthy();
    expect(pager.get(150)).toBeUndefined();
    pager.retry();
    await pager.at(150);
    expect(pager.error).toBeUndefined();
    expect(pager.get(150)?.id).toBeDefined();
  });

  it('reads a page again after reload(), showing the old one until then and dropping answers from before it', async () => {
    const mock = new MockApi('normal', 0);
    // The dataset on disk: its first image's file, changed by the test; an answer can be held back.
    let first = 'before.jpg';
    let hold: Promise<void> | null = null;
    const reads: number[] = [];
    const api: Partial<Api> = {
      datasetImages: async (id, q) => {
        const name = first;
        reads.push(q.offset ?? 0);
        const page = await mock.datasetImages(id, q);
        if (hold) await hold;
        return { ...page, images: page.images.map((im, k) => (k === 0 && q.offset === 0 ? { ...im, file: name } : im)) };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    await pager.at(0);
    expect(pager.get(0)?.file).toBe('before.jpg');

    // A read made before the change answers after reload(): it is dropped, not shown as the new state.
    let release!: () => void;
    hold = new Promise<void>((r) => (release = r));
    const early = pager.at(250);
    first = 'after.jpg';
    pager.reload();
    hold = null;
    // Until its new read answers, the page shows what it had (nothing blinks).
    expect(pager.get(0)?.file).toBe('before.jpg');
    expect(pager.indexOf(pager.get(0)!.id)).toBe(0);
    await vi.waitFor(() => expect(pager.get(0)?.file).toBe('after.jpg'));
    release();
    await early.catch(() => undefined);
    expect(pager.get(250)).toBeUndefined();
    // Only what was read again (page 0, by the reader above) was asked for: not every page ever loaded.
    expect(reads.filter((o) => o === 0)).toHaveLength(2);
    expect([pager.revision, pager.revisionOf(0), pager.revisionOf(250), pager.refreshed]).toEqual([1, 1, 0, 1]);
  });

  it('keeps a page whose read after reload() failed for Retry, which reads it again', async () => {
    const mock = new MockApi('normal', 0);
    let fail = false;
    const api: Partial<Api> = { datasetImages: (id, q) => (fail ? Promise.reject(new Error('offline')) : mock.datasetImages(id, q)) };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    await pager.at(0);
    fail = true;
    pager.reload();
    pager.get(0);
    await vi.waitFor(() => expect(pager.error).toBeTruthy());
    // What it showed stays; reading it again is up to Retry (no loop of failing reads meanwhile).
    expect(pager.get(0)?.id).toBeDefined();
    const read = vi.spyOn(api, 'datasetImages');
    fail = false;
    pager.retry();
    await vi.waitFor(() => expect(pager.error).toBeUndefined());
    expect(read).toHaveBeenCalledOnce();
  });

  it('reads a page that was on its way at reload() again, for the reader that waited for it', async () => {
    const mock = new MockApi('normal', 0);
    let release!: () => void;
    const hold = new Promise<void>((r) => (release = r));
    let first = true;
    const api: Partial<Api> = {
      datasetImages: async (id, q) => {
        if (q.offset === 100 && first) {
          first = false;
          await hold;
        }
        return mock.datasetImages(id, q);
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    await pager.at(0);
    pager.ensure(100, 120);
    pager.reload();
    release();
    // The row that asked before the change asks again as it reads (no new range is needed).
    expect(pager.get(100)).toBeUndefined();
    await vi.waitFor(() => expect(pager.get(100)?.id).toBeDefined());
  });

  it('does not take a place read before a change for an image, while its page cannot be read again', async () => {
    const mock = new MockApi('normal', 0);
    let fail = false;
    const api: Partial<Api> = { datasetImages: (id, q) => (fail ? Promise.reject(new Error('offline')) : mock.datasetImages(id, q)) };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    const last = await pager.at(99);
    fail = true;
    pager.reload();
    await expect(pager.find(last!.id)).rejects.toThrow('offline');
  });

  it('reads an empty filter again on reload(), where no row asks for a page', async () => {
    const mock = new MockApi('normal', 0);
    let none = true;
    const api: Partial<Api> = {
      datasetImages: async (id, q) => (none ? { total: 0, images: [] } : mock.datasetImages(id, q)),
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_fasteners', undefined, 2);
    await pager.at(0);
    expect(pager.total).toBe(0);
    none = false;
    pager.reload();
    await vi.waitFor(() => expect(pager.total).toBeGreaterThan(0));
  });
});
