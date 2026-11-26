import { computed, createRoot, effect, flush } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Api } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { DatasetImage } from '$lib/api/types';
import { ImagePager, MAX_CACHED_PAGES, PAGE } from './pager';

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

  it('isolates reactivity per page so loading page 1 does not wake readers of page 0', async () => {
    const mock = new MockApi('normal', 0);
    hub.api = { datasetImages: (id, q) => mock.datasetImages(id, q) } as Api;
    const pager = new ImagePager('ds_fasteners', undefined, undefined);
    let page0Runs = 0;
    let page1Runs = 0;
    let page0RevRuns = 0;
    const stop = createRoot((dispose) => {
      effect(() => {
        void pager.get(0);
        page0Runs++;
      });
      effect(() => {
        void pager.revisionOf(0);
        page0RevRuns++;
      });
      effect(() => {
        void pager.get(PAGE);
        page1Runs++;
      });
      return dispose;
    });
    try {
      flush();
      expect([page0Runs, page0RevRuns, page1Runs]).toEqual([1, 1, 1]);
      await pager.at(0);
      flush();
      expect([page0Runs, page0RevRuns, page1Runs]).toEqual([2, 2, 1]);
      // Loading page 1 must only wake page 1's reader, never page 0's.
      await pager.at(PAGE);
      flush();
      expect([page0Runs, page0RevRuns, page1Runs]).toEqual([2, 2, 2]);
    } finally {
      stop();
    }
  });

  it('evicts the least-recently-used page once the LRU cap is exceeded and cleans its index entries', async () => {
    const total = 1000;
    const reads: number[] = [];
    const makeImg = (idx: number): DatasetImage => ({
      id: `img_${idx}`,
      file: `img_${idx}.jpg`,
      split: 'train',
      width: 640,
      height: 480,
      boxes: [],
    });
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        reads.push(off);
        const end = Math.min(total, off + lim);
        return { total, images: Array.from({ length: end - off }, (_, k) => makeImg(off + k)) };
      },
    };
    hub.api = api as Api;
    expect(MAX_CACHED_PAGES).toBe(50);
    const pager = new ImagePager('ds_big', undefined, undefined, 0, 3);
    await pager.at(0); // cache: [0]
    await pager.at(PAGE); // cache: [0, 1]
    await pager.at(2 * PAGE); // cache: [0, 1, 2]
    expect(pager.cachedPages).toBe(3);
    // Touch page 0 so page 1 becomes the least-recently-used page.
    expect(pager.get(0)?.id).toBe('img_0');
    await pager.at(3 * PAGE); // evicts page 1; cache: [2, 0, 3]
    expect(pager.cachedPages).toBe(3);
    expect(pager.get(0)?.id).toBe('img_0');
    expect(pager.get(2 * PAGE)?.id).toBe(`img_${2 * PAGE}`);
    expect(pager.get(3 * PAGE)?.id).toBe(`img_${3 * PAGE}`);
    // Page 1 was evicted, and its index entries were removed too.
    expect(pager.get(PAGE)).toBeUndefined();
    expect(pager.indexOf(`img_${PAGE}`)).toBe(-1);
    // Fetching page 1 again loads it from the API cleanly.
    const before = reads.length;
    expect((await pager.at(PAGE))?.id).toBe(`img_${PAGE}`);
    expect(reads.length).toBe(before + 1);
    expect(pager.cachedPages).toBe(3);
  });

  it('requests visible pages before leading and trailing overscan pages in ensure()', async () => {
    const reads: number[] = [];
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        reads.push(off / PAGE);
        return {
          total: 1000,
          images: Array.from({ length: PAGE }, (_, k) => ({
            id: `img_${off + k}`,
            file: `img_${off + k}.jpg`,
            split: 'train',
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_big', undefined, undefined);
    await pager.at(0);
    reads.length = 0;
    // Rendered range spans pages 2..5, with viewport covering pages 3..4 (page 2 leading overscan, page 5 trailing).
    pager.ensure(2 * PAGE, 6 * PAGE, 3 * PAGE, 5 * PAGE);
    await vi.waitFor(() => expect(reads).toHaveLength(4));
    expect(reads).toEqual([3, 4, 2, 5]);
  });

  it('keeps observed visible pages cached and responsive across a missing-image find() scan and reload()', async () => {
    const total = 1200;
    let ver = 'v1';
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        const end = Math.min(total, off + lim);
        return {
          total,
          images: Array.from({ length: end - off }, (_, k) => ({
            id: `img_${off + k}`,
            file: `${ver}_${off + k}.jpg`,
            split: 'train' as const,
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_big', undefined, 0, 0, 3);
    let seen: string | undefined;
    const stop = createRoot((dispose) => {
      effect(() => {
        seen = pager.get(0)?.file;
      });
      return dispose;
    });
    try {
      await pager.at(0);
      flush();
      expect(seen).toBe('v1_0.jpg');
      // Full scan for a missing image must not evict page 0 or break its reactive subscription.
      expect(await pager.find('img_missing')).toBe(-1);
      expect(pager.get(0)?.file).toBe('v1_0.jpg');
      expect(pager.indexOf('img_0')).toBe(0);
      // Reload updates the observed page cleanly.
      ver = 'v2';
      pager.reload();
      flush();
      await vi.waitFor(() => {
        flush();
        expect(seen).toBe('v2_0.jpg');
      });
    } finally {
      stop();
    }
  });

  it('returns each concurrent at() result even when another response evicts its page under a 1-page cap', async () => {
    let releaseA!: () => void;
    let releaseB!: () => void;
    const gateA = new Promise<void>((r) => (releaseA = r));
    const gateB = new Promise<void>((r) => (releaseB = r));
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        if (off === 0) await gateA;
        if (off === PAGE) await gateB;
        return {
          total: 500,
          images: Array.from({ length: PAGE }, (_, k) => ({
            id: `img_${off + k}`,
            file: `img_${off + k}.jpg`,
            split: 'train' as const,
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_big', undefined, undefined, 0, 1);
    const a = pager.at(0);
    const b = pager.at(PAGE);
    releaseA();
    releaseB();
    const [imgA, imgB] = await Promise.all([a, b]);
    expect(imgA?.id).toBe('img_0');
    expect(imgB?.id).toBe(`img_${PAGE}`);
  });

  it('never self-evicts a newly stored page when all previously cached pages are observed at the cap', async () => {
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        return {
          total: 600,
          images: Array.from({ length: lim }, (_, k) => ({
            id: `img_${off + k}`,
            file: `img_${off + k}.jpg`,
            split: 'train' as const,
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_big', undefined, undefined, 0, 1);
    const stop = createRoot((dispose) => {
      effect(() => {
        void pager.get(0);
      });
      return dispose;
    });
    try {
      await pager.at(0);
      flush();
      // Page 0 is observed and fills maxPages=1; finding img_500 must keep page 5 (not self-evict it).
      expect(await pager.find('img_500')).toBe(500);
      expect(pager.indexOf('img_500')).toBe(500);
      expect(pager.get(500)?.id).toBe('img_500');
    } finally {
      stop();
    }
  });

  it('re-subscribes a retained computed after its consumer detaches and reattaches across a load and reload', async () => {
    let tag = 'v1';
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        return {
          total: 200,
          images: Array.from({ length: lim }, (_, k) => ({
            id: `img_${off + k}`,
            file: `${tag}_${off + k}.jpg`,
            split: 'train' as const,
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_detach', undefined, undefined);
    const first = computed(() => pager.get(0));
    const seen: (string | undefined)[] = [];

    // 1. Observe while page 0 is unloaded, then detach.
    const stop1 = createRoot((dispose) => {
      effect(() => {
        seen.push(first.get()?.file);
      });
      return dispose;
    });
    flush();
    expect(seen).toEqual([undefined]);
    stop1();

    // 2. Load page 0 while detached.
    await pager.at(0);
    flush();

    // 3. Reattach: must see the loaded image immediately and keep tracking future reloads.
    const stop2 = createRoot((dispose) => {
      effect(() => {
        seen.push(first.get()?.file);
      });
      return dispose;
    });
    try {
      flush();
      expect(seen.at(-1)).toBe('v1_0.jpg');
      tag = 'v2';
      pager.reload();
      flush();
      await Promise.resolve();
      await Promise.resolve();
      flush();
      expect(seen.at(-1)).toBe('v2_0.jpg');
    } finally {
      stop2();
    }
  });

  it('retries find() cleanly when reload() interrupts the initial page-0 load without a redundant scan', async () => {
    let releaseFirst!: () => void;
    let releaseSecond!: () => void;
    const firstGate = new Promise<void>((r) => (releaseFirst = r));
    const secondGate = new Promise<void>((r) => (releaseSecond = r));
    let calls = 0;
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        calls++;
        const my = calls;
        if (my === 1) await firstGate;
        if (my === 2) await secondGate;
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        return {
          total: 300,
          images: Array.from({ length: Math.min(lim, Math.max(0, 300 - off)) }, (_, k) => ({
            id: `alpha_${off + k}`,
            file: `alpha_${off + k}.jpg`,
            split: 'train' as const,
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_reload_find', undefined, 0);
    // Start finding an image whose name doesn't match the numeric guess: enters initial #load(0).
    const found = pager.find('alpha_3');
    pager.reload();
    // Release call 1 and flush pending microtasks completely while call 2 is still pending so `total` stays null.
    releaseFirst();
    await new Promise((r) => setTimeout(r, 0));
    expect(pager.total).toBeNull();
    releaseSecond();
    expect(await found).toBe(3);
    expect(calls).toBe(2);

    // Cold single-page class-filtered lookup must find an image on page 0 without skipping the only fresh page.
    let smallCalls = 0;
    const smallApi: Partial<Api> = {
      datasetImages: async () => {
        smallCalls++;
        return {
          total: 10,
          images: Array.from({ length: 10 }, (_, k) => ({
            id: `img_${k}`,
            file: `img_${k}.jpg`,
            split: 'train' as const,
            width: 640,
            height: 480,
            boxes: [],
          })),
        };
      },
    };
    hub.api = smallApi as Api;
    const coldClass = new ImagePager('ds_small_class', undefined, 0);
    expect(await coldClass.find('img_3')).toBe(3);
    expect(smallCalls).toBe(1);

    // Once disposed, further find() and uncached at() calls bail out without issuing new requests.
    const beforeDispose = calls;
    pager.dispose();
    expect(await pager.find('alpha_missing')).toBe(-1);
    expect(await pager.at(250)).toBeUndefined();
    expect(calls).toBe(beforeDispose);
  });

  it('dispose() resolves an in-flight find() scan to -1 and prevents an un-aborted page response from mutating or returning from at()', async () => {
    let resolvePage0!: (v: { total: number; images: DatasetImage[] }) => void;
    const img = (k: number): DatasetImage => ({
      id: `beta_${k}`,
      file: `beta_${k}.jpg`,
      split: 'train',
      width: 640,
      height: 480,
      boxes: [],
    });

    // Case 1: in-flight page load that ignores AbortSignal must not cache or return after dispose().
    hub.api = {
      datasetImages: () =>
        new Promise((resolve) => {
          resolvePage0 = resolve;
        }),
    } as unknown as Api;
    const p1 = new ImagePager('ds_dispose_page', undefined, undefined);
    const pendingAt = p1.at(0);
    p1.dispose();
    resolvePage0({ total: 10, images: [img(0)] });
    expect(await pendingAt).toBeUndefined();
    expect(p1.get(0)).toBeUndefined();
    expect(p1.total).toBeNull();

    // Case 2: in-flight find() scan aborted by dispose() resolves cleanly to -1 instead of rejecting with AbortError.
    let scanCalls = 0;
    let markScanStarted!: () => void;
    const scanStarted = new Promise<void>((r) => (markScanStarted = r));
    const scanApi: Partial<Api> = {
      datasetImages: (_id, _q, signal) => {
        scanCalls++;
        if (scanCalls === 1) return Promise.resolve({ total: 500, images: Array.from({ length: PAGE }, (_, k) => img(k)) });
        return new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
          markScanStarted();
        });
      },
    };
    hub.api = scanApi as Api;
    const p2 = new ImagePager('ds_dispose_scan', undefined, 0);
    const pendingFind = p2.find('beta_450');
    await scanStarted;
    expect(scanCalls).toBe(2);
    p2.dispose();
    expect(await pendingFind).toBe(-1);
  });
});
