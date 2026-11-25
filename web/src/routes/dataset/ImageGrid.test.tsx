// @vitest-environment happy-dom
import { flush } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import type { Api } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { DatasetImage } from '$lib/api/types';
import ImageGrid from './ImageGrid';
import { ImagePager, PAGE } from './pager';

const makeImg = (idx: number): DatasetImage => ({
  id: `img_${idx}`,
  file: `img_${idx}.jpg`,
  split: 'train',
  width: 640,
  height: 480,
  boxes: [],
});

describe('ImageGrid visible-over-overscan thumbnail priority', () => {
  const saved = hub.api;
  let gridTop = 0;

  beforeEach(() => {
    vi.useFakeTimers();
    gridTop = 0;
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('grid') ? 800 : 0;
    });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const top = this.classList.contains('grid') ? gridTop : 0;
      return { top, bottom: top, left: 0, right: 0, width: 800, height: 0, x: 0, y: top, toJSON: () => ({}) } as DOMRect;
    });
    vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(460);
  });

  afterEach(() => {
    hub.api = saved;
    vi.restoreAllMocks();
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('requests visible row thumbnails immediately and defers overscan thumbnails by one frame', async () => {
    const requested: string[] = [];
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        return { total: 400, images: Array.from({ length: lim }, (_, k) => makeImg(off + k)) };
      },
      imageUrl: async (_ds, id) => {
        requested.push(id);
        return `data:image/png;base64,${id}`;
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_1', undefined, undefined);
    await pager.at(0);
    await pager.at(100);
    const { unmount } = render(() => (
      <ImageGrid pager={pager} classes={['car']} density="m" boxes={false} only={null} onopen={() => {}} />
    ));
    try {
      // Let initial layout settle at row 0.
      flush();
      await vi.advanceTimersByTimeAsync(40);
      flush();
      requested.length = 0;
      // Jump to row 10 (4 cols of 194x146 + 8px gap = 154px/row; rows 10..12 = items 40..51 in view, rows 8..9 & 13..14 overscan).
      gridTop = -10 * 154;
      window.dispatchEvent(new Event('scroll'));
      // Advance just enough for TileGrid's scroll rAF to measure and mount the new rows (one frame).
      await vi.advanceTimersByTimeAsync(16);
      flush();
      const beforeOverscanFrame = [...requested];
      expect(beforeOverscanFrame.length).toBeGreaterThan(0);
      expect(beforeOverscanFrame.every((id) => Number(id.replace('img_', '')) >= 40 && Number(id.replace('img_', '')) < 52)).toBe(true);
      // After one more frame, overscan tiles warm up too.
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(requested.length).toBeGreaterThan(beforeOverscanFrame.length);
      expect(requested.some((id) => Number(id.replace('img_', '')) < 40)).toBe(true);
      expect(requested.some((id) => Number(id.replace('img_', '')) >= 52)).toBe(true);
    } finally {
      unmount();
    }
  });

  it('keeps visible priority over leading overscan when uncached page metadata resolves asynchronously', async () => {
    const requested: string[] = [];
    let releasePage!: () => void;
    const gate = new Promise<void>((r) => (releasePage = r));
    const api: Partial<Api> = {
      datasetImages: async (_id, q) => {
        const off = q.offset ?? 0;
        const lim = q.limit ?? PAGE;
        if (off === 200) await gate;
        return { total: 400, images: Array.from({ length: lim }, (_, k) => makeImg(off + k)) };
      },
      imageUrl: async (_ds, id) => {
        requested.push(id);
        return `data:image/png;base64,${id}`;
      },
    };
    hub.api = api as Api;
    const pager = new ImagePager('ds_1', undefined, undefined);
    await pager.at(0);
    const { unmount } = render(() => (
      <ImageGrid pager={pager} classes={['car']} density="m" boxes={false} only={null} onopen={() => {}} />
    ));
    try {
      flush();
      await vi.advanceTimersByTimeAsync(40);
      flush();
      requested.length = 0;
      // Jump into uncached page 2 (rows 55..57 = items 220..231 in view; rows 53..54 = items 212..219 leading overscan).
      gridTop = -55 * 154;
      window.dispatchEvent(new Event('scroll'));
      // Let multiple frames pass while page 2 is still in flight.
      await vi.advanceTimersByTimeAsync(64);
      flush();
      expect(requested).toEqual([]);
      // Now page 2 metadata resolves: visible tiles must request immediately before leading overscan tiles.
      releasePage();
      await vi.advanceTimersByTimeAsync(0);
      flush();
      const immediate = [...requested];
      expect(immediate.length).toBeGreaterThan(0);
      expect(immediate.every((id) => Number(id.replace('img_', '')) >= 220 && Number(id.replace('img_', '')) < 232)).toBe(true);
      // One frame later, overscan tiles warm up.
      await vi.advanceTimersByTimeAsync(20);
      flush();
      expect(requested.some((id) => Number(id.replace('img_', '')) < 220)).toBe(true);
    } finally {
      unmount();
    }
  });
});
