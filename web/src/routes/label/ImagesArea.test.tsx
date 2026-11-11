// @vitest-environment happy-dom
// The job's virtualised grid does no wasted work: a tile is made once per image it shows, keyed by the image's id.
// Scrolling makes tiles only for the rows that come into view (the rows that stay keep theirs), an image replaced by a
// newer copy of itself (the agent labeled it) updates its tile in place without loading again, and when the images
// shift (one leaves the filter) the tiles move with them. Tiles are counted by their image requests (ui/DatasetImage
// asks for its URL when it is made, and again whenever its image changes).
import { flush, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import type { Api } from '$lib/api/client';
import { hub } from '$lib/api/index';
import type { LiveLabelJob } from '$lib/api/liveLabel';
import type { LabelImage } from '$lib/api/types';
import { gridView } from '$lib/ui/gridView';
import ImagesArea from './ImagesArea';
import { Uploads } from './uploads';

const img = (n: number): LabelImage => ({ id: `im${n}`, file: `im${n}.jpg`, width: 640, height: 480, status: 'pending', boxes: [], source: null, error: null });

/** The grid's top edge in the viewport (TileGrid reads it on scroll). */
let gridTop = 0;
const frame = () => new Promise((r) => requestAnimationFrame(() => r(undefined)));

describe('the job grid', () => {
  const saved = hub.api;
  let urls: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    gridTop = 0;
    gridView.density = 'm';
    urls = vi.fn(async (_job: string, id: string) => `data:image/png;base64,${id}`);
    hub.api = { labelImageUrl: urls } as unknown as Api;
    // A grid 800 px wide (4 columns of 194 x 146 at density M), in a 768 px tall window.
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('grid') ? 800 : 0;
    });
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const top = this.classList.contains('grid') ? gridTop : 0;
      return { top, bottom: top, left: 0, right: 0, width: 0, height: 0, x: 0, y: top, toJSON: () => ({}) } as DOMRect;
    });
    vi.spyOn(window, 'innerHeight', 'get').mockReturnValue(768);
  });
  afterEach(() => {
    hub.api = saved;
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  it('makes tiles only for the rows that come into view, updates an image in place, and moves tiles with their images', async () => {
    const images = signal(Array.from({ length: 300 }, (_, i) => img(i)));
    const live = { mark: () => 0, act: async () => undefined, put: () => {} } as unknown as LiveLabelJob;
    const { container, unmount } = render(() => (
      <ImagesArea
        jobId="lb_1"
        classes={['car']}
        images={images.get()}
        show="all"
        uploads={new Uploads(() => 'lb_1', live)}
        onshow={() => {}}
        onopen={() => {}}
        onbrowse={() => {}}
        onadd={() => {}}
      />
    ));
    flush();
    await frame();
    flush();
    const cells = () => container.querySelectorAll('.cell').length;
    // Rows 0-6 (5 in view, 2 of overscan): 28 tiles, each made once. (TileGrid's first pass, before it knows its
    // width, has 8 rows of one column: the eighth row's tile, im28, is made and dropped.)
    expect(cells()).toBe(28);
    expect(urls.mock.calls.map((c) => c[1]).sort()).toEqual([...Array.from({ length: 28 }, (_, i) => `im${i}`), 'im28'].sort());
    const made = urls.mock.calls.length;
    const kept = container.querySelector('[data-i="8"] .tile');

    // Three rows down: rows 1-9 are in the DOM. Rows 1-6 stay as they were; only rows 7-9 (12 tiles) are new.
    gridTop = -3 * 154;
    window.dispatchEvent(new Event('scroll'));
    await frame();
    flush();
    expect(cells()).toBe(36);
    expect(urls).toHaveBeenCalledTimes(made + 12);
    expect(container.querySelector('[data-i="8"] .tile')).toBe(kept);

    // The agent takes image 5: a new copy of it under the same id. Its tile shows the live mark, made again: never.
    images.set(images.peek().map((im, i) => (i === 5 ? { ...im, status: 'labeling' } : im)));
    flush();
    expect(urls).toHaveBeenCalledTimes(made + 12);
    expect(container.querySelector('[data-i="5"] .mark.live')?.textContent).toBe('Labeling');
    expect(container.querySelector('[data-i="8"] .tile')).toBe(kept);

    // Image 1 leaves (labeled, out of "To label"; removed): every image after it moves one cell back, and its tile with
    // it. Only the image that comes into the last cell of row 9 gets a tile made.
    images.set(images.peek().filter((_, i) => i !== 1));
    flush();
    await Promise.resolve();
    expect(urls).toHaveBeenCalledTimes(made + 13);
    expect(container.querySelector('[data-i="7"] .tile')).toBe(kept);
    expect(container.querySelector('[data-i="4"] .mark.live')?.textContent).toBe('Labeling');
    expect(container.querySelectorAll('.tile')).toHaveLength(36);
    unmount();
  });
});
