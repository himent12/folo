// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { render as fwRender } from '../../../framework/testing';
import DatasetImage, { type DatasetImageProps } from './DatasetImage';

/** Port of `mount(DatasetImage, { target, props })`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof DatasetImage, o: { target: HTMLElement; props: DatasetImageProps }) => fwRender(() => <DatasetImage {...o.props} />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

let target: HTMLElement;
let c: { unmount(): void } | undefined;

// The boxes are placed in screen pixels: give every element a size, since this DOM does no layout.
const size = { clientWidth: Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth'), clientHeight: Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight') };
beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 200 });
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => 200 });
});
afterAll(() => {
  if (size.clientWidth) Object.defineProperty(HTMLElement.prototype, 'clientWidth', size.clientWidth);
  if (size.clientHeight) Object.defineProperty(HTMLElement.prototype, 'clientHeight', size.clientHeight);
});

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 2000, interval: 5 },
  );
  return v as T;
}

function render(imageUrl: () => Promise<string>, inside?: string) {
  const api = new MockApi('normal', 0);
  api.imageUrl = vi.fn(imageUrl);
  hub.api = api;
  target = document.createElement(inside ?? 'div');
  document.body.appendChild(target);
  c = mount(DatasetImage, {
    target,
    props: { datasetId: 'ds', imageId: 'img', width: 100, height: 100, classes: ['a'], boxes: [{ class: 0, x1: 10, y1: 10, x2: 50, y2: 50 }] },
  });
  flushSync();
  return api.imageUrl as ReturnType<typeof vi.fn>;
}

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('DatasetImage', () => {
  it('holds a skeleton and draws no boxes until the image has loaded', async () => {
    render(async () => 'data:,a');
    const img = await until(() => target.querySelector('img'), 'the image');
    expect(target.querySelector('.cover .sk')).not.toBeNull();
    expect(target.querySelector('.box')).toBeNull();
    expect(target.querySelector('.dsimg')?.getAttribute('aria-busy')).toBe('true');
    img.dispatchEvent(new Event('load'));
    await until(() => target.querySelector('.box'), 'the boxes');
    expect(target.querySelector('.cover')).toBeNull();
    expect(target.querySelector('.dsimg')?.getAttribute('aria-busy')).toBe('false');
  });

  it('says the image is unavailable when its URL cannot be resolved, and retries', async () => {
    let fail = true;
    const url = render(async () => {
      if (fail) throw new Error('offline');
      return 'data:,b';
    });
    const failed = await until(() => target.querySelector('.failed'), 'the failure');
    expect(failed.textContent).toContain('Image unavailable');
    expect(target.querySelector('.box')).toBeNull();
    fail = false;
    const retry = failed.querySelector<HTMLButtonElement>('button.retry')!;
    expect(retry.textContent).toBe('Retry');
    retry.click();
    const img = await until(() => target.querySelector('img'), 'the image');
    expect(url).toHaveBeenCalledTimes(2);
    expect(target.querySelector('.failed')).toBeNull();
    img.dispatchEvent(new Event('load'));
    await until(() => target.querySelector('.box'), 'the boxes');
  });

  it('tries a failed image again by itself when the server answers again after an outage', async () => {
    const back = signal(0);
    const spy = vi.spyOn(hub, 'back', 'get').mockImplementation(() => back.get());
    try {
      let fail = true;
      const url = render(async () => {
        if (fail) throw new Error('offline');
        return 'data:,c';
      });
      await until(() => target.querySelector('.failed'), 'the failure');
      fail = false;
      back.set(1);
      const img = await until(() => target.querySelector('img'), 'the image');
      expect(url).toHaveBeenCalledTimes(2);
      img.dispatchEvent(new Event('load'));
      await until(() => target.querySelector('.box'), 'the boxes');
      // A loaded image is left alone by the next recovery.
      back.set(2);
      flushSync();
      expect(url).toHaveBeenCalledTimes(2);
    } finally {
      spy.mockRestore();
    }
  });

  it('tries again an image that was still loading when the server came back and failed after (stress s1-datasets-8)', async () => {
    const back = signal(0);
    const spy = vi.spyOn(hub, 'back', 'get').mockImplementation(() => back.get());
    try {
      const url = render(async () => 'data:,d');
      const first = await until(() => target.querySelector('img'), 'the image');
      // The recovery is seen while the image loads; the browser reports its (outage-time) failure only after.
      back.set(1);
      flushSync();
      expect(url).toHaveBeenCalledTimes(1);
      first.dispatchEvent(new Event('error'));
      const second = await until(() => {
        const im = target.querySelector('img');
        return im !== first && im;
      }, 'the image again');
      expect(url).toHaveBeenCalledTimes(2);
      expect(target.querySelector('.failed')).toBeNull();
      // A failure of an attempt made after the recovery is shown, not retried for ever.
      second.dispatchEvent(new Event('error'));
      await until(() => target.querySelector('.failed'), 'the failure');
      expect(url).toHaveBeenCalledTimes(2);
    } finally {
      spy.mockRestore();
    }
  });

  it('fetches the image again under a new dataset revision (the same id may name another file now)', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = vi.fn(async () => '/api/datasets/ds/images/0?thumb=256');
    hub.api = api;
    target = document.createElement('div');
    document.body.appendChild(target);
    const rev = signal(0);
    c = fwRender(() => <DatasetImage datasetId="ds" imageId="0" rev={rev.get()} width={100} height={100} classes={['a']} />, target);
    const src = () => target.querySelector('img')?.getAttribute('src');
    expect(await until(src, 'the image')).toBe('/api/datasets/ds/images/0?thumb=256');
    rev.set(2);
    await until(() => src()?.endsWith('&rev=2'), 'the new revision');
    expect(api.imageUrl).toHaveBeenCalledTimes(2);
  });

  it('stops loading an image that leaves the page: a retry, or the tile scrolled away (stress s1-datasets-2)', async () => {
    render(async () => '/api/datasets/ds/images/img?thumb=256');
    const first = await until(() => target.querySelector('img'), 'the image');
    expect(first.getAttribute('src')).toBe('/api/datasets/ds/images/img?thumb=256');
    // Chrome goes on loading a detached <img> that keeps its src: the one taken down must have none.
    first.dispatchEvent(new Event('error'));
    await until(() => target.querySelector('.failed'), 'the failure');
    expect(first.isConnected).toBe(false);
    expect(first.hasAttribute('src')).toBe(false);
    target.querySelector<HTMLButtonElement>('button.retry')!.click();
    const second = await until(() => target.querySelector('img'), 'the image again');
    expect(second).not.toBe(first);
    unmount(c!);
    c = undefined;
    expect(second.hasAttribute('src')).toBe(false);
  });

  it('says so when the image itself fails to load', async () => {
    render(async () => 'data:,c');
    const img = await until(() => target.querySelector('img'), 'the image');
    img.dispatchEvent(new Event('error'));
    await until(() => target.querySelector('.failed'), 'the failure');
    expect(target.querySelector('img')).toBeNull();
    expect(target.querySelector('.box')).toBeNull();
  });

  it('inside a button, the failed tile retries instead of picking, by pointer or keyboard', async () => {
    const url = render(async () => {
      throw new Error('offline');
    }, 'button');
    // The picker's handler, delegated as Svelte delegates onclick: it sees a click only once it bubbles past the tile.
    const onclick = vi.fn();
    document.body.addEventListener('click', onclick);
    const failed = await until(() => target.querySelector('.failed'), 'the failure');
    expect(failed.querySelector('button')).toBeNull();
    // The tile is the one control, and its name says what it does now.
    const tile = target;
    expect(tile.textContent).toContain('Image unavailable');
    expect(tile.textContent).toContain('Retry');
    expect(failed.querySelector('[aria-hidden="true"].retry')).toBeNull();
    // Enter / Space on a button fire click: activating the tile retries and never reaches the picker.
    tile.click();
    expect(onclick).not.toHaveBeenCalled();
    await until(() => url.mock.calls.length === 2, 'a second attempt');
    // Clicking the Retry label itself does the same.
    await until(() => target.querySelector('.failed'), 'the failure again');
    target.querySelector<HTMLElement>('.retry')!.click();
    expect(onclick).not.toHaveBeenCalled();
    await until(() => url.mock.calls.length === 3, 'a third attempt');
    document.body.removeEventListener('click', onclick);
  });

  it("says Retry in a labelled tile's own name while the image is unavailable, and restores it after", async () => {
    let fail = true;
    const url = render(async () => {
      if (fail) throw new Error('offline');
      return 'data:,a';
    }, 'button');
    target.setAttribute('aria-label', 'Use img_0004.jpg');
    // The label is read when the failure lands, so set it before: re-render by retrying once.
    await until(() => target.querySelector('.failed'), 'the failure');
    target.click();
    await until(() => url.mock.calls.length === 2 && target.querySelector('.failed'), 'the second failure');
    expect(target.getAttribute('aria-label')).toBe('Use img_0004.jpg: image unavailable, activate to retry');
    fail = false;
    target.click();
    await until(() => !target.querySelector('.failed'), 'the recovery');
    expect(target.getAttribute('aria-label')).toBe('Use img_0004.jpg');
  });
});
