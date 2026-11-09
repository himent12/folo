// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import DatasetImage from './DatasetImage.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

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

  it('says so when the image itself fails to load', async () => {
    render(async () => 'data:,c');
    const img = await until(() => target.querySelector('img'), 'the image');
    img.dispatchEvent(new Event('error'));
    await until(() => target.querySelector('.failed'), 'the failure');
    expect(target.querySelector('img')).toBeNull();
    expect(target.querySelector('.box')).toBeNull();
  });

  it('inside a button, keeps Retry out of the tab order and the click from the tile', async () => {
    const url = render(async () => {
      throw new Error('offline');
    }, 'button');
    const onclick = vi.fn();
    target.addEventListener('click', onclick);
    const failed = await until(() => target.querySelector('.failed'), 'the failure');
    expect(failed.querySelector('button')).toBeNull();
    failed.querySelector<HTMLElement>('.retry')!.click();
    expect(onclick).not.toHaveBeenCalled();
    await until(() => url.mock.calls.length === 2, 'a second attempt');
  });
});
