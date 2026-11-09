// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import Inspector from './Inspector.svelte';
import { ImagePager } from './pager.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

beforeEach(() => {
  hub.api = new MockApi('normal', 0);
  hub.mode = 'mock';
  // happy-dom has no modal dialogs: opening is enough for the content to render.
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  window.matchMedia ??= (() => ({ matches: false, addEventListener() {}, removeEventListener() {} })) as unknown as typeof window.matchMedia;
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('Inspector labels', () => {
  it('keeps hover and selection apart: a clicked label is selected (aria-current) until clicked again', async () => {
    const dataset = await hub.api.dataset('ds_fasteners');
    const pager = new ImagePager(dataset.id, undefined, undefined);
    const first = await pager.at(0);
    expect(first?.boxes.length).toBeGreaterThan(1);
    c = mount(Inspector, {
      target,
      props: { pager, classes: dataset.classes, imageId: first!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
    });
    const labels = () => [...target.querySelectorAll<HTMLButtonElement>('.labels button')];
    await vi.waitFor(() => {
      flushSync();
      if (labels().length < 2) throw new Error('labels not rendered yet');
    });
    const [a, b] = labels();
    b!.dispatchEvent(new PointerEvent('pointerenter'));
    flushSync();
    expect(b!.classList.contains('hot')).toBe(true);
    expect(b!.classList.contains('on')).toBe(false);
    expect(b!.getAttribute('aria-current')).toBeNull();

    a!.click();
    flushSync();
    expect(a!.classList.contains('on')).toBe(true);
    expect(a!.getAttribute('aria-current')).toBe('true');
    expect(target.querySelectorAll('.labels [aria-current]')).toHaveLength(1);

    a!.click();
    flushSync();
    expect(a!.getAttribute('aria-current')).toBeNull();
    pager.dispose();
  });
});
