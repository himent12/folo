// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import Inspector from './Inspector';
import { ImagePager } from './pager';

/** Port of `mount(C, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = <P extends object>(C: (p: P) => JSX.Element, o: { target: HTMLElement; props: P }) => render(() => C(o.props), o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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

  it('never leaves a hover behind a tap, and keeps Shift + arrows for panning', async () => {
    const dataset = await hub.api.dataset('ds_fasteners');
    const pager = new ImagePager(dataset.id, undefined, undefined);
    const first = await pager.at(0);
    const onnavigate = vi.fn();
    c = mount(Inspector, {
      target,
      props: { pager, classes: dataset.classes, imageId: first!.id, only: null, onnavigate, onclose: () => {}, onmissing: () => {} },
    });
    const labels = () => [...target.querySelectorAll<HTMLButtonElement>('.labels button')];
    await vi.waitFor(() => {
      flushSync();
      if (labels().length < 2) throw new Error('labels not rendered yet');
    });
    labels()[1]!.dispatchEvent(new PointerEvent('pointerenter', { pointerType: 'touch' }));
    flushSync();
    expect(labels()[1]!.classList.contains('hot')).toBe(false);

    // The canvas is focusable and described by its keys; Shift + arrows pan it and never change the image.
    const canvas = target.querySelector<HTMLElement>('.viewport')!;
    expect(canvas.tabIndex).toBe(0);
    expect(document.getElementById(canvas.getAttribute('aria-describedby')!)?.textContent).toMatch(/Shift and arrows: pan/);
    canvas.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', shiftKey: true, bubbles: true }));
    await new Promise((r) => setTimeout(r, 20));
    expect(onnavigate).not.toHaveBeenCalled();
    canvas.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    await vi.waitFor(() => expect(onnavigate).toHaveBeenCalled());

    // The full path is copyable.
    expect(target.querySelector('dd.path button.copy')?.getAttribute('aria-label')).toBe(`Copy the path ${first!.file}`);
    pager.dispose();
  });

  it('draws the selected label as the selected box (the others recede) and the hovered one apart from it', async () => {
    // The boxes are placed in screen pixels: give every element a size, since this DOM does no layout.
    const cw = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
    const chh = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight');
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 800 });
    Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => 600 });
    try {
      // The mock renders its images on a canvas, which this DOM lacks: any URL will do.
      hub.api.imageUrl = () => Promise.resolve('data:image/png;base64,');
      const dataset = await hub.api.dataset('ds_fasteners');
      const pager = new ImagePager(dataset.id, undefined, undefined);
      const first = await pager.at(0);
      expect(first?.boxes.length).toBeGreaterThan(2);
      c = mount(Inspector, {
        target,
        props: { pager, classes: dataset.classes, imageId: first!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
      });
      // The boxes draw once the image has loaded.
      const img = await vi.waitFor(() => {
        flushSync();
        const el = target.querySelector<HTMLImageElement>('.stage img');
        if (!el) throw new Error('image not rendered yet');
        return el;
      });
      img.dispatchEvent(new Event('load'));
      flushSync();
      const box = (i: number) => target.querySelector<HTMLElement>(`.stage .box[data-box="${i}"]`)!;
      const labels = [...target.querySelectorAll<HTMLButtonElement>('.labels button')];
      expect(box(0)).not.toBeNull();

      labels[0]!.click();
      flushSync();
      expect(box(0).classList.contains('selected')).toBe(true);
      for (let i = 1; i < first!.boxes.length; i++) expect(box(i).classList.contains('recede')).toBe(true);

      // Hovering another label: that box takes its hover fill and steps forward; the selection keeps its stroke.
      labels[0]!.dispatchEvent(new FocusEvent('blur'));
      labels[2]!.dispatchEvent(new PointerEvent('pointerenter'));
      flushSync();
      expect(box(2).classList.contains('active')).toBe(true);
      expect(box(2).classList.contains('selected')).toBe(false);
      expect(box(2).classList.contains('recede')).toBe(false);
      expect(box(0).classList.contains('selected')).toBe(true);
      expect(box(0).classList.contains('dim')).toBe(false);
      expect(box(1).classList.contains('recede')).toBe(true);

      // Let go of the selection: nothing recedes.
      labels[0]!.click();
      labels[0]!.dispatchEvent(new PointerEvent('pointerleave', { bubbles: true }));
      target.querySelector('.labels')!.dispatchEvent(new PointerEvent('pointerleave'));
      flushSync();
      expect(target.querySelectorAll('.stage .box.selected, .stage .box.recede')).toHaveLength(0);
      pager.dispose();
    } finally {
      if (cw) Object.defineProperty(HTMLElement.prototype, 'clientWidth', cw);
      if (chh) Object.defineProperty(HTMLElement.prototype, 'clientHeight', chh);
    }
  });
});
