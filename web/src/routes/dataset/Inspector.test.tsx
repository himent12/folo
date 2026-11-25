// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
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
  it("lists only the image's classes in the legend of a dataset with many (studio-data F3: COCO has 80)", async () => {
    const dataset = await hub.api.dataset('ds_fasteners');
    const pager = new ImagePager(dataset.id, undefined, undefined);
    const first = await pager.at(0);
    const many = [...dataset.classes, ...Array.from({ length: 80 - dataset.classes.length }, (_, i) => `extra ${i}`)];
    const here = new Set(first!.boxes.map((b) => b.class));
    c = mount(Inspector, {
      target,
      props: { pager, classes: many, imageId: first!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
    });
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('.legend li')) throw new Error('legend not rendered yet');
    });
    const legend = [...target.querySelectorAll('.legend li')].map((l) => l.textContent?.trim());
    expect(legend).toHaveLength(here.size);
    expect(target.querySelector('.others')?.textContent).toBe(`${80 - here.size} other classes, none in this image`);
  });

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

  it('drops the neighbour prefetches still loading when it closes (stress s1-datasets-2)', async () => {
    const Original = globalThis.Image;
    const made: HTMLImageElement[] = [];
    // A full-size image the browser is still fetching.
    globalThis.Image = class extends Original {
      constructor() {
        super();
        made.push(this);
      }
      override get complete() {
        return false;
      }
    } as typeof Image;
    try {
      hub.api.imageUrl = (_ds: string, id: string) => Promise.resolve(`/api/datasets/ds_fasteners/images/${id}`);
      const dataset = await hub.api.dataset('ds_fasteners');
      const pager = new ImagePager(dataset.id, undefined, undefined);
      const second = await pager.at(1);
      await pager.at(2);
      c = mount(Inspector, {
        target,
        props: { pager, classes: dataset.classes, imageId: second!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
      });
      await vi.waitFor(() => {
        flushSync();
        if (made.filter((im) => im.getAttribute('src')).length < 2) throw new Error('neighbours not prefetched yet');
      });
      unmount(c);
      c = undefined;
      expect(made.filter((im) => im.hasAttribute('src'))).toHaveLength(0);
      pager.dispose();
    } finally {
      globalThis.Image = Original;
    }
  });

  it('fades its key hint off the image once the viewer is used, and keeps the split on one line', async () => {
    const dataset = await hub.api.dataset('ds_fasteners');
    const pager = new ImagePager(dataset.id, undefined, undefined);
    const first = await pager.at(0);
    c = mount(Inspector, {
      target,
      props: { pager, classes: dataset.classes, imageId: first!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
    });
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('.split')) throw new Error('not rendered yet');
    });
    const hint = target.querySelector('.hint')!;
    expect(hint.classList.contains('touched')).toBe(false);
    target.querySelector<HTMLElement>('.viewport')!.dispatchEvent(new KeyboardEvent('keydown', { key: '+', bubbles: true }));
    flushSync();
    expect(hint.classList.contains('touched')).toBe(true);
    // "train" and, beside it (hidden on phones), " split".
    expect(target.querySelector('.split .split-word')?.textContent).toBe(' split');
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

  it('toggles bounding boxes with B and its toolbar button, and restores them when picking a label', async () => {
    const cw = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientWidth');
    const chh = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight');
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', { configurable: true, get: () => 800 });
    Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => 600 });
    try {
      hub.api.imageUrl = () => Promise.resolve('data:image/png;base64,');
      const dataset = await hub.api.dataset('ds_fasteners');
      const pager = new ImagePager(dataset.id, undefined, undefined);
      const first = await pager.at(0);
      c = mount(Inspector, {
        target,
        props: { pager, classes: dataset.classes, imageId: first!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
      });
      const img = await vi.waitFor(() => {
        flushSync();
        const el = target.querySelector<HTMLImageElement>('.stage img');
        if (!el) throw new Error('image not rendered yet');
        return el;
      });
      img.dispatchEvent(new Event('load'));
      flushSync();

      const btn = () => target.querySelector<HTMLButtonElement>('.head-zoom button[aria-keyshortcuts="B"]')!;
      expect(btn().getAttribute('aria-pressed')).toBe('true');
      expect(btn().getAttribute('aria-label')).toBe('Show boxes');
      expect(target.querySelectorAll('.stage .box').length).toBeGreaterThan(0);

      // Press B: boxes hide, button keeps stable accessible name and flips aria-pressed, viewport notes boxes hidden.
      target.querySelector<HTMLElement>('.viewport')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'b', bubbles: true }));
      flushSync();
      expect(btn().getAttribute('aria-pressed')).toBe('false');
      expect(btn().getAttribute('aria-label')).toBe('Show boxes');
      expect(target.querySelectorAll('.stage .box')).toHaveLength(0);
      expect(target.querySelector('.viewport')?.getAttribute('aria-label')).toContain('(boxes hidden)');

      // Clicking the toolbar button restores boxes.
      btn().click();
      flushSync();
      expect(btn().getAttribute('aria-pressed')).toBe('true');
      expect(target.querySelectorAll('.stage .box').length).toBeGreaterThan(0);

      // Selecting a label, hiding boxes with B, then picking the SAME label again restores boxes instead of leaving them hidden.
      const firstLabel = target.querySelector<HTMLButtonElement>('.labels button')!;
      firstLabel.click();
      flushSync();
      expect(firstLabel.getAttribute('aria-current')).toBe('true');
      btn().click();
      flushSync();
      expect(target.querySelectorAll('.stage .box')).toHaveLength(0);
      firstLabel.click();
      flushSync();
      expect(btn().getAttribute('aria-pressed')).toBe('true');
      expect(firstLabel.getAttribute('aria-current')).toBe('true');
      expect(target.querySelectorAll('.stage .box').length).toBeGreaterThan(0);
      pager.dispose();
    } finally {
      if (cw) Object.defineProperty(HTMLElement.prototype, 'clientWidth', cw);
      if (chh) Object.defineProperty(HTMLElement.prototype, 'clientHeight', chh);
    }
  });

  it('zooms in on a clean single click at Fit without breaking drag or double-click', async () => {
    const dataset = await hub.api.dataset('ds_fasteners');
    const pager = new ImagePager(dataset.id, undefined, undefined);
    const first = await pager.at(0);
    c = mount(Inspector, {
      target,
      props: { pager, classes: dataset.classes, imageId: first!.id, only: null, onnavigate: () => {}, onclose: () => {}, onmissing: () => {} },
    });
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('.stage')) throw new Error('stage not rendered');
    });
    const vp = target.querySelector<HTMLElement>('.viewport')!;
    const zoomBtn = () => target.querySelector<HTMLButtonElement>('.head-zoom button.zoom')!;
    expect(zoomBtn().textContent).toBe('Fit');

    // A drag (> 4px movement) at Fit does not trigger click-to-zoom.
    vp.dispatchEvent(new PointerEvent('pointerdown', { pointerId: 1, button: 0, pointerType: 'mouse', clientX: 100, clientY: 100, bubbles: true }));
    vp.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1, button: 0, pointerType: 'mouse', clientX: 120, clientY: 100, bubbles: true }));
    vp.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1, button: 0, pointerType: 'mouse', clientX: 120, clientY: 100, bubbles: true }));
    flushSync();
    expect(zoomBtn().textContent).toBe('Fit');

    // A pointerup > 4px away without any intervening pointermove does not trigger click-to-zoom.
    vp.dispatchEvent(new PointerEvent('pointerdown', { pointerId: 2, button: 0, pointerType: 'mouse', clientX: 100, clientY: 100, bubbles: true }));
    vp.dispatchEvent(new PointerEvent('pointerup', { pointerId: 2, button: 0, pointerType: 'mouse', clientX: 112, clientY: 100, bubbles: true }));
    flushSync();
    expect(zoomBtn().textContent).toBe('Fit');

    // A touch tap does not trigger mouse click-to-zoom (touch uses pinch / double-tap).
    vp.dispatchEvent(new PointerEvent('pointerdown', { pointerId: 3, button: 0, pointerType: 'touch', clientX: 100, clientY: 100, bubbles: true }));
    vp.dispatchEvent(new PointerEvent('pointerup', { pointerId: 3, button: 0, pointerType: 'touch', clientX: 100, clientY: 100, bubbles: true }));
    flushSync();
    expect(zoomBtn().textContent).toBe('Fit');

    // A clean mouse click at Fit zooms in immediately.
    vp.dispatchEvent(new PointerEvent('pointerdown', { pointerId: 4, button: 0, pointerType: 'mouse', clientX: 120, clientY: 80, bubbles: true }));
    vp.dispatchEvent(new PointerEvent('pointerup', { pointerId: 4, button: 0, pointerType: 'mouse', clientX: 121, clientY: 81, bubbles: true }));
    vp.dispatchEvent(new MouseEvent('click', { detail: 1, clientX: 121, clientY: 81, bubbles: true }));
    flushSync();
    expect(vp.classList.contains('zoomed')).toBe(true);
    const zoomedOnce = zoomBtn().textContent;
    expect(zoomedOnce).not.toBe('Fit');

    // A native dblclick completing that same click sequence (even after > 350ms) does not double-zoom past 2.5x.
    vp.dispatchEvent(new MouseEvent('click', { detail: 2, clientX: 121, clientY: 81, bubbles: true }));
    vp.dispatchEvent(new MouseEvent('dblclick', { detail: 2, clientX: 121, clientY: 81, bubbles: true }));
    flushSync();
    expect(zoomBtn().textContent).toBe(zoomedOnce);

    // A subsequent deliberate double-click while zoomed (starting with a fresh click detail=1) zooms further.
    vp.dispatchEvent(new MouseEvent('click', { detail: 1, clientX: 121, clientY: 81, bubbles: true }));
    vp.dispatchEvent(new MouseEvent('click', { detail: 2, clientX: 121, clientY: 81, bubbles: true }));
    vp.dispatchEvent(new MouseEvent('dblclick', { detail: 2, clientX: 121, clientY: 81, bubbles: true }));
    flushSync();
    expect(zoomBtn().textContent).not.toBe(zoomedOnce);
    expect(zoomBtn().textContent).not.toBe('Fit');

    // Pressing 0 returns cleanly to Fit.
    vp.dispatchEvent(new KeyboardEvent('keydown', { key: '0', bubbles: true }));
    flushSync();
    expect(zoomBtn().textContent).toBe('Fit');
    expect(vp.classList.contains('zoomed')).toBe(false);
    pager.dispose();
  });

  it('accumulates rapid J/K presses across uncached async pages and discards stale or post-close resolutions', async () => {
    const dataset = await hub.api.dataset('ds_fasteners');
    const pager = new ImagePager(dataset.id, undefined, undefined);
    const img0 = (await pager.at(0))!;
    const img1 = (await pager.at(1))!;
    const img2 = (await pager.at(2))!;

    const resolvers = new Map<number, (v: typeof img0) => void>();
    vi.spyOn(pager, 'at').mockImplementation((i: number) => {
      return new Promise((resolve) => {
        resolvers.set(i, resolve);
      });
    });

    const imageId = signal<string | null>(img0.id);
    const onnavigate = vi.fn((nextId: string) => imageId.set(nextId));
    c = render(
      () => (
        <Inspector
          pager={pager}
          classes={dataset.classes}
          imageId={imageId.get()}
          only={null}
          onnavigate={onnavigate}
          onclose={() => imageId.set(null)}
          onmissing={() => {}}
        />
      ),
      target,
    );
    await vi.waitFor(() => {
      flushSync();
      if (!target.querySelector('.viewport')) throw new Error('waiting for inspector');
    });
    const dialog = target.querySelector<HTMLDialogElement>('dialog')!;

    // Two rapid J presses before any page responds accumulate 0 -> 1 -> 2.
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true }));
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true }));
    expect(resolvers.has(1)).toBe(true);
    expect(resolvers.has(2)).toBe(true);

    // Resolve the newer target (2) first, then the stale earlier one (1): only img2 navigates.
    resolvers.get(2)!(img2);
    await Promise.resolve();
    flushSync();
    expect(onnavigate).toHaveBeenCalledTimes(1);
    expect(onnavigate).toHaveBeenLastCalledWith(img2.id);

    resolvers.get(1)!(img1);
    await Promise.resolve();
    flushSync();
    expect(onnavigate).toHaveBeenCalledTimes(1);

    // Stepping K then immediately J back to img2 cancels the pending fetch for img1.
    onnavigate.mockClear();
    resolvers.clear();
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', bubbles: true }));
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'j', bubbles: true }));
    expect(resolvers.has(1)).toBe(true);
    resolvers.get(1)!(img1);
    await Promise.resolve();
    flushSync();
    expect(onnavigate).not.toHaveBeenCalled();

    // Closing the inspector while a step is in flight discards the late resolution.
    resolvers.clear();
    dialog.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', bubbles: true }));
    expect(resolvers.has(1)).toBe(true);
    imageId.set(null);
    flushSync();
    resolvers.get(1)!(img1);
    await Promise.resolve();
    flushSync();
    expect(onnavigate).not.toHaveBeenCalled();
    pager.dispose();
  });
});
