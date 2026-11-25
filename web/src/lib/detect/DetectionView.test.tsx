// @vitest-environment happy-dom
import { flush, signal } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import DetectionView, { type ViewBox } from './DetectionView';

const BOXES: ViewBox[] = [
  { class: 0, x1: 10, y1: 10, x2: 120, y2: 120, score: 0.91 },
  { class: 1, x1: 180, y1: 40, x2: 300, y2: 180, score: 0.77 },
];

describe('DetectionView', () => {
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('view') ? 640 : 0;
    });
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockImplementation(function (this: HTMLElement) {
      return this.classList.contains('view') ? 480 : 0;
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  it('sets decoding="async" on the image element', () => {
    const { container, unmount } = render(() => (
      <DetectionView src="data:image/png;base64,AA==" width={640} height={480} boxes={BOXES} classes={['cat', 'dog']} />
    ));
    try {
      flush();
      const img = container.querySelector('img');
      expect(img).not.toBeNull();
      expect(img!.getAttribute('decoding')).toBe('async');
    } finally {
      unmount();
    }
  });

  it('delegates box hover and click on .layer without flickering across box children', () => {
    const hovered: Array<number | null> = [];
    const selected: number[] = [];
    const { container, unmount } = render(() => (
      <DetectionView
        src="data:image/png;base64,AA=="
        width={640}
        height={480}
        boxes={BOXES}
        classes={['cat', 'dog']}
        onhover={(i) => hovered.push(i)}
        onselect={(i) => selected.push(i)}
      />
    ));
    try {
      flush();
      const box0 = container.querySelector<HTMLElement>('.box[data-box="0"]')!;
      const box1 = container.querySelector<HTMLElement>('.box[data-box="1"]')!;
      const chip0 = box0.querySelector<HTMLElement>('.chip')!;
      const img = container.querySelector<HTMLElement>('img')!;
      expect(box0).not.toBeNull();
      expect(box1).not.toBeNull();
      expect(chip0).not.toBeNull();

      // Enter box 0 from background -> hover 0.
      box0.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, relatedTarget: img }));
      expect(hovered).toEqual([0]);

      // Move from box 0 into its chip child -> no spurious null or duplicate hover.
      box0.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, relatedTarget: chip0 }));
      chip0.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, relatedTarget: box0 }));
      expect(hovered).toEqual([0]);

      // Move directly from box 0 to box 1 -> transitions cleanly without flashing null.
      box0.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, relatedTarget: box1 }));
      box1.dispatchEvent(new PointerEvent('pointerover', { bubbles: true, relatedTarget: box0 }));
      expect(hovered).toEqual([0, 1]);

      // Leave box 1 to background -> hover null.
      box1.dispatchEvent(new PointerEvent('pointerout', { bubbles: true, relatedTarget: img }));
      expect(hovered).toEqual([0, 1, null]);

      // Clicking a box or its chip delegates to onselect with the box index.
      chip0.click();
      box1.click();
      expect(selected).toEqual([0, 1]);
    } finally {
      unmount();
    }
  });

  it('awaits async img.decode() for real bitmaps, cancels stale decodes on src swap/error/unmount, and forwards fetchpriority', async () => {
    const resolvers: Array<{ res: () => void; rej: (e: Error) => void }> = [];
    vi.spyOn(HTMLImageElement.prototype, 'naturalWidth', 'get').mockReturnValue(640);
    vi.spyOn(HTMLImageElement.prototype, 'decode').mockImplementation(
      () =>
        new Promise<void>((res, rej) => {
          resolvers.push({ res, rej });
        }),
    );

    const src = signal<string | null>('/img/a.png');
    const outcomes: boolean[] = [];
    const { container, unmount } = render(() => (
      <DetectionView
        src={src.get()}
        priority="low"
        width={640}
        height={480}
        boxes={BOXES}
        classes={['cat', 'dog']}
        onload={(ok) => outcomes.push(ok)}
      />
    ));
    let unmounted = false;
    try {
      flush();
      const img = container.querySelector<HTMLImageElement>('img')!;
      expect(img.getAttribute('fetchpriority')).toBe('low');
      expect(resolvers.length).toBe(0);

      // 1. Active decode succeeds -> reveals .loaded and reports onload(true).
      img.dispatchEvent(new Event('load'));
      flush();
      expect(resolvers.length).toBe(1);
      expect(img.classList.contains('loaded')).toBe(false);
      expect(outcomes).toEqual([]);
      resolvers[0]!.res();
      await Promise.resolve();
      flush();
      expect(img.classList.contains('loaded')).toBe(true);
      expect(outcomes).toEqual([true]);

      // 2. Rapid src swap cancels an in-flight decode from the previous image.
      src.set('/img/b.png');
      flush();
      img.dispatchEvent(new Event('load'));
      expect(resolvers.length).toBe(2);
      src.set('/img/c.png');
      flush();
      resolvers[1]!.res();
      await Promise.resolve();
      flush();
      expect(img.classList.contains('loaded')).toBe(false);
      expect(outcomes).toEqual([true]);

      // 3. An active error invalidates any pending decode so it cannot settle as success later.
      img.dispatchEvent(new Event('load'));
      expect(resolvers.length).toBe(3);
      img.dispatchEvent(new Event('error'));
      flush();
      expect(outcomes).toEqual([true, false]);
      resolvers[2]!.res();
      await Promise.resolve();
      flush();
      expect(img.classList.contains('loaded')).toBe(false);
      expect(outcomes).toEqual([true, false]);

      // 4. An active decode rejection reports onload(false).
      src.set('/img/d.png');
      flush();
      img.dispatchEvent(new Event('load'));
      expect(resolvers.length).toBe(4);
      resolvers[3]!.rej(new Error('corrupt'));
      await Promise.resolve();
      flush();
      expect(outcomes).toEqual([true, false, false]);

      // 5. Unmounting while a decode is pending cancels it so late settlement is ignored.
      src.set('/img/e.png');
      flush();
      img.dispatchEvent(new Event('load'));
      expect(resolvers.length).toBe(5);
      unmount();
      unmounted = true;
      resolvers[4]!.res();
      await Promise.resolve();
      flush();
      expect(outcomes).toEqual([true, false, false]);
    } finally {
      if (!unmounted) unmount();
    }
  });
});
