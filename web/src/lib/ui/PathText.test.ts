// @vitest-environment happy-dom
import { flush } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import PathText, { splitPath } from './PathText';

describe('splitPath', () => {
  it('keeps the folder and the one holding it whole, so folders of one name in two places read apart', () => {
    expect(splitPath('D:/folo-sweep3/studio-data/work/messy/place-a/cells')).toEqual({
      head: 'D:/folo-sweep3/studio-data/work/messy',
      tail: '/place-a/cells',
    });
    expect(splitPath('D:/folo-sweep3/studio-data/work/messy/place-b/cells').tail).toBe('/place-b/cells');
  });

  it('cuts nothing off a short path, and reads back slashes and a trailing separator', () => {
    expect(splitPath('D:/data/coco')).toEqual({ head: '', tail: 'D:/data/coco' });
    expect(splitPath('/data/coco/')).toEqual({ head: '', tail: '/data/coco' });
    expect(splitPath('C:\\Users\\me\\data\\bccd')).toEqual({ head: 'C:\\Users\\me', tail: '\\data\\bccd' });
    expect(splitPath('/')).toEqual({ head: '', tail: '/' });
    expect(splitPath('D:/')).toEqual({ head: '', tail: 'D:/' });
    expect(splitPath('C:\\')).toEqual({ head: '', tail: 'C:\\' });
  });
});

describe('PathText ResizeObserver deferral', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('defers .cut class toggles from ResizeObserver via coalesced rAF and cancels on unmount', async () => {
    vi.useFakeTimers();
    const roCallbacks: ResizeObserverCallback[] = [];
    const disconnect = vi.fn();
    class FakeResizeObserver {
      constructor(cb: ResizeObserverCallback) {
        roCallbacks.push(cb);
      }
      observe() {}
      unobserve() {}
      disconnect = disconnect;
    }
    const origRO = globalThis.ResizeObserver;
    globalThis.ResizeObserver = FakeResizeObserver as unknown as typeof ResizeObserver;

    const rafSpy = vi.spyOn(window, 'requestAnimationFrame');
    const cafSpy = vi.spyOn(window, 'cancelAnimationFrame');

    const { container, unmount } = render(() => PathText({ path: 'D:/folo-sweep3/studio-data/work/messy/place-a/cells' }));
    try {
      flush();
      const root = container.querySelector('.ui-path') as HTMLElement;
      const head = root.querySelector('.head') as HTMLElement;
      let headScrollW = 100;
      let headClientW = 100;
      Object.defineProperty(head, 'scrollWidth', { get: () => headScrollW });
      Object.defineProperty(head, 'clientWidth', { get: () => headClientW });

      expect(root.classList.contains('cut')).toBe(false);
      expect(roCallbacks.length).toBe(1);

      headScrollW = 240;
      headClientW = 100;
      roCallbacks[0]!([], {} as ResizeObserver);
      roCallbacks[0]!([], {} as ResizeObserver);

      expect(rafSpy).toHaveBeenCalledTimes(1);
      expect(root.classList.contains('cut')).toBe(false);

      await vi.advanceTimersByTimeAsync(20);
      expect(root.classList.contains('cut')).toBe(true);

      roCallbacks[0]!([], {} as ResizeObserver);
      unmount();
      expect(cafSpy).toHaveBeenCalled();
      expect(disconnect).toHaveBeenCalledTimes(1);
    } finally {
      globalThis.ResizeObserver = origRO;
    }
  });
});
