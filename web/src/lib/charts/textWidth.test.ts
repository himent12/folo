// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { labelWidth } from './endLabels';
import { resetTextMeasurerCache, textMeasurer } from './textWidth';

afterEach(() => {
  resetTextMeasurerCache();
  delete document.documentElement.dataset.theme;
  delete document.documentElement.dataset.themeId;
  document.documentElement.style.cssText = '';
  vi.restoreAllMocks();
});

describe('textMeasurer', () => {
  it('falls back to labelWidth when 2D canvas is unavailable', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const m = textMeasurer('semibold');
    expect(m('mAP@50')).toBe(labelWidth('mAP@50'));
  });

  it('reuses a single shared canvas and caches per-font widths across measurers', () => {
    const measureText = vi.fn((t: string) => ({ width: t.length * 7.2 }) as TextMetrics);
    const fakeCtx = { font: '', measureText } as unknown as CanvasRenderingContext2D;
    const origCreate = document.createElement.bind(document);
    let canvases = 0;
    vi.spyOn(document, 'createElement').mockImplementation(((tag: string, opts?: ElementCreationOptions) => {
      const el = origCreate(tag, opts);
      if (tag.toLowerCase() === 'canvas') {
        canvases++;
        vi.spyOn(el as HTMLCanvasElement, 'getContext').mockReturnValue(fakeCtx);
      }
      return el;
    }) as typeof document.createElement);

    vi.spyOn(window, 'getComputedStyle').mockReturnValue({
      getPropertyValue: (prop: string) =>
        prop === '--weight-semibold'
          ? '600'
          : prop === '--weight-regular'
            ? '400'
            : prop === '--type-sm-size'
              ? '12px'
              : prop === '--font-sans'
                ? 'Geist'
                : '',
    } as CSSStyleDeclaration);

    const bold1 = textMeasurer('semibold');
    const bold2 = textMeasurer('semibold');
    const reg = textMeasurer('regular');

    expect(bold1('loss')).toBe(Math.ceil(4 * 7.2) + 4);
    expect(bold2('loss')).toBe(Math.ceil(4 * 7.2) + 4);
    expect(reg('loss')).toBe(Math.ceil(4 * 7.2));
    expect(reg('loss')).toBe(Math.ceil(4 * 7.2));

    expect(canvases).toBe(1);
    // One call for semibold 'loss', one call for regular 'loss' (second calls hit cache).
    expect(measureText).toHaveBeenCalledTimes(2);
  });

  it('re-resolves font tokens when theme attributes change', () => {
    let size = '12px';
    const measureText = vi.fn(() => ({ width: size === '14px' ? 40 : 30 }) as TextMetrics);
    const fakeCtx = { font: '', measureText } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(fakeCtx);
    const getCS = vi.spyOn(window, 'getComputedStyle').mockImplementation(
      () =>
        ({
          getPropertyValue: (prop: string) =>
            prop === '--weight-regular' ? '400' : prop === '--type-sm-size' ? size : prop === '--font-sans' ? 'Geist' : '',
        }) as CSSStyleDeclaration,
    );

    const m = textMeasurer('regular');
    document.documentElement.dataset.theme = 'dark';
    expect(m('box')).toBe(30);
    expect(getCS).toHaveBeenCalledTimes(1);

    // Same theme: font stays cached.
    expect(m('cls')).toBe(30);
    expect(getCS).toHaveBeenCalledTimes(1);

    // Theme changes: re-reads computed font and measures against the new font key.
    size = '14px';
    document.documentElement.dataset.theme = 'light';
    expect(m('box')).toBe(40);
    expect(getCS).toHaveBeenCalledTimes(2);
  });

  it('invalidates cached widths when web fonts finish loading', async () => {
    let loaded = false;
    const measureText = vi.fn(() => ({ width: loaded ? 40 : 30 }) as TextMetrics);
    const fakeCtx = { font: '', measureText } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(fakeCtx);
    vi.spyOn(window, 'getComputedStyle').mockReturnValue({
      getPropertyValue: (prop: string) => (prop === '--weight-regular' ? '400' : prop === '--type-sm-size' ? '12px' : prop === '--font-sans' ? 'Geist' : ''),
    } as CSSStyleDeclaration);

    const listeners = new Set<() => void>();
    let resolveReady!: () => void;
    const ready = new Promise<void>((r) => (resolveReady = r));
    const origFonts = Object.getOwnPropertyDescriptor(document, 'fonts');
    Object.defineProperty(document, 'fonts', {
      configurable: true,
      value: {
        ready,
        addEventListener: (_type: string, fn: () => void) => listeners.add(fn),
        removeEventListener: (_type: string, fn: () => void) => listeners.delete(fn),
      },
    });
    try {
      const m = textMeasurer('regular');
      expect(m('mAP@50')).toBe(30);
      expect(measureText).toHaveBeenCalledTimes(1);

      // Font finishes loading and fires loadingdone: cache clears and remeasures.
      loaded = true;
      for (const fn of listeners) fn();
      expect(m('mAP@50')).toBe(40);
      expect(measureText).toHaveBeenCalledTimes(2);

      // Resolving document.fonts.ready also invalidates stale fallback measurements.
      loaded = false;
      resolveReady();
      await ready;
      expect(m('mAP@50')).toBe(30);
      expect(measureText).toHaveBeenCalledTimes(3);
    } finally {
      resetTextMeasurerCache();
      if (origFonts) Object.defineProperty(document, 'fonts', origFonts);
      else delete (document as unknown as Record<string, unknown>).fonts;
    }
  });
});
