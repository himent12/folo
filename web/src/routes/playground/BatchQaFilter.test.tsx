// @vitest-environment happy-dom
// Playground batch filmstrip QA filtering (All / With detections / Empty / Specific class), live match counts,
// filtered stepping & keyboard navigation, zero-match reset, and filtered JSON / CSV / Summary CSV exports.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import type { Api } from '$lib/api/client';
import { MockApi } from '$lib/api/mock';
import type { Prediction } from '$lib/api/types';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { session } from './session';

const PNG_HEAD = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]);

let target: HTMLElement;
let mounted: { unmount(): void } | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
  let value: T | null | undefined | false;
  await vi.waitFor(
    () => {
      flushSync();
      value = get();
      if (!value) throw new Error(`waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return value as T;
}

function makeImageFile(name: string): File {
  return new File([PNG_HEAD, new TextEncoder().encode(name)], name, { type: 'image/png' });
}

function makePred(modelId: string, boxes: Prediction['boxes']): Prediction {
  return {
    model_id: modelId,
    image: { width: 320, height: 240 },
    boxes,
    timing: { preprocess_ms: 1, inference_ms: 5, postprocess_ms: 1, total_ms: 7 },
    device: 'cpu',
  };
}

function setupPlayground(predictImpl: Api['predict']) {
  vi.stubGlobal('createImageBitmap', async () => ({ width: 320, height: 240, close: () => {} }));
  let urlIdx = 0;
  vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:qa-${++urlIdx}`);
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

  const api = new MockApi('normal', 0);
  api.imageUrl = async () => 'data:,';
  api.predict = predictImpl;
  hub.api = api;
  hub.mode = 'mock';

  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse('#/playground?model=m_hard_s');
  mounted = render(() => <Playground />, target);
}

function selectBatchFiles(files: File[]) {
  const input = target.querySelector<HTMLInputElement>('input[type="file"][multiple]')!;
  Object.defineProperty(input, 'files', { value: files, configurable: true });
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  session.set(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('Playground batch QA filter and summary export', () => {
  it('filters filmstrip by With detections / Empty / Specific class with live match counts and filtered exports', async () => {
    const f1 = makeImageFile('a.png');
    const f2 = makeImageFile('bb.png');
    const f3 = makeImageFile('ccc.png');
    const bySize = new Map<number, Prediction['boxes']>([
      [f1.size, [{ class: 0, name: 'circle', score: 0.92, x1: 10, y1: 10, x2: 50, y2: 50 }]],
      [f2.size, []],
      [
        f3.size,
        [
          { class: 1, name: 'square', score: 0.88, x1: 20, y1: 20, x2: 70, y2: 70 },
          { class: 0, name: 'circle', score: 0.55, x1: 80, y1: 20, x2: 120, y2: 60 },
        ],
      ],
    ]);

    setupPlayground(async (input, opts) => {
      const sz = input.kind === 'file' ? input.file.size : 0;
      return makePred(opts.model_id, bySize.get(sz) ?? []);
    });

    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([f1, f2, f3]);
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '1', 'img 0 done');

    // Run all 3 images
    target.querySelector<HTMLButtonElement>('button[data-batch-run]')!.click();
    await until(() => target.querySelector('[data-batch-summary]')?.textContent?.includes('3/3 run'), 'all 3 run');

    // QA filter pills show live counts: All 3, With detections 2, Empty 1
    expect(target.querySelector('[data-qa-filter="all"] .qa-count')?.textContent?.trim()).toBe('3');
    expect(target.querySelector('[data-qa-filter="with-detections"] .qa-count')?.textContent?.trim()).toBe('2');
    expect(target.querySelector('[data-qa-filter="empty"] .qa-count')?.textContent?.trim()).toBe('1');

    // Filter to Empty -> auto-selects empty image (index 1, bb.png) and shows 1 thumbnail
    target.querySelector<HTMLButtonElement>('[data-qa-filter="empty"]')!.click();
    flushSync();
    expect(target.querySelectorAll('[data-batch-filmstrip] [role="option"]')).toHaveLength(1);
    expect(target.querySelector('.src-name')?.textContent).toBe('bb.png');
    expect(target.querySelector('[data-batch-match-count]')?.textContent).toContain('1 of 3');
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('1 / 1');

    // Summary CSV export works cleanly even on Empty filter and writes expected filename + CSV payload
    const downloads: { filename: string; blob: Blob }[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      if (b instanceof Blob) downloads.push({ filename: '', blob: b });
      return `blob:qa-dl-${downloads.length}`;
    });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      if (downloads.length) downloads[downloads.length - 1]!.filename = this.download;
    });
    const summaryBtn = target.querySelector<HTMLButtonElement>('button[data-batch-export="summary-csv"]')!;
    expect(summaryBtn.disabled).toBe(false);
    summaryBtn.click();
    expect(downloads).toHaveLength(1);
    expect(downloads[0]!.filename).toContain('-empty-summary-1img.csv');
    const emptyCsv = await downloads[0]!.blob.text();
    expect(emptyCsv).toContain('filename,status,detections,classes,top_confidence,total_ms,image_width,image_height,error');
    expect(emptyCsv).toContain('bb.png,done,0,,,7,320,240,');

    // Filter to With detections -> shows indices 0 and 2; focused Next button reaching the last match moves focus to the selected thumbnail (Playtest R24 P3-1)
    target.querySelector<HTMLButtonElement>('[data-qa-filter="with-detections"]')!.click();
    flushSync();
    const thumbs = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-filmstrip] [role="option"]')];
    expect(thumbs.map((el) => el.getAttribute('data-batch-idx'))).toEqual(['0', '2']);
    expect(target.querySelector('.src-name')?.textContent).toBe('a.png');
    const nextBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!;
    const prevBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Previous image"]')!;
    nextBtn.focus();
    nextBtn.click();
    await vi.waitFor(() => {
      expect(nextBtn.disabled).toBe(true);
      expect(document.activeElement).toBe(target.querySelector('[data-batch-idx="2"]'));
    });
    expect(target.querySelector('.src-name')?.textContent).toBe('ccc.png');
    prevBtn.focus();
    prevBtn.click();
    await vi.waitFor(() => {
      expect(prevBtn.disabled).toBe(true);
      expect(document.activeElement).toBe(target.querySelector('[data-batch-idx="0"]'));
    });
    expect(target.querySelector('.src-name')?.textContent).toBe('a.png');

    const strip = target.querySelector<HTMLElement>('[data-batch-filmstrip]')!;
    strip.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true, cancelable: true }));
    flushSync();
    expect(target.querySelector('.src-name')?.textContent).toBe('ccc.png');
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('2 / 2');
    strip.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true, cancelable: true }));
    flushSync();
    expect(target.querySelector('.src-name')?.textContent).toBe('a.png');
    strip.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
    flushSync();
    expect(target.querySelector('.src-name')?.textContent).toBe('ccc.png');

    // Filter to specific class 'square' (classId 1) via the class selector -> shows only index 2 (ccc.png)
    const classSelect = target.querySelector<HTMLSelectElement>('[data-batch-class-filter] select')!;
    classSelect.value = '1';
    classSelect.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelectorAll('[data-batch-filmstrip] [role="option"]')).toHaveLength(1);
    expect(target.querySelector('[data-batch-filmstrip] [role="option"]')?.getAttribute('data-batch-idx')).toBe('2');

    // Raise confidence above 0.90 so 'square' (0.88) has 0 matches -> empty state with Show all button and disabled exports
    const confInput = target.querySelector<HTMLInputElement>('input[type="range"]')!;
    confInput.value = '0.91';
    confInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('[data-batch-empty]')).not.toBeNull();
    expect(target.querySelector<HTMLButtonElement>('button[data-batch-export="json"]')!.disabled).toBe(true);
    expect(target.querySelector<HTMLButtonElement>('button[data-batch-export="summary-csv"]')!.disabled).toBe(true);

    // Clicking Show all resets cleanly back to all 3 thumbnails and restores focus to the persistent All pill
    const showAllBtn = target.querySelector<HTMLButtonElement>('[data-batch-empty] button')!;
    showAllBtn.focus();
    showAllBtn.click();
    flushSync();
    await Promise.resolve();
    expect(target.querySelectorAll('[data-batch-filmstrip] [role="option"]')).toHaveLength(3);
    expect(document.activeElement).toBe(target.querySelector('[data-qa-filter="all"]'));
  });

  it('enables Summary CSV on an all-failed batch and exports failure diagnostics while keeping detection exports disabled', async () => {
    setupPlayground(async () => {
      throw new Error('GPU timeout');
    });
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('fail1.png'), makeImageFile('fail22.png')]);
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '!', 'img 0 failed');

    expect(target.querySelector<HTMLButtonElement>('button[data-batch-export="json"]')!.disabled).toBe(true);
    expect(target.querySelector<HTMLButtonElement>('button[data-batch-export="csv"]')!.disabled).toBe(true);
    const summaryBtn = target.querySelector<HTMLButtonElement>('button[data-batch-export="summary-csv"]')!;
    expect(summaryBtn.disabled).toBe(false);

    const blobs: Blob[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      if (b instanceof Blob) blobs.push(b);
      return 'blob:fail-summary';
    });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    summaryBtn.click();
    expect(blobs).toHaveLength(1);
    const csv = await blobs[0]!.text();
    expect(csv).toContain('fail1.png,error,0,,,,320,240,GPU timeout');
  });

  it('completes capped non-matching items before filtered export and aborts stale export if QA filter changes mid-flight', async () => {
    // 1000 overlapping circle candidates hit FETCH.max_det=1000 and collapse under NMS, hiding a lower-scoring square past 1000.
    const dense1000: Prediction['boxes'] = Array.from({ length: 1000 }, (_, i) => ({
      class: 0,
      name: 'circle',
      score: Number((0.99 - i * 0.0005).toFixed(4)),
      x1: 10,
      y1: 10,
      x2: 40,
      y2: 40,
    }));
    const fullWithSquare: Prediction['boxes'] = [
      ...dense1000,
      { class: 1, name: 'square', score: 0.4, x1: 5, y1: 5, x2: 15, y2: 15 },
    ];
    const fDense = makeImageFile('dense-cap.png');
    const fSquare = makeImageFile('plain-square.png');
    let allAttempts = 0;
    let resolveDeferredAll: ((p: Prediction) => void) | undefined;
    let deferredSignal: AbortSignal | undefined;

    setupPlayground((input, opts, signal) => {
      const sz = input.kind === 'file' ? input.file.size : 0;
      if (sz === fSquare.size) {
        return Promise.resolve(makePred(opts.model_id, [{ class: 1, name: 'square', score: 0.85, x1: 10, y1: 10, x2: 50, y2: 50 }]));
      }
      if (opts.max_det === 30_000) {
        allAttempts++;
        // First automatic ALL completion fails so item 0 stays capped at 1000 circles (no square yet).
        if (allAttempts === 1) return Promise.reject(new Error('transient ALL failure'));
        // Second ALL completion (during export) stays pending so we can test filter-change abort, then third resolves with square!
        if (allAttempts === 2) {
          deferredSignal = signal;
          return new Promise<Prediction>((resolve, reject) => {
            signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
            resolveDeferredAll = resolve;
          });
        }
        return Promise.resolve(makePred(opts.model_id, fullWithSquare));
      }
      return Promise.resolve(makePred(opts.model_id, dense1000));
    });

    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([fDense, fSquare]);
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent?.includes('+'), 'item 0 capped');
    target.querySelector<HTMLButtonElement>('button[data-batch-run]')!.click();
    await until(() => target.querySelector('[data-batch-summary]')?.textContent?.includes('2/2 run'), 'both run');

    // Filter to class 'square' -> initially only item 1 matches because item 0 is capped without square
    const classSelect = target.querySelector<HTMLSelectElement>('[data-batch-class-filter] select')!;
    classSelect.value = '1';
    classSelect.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelectorAll('[data-batch-filmstrip] [role="option"]')).toHaveLength(1);

    const blobs: Blob[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      if (b instanceof Blob) blobs.push(b);
      return 'blob:filtered-json';
    });
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    // Click Batch JSON -> triggers async ALL completion for capped item 0; switch QA filter to Empty mid-flight -> aborts cleanly
    const jsonBtn = target.querySelector<HTMLButtonElement>('button[data-batch-export="json"]')!;
    jsonBtn.click();
    await until(() => !!resolveDeferredAll, 'export ALL in flight for capped item 0');
    target.querySelector<HTMLButtonElement>('[data-qa-filter="empty"]')!.click();
    flushSync();
    expect(deferredSignal?.aborted).toBe(true);
    await Promise.resolve();
    expect(clickSpy).toHaveBeenCalledTimes(0);

    // Switch back to class 'square' and export again -> completes capped item 0 with square and includes BOTH images in JSON!
    classSelect.value = '1';
    classSelect.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    jsonBtn.click();
    await until(() => clickSpy.mock.calls.length === 1, 'filtered JSON exported after ALL completion');
    const payload = JSON.parse(await blobs.at(-1)!.text()) as { images: { filename: string }[] };
    expect(payload.images.map((im) => im.filename)).toEqual(['dense-cap.png', 'plain-square.png']);
  });
});
