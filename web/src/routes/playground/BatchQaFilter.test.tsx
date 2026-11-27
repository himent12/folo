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

    // Summary CSV export works cleanly even on Empty filter
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const summaryBtn = target.querySelector<HTMLButtonElement>('button[data-batch-export="summary-csv"]')!;
    expect(summaryBtn.disabled).toBe(false);
    summaryBtn.click();
    expect(clickSpy).toHaveBeenCalledTimes(1);

    // Filter to With detections -> shows indices 0 and 2, Next steps directly from index 0 to index 2
    target.querySelector<HTMLButtonElement>('[data-qa-filter="with-detections"]')!.click();
    flushSync();
    const thumbs = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-filmstrip] [role="option"]')];
    expect(thumbs.map((el) => el.getAttribute('data-batch-idx'))).toEqual(['0', '2']);
    expect(target.querySelector('.src-name')?.textContent).toBe('a.png');
    target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!.click();
    flushSync();
    expect(target.querySelector('.src-name')?.textContent).toBe('ccc.png');
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('2 / 2');

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

    // Clicking Show all resets cleanly back to all 3 thumbnails
    target.querySelector<HTMLButtonElement>('[data-batch-empty] button')!.click();
    flushSync();
    expect(target.querySelectorAll('[data-batch-filmstrip] [role="option"]')).toHaveLength(3);
  });
});
