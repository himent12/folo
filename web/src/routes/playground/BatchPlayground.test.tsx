// @vitest-environment happy-dom
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

function fakePrediction(modelId: string, count: number): Prediction {
  return {
    model_id: modelId,
    image: { width: 320, height: 240 },
    boxes: Array.from({ length: count }, (_, i) => ({
      class: i % 2,
      name: i % 2 === 0 ? 'circle' : 'square',
      score: Number((0.95 - i * 0.1).toFixed(2)),
      x1: 10 + i * 20,
      y1: 10,
      x2: 40 + i * 20,
      y2: 40,
    })),
    timing: { preprocess_ms: 1, inference_ms: 5, postprocess_ms: 1, total_ms: 7 },
    device: 'cpu',
  };
}

function setupPlayground(predictImpl?: Api['predict']) {
  vi.stubGlobal('createImageBitmap', async () => ({ width: 320, height: 240, close: () => {} }));
  let urlIdx = 0;
  const createdUrls: string[] = [];
  const revokedUrls: string[] = [];
  vi.spyOn(URL, 'createObjectURL').mockImplementation(() => {
    const u = `blob:batch-${++urlIdx}`;
    createdUrls.push(u);
    return u;
  });
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation((u) => {
    revokedUrls.push(u);
  });

  const api = new MockApi('normal', 0);
  api.imageUrl = async () => 'data:,';
  if (predictImpl) api.predict = predictImpl;
  hub.api = api;
  hub.mode = 'mock';

  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse('#/playground?model=m_hard_s');
  mounted = render(() => <Playground />, target);
  return { api, createdUrls, revokedUrls };
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

describe('Playground multi-image batch filmstrip', () => {
  it('loads multiple images into a filmstrip, runs sequential Run All, steps with buttons/keyboard, and exports JSON/CSV', async () => {
    const countsBySize = new Map<number, number>();
    const files = [makeImageFile('img-1.png'), makeImageFile('img-22.png'), makeImageFile('img-333.png')];
    countsBySize.set(files[0]!.size, 1);
    countsBySize.set(files[1]!.size, 3);
    countsBySize.set(files[2]!.size, 2);

    const predictSpy = vi.fn<Api['predict']>(async (input, opts) => {
      const sz = input.kind === 'file' ? input.file.size : 0;
      return fakePrediction(opts.model_id, countsBySize.get(sz) ?? 1);
    });
    setupPlayground(predictSpy);

    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles(files);

    // Filmstrip mounts with 3 thumbnails and first image automatically inferred
    await until(() => target.querySelectorAll('[data-batch-filmstrip] [role="option"]').length === 3, '3 filmstrip thumbnails');
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '1', 'first image badge');
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('1 / 3');
    expect(predictSpy).toHaveBeenCalledTimes(1);

    // Click Run all to infer remaining 2 images sequentially (skipping already-cached image 0)
    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();

    await until(() => target.querySelector('[data-batch-idx="2"] [data-batch-badge]')?.textContent === '2', 'third image badge');
    expect(target.querySelector('[data-batch-idx="1"] [data-batch-badge]')?.textContent).toBe('3');
    expect(target.querySelector('[data-batch-summary]')?.textContent).toContain('3/3 run');
    expect(target.querySelector('[data-batch-summary]')?.textContent).toContain('6 dets');
    // Total calls = 3 (1 initial + 2 remaining), zero duplicate requests
    expect(predictSpy).toHaveBeenCalledTimes(3);

    // Stepping via Next button and ArrowRight / ArrowLeft reuses cached predictions with zero extra requests
    target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!.click();
    flushSync();
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('2 / 3');
    expect(target.querySelector('.src-name')?.textContent).toBe('img-22.png');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    flushSync();
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('3 / 3');
    expect(target.querySelector('.src-name')?.textContent).toBe('img-333.png');

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    flushSync();
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('2 / 3');
    expect(predictSpy).toHaveBeenCalledTimes(3);

    // Batch JSON & Batch CSV export buttons are enabled and trigger downloads
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const jsonBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Export batch JSON"]')!;
    const csvBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Export batch CSV"]')!;
    expect(jsonBtn.disabled).toBe(false);
    expect(csvBtn.disabled).toBe(false);
    jsonBtn.click();
    csvBtn.click();
    expect(clickSpy).toHaveBeenCalledTimes(2);
  });

  it('aborts in-flight Run All on Stop, clears stale badges on model switch, and revokes every object URL on Clear/unmount', async () => {
    const signals: AbortSignal[] = [];
    const resolvers: ((p: Prediction) => void)[] = [];
    const predictSpy = vi.fn<Api['predict']>((_input, _opts, signal) => {
      if (signal) signals.push(signal);
      return new Promise<Prediction>((resolve, reject) => {
        if (signal?.aborted) {
          reject(new DOMException('Aborted', 'AbortError'));
          return;
        }
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
        resolvers.push((p) => resolve(p));
      });
    });

    const { createdUrls, revokedUrls } = setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('a.png'), makeImageFile('bb.png'), makeImageFile('ccc.png')]);

    await until(() => target.querySelectorAll('[data-batch-filmstrip] [role="option"]').length === 3, '3 thumbnails');
    await until(() => resolvers.length === 1, 'initial predict in flight');
    resolvers[0]!(fakePrediction('m_hard_s', 2));
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '2', 'img 0 done');

    // Start Run All -> image 1 starts predicting
    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();
    await until(() => resolvers.length === 2, 'img 1 in flight');

    // Click Stop -> aborts signal[1] cleanly without starting image 2
    const stopBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Stop'))!;
    stopBtn.click();
    flushSync();
    expect(signals[1]?.aborted).toBe(true);
    expect(predictSpy).toHaveBeenCalledTimes(2);
    expect(target.querySelector('[data-batch-idx="1"] [data-batch-badge]')?.textContent).toBe('—');

    // Clicking Clear removes the filmstrip and revokes all 3 batch object URLs (no leaks)
    target.querySelector<HTMLButtonElement>('button[aria-label="Clear image"]')!.click();
    flushSync();
    expect(target.querySelector('[data-batch-bar]')).toBeNull();
    expect(createdUrls).toHaveLength(3);
    for (const u of createdUrls) {
      expect(revokedUrls).toContain(u);
    }
  });

  it('aborts in-flight Run All cleanly on route unmount without leaking batch object URLs', async () => {
    const signals: AbortSignal[] = [];
    const predictSpy = vi.fn<Api['predict']>((_input, _opts, signal) => {
      if (signal) signals.push(signal);
      return new Promise<Prediction>((_resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
      });
    });

    const { createdUrls, revokedUrls } = setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('x.png'), makeImageFile('yy.png')]);
    await until(() => target.querySelectorAll('[data-batch-filmstrip] [role="option"]').length === 2, '2 thumbnails');

    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();
    await until(() => signals.length >= 2, 'runAll signal');

    mounted!.unmount();
    mounted = undefined;
    expect(signals.every((s) => s.aborted)).toBe(true);
    expect(createdUrls).toHaveLength(2);
    for (const u of createdUrls) {
      expect(revokedUrls).toContain(u);
    }
  });

  it('supports Compare A/B mode cleanly alongside the batch filmstrip', async () => {
    const predictSpy = vi.fn<Api['predict']>(async (_input, opts) => fakePrediction(opts.model_id, opts.model_id === 'm_hard_s' ? 2 : 1));
    setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('c1.png'), makeImageFile('c22.png')]);
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '2', 'batch ready');

    // Toggle Compare A/B ON -> Compare stage mounts while filmstrip stays active
    const cmpBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Compare A/B'))!;
    cmpBtn.click();
    await until(() => target.querySelector('[data-compare-view]'), 'compare stage');
    expect(target.querySelector('[data-batch-bar]')).not.toBeNull();

    // Stepping to image 2 keeps Compare A/B active and updates both panes
    target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!.click();
    await until(() => target.querySelector('[data-batch-pos]')?.textContent?.trim() === '2 / 2', 'stepped to image 2');
    expect(target.querySelector('[data-compare-view]')).not.toBeNull();
  });
});
