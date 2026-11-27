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

  it('cancels in-flight single predict synchronously on Run All (no duplicate call) and aborts on Compare toggle or threshold change', async () => {
    const calls: { signal?: AbortSignal; resolve: (p: Prediction) => void; reject: (e: unknown) => void }[] = [];
    const predictSpy = vi.fn<Api['predict']>((_input, _opts, signal) => {
      return new Promise<Prediction>((resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
        calls.push({ signal, resolve, reject });
      });
    });

    setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('p1.png'), makeImageFile('p22.png'), makeImageFile('p333.png')]);
    await until(() => calls.length === 1, 'initial single predict started');

    // Click Run all while initial single-image request is STILL pending -> single request aborted BEFORE batch request starts
    const runAllBtn = () => [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn().click();
    flushSync();
    expect(calls[0]!.signal?.aborted).toBe(true);
    await until(() => calls.length === 2, 'batch runAll started for item 0');
    expect(predictSpy).toHaveBeenCalledTimes(2);

    // Resolve item 0 -> batch advances to item 1
    calls[1]!.resolve(fakePrediction('m_hard_s', 2));
    await until(() => calls.length === 3, 'batch runAll started for item 1');

    // Step to completed item 0 while item 1 is running -> stage busy indicator must NOT stick over completed item 0
    expect(target.querySelector('.stage .busy')).toBeNull();
    // Step onto running item 1 -> stage busy indicator reflects item 1 is running
    target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!.click();
    flushSync();
    expect(target.querySelector('.stage .busy')).not.toBeNull();
    // Step back to completed item 0 -> busy indicator clears immediately
    target.querySelector<HTMLButtonElement>('button[aria-label="Previous image"]')!.click();
    flushSync();
    expect(target.querySelector('.stage .busy')).toBeNull();

    // Toggling Compare A/B ON aborts the in-flight Run All cleanly
    const cmpBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Compare A/B'))!;
    cmpBtn.click();
    flushSync();
    expect(calls[2]!.signal?.aborted).toBe(true);
    expect([...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].some((b) => b.textContent?.includes('Stop'))).toBe(false);
  });

  it('aborts Run All on threshold change while keeping completed items cached, invalidates on model switch, and falls back to single-image mode when 1 of 2 files fails decode', async () => {
    const signals: AbortSignal[] = [];
    const resolvers: ((p: Prediction) => void)[] = [];
    const predictSpy = vi.fn<Api['predict']>((_input, _opts, signal) => {
      if (signal) signals.push(signal);
      return new Promise<Prediction>((resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
        resolvers.push(resolve);
      });
    });

    setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('t1.png'), makeImageFile('t22.png')]);
    await until(() => resolvers.length === 1, 'item 0 started');
    resolvers[0]!(fakePrediction('m_hard_s', 2));
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '2', 'item 0 cached');

    // Start Run All -> item 1 starts in flight; changing confidence slider aborts item 1 while keeping item 0 cached
    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();
    await until(() => resolvers.length === 2, 'item 1 in flight');
    const confInput = target.querySelector<HTMLInputElement>('input[type="range"]')!;
    confInput.value = '0.5';
    confInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(signals[1]?.aborted).toBe(true);
    expect(target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent).toBe('2');

    // Switching model via URL query invalidates stale badges on batch items
    router.route = parse('#/playground?model=m_shapes_n');
    flushSync();
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent !== '2', 'stale badge cleared on model switch');

    // Dropping 2 files where 1 fails decode falls back cleanly to single-image mode (no filmstrip)
    let decodeIdx = 0;
    vi.stubGlobal('createImageBitmap', async () => {
      decodeIdx++;
      if (decodeIdx === 1) throw new Error('corrupt');
      return { width: 100, height: 100, close: () => {} };
    });
    selectBatchFiles([makeImageFile('bad.png'), makeImageFile('good.png')]);
    await until(() => target.querySelector('.src-name')?.textContent === 'good.png', 'single valid image loaded');
    expect(target.querySelector('[data-batch-bar]')).toBeNull();
  });

  it('preserves error state without firing a duplicate request when Run All fails on the active image', async () => {
    let failNow = false;
    const predictSpy = vi.fn<Api['predict']>(async (_input, opts) => {
      if (failNow) throw new Error('Server 500 boom');
      return fakePrediction(opts.model_id, 1);
    });
    setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('e1.png'), makeImageFile('e22.png')]);
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '1', 'img 0 ok');

    // Step to item 1 with failure active -> fails once on single predict
    failNow = true;
    target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!.click();
    await until(() => target.querySelector('[data-batch-idx="1"] [data-batch-badge]')?.textContent === '!', 'img 1 failed');
    const callsBeforeRunAll = predictSpy.mock.calls.length;

    // Click Run all -> retries failed item 1 once, fails, sets error notice, and does NOT trigger a duplicate useSinglePredict call
    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();
    await until(() => predictSpy.mock.calls.length === callsBeforeRunAll + 1, 'runAll retried item 1 once');
    await new Promise((r) => setTimeout(r, 30));
    flushSync();
    expect(predictSpy.mock.calls.length).toBe(callsBeforeRunAll + 1);
    expect(target.querySelector('.notice')).not.toBeNull();
  });

  it('prevents an older batch decode from superseding a newer single-image pick and revokes superseded batch URLs', async () => {
    let resolveSecondBatchImage!: (bmp: { width: number; height: number; close(): void }) => void;
    let calls = 0;
    const { createdUrls, revokedUrls } = setupPlayground(async (_input, opts) => fakePrediction(opts.model_id, 1));
    vi.stubGlobal('createImageBitmap', () => {
      calls++;
      // Let slow-a.png decode and allocate an object URL, then pause on slow-b.png so winner.png supersedes it.
      if (calls === 2) {
        return new Promise((r) => {
          resolveSecondBatchImage = r;
        });
      }
      return Promise.resolve({ width: 100, height: 100, close: () => {} });
    });

    await until(() => target.querySelector('.model-pick'), 'model picker');
    // Start a slow 2-image batch load; wait until slow-a.png created its object URL and slow-b.png is pending
    selectBatchFiles([makeImageFile('slow-a.png'), makeImageFile('slow-b.png')]);
    await until(() => createdUrls.length === 1 && calls === 2, 'slow-a URL allocated while slow-b decodes');
    const supersededBatchUrl = createdUrls[0]!;

    // Immediately supersede it with a single image pick
    selectBatchFiles([makeImageFile('winner.png')]);
    await until(() => target.querySelector('.src-name')?.textContent === 'winner.png', 'single image won');
    const winnerUrl = createdUrls[createdUrls.length - 1]!;

    // Now let slow-b.png resolve -> superseded batch must revoke slow-a's URL and NOT replace winner.png
    resolveSecondBatchImage({ width: 320, height: 240, close: () => {} });
    await until(() => revokedUrls.includes(supersededBatchUrl), 'superseded batch URL revoked');
    flushSync();
    expect(target.querySelector('.src-name')?.textContent).toBe('winner.png');
    expect(target.querySelector('[data-batch-bar]')).toBeNull();
    expect(revokedUrls).not.toContain(winnerUrl);
  });

  it('lets Model A resume cleanly when Compare A/B aborts Run All mid-first-frame, and does not replay a failed ALL request', async () => {
    // Part 1: toggling Compare while Run All is running on uncached item 0 aborts Run All AND lets Model A resume.
    const calls: { model: string; maxDet: number | undefined; signal?: AbortSignal; resolve: (p: Prediction) => void; reject: (e: unknown) => void }[] = [];
    const predictSpy = vi.fn<Api['predict']>((_input, opts, signal) => {
      return new Promise<Prediction>((resolve, reject) => {
        signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true });
        calls.push({ model: opts.model_id, maxDet: opts.max_det, signal, resolve, reject });
      });
    });

    setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('cmp1.png'), makeImageFile('cmp22.png')]);
    await until(() => calls.length === 1, 'initial single predict');

    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();
    await until(() => calls.length === 2, 'batch runAll started on item 0');

    // Toggle Compare ON while item 0 is still running in Run All -> aborts Run All, Model A resumes cleanly (not blank!)
    const cmpBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Compare A/B'))!;
    cmpBtn.click();
    await until(() => calls[1]!.signal?.aborted && calls.filter((c) => c.model === 'm_hard_s').length >= 3, 'Model A resumed after Compare toggle');
    // Resolve the resumed Model A request -> badge updates to 2
    const resumedA = calls.filter((c) => c.model === 'm_hard_s').at(-1)!;
    resumedA.resolve(fakePrediction('m_hard_s', 2));
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent === '2', 'Model A populated after Compare');

    // Part 2: failed ALL completion during Run All keeps capped result and does NOT automatically replay ALL on exit
    mounted!.unmount();
    mounted = undefined;
    target.remove();
    session.set(null);
    vi.restoreAllMocks();

    const dense1000: Prediction['boxes'] = Array.from({ length: 1000 }, (_, i) => ({
      class: 0,
      name: 'circle',
      score: Number((0.99 - i * 0.0005).toFixed(4)),
      x1: (i % 20) * 15,
      y1: Math.floor(i / 20) * 10,
      x2: (i % 20) * 15 + 10,
      y2: Math.floor(i / 20) * 10 + 8,
    }));
    let allCalls = 0;
    const denseSpy = vi.fn<Api['predict']>(async (_input, opts) => {
      if (opts.max_det === 30_000) {
        allCalls++;
        throw new Error('ALL temporarily failed');
      }
      return {
        model_id: opts.model_id,
        image: { width: 320, height: 240 },
        boxes: dense1000,
        timing: { preprocess_ms: 1, inference_ms: 5, postprocess_ms: 1, total_ms: 7 },
        device: 'cpu',
      };
    });
    setupPlayground(denseSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker 2');
    selectBatchFiles([makeImageFile('cap1.png'), makeImageFile('cap22.png')]);
    await until(() => target.querySelector('[data-batch-idx="0"] [data-batch-badge]')?.textContent?.includes('+'), 'capped + badge on item 0');
    const allCallsBefore = allCalls;

    // Select item 1 and immediately start Run All -> item 1 FETCH succeeds, ALL fails once, Run All exits WITHOUT auto-replaying ALL
    target.querySelector<HTMLButtonElement>('button[aria-label="Next image"]')!.click();
    flushSync();
    const runAll2 = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAll2.click();
    await until(() => target.querySelector('[data-batch-idx="1"] [data-batch-badge]')?.textContent?.includes('+'), 'item 1 finished with capped + badge');
    await new Promise((r) => setTimeout(r, 30));
    flushSync();
    expect(allCalls).toBe(allCallsBefore + 1);
  });

  it('ignores keyboard stepping when focus is inside an input and preserves Alt+ArrowLeft inside filmstrip', async () => {
    setupPlayground(async (_input, opts) => fakePrediction(opts.model_id, 1));
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('k1.png'), makeImageFile('k22.png')]);
    await until(() => target.querySelectorAll('[data-batch-filmstrip] [role="option"]').length === 2, '2 thumbs');

    const sliderInput = target.querySelector<HTMLInputElement>('input[type="range"]')!;
    sliderInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    flushSync();
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('1 / 2');

    // Alt+ArrowRight inside the filmstrip must not be hijacked
    const strip = target.querySelector<HTMLElement>('[data-batch-filmstrip]')!;
    const altEv = new KeyboardEvent('keydown', { key: 'ArrowRight', altKey: true, bubbles: true, cancelable: true });
    strip.dispatchEvent(altEv);
    flushSync();
    expect(altEv.defaultPrevented).toBe(false);
    expect(target.querySelector('[data-batch-pos]')?.textContent?.trim()).toBe('1 / 2');
  });

  it('routes mixed empty-MIME image files plus a video file into batch mode rather than dropping the images', async () => {
    setupPlayground(async (_input, opts) => fakePrediction(opts.model_id, 1));
    await until(() => target.querySelector('.model-pick'), 'model picker');

    const imgA = new File([PNG_HEAD, 'a'], 'no-mime-a.png', { type: '' });
    const imgB = new File([PNG_HEAD, 'bb'], 'no-mime-b.png', { type: '' });
    const vid = new File(['ftyp'], 'clip.mp4', { type: 'video/mp4' });
    selectBatchFiles([imgA, imgB, vid]);

    await until(() => target.querySelectorAll('[data-batch-filmstrip] [role="option"]').length === 2, 'batch loaded 2 empty-MIME images');
    expect(target.querySelector('[data-media-bar]')).toBeNull();
  });

  it('syncs full-candidate (ALL) completion into batch cache and exports without truncation', async () => {
    const dense1000: Prediction['boxes'] = Array.from({ length: 1000 }, (_, i) => ({
      class: 0,
      name: 'circle',
      score: Number((0.99 - i * 0.0005).toFixed(4)),
      x1: (i % 20) * 15,
      y1: Math.floor(i / 20) * 10,
      x2: (i % 20) * 15 + 10,
      y2: Math.floor(i / 20) * 10 + 8,
    }));
    const full1005: Prediction['boxes'] = [
      ...dense1000,
      { class: 1, name: 'square', score: 0.4, x1: 5, y1: 5, x2: 15, y2: 15 },
    ];
    let sawAllRequest = false;
    const predictSpy = vi.fn<Api['predict']>(async (_input, opts) => {
      if (opts.max_det === 30_000) {
        sawAllRequest = true;
        return {
          model_id: opts.model_id,
          image: { width: 320, height: 240 },
          boxes: full1005,
          timing: { preprocess_ms: 1, inference_ms: 5, postprocess_ms: 1, total_ms: 7 },
          device: 'cpu',
        };
      }
      return {
        model_id: opts.model_id,
        image: { width: 320, height: 240 },
        boxes: dense1000,
        timing: { preprocess_ms: 1, inference_ms: 5, postprocess_ms: 1, total_ms: 7 },
        device: 'cpu',
      };
    });

    setupPlayground(predictSpy);
    await until(() => target.querySelector('.model-pick'), 'model picker');
    selectBatchFiles([makeImageFile('dense1.png'), makeImageFile('dense22.png')]);
    await until(() => sawAllRequest, 'askAllWhenCut fetched ALL candidates for dense active image');

    // Run all on item 2 (also dense at default conf 0.25 <= floor ~0.49) fetches ALL and completes cleanly
    const runAllBtn = [...target.querySelectorAll<HTMLButtonElement>('[data-batch-bar] button')].find((b) => b.textContent?.includes('Run all'))!;
    runAllBtn.click();
    await until(() => target.querySelector('[data-batch-summary]')?.textContent?.includes('2/2 run'), 'both dense images completed');
    expect(predictSpy.mock.calls.filter((c) => c[1].max_det === 30_000).length).toBeGreaterThanOrEqual(2);
  });
});
