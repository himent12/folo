// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { pendingBlobRevokeCount, revokePendingBlobUrls } from '$lib/tableExport';
import Playground from '../Playground';
import * as exportMod from './export';
import { session } from './session';

let target: HTMLElement;
let handle: { unmount(): void } | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'condition'): Promise<T> {
  let v: T | null | undefined | false = undefined;
  await vi.waitFor(
    () => {
      flushSync();
      v = get();
      if (!v) throw new Error(`still waiting for ${what}`);
    },
    { timeout: 4000, interval: 10 },
  );
  return v as T;
}

async function openPlayground(path = '#/playground?model=m_hard_s') {
  const api = new MockApi('normal', 0);
  api.imageUrl = async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
  const predictSpy = vi.spyOn(api, 'predict');
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  history.replaceState(history.state, '', path);
  router.route = parse(path);
  handle = render(() => <Playground />, target);
  (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample image')).click();
  await until(() => target.querySelector('#pg-panel tbody tr'), 'initial detections');
  return { api, predictSpy };
}

afterEach(() => {
  handle?.unmount();
  handle = undefined;
  target?.remove();
  session.set(null);
  revokePendingBlobUrls();
  vi.restoreAllMocks();
});

describe('Playground: Compare A/B and Export', () => {
  it('makes zero extra prediction calls while Compare A/B is off, then runs Model B when toggled on', async () => {
    const { predictSpy } = await openPlayground('#/playground?model=m_hard_s');
    // Single mode: exactly one prediction call.
    expect(predictSpy).toHaveBeenCalledTimes(1);
    expect(target.querySelector('[data-compare-view]')).toBeNull();

    // Toggle Compare A/B on.
    const cmpBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Compare A/B'))!;
    expect(cmpBtn).toBeDefined();
    expect(cmpBtn.getAttribute('aria-pressed')).toBe('false');
    cmpBtn.click();
    flushSync();

    expect(cmpBtn.getAttribute('aria-pressed')).toBe('true');
    await until(() => target.querySelector('[data-compare-view="side-by-side"]'), 'side-by-side compare stage');
    await until(() => predictSpy.mock.calls.length === 2, 'Model B prediction call');
    expect(target.querySelector('#pg-compare-summary')?.textContent).toContain('A/B Summary');
    expect(target.querySelector('[data-compare-side="a"]')).not.toBeNull();
    expect(target.querySelector('[data-compare-side="b"]')).not.toBeNull();

    // Switch to Split layout and verify keyboard slider movement.
    const splitBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.trim() === 'Split')!;
    splitBtn.click();
    flushSync();
    const divider = await until(() => target.querySelector<HTMLElement>('.split-divider[role="slider"]'), 'split divider');
    expect(divider.getAttribute('aria-valuenow')).toBe('50');
    divider.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', shiftKey: true, bubbles: true }));
    flushSync();
    expect(divider.getAttribute('aria-valuenow')).toBe('60');

    // Synchronized zoom: clicking Zoom in updates the shared zoom readout.
    const zoomIn = target.querySelector<HTMLButtonElement>('[data-zoom="in"]')!;
    zoomIn.click();
    flushSync();
    expect(target.querySelector('.zoom-pct')?.textContent).toBe('150%');

    // Export JSON in Compare mode downloads structured A/B payload.
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:compare-json');
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const jsonBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Export detections JSON"]')!;
    jsonBtn.click();
    flushSync();
    expect(createUrl).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
  });

  it('swaps Model A and Model B cleanly and allows toggling Compare off without extra calls', async () => {
    const { predictSpy } = await openPlayground('#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n');
    await until(() => target.querySelector('[data-compare-view="side-by-side"]'), 'compare stage from URL');
    await until(() => predictSpy.mock.calls.length >= 2, 'both models run');

    const selects = () => [...target.querySelectorAll<HTMLSelectElement>('aside.rail select')];
    const beforeA = selects()[0]!.value;
    const beforeB = selects()[1]!.value;
    expect(beforeA).toBe('m_hard_s');
    expect(beforeB).toBe('m_shapes_n');

    // Swap should reuse swapped predictions without firing redundant network calls and write both models atomically.
    const callsBeforeSwap = predictSpy.mock.calls.length;
    const swapBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Swap Model A and Model B"]')!;
    swapBtn.click();
    flushSync();
    expect(selects()[0]!.value).toBe(beforeB);
    expect(selects()[1]!.value).toBe(beforeA);
    expect(router.route.query.get('model')).toBe(beforeB);
    expect(router.route.query.get('modelB')).toBe(beforeA);
    await new Promise((r) => setTimeout(r, 20));
    expect(predictSpy.mock.calls.length).toBe(callsBeforeSwap);

    // Toggle Compare off returns cleanly to single-model view.
    const cmpBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Compare A/B'))!;
    cmpBtn.click();
    flushSync();
    expect(target.querySelector('[data-compare-view]')).toBeNull();
    expect(target.querySelector('#pg-compare-summary')).toBeNull();
  });

  it('keeps CompareStage mounted across Model A updates and resets shared zoom when switching images', async () => {
    await openPlayground('#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n');
    const stage = await until(() => target.querySelector('[data-compare-view="side-by-side"]'), 'compare stage');

    // Zoom in while in Compare mode.
    target.querySelector<HTMLButtonElement>('[data-zoom="in"]')!.click();
    flushSync();
    expect(target.querySelector('.zoom-pct')?.textContent).toBe('150%');

    // Switching Model A should NOT remount the CompareStage node.
    const selectA = target.querySelectorAll<HTMLSelectElement>('aside.rail select')[0]!;
    selectA.value = 'm_shapes_s';
    selectA.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(target.querySelector('[data-compare-view="side-by-side"]')).toBe(stage);

    // Clearing the image and loading another resets shared zoom back to Fit.
    target.querySelector<HTMLButtonElement>('button[aria-label="Clear image"]')!.click();
    flushSync();
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample image after clear')).click();
    await until(() => target.querySelector('[data-compare-view="side-by-side"]'), 'compare stage on new image');
    expect(target.querySelector('.zoom-pct')?.textContent).toBe('Fit');
  });

  it('shows unavailable summary and Retry B without fabricating zero counts when Model B fails', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const origPredict = api.predict.bind(api);
    let failB = true;
    api.predict = async (input, opts) => {
      if (opts.model_id === 'm_shapes_n' && failB) throw new Error('Model B offline');
      return origPredict(input, opts);
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    history.replaceState(history.state, '', '#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n');
    router.route = parse('#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n');
    handle = render(() => <Playground />, target);
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample image')).click();
    const retryB = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Retry B')),
      'Retry B button',
    );
    // Summary omits Δ pill when Model B is unavailable.
    expect(target.querySelector('.ab-sum .delta')).toBeNull();
    // JSON tab stays accessible even when inspecting the failed side.
    const sideBBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.includes('Model B'))!;
    sideBBtn.click();
    flushSync();
    target.querySelector<HTMLButtonElement>('#pg-json')!.click();
    flushSync();
    await until(() => target.querySelector('#pg-panel pre'), 'compare JSON while B failed');

    // Retrying Model B recovers cleanly.
    failB = false;
    retryB.click();
    await until(() => target.querySelector('.ab-sum .delta') !== null, 'delta restored after Retry B');
  });

  it('stops a pending camera MediaStream if Stop camera is clicked before getUserMedia resolves', async () => {
    const stop = vi.fn();
    let resolveStream!: (s: MediaStream) => void;
    const pending = new Promise<MediaStream>((r) => (resolveStream = r));
    const origMediaDevices = navigator.mediaDevices;
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(() => pending) },
    });
    try {
      const api = new MockApi('normal', 0);
      hub.api = api;
      hub.mode = 'mock';
      target = document.createElement('div');
      document.body.appendChild(target);
      history.replaceState(history.state, '', '#/playground');
      router.route = parse('#/playground');
      handle = render(() => <Playground />, target);

      const camBtn = await until(
        () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
        'live camera button',
      );
      camBtn.click();
      flushSync();
      const bar = await until(() => target.querySelector('[data-media-bar="webcam"]'), 'webcam bar');

      // Stop camera while getUserMedia is still pending.
      const stopBtn = [...bar.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Stop camera'))!;
      stopBtn.click();
      flushSync();

      // Resolving the stale getUserMedia call must immediately stop every track.
      resolveStream({ getTracks: () => [{ stop }] } as unknown as MediaStream);
      await new Promise((r) => setTimeout(r, 10));
      expect(stop).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: origMediaDevices });
    }
  });

  it('degrades gracefully inline when Live camera is clicked without getUserMedia support', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    history.replaceState(history.state, '', '#/playground');
    router.route = parse('#/playground');
    handle = render(() => <Playground />, target);

    const camBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Live camera')),
      'live camera button',
    );
    camBtn.click();
    flushSync();

    const alert = await until(() => target.querySelector('[data-live-media="webcam"] [role="alert"]'), 'inline camera error');
    expect(alert.textContent).toContain('Camera access is unavailable');
  });

  it('hides a B-only class without affecting unrelated classes in Model A', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const origModels = api.models.bind(api);
    api.models = async () => {
      const list = await origModels();
      return list.map((m) =>
        m.id === 'm_hard_s'
          ? { ...m, classes: ['circle', 'square'] }
          : m.id === 'm_shapes_s'
            ? { ...m, classes: ['circle', 'star'] }
            : m.id === 'm_shapes_n'
              ? { ...m, classes: ['car', 'truck'] }
              : m,
      );
    };
    api.predict = async (_input, opts) => ({
      model_id: opts.model_id,
      image: { width: 640, height: 480 },
      boxes:
        opts.model_id === 'm_shapes_n'
          ? [
              { class: 0, name: 'car', score: 0.9, x1: 20, y1: 20, x2: 80, y2: 80 },
              { class: 1, name: 'truck', score: 0.85, x1: 90, y1: 20, x2: 140, y2: 80 },
            ]
          : [{ class: 0, name: 'circle', score: 0.9, x1: 10, y1: 10, x2: 50, y2: 50 }],
      timing: { preprocess_ms: 1, inference_ms: 5, postprocess_ms: 1, total_ms: 7 },
      device: 'gpu:0',
    });
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    history.replaceState(history.state, '', '#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n');
    router.route = parse('#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n');
    handle = render(() => <Playground />, target);
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample image')).click();
    await until(() => target.querySelector('[data-compare-view="side-by-side"]'), 'compare stage');
    await until(() => target.querySelector('.ab-sum .delta') !== null, 'both predictions loaded');

    // Switch inspect side to Model B, open Classes tab, and hide 'car' (index 0 in B, absent from A).
    const sideBBtn = [...target.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.includes('Model B'))!;
    sideBBtn.click();
    flushSync();
    const classesTab = [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((t) => t.textContent?.includes('Classes'))!;
    classesTab.click();
    flushSync();
    const carToggle = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('#pg-panel ul.counts button')].find((b) => b.textContent?.includes('car')),
      'car class toggle',
    );
    carToggle.click();
    flushSync();

    // Model B's 'car' is now hidden (countB === 1, only 'truck' visible), while Model A's 'circle' stays visible (countA === 1).
    const detKpi = target.querySelector('.ab-sum .k-vals')!;
    const readKpi = () => [...detKpi.querySelectorAll(':scope > span')].map((s) => s.textContent?.replace(/\s+/g, ' ').trim());
    expect(readKpi()).toEqual(['A 1', 'vs', 'B 1', 'Δ 0']);

    // Switching Model A to another model that still lacks 'car' preserves Model B's hidden 'car' filter AND keeps focus on the picker.
    const selectA = target.querySelectorAll<HTMLSelectElement>('aside.rail select')[0]!;
    selectA.focus();
    selectA.value = 'm_shapes_s';
    selectA.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(selectA.value).toBe('m_shapes_s');
    expect(document.activeElement).toBe(selectA);
    await until(
      () => readKpi()[0] === 'A 1' && readKpi()[2] === 'B 1',
      'Model A settled and B car stayed hidden after Model A switch to model without car',
    );
    expect(readKpi()).toEqual(['A 1', 'vs', 'B 1', 'Δ 0']);

    // Switching Model A to a model that includes 'car' clears the now-shared B-only hidden name so 'car' is never stuck hidden.
    selectA.value = 'm_shapes_n';
    selectA.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(selectA.value).toBe('m_shapes_n');
    expect(document.activeElement).toBe(selectA);
    await until(() => readKpi()[2] === 'B 2', 'B car unhidden after Model A switch to model with car');
  });

  it('retains keyboard focus on both Compare model pickers when changing selections', async () => {
    await openPlayground('#/playground?model=m_hard_s&compare=1');
    await until(() => target.querySelectorAll('aside.rail select').length === 2, 'two compare model pickers');
    const [selectA, selectB] = [...target.querySelectorAll<HTMLSelectElement>('aside.rail select')];
    expect(selectA).toBeTruthy();
    expect(selectB).toBeTruthy();

    selectA!.focus();
    const nextA = [...selectA!.options].find((o) => o.value !== selectA!.value)!.value;
    selectA!.value = nextA;
    selectA!.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(document.activeElement).toBe(selectA);

    selectB!.focus();
    const nextB = [...selectB!.options].find((o) => o.value !== selectB!.value)!.value;
    selectB!.value = nextB;
    selectB!.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(document.activeElement).toBe(selectB);
  });

  it('shows Both / Only A / Only B class-diff breakdown and one-click Compare JSON & side-by-side PNG exports without timer leaks', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    api.predict = async (_input, opts) => ({
      model_id: opts.model_id,
      image: { width: 640, height: 480 },
      boxes:
        opts.model_id === 'm_hard_s'
          ? [
              { class: 0, name: 'circle', score: 0.92, x1: 10, y1: 10, x2: 60, y2: 60 },
              { class: 1, name: 'square', score: 0.86, x1: 70, y1: 10, x2: 120, y2: 60 },
            ]
          : [
              { class: 0, name: 'circle', score: 0.95, x1: 12, y1: 12, x2: 62, y2: 62 },
              { class: 2, name: 'triangle', score: 0.88, x1: 140, y1: 20, x2: 190, y2: 70 },
            ],
      timing: { preprocess_ms: 1, inference_ms: 6, postprocess_ms: 1, total_ms: 8 },
      device: 'gpu:0',
    });
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    history.replaceState(history.state, '', '#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n&view=split');
    router.route = parse('#/playground?model=m_hard_s&compare=1&modelB=m_shapes_n&view=split');
    handle = render(() => <Playground />, target);
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample image')).click();
    const diffStrip = await until(() => target.querySelector<HTMLElement>('[data-class-diff]'), 'class diff strip');

    const pills = [...diffStrip.querySelectorAll<HTMLButtonElement>('button.diff-pill')];
    expect(pills.map((p) => p.textContent?.replace(/\s+/g, ' ').trim())).toEqual(['Both 1', 'Only A 1', 'Only B 1']);

    // Filtering by Only A shows only 'square'; clicking again resets to all 3 classes.
    pills[1]!.click();
    flushSync();
    const clsNames = () => [...target.querySelectorAll('.cls-table tbody td.lead .truncate')].map((el) => el.textContent?.trim());
    expect(clsNames()).toEqual(['square']);
    pills[1]!.click();
    flushSync();
    expect(clsNames().sort()).toEqual(['circle', 'square', 'triangle']);

    // One-click Side-by-side PNG inside CompareSummary exports side-by-side even when Split view is active.
    const pngSpy = vi.spyOn(exportMod, 'exportComparePng').mockResolvedValue(true);
    const pngBtn = target.querySelector<HTMLButtonElement>('[data-compare-export] button[aria-label="Export side-by-side annotated PNG"]')!;
    pngBtn.click();
    await vi.waitFor(() => expect(pngSpy).toHaveBeenCalledTimes(1));
    expect(pngSpy.mock.calls[0]![0].view).toBe('side-by-side');

    // One-click Compare JSON exports structured payload with class_diff and revokes cleanly.
    const created: Blob[] = [];
    const revoked: string[] = [];
    vi.spyOn(URL, 'createObjectURL').mockImplementation((b) => {
      created.push(b as Blob);
      return `blob:cmp-summary/${created.length}`;
    });
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((u) => revoked.push(u));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    const jsonBtn = target.querySelector<HTMLButtonElement>('[data-compare-export] button[aria-label="Export A/B compare JSON"]')!;
    jsonBtn.click();
    flushSync();
    expect(created).toHaveLength(1);
    const parsed = JSON.parse(await created[0]!.text());
    expect(parsed.mode).toBe('compare');
    expect(parsed.comparison.class_diff).toEqual({
      both: ['circle'],
      only_a: ['square'],
      only_b: ['triangle'],
    });
    expect(pendingBlobRevokeCount()).toBe(1);
    revokePendingBlobUrls();
    expect(pendingBlobRevokeCount()).toBe(0);
    expect(revoked).toEqual(['blob:cmp-summary/1']);
  });
});
