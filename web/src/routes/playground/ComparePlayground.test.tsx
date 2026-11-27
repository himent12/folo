// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../../framework/testing';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
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

    const swapBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Swap Model A and Model B"]')!;
    swapBtn.click();
    flushSync();
    expect(selects()[0]!.value).toBe(beforeB);
    expect(selects()[1]!.value).toBe(beforeA);

    // Toggle Compare off returns cleanly to single-model view.
    const cmpBtn = [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Compare A/B'))!;
    cmpBtn.click();
    flushSync();
    expect(target.querySelector('[data-compare-view]')).toBeNull();
    expect(target.querySelector('#pg-compare-summary')).toBeNull();
  });

  it('opens the Video / camera frame capture dialog cleanly on the empty Playground', async () => {
    const api = new MockApi('normal', 0);
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground');
    handle = render(() => <Playground />, target);

    const captureBtn = await until(
      () => [...target.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.includes('Video / camera')),
      'capture button',
    );
    captureBtn.click();
    flushSync();

    const dlg = await until(() => target.querySelector('dialog[open]'), 'capture dialog');
    expect(dlg.textContent).toContain('Capture frame');
    // Switching to Camera tab degrades gracefully in happy-dom (no getUserMedia).
    const camTab = [...dlg.querySelectorAll<HTMLButtonElement>('[role="radio"]')].find((b) => b.textContent?.trim() === 'Camera')!;
    camTab.click();
    flushSync();
    expect(dlg.textContent).toContain('Camera capture is unavailable');
  });
});
