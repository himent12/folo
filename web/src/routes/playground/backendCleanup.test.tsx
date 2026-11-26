// @vitest-environment happy-dom
// Playground backend-switch cleanup, missing-weights clearing, and viewport sheet reset tests.
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, type Rendered } from '../../../framework/testing';
import { ApiError } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import Playground from '../Playground';
import { session } from './session';

let target: HTMLElement;
let c: Rendered | undefined;
const origWidth = window.innerWidth;
const setWidth = (width: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width });

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

afterEach(() => {
  c?.unmount();
  c = undefined;
  target?.remove();
  session.set(null);
  setWidth(origWidth);
});

describe('Playground backend switch and viewport cleanup', () => {
  it('clears stale server boxes and timing when switching to browser backend without server weights', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    api.predictInPage = async () => {
      throw new ApiError(501, 'not_implemented', 'This server cannot send weights.');
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    c = render(() => <Playground />, target);

    // Run on server first so boxes and timing populate.
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample')).click();
    await until(() => target.querySelector('#pg-panel tbody tr'), 'server detections');
    expect(target.querySelectorAll('#pg-panel tbody tr').length).toBeGreaterThan(0);

    // Switch Run on to CPU -> server cannot send weights -> stale server boxes & timing must clear.
    const group = target.querySelector('[role="radiogroup"][aria-label="Run the model on"]')!;
    const cpu = await until(
      () => [...group.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'CPU' && !b.disabled),
      'CPU option',
    );
    cpu.click();
    await until(() => target.textContent?.includes('Choose .folo file'), 'weights missing notice');
    expect(target.querySelector('#pg-panel tbody tr')).toBeNull();
    expect(target.querySelector('.vb-facts, .vfoot')?.textContent).toContain('Waiting for the model file');
    expect(target.querySelector('section[aria-labelledby="pg-timing"]')?.textContent).toContain('Timing appears after a run.');
  });

  it('clears hidden class filters when switching models even if a pending backend switch already cleared pred', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    api.predictInPage = async () => {
      throw new ApiError(501, 'not_implemented', 'This server cannot send weights.');
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    c = render(() => <Playground />, target);

    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample')).click();
    await until(() => target.querySelector('#pg-panel tbody tr'), 'detections');
    // Hide the first class on model A.
    const classesTab = [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((b) => b.textContent?.includes('Classes'))!;
    classesTab.click();
    flushSync();
    const firstToggle = target.querySelector<HTMLButtonElement>('#pg-panel button[aria-pressed="true"]')!;
    firstToggle.click();
    flushSync();
    expect(target.querySelector('.vb-facts')?.textContent).toContain('1 class hidden');

    // Switch to CPU (clears pred while weights are missing), then switch to model B and back to server.
    const group = target.querySelector('[role="radiogroup"][aria-label="Run the model on"]')!;
    const cpu = await until(
      () => [...group.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'CPU' && !b.disabled),
      'CPU option',
    );
    cpu.click();
    await until(() => target.textContent?.includes('Choose .folo file'), 'weights missing');

    const select = target.querySelector<HTMLSelectElement>('select[aria-label="Model"]')!;
    const otherModel = [...select.options].find((o) => o.value && o.value !== 'm_hard_s')!.value;
    select.value = otherModel;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    const serverBtn = [...group.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Server')!;
    serverBtn.click();
    await until(() => target.querySelector('#pg-panel button[aria-pressed]'), 'model B classes');
    expect(target.querySelector('.vb-facts')?.textContent).not.toContain('hidden');
  });

  it('closes the narrow thresholds sheet when widening to desktop so narrowing again does not re-open it', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    c = render(() => <Playground />, target);

    await until(() => target.querySelector('aside.rail'), 'initial desktop rail');
    setWidth(390);
    await until(() => !target.querySelector('aside.rail'), 'phone layout');

    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'sample')).click();
    await until(() => target.querySelector('#pg-panel tbody tr'), 'detections');
    const more = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('Thresholds'))!;
    more.click();
    await until(() => target.querySelector('#pg-more'), 'sheet open');

    setWidth(1280);
    await until(() => target.querySelector('aside.rail'), 'desktop rail');
    expect(target.querySelector('#pg-more')).toBeNull();

    setWidth(390);
    await until(() => !target.querySelector('aside.rail'), 'phone layout again');
    expect(target.querySelector('#pg-more')).toBeNull();
  });
});
