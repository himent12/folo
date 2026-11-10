// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index.svelte';
import { MockApi, type Scenario } from '$lib/api/mock';
import { resetWasmProbe } from '$lib/inference/browser.svelte';
import { prefs } from '$lib/prefs.svelte';
import { parse, router } from '$lib/router.svelte';
import Playground from './Playground.svelte';

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what = 'the condition'): Promise<T> {
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

/** Opens the Playground on `model` and runs it on the first sample image. */
async function run(scenario: Scenario, model: string) {
  const api = new MockApi(scenario, 0);
  api.imageUrl = async () => 'data:,';
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  router.route = parse(`#/playground?model=${model}`);
  c = mount(Playground, { target });
  (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
}

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target?.remove();
});

describe('Playground: a failed prediction', () => {
  it('shows the error once, above the image, and moves focus to it', async () => {
    await run('oom', 'm_shapes_s');
    const notice = await until(() => target.querySelector('.notice'), 'the notice');
    // Above the image, where the result would be, not over it: the stage holds only the image. The same at every width.
    expect(target.querySelector('.stage .notice, .stage [role="alert"]')).toBeNull();
    expect(notice.nextElementSibling?.classList.contains('stage')).toBe(true);
    expect(notice.previousElementSibling?.classList.contains('vbar')).toBe(true);
    const title = notice.querySelector<HTMLElement>('.title')!;
    expect(title.textContent).toBe('Prediction failed');
    expect(document.activeElement).toBe(title);
    // One authoritative block: the caption under it adds only which model ran where.
    const foot = target.querySelector('.vfoot')?.textContent?.replace(/\s+/g, ' ').trim();
    expect(foot).not.toContain('Prediction failed');
    expect(foot).not.toContain('out of GPU memory');
    expect(foot).toContain('shapes-s-best on the server');
    expect(foot).not.toContain('Waiting');
    // The detections panel says why it is empty.
    expect(target.querySelector('#pg-panel .placeholder')).not.toBeNull();
    // The ways out: this browser first, then the smaller model, then Retry.
    expect([...notice.querySelectorAll('button')].map((b) => b.textContent?.trim()).filter((t) => t !== 'Technical details')).toEqual([
      'Run in this browser',
      'Use shapes-n-best',
      'Retry',
    ]);
  });
});

describe('Playground on a phone', () => {
  const width = window.innerWidth;
  afterEach(() => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width });
  });

  it('opens Thresholds as a sheet', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 390 });
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const btn = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('Thresholds'))!;
    btn.click();
    const sheet = await until(() => target.querySelector('#pg-more'), 'the sheet');
    expect(sheet.classList.contains('pop')).toBe(false);
    expect(sheet.querySelector('.slider input')).not.toBeNull();
  });
});

describe('Playground on a tablet', () => {
  const width = window.innerWidth;
  afterEach(() => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width });
  });

  // Bottom sheets are for phones (DESIGN_DIRECTION 9): round 24's tablet sheet covered Classes and Timing, clipped at
  // the viewport. More settings is a popover anchored under its button; focus moves in, and Escape brings it back.
  it('opens More settings as an anchored popover, not a sheet, and Escape returns focus to the button', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 820 });
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const btn = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('More settings'))!;
    btn.click();
    const pop = await until(() => target.querySelector<HTMLElement>('#pg-more'), 'the popover');
    expect(pop.classList.contains('pop')).toBe(true);
    expect(pop.querySelector('.sheet-handle')).toBeNull();
    await until(() => document.activeElement === pop, 'focus in the popover');
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    flushSync();
    expect(target.querySelector('#pg-more')).toBeNull();
    expect(document.activeElement).toBe(btn);
  });
});

describe('Playground: detections', () => {
  it('picks a row with aria-selected, its score the same in the table and the picked line', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelectorAll('#pg-panel tbody tr').length > 1, 'the detections');
    const rows = () => [...target.querySelectorAll<HTMLTableRowElement>('#pg-panel tbody tr')];
    expect(rows().some((r) => r.hasAttribute('aria-selected'))).toBe(false);
    const cell = rows()[1]!.querySelector('.score .num')!.textContent!;
    rows()[1]!.click();
    flushSync();
    expect(rows().map((r) => r.getAttribute('aria-selected'))).toEqual(rows().map((_, i) => (i === 1 ? 'true' : null)));
    expect(rows()[1]!.getAttribute('aria-label')).toContain(`score ${cell},`);
    expect(target.querySelector('.pick-line')?.textContent).toContain(cell);
  });
});

describe('Playground without the in-browser runtime (DOGFOOD-3 S5)', () => {
  async function open(scenario: Scenario) {
    hub.api = new MockApi(scenario, 0);
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_s');
    c = mount(Playground, { target });
    return until(() => target.querySelector('[role="radiogroup"][aria-label="Run the model on"]'), 'Run on');
  }
  const option = (g: Element, name: string) => [...g.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === name)!;

  it('offers WebGPU and CPU off, with the reason, where the build has none', async () => {
    const g = await open('no-wasm');
    await until(() => option(g, 'CPU').disabled, 'the CPU choice to turn off');
    expect(option(g, 'Server').disabled).toBe(false);
    expect(option(g, 'WebGPU').disabled).toBe(true);
    expect(target.querySelector('.runon')?.getAttribute('aria-describedby')).toBe('runon-note');
  });

  it('offers the CPU where the build has it', async () => {
    const g = await open('normal');
    await until(() => target.querySelector('select, .model-pick'), 'the pickers');
    await new Promise((r) => setTimeout(r, 50));
    flushSync();
    expect(option(g, 'CPU').disabled).toBe(false);
    expect(target.querySelector('#runon-note')?.textContent ?? '').not.toContain('runtime');
  });

  // Review of round 25: a Studio without its own runtime files, connected to another server whose build lists
  // `web_wasm`, kept the browser choices on and failed when a model loaded. That server's word is not this page's: the
  // page asks its own files, and they are missing.
  it("asks its own files when another server lists web_wasm", async () => {
    const url = prefs.serverUrl;
    prefs.serverUrl = 'http://gpu-box:7878';
    resetWasmProbe();
    const fetch = vi.fn(async () => new Response(null, { status: 404 }));
    vi.stubGlobal('fetch', fetch);
    try {
      const api = new MockApi('normal', 0);
      Object.defineProperty(api, 'mock', { value: false });
      hub.api = api;
      hub.mode = 'live';
      target = document.createElement('div');
      document.body.appendChild(target);
      router.route = parse('#/playground?model=m_shapes_s');
      c = mount(Playground, { target });
      const g = await until(() => target.querySelector('[role="radiogroup"][aria-label="Run the model on"]'), 'Run on');
      expect((await api.system()).features).toContain('web_wasm');
      await until(() => option(g, 'CPU').disabled, 'the CPU choice to turn off');
      expect(fetch).toHaveBeenCalledWith(expect.stringContaining('folo-wasm/folo_wasm_bg.wasm'), { method: 'HEAD' });
      expect(option(g, 'WebGPU').disabled).toBe(true);
      expect(target.querySelector('#runon-note')?.textContent).toBe('No in-browser runtime');
    } finally {
      vi.unstubAllGlobals();
      resetWasmProbe();
      prefs.serverUrl = url;
    }
  });
});
