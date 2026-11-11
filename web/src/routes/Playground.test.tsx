// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi, type Scenario } from '$lib/api/mock';
import { resetWasmProbe } from '$lib/inference/browser';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import Playground from './Playground';
import { session } from './playground/session';

/** Port of `mount(Playground, { target })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof Playground, o: { target: HTMLElement }) => render(() => <Playground />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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
  // Each test starts a fresh tab: no image or thresholds kept from the one before.
  session.set(null);
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
    await until(() => target.querySelector('.model-pick'), 'the model picker');
    return target.querySelector('[role="radiogroup"][aria-label="Run the model on"]');
  }
  const option = (g: Element, name: string) => [...g.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === name)!;

  // models F14: options this build can never use are not shown (it runs on the server, and nothing needs saying).
  it('offers no browser choice where the build has none, and says nothing about it', async () => {
    await open('no-wasm');
    await until(() => !target.querySelector('.runon'), 'the Run on choice to go');
    expect(target.textContent).not.toContain('in-browser runtime');
  });

  it('offers the CPU where the build has it', async () => {
    await open('normal');
    const g = await until(() => target.querySelector('[role="radiogroup"][aria-label="Run the model on"]'), 'Run on');
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
      await until(() => target.querySelector('.model-pick'), 'the model picker');
      expect((await api.system()).features).toContain('web_wasm');
      await until(() => !target.querySelector('.runon'), 'the Run on choice to go');
      expect(fetch).toHaveBeenCalledWith(expect.stringContaining('folo-wasm/folo_wasm_bg.wasm'), { method: 'HEAD' });
    } finally {
      vi.unstubAllGlobals();
      resetWasmProbe();
      prefs.serverUrl = url;
    }
  });
});

describe('Playground: an end-to-end (o2o) model', () => {
  // An o2o model predicts without NMS (docs/MODEL.md §6.6; folo-vision's predictor takes its top-k): the server keeps
  // two overlapping boxes of one class, and so must the Playground's own filtering, which ran class-aware NMS at the
  // IoU slider's 0.7 for every model.
  async function detections(model: string) {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    const head = (await api.model(model)).head;
    const box = (x: number, score: number) => ({ class: 0, name: 'circle', score, x1: x, y1: 0, x2: x + 100, y2: 100 });
    api.predict = async (_input, opts) => ({
      model_id: opts.model_id,
      image: { width: 640, height: 640 },
      boxes: [box(0, 0.9), box(5, 0.8)], // IoU 0.90
      timing: { preprocess_ms: 1, inference_ms: 1, postprocess_ms: 1, total_ms: 3 },
      device: 'gpu:0',
    });
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse(`#/playground?model=${model}`);
    c = mount(Playground, { target });
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    return { head, rows: target.querySelectorAll('#pg-panel tbody tr').length };
  }

  it('shows every box the server sent, overlapping ones of one class too', async () => {
    expect(await detections('m_shapes_s')).toEqual({ head: 'o2o', rows: 2 });
  });

  it('still suppresses them for an NMS (o2m) model', async () => {
    expect(await detections('m_shapes_n')).toEqual({ head: 'o2m', rows: 1 });
  });
});

describe('Playground: the session', () => {
  it('keeps the image, its result and the thresholds when the page is left and opened again', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const conf = target.querySelector<HTMLInputElement>('.slider input[type="range"]')!;
    conf.value = '0.5';
    conf.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const rows = target.querySelectorAll('#pg-panel tbody tr').length;
    unmount(c!);
    target.remove();
    const predict = vi.spyOn(hub.api, 'predict');
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground');
    c = mount(Playground, { target });
    await until(() => target.querySelectorAll('#pg-panel tbody tr').length === rows, 'the same detections');
    expect(target.querySelector<HTMLInputElement>('.slider input[type="range"]')!.value).toBe('0.5');
    expect(target.querySelector('button.sample')).toBeNull();
    // The kept result is shown as it was: the same answer is not asked for twice.
    expect(predict).not.toHaveBeenCalled();
  });

  it('opens a linked dataset image with its thresholds', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';
    const ds = (await api.datasets())[0]!;
    const img = (await api.datasetImages(ds.id, { split: 'val', limit: 3 })).images[2]!;
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse(`#/playground?model=m_hard_s&dataset=${ds.id}&image=${encodeURIComponent(img.id)}&split=val&conf=0.4`);
    c = mount(Playground, { target });
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    expect(session.get()?.source).toMatchObject({ kind: 'dataset', img: { id: img.id }, at: 2 });
    expect(target.querySelector<HTMLInputElement>('.slider input[type="range"]')!.value).toBe('0.4');
  });
});

describe('Playground: a failure is said once', () => {
  it('shows a failed prediction in place, with no toast', async () => {
    const { toast } = await import('$lib/ui/toast');
    toast.items = [];
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    api.predict = async () => {
      throw new ApiError(422, 'image_decode_failed', 'not an image', 'send a JPEG, PNG, BMP or WebP file');
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    c = mount(Playground, { target });
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
    await until(() => target.querySelector('.notice'), 'the notice');
    expect(target.querySelector('.notice .title')?.textContent).toBe('Prediction failed');
    expect(toast.items).toEqual([]);
  });
});

describe('Playground: a cold first prediction', () => {
  it("says the first call's time includes loading the model, and the next one does not", async () => {
    await run('normal', 'm_hard_s');
    const note = await until(() => target.querySelector('p.cold'), 'the cold note');
    expect(note.textContent).toBe('Includes loading the model: the next image runs faster.');
    // A first call's rate would be the warm-up's: none is given.
    expect(note.parentElement!.querySelector('.rate')).toBeNull();
    // The model is warm now: the next call is not cold.
    const api = hub.api as MockApi;
    const ds = (await api.datasets())[0]!;
    const img = (await api.datasetImages(ds.id, { split: 'val', limit: 2 })).images[1]!;
    const p = await api.predict({ kind: 'dataset', dataset_id: ds.id, image_id: img.id }, { model_id: 'm_hard_s' });
    expect(p.cold).toBeUndefined();
  });
});
