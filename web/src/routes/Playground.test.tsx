// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from '../../framework/testing';
import { hub } from '$lib/api/index';
import { ApiError } from '$lib/api/client';
import { MockApi, type Scenario } from '$lib/api/mock';
import { resetWasmProbe } from '$lib/inference/browser';
import { prefs } from '$lib/prefs';
import { betterModel } from '$lib/scoreSummary';
import { toast } from '$lib/ui/toast';
import { parse, router } from '$lib/router';
import Playground from './Playground';
import { URL_DELAY_MS } from './playground/keepSession';
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
    const foot = target.querySelector('.vfoot, .vb-facts')?.textContent?.replace(/\s+/g, ' ').trim();
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
    expect(sheet.querySelectorAll('.slider').length).toBe(3);
  });

  it('hides the IoU slider and footer IoU readout for an end-to-end (o2o) model on phones', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 390 });
    await run('normal', 'm_shapes_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    expect(target.querySelector('.vfoot')?.textContent).not.toContain('IoU');
    const btn = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('Thresholds'))!;
    btn.click();
    const sheet = await until(() => target.querySelector('#pg-more'), 'the sheet');
    expect(sheet.querySelectorAll('.slider').length).toBe(2);
    expect(sheet.querySelector('.no-nms')?.textContent).toContain('Not used · end-to-end model');
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
    expect(target.querySelector('.qconf .qiou')).not.toBeNull();
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

  it('hides the IoU slider in the quick strip and explains it in More settings for an end-to-end (o2o) model on tablets', async () => {
    (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: 820 });
    await run('normal', 'm_shapes_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const qconf = target.querySelector('.qconf')!;
    expect(qconf.classList.contains('single')).toBe(true);
    expect(qconf.querySelector('.qiou')).toBeNull();
    expect(qconf.querySelectorAll('.slider').length).toBe(1);
    expect(target.querySelector('.vfoot')?.textContent).not.toContain('IoU');
    const btn = [...target.querySelectorAll<HTMLButtonElement>('.more-btn button')].find((b) => b.textContent?.includes('More settings'))!;
    btn.click();
    const pop = await until(() => target.querySelector<HTMLElement>('#pg-more'), 'the popover');
    expect(pop.querySelector('.no-nms')?.textContent).toContain('Not used · end-to-end model');
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
    await until(() => !option(g, 'CPU').disabled, 'the CPU offered');
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

describe('Playground: a busy image (more candidates than the first answer holds)', () => {
  // The first answer holds the 1,000 highest-scoring candidates (FETCH). On a busy image they are near-duplicates of a
  // few objects, so filtering them at a low confidence lost boxes the server finds (models F5: 194 shown where
  // POST /api/predict at the same thresholds gave 300).
  const box = (x: number, score: number) => ({ class: 0, name: 'circle', score, x1: x, y1: 0, x2: x + 100, y2: 100 });
  // 1,000 copies of 3 objects, scores 0.90 down to `floor`; all candidates add 5 more objects at 0.15.
  const first = (floor: number) => Array.from({ length: 1000 }, (_, i) => box((i % 3) * 200, 0.9 - ((0.9 - floor) * i) / 999));
  const all = (floor: number) => [...first(floor), ...[1, 2, 3, 4, 5].map((k) => box(600 + k * 200, 0.15))];

  /**
   * `first`: what the first request for every candidate does: never answer (the page is left while it runs) or fail
   * (the server is busy).
   */
  async function open(floor: number, firstAll: 'answer' | 'hang' | 'fail' = 'answer') {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    const asked: number[] = [];
    api.predict = async (_input, opts) => {
      asked.push(opts.max_det ?? 300);
      if ((opts.max_det ?? 300) > 1000 && asked.filter((n) => n > 1000).length === 1) {
        if (firstAll === 'hang') await new Promise(() => {});
        if (firstAll === 'fail') throw new ApiError(503, 'busy', 'The GPU is busy.');
      }
      return {
        model_id: opts.model_id,
        image: { width: 2000, height: 640 },
        boxes: (opts.max_det ?? 300) > 1000 ? all(floor) : first(floor),
        timing: { preprocess_ms: 1, inference_ms: 1, postprocess_ms: 1, total_ms: 3 },
        device: 'gpu:0',
      };
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_n');
    c = mount(Playground, { target });
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    return asked;
  }
  const rows = () => target.querySelectorAll('#pg-panel tbody tr').length;
  const foot = () => target.querySelector('.vfoot, .vb-facts')?.textContent?.replace(/\s+/g, ' ') ?? '';
  function setConf(v: number) {
    const conf = target.querySelector<HTMLInputElement>('.slider input[type="range"]')!;
    conf.value = String(v);
    conf.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
  }

  it('asks once for every candidate when the confidence goes under the cut, and shows what they hold', async () => {
    const asked = await open(0.2);
    // Above the cut the first answer is exact: no second request.
    expect(rows()).toBe(3);
    expect(asked).toEqual([1000]);
    expect(foot()).toContain('of 1,000+ candidates');
    setConf(0.1);
    await until(() => rows() === 8, 'the boxes only all candidates hold');
    expect(asked).toEqual([1000, 30_000]);
    expect(foot()).toContain('of 1,005 candidates');
    // Further down the answer is complete: nothing more is asked.
    setConf(0.05);
    setConf(0.01);
    await new Promise((r) => setTimeout(r, 20));
    expect(asked).toEqual([1000, 30_000]);
  });

  it('asks again for a kept answer when the page is opened again, once the model list is in', async () => {
    const asked = await open(0.2, 'hang');
    setConf(0.1);
    await until(() => asked.length === 2, 'the request for every candidate');
    unmount(c!);
    target.remove();
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_n');
    c = mount(Playground, { target });
    await until(() => rows() === 8, `every candidate, asked for again (asked ${asked}, rows ${rows()})`);
    expect(asked).toEqual([1000, 30_000, 30_000]);
  });

  it('asks for the model file when the browser cannot get every candidate from the server weights', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    // In the browser: the first answer came (from a .folo picked earlier, not kept), the server has no weights to send.
    api.predictInPage = async (_input, opts) => {
      if ((opts.max_det ?? 300) > 1000) throw new ApiError(501, 'not_implemented', 'This server cannot send weights.');
      const p = { model_id: opts.model_id, image: { width: 2000, height: 640 }, boxes: first(0.2), device: 'cpu' };
      return { ...p, timing: { preprocess_ms: 1, inference_ms: 1, postprocess_ms: 1, total_ms: 3 } };
    };
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_shapes_n');
    c = mount(Playground, { target });
    const g = await until(() => target.querySelector('[role="radiogroup"][aria-label="Run the model on"]'), 'Run on');
    const cpu = await until(() => [...g.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'CPU' && !b.disabled), 'the CPU');
    cpu.click();
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
    await until(() => rows() === 3, 'the first answer');
    setConf(0.1);
    await until(() => target.textContent?.includes('Choose .folo file'), 'the request for the model file');
  });

  it('says when every candidate could not be loaded, and Retry asks again', async () => {
    const shown = vi.spyOn(toast, 'show');
    const asked = await open(0.2, 'fail');
    setConf(0.1);
    const t = await until(() => shown.mock.calls.map((c) => c[0]).find((x) => x.title === 'Could not load every candidate'), 'the error');
    expect(t.detail).toContain('below 0.20 may be missing');
    expect(rows()).toBe(3);
    expect(foot()).toContain('1,000+ candidates');
    t.action!.run();
    await until(() => rows() === 8, 'every candidate, after Retry');
    expect(asked).toEqual([1000, 30_000, 30_000]);
  });

  it('takes the failure back with the answer it is about: another image, or the page left', async () => {
    const live = () => toast.items.filter((t) => t.title === 'Could not load every candidate').length;
    await open(0.2, 'fail');
    setConf(0.1);
    await until(() => live() === 1, 'the error');
    target.querySelector<HTMLButtonElement>('button[aria-label="Clear image"]')!.click();
    await until(() => live() === 0, 'the error to go with the image');
    unmount(c!);
    c = undefined;
    target.remove();
    await open(0.2, 'fail');
    setConf(0.1);
    await until(() => live() === 1, 'the error again');
    unmount(c!);
    c = undefined;
    expect(live()).toBe(0);
  });

  it('does not complete an answer that a new prediction is replacing (another backend)', async () => {
    const asked = await open(0.2);
    const inPage: number[] = [];
    (hub.api as MockApi).predictInPage = (_input, opts) => {
      inPage.push(opts.max_det ?? 300);
      return new Promise(() => {});
    };
    const g = target.querySelector('[role="radiogroup"][aria-label="Run the model on"]')!;
    const cpu = await until(() => [...g.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'CPU' && !b.disabled), 'the CPU');
    cpu.click();
    await until(() => inPage.length === 1, 'the prediction on the CPU');
    setConf(0.1);
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(inPage).toEqual([1000]);
    expect(asked).toEqual([1000]);
  });

  it('keeps the first answer above its cut, whatever the slider did before', async () => {
    const asked = await open(0.2);
    setConf(0.5);
    setConf(0.21);
    await new Promise((r) => setTimeout(r, 20));
    expect(asked).toEqual([1000]);
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

  const link = (ds: string, img: string, at: number, rest: string) => parse(`#/playground?${rest}&dataset=${ds}&image=${encodeURIComponent(img)}&split=val&at=${at}`);

  // review R1: only the query changing keeps the page mounted (an edited URL, Back, a link to itself); it follows the
  // URL, the model too after a pick in the list, and its own later write never puts the old state back.
  it('follows a URL changed while it is open', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';
    const ds = (await api.datasets())[0]!;
    const [, a, b] = (await api.datasetImages(ds.id, { split: 'val', limit: 3 })).images;
    const predict = vi.spyOn(api, 'predict');
    const last = () => predict.mock.calls.at(-1);
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = link(ds.id, a!.id, 1, 'model=m_hard_s&conf=0.4');
    c = mount(Playground, { target });
    await until(() => target.querySelector('#pg-panel tbody tr') && last()?.[1].model_id === 'm_hard_s', "A's detections");
    const select = target.querySelector<HTMLSelectElement>('.model-pick select')!;
    select.value = 'm_shapes_n';
    select.dispatchEvent(new Event('change', { bubbles: true }));
    await until(() => last()?.[1].model_id === 'm_shapes_n', 'the picked model');

    // As an edited URL does: the browser moves, and the router follows (`hashchange`).
    const to = link(ds.id, b!.id, 2, 'model=m_hard_s&conf=0.8');
    location.hash = `#${to.path}?${to.query}`;
    await until(() => router.route.query.get('image') === b!.id, 'the route');
    const onB = () => (last()?.[0] as { image_id?: string } | undefined)?.image_id === b!.id && last()?.[1].model_id === 'm_hard_s';
    await until(() => onB() && session.get()?.pred?.model_id === 'm_hard_s', "B's detections by the linked model");
    expect(session.get()?.source).toMatchObject({ img: { id: b!.id } });
    expect(target.querySelector<HTMLInputElement>('.slider input[type="range"]')!.value).toBe('0.8');
    await new Promise((r) => setTimeout(r, URL_DELAY_MS + 100));
    flushSync();
    expect(Object.fromEntries(router.route.query)).toMatchObject({ model: 'm_hard_s', image: b!.id, conf: '0.8' });
    expect(session.get()?.source).toMatchObject({ img: { id: b!.id } });
  });

  // review studio-2 R2: a URL leaves out the thresholds that are the defaults, so one followed while the page is open
  // that names none means the defaults, not the values on screen; its own later write keeps them so.
  it('takes the defaults from a followed URL that names no thresholds, on Back and Forward', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';
    const ds = (await api.datasets())[0]!;
    const [, a] = (await api.datasetImages(ds.id, { split: 'val', limit: 2 })).images;
    target = document.createElement('div');
    document.body.appendChild(target);
    const plain = link(ds.id, a!.id, 1, 'model=m_hard_s');
    const tuned = link(ds.id, a!.id, 1, 'model=m_hard_s&conf=0.8&iou=0.9&max=20');
    location.hash = `#${plain.path}?${plain.query}`;
    router.route = plain;
    c = mount(Playground, { target });
    await until(() => target.querySelector('#pg-panel tbody tr'), "A's detections");
    const thresholds = () => {
      const s = session.get()!;
      return [s.conf, s.iou, s.maxDet];
    };
    const written = () => ['conf', 'iou', 'max'].map((p) => router.route.query.get(p));
    const settled = async () => {
      await new Promise((r) => setTimeout(r, URL_DELAY_MS + 100));
      flushSync();
    };
    expect(thresholds()).toEqual([0.25, 0.7, 300]);

    // A link to the tuned URL, then Back to the plain one and Forward again: the browser moves, the router follows.
    location.hash = `#${tuned.path}?${tuned.query}`;
    await until(() => router.route.query.get('conf') === '0.8', 'the tuned route');
    await until(() => thresholds()[0] === 0.8, 'the tuned thresholds');
    expect(thresholds()).toEqual([0.8, 0.9, 20]);
    await settled();
    expect(written()).toEqual(['0.8', '0.9', '20']);

    history.back();
    await until(() => router.route.query.get('conf') === null, 'the plain route');
    await until(() => thresholds()[0] === 0.25, 'the default thresholds');
    expect(thresholds()).toEqual([0.25, 0.7, 300]);
    expect(target.querySelector<HTMLInputElement>('.slider input[type="range"]')!.value).toBe('0.25');
    // The page's own write a moment later names no threshold either (it would put 0.8 back otherwise).
    await settled();
    expect(written()).toEqual([null, null, null]);

    history.forward();
    await until(() => router.route.query.get('conf') === '0.8', 'the tuned route again');
    await until(() => thresholds()[0] === 0.8, 'the tuned thresholds again');
    expect(thresholds()).toEqual([0.8, 0.9, 20]);
    await settled();
    expect(written()).toEqual(['0.8', '0.9', '20']);
  });

  // review R2: a linked image still being looked up is a choice like any other; one made after it wins.
  it('keeps an image chosen while a linked one loads', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    let release!: () => void;
    const slow = new Promise<void>((r) => (release = r));
    const dataset = api.dataset.bind(api);
    api.dataset = async (id) => (await slow, dataset(id));
    hub.api = api;
    hub.mode = 'mock';
    const ds = (await api.datasets())[0]!;
    const linked = (await api.datasetImages(ds.id, { split: 'val', limit: 6 })).images[5]!;
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = link(ds.id, linked.id, 5, 'model=m_hard_s');
    c = mount(Playground, { target });
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
    await until(() => target.querySelector('#pg-panel tbody tr'), "the sample's detections");
    const chosen = (session.get()?.source as { img: { id: string } }).img.id;
    expect(chosen).not.toBe(linked.id);
    release();
    await new Promise((r) => setTimeout(r, URL_DELAY_MS + 100));
    flushSync();
    expect(session.get()?.source).toMatchObject({ img: { id: chosen } });
    expect(router.route.query.get('image')).toBe(chosen);
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

describe('Playground: the model caption', () => {
  // models F17: a model without a score kept no caption line, and the workspace moved up 20 px on switching to it.
  it('keeps the line under the model select for a model without a score', async () => {
    const api = new MockApi('normal', 0);
    const models = api.models.bind(api);
    api.models = async () => (await models()).map((m) => (m.id === 'm_hard_s' ? { ...m, metrics: null } : m));
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    c = mount(Playground, { target });
    const meta = await until(() => target.querySelector('#model-meta'), 'the caption');
    expect(meta.textContent).toBe('Not evaluated');
  });
});

describe('Playground: which model it opens on', () => {
  const open = async (api: MockApi, link: string) => {
    hub.api = api;
    hub.mode = 'mock';
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse(link);
    c = mount(Playground, { target });
    return until(() => target.querySelector<HTMLSelectElement>('select')?.value, 'a model');
  };

  // v-desktop N8: Studio opened on the first group's model (a newest checkpoint at 0.000) where Desktop opens on the best.
  it('opens on the best model by the rule Desktop and Overview use, not the first listed', async () => {
    const api = new MockApi('normal', 0);
    const models = api.models.bind(api);
    // The models of the dataset listed first (by name) learned nothing: the picker's first group is at 0.000.
    api.models = async () => (await models()).map((m) => (m.name.startsWith('fasteners') && m.metrics ? { ...m, metrics: { ...m.metrics, map50_95: 0 } } : m));
    const best = [...(await api.models())].sort(betterModel)[0]!;
    expect(best.name).not.toMatch(/^fasteners/);
    expect(await open(api, '#/playground')).toBe(best.id);
  });

  // v-models-label N6: a link to a deleted model showed another one without a word.
  it('says which model it shows when the link names one that is gone', async () => {
    toast.items = [];
    const api = new MockApi('normal', 0);
    const shown = await open(api, '#/playground?model=m_gone&conf=0.4');
    const name = (await api.models()).find((m) => m.id === shown)!.name;
    await until(() => toast.items.length, 'the note');
    expect(toast.items.map((t) => [t.title, t.detail])).toEqual([['The linked model was not found', `It was deleted or the link is wrong. Showing ${name} instead.`]]);
    toast.items = [];
  });

  // v-models-label N2: "Unsupported image" stayed 20 s over the results of the images that followed.
  it('takes back an image error once the next image is accepted', async () => {
    toast.items = [];
    // What the Playground says of a file it cannot open (setFile), then a sample image taken.
    toast.error(new Error('fake.jpg could not be opened'), 'Unsupported image', { key: 'image' });
    toast.info('Something else');
    await run('normal', 'm_shapes_s');
    expect(toast.items.map((t) => t.title)).toEqual(['Something else']);
    toast.items = [];
  });
});

describe('Playground: the rail (from 1024 px)', () => {
  const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, ' ').trim() ?? '';
  const tab = (name: string) => [...target.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((t) => t.textContent?.includes(name))!;

  it('holds the model, thresholds, what is drawn, the detections and the timing; the header has Open image', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const rail = target.querySelector('aside.rail')!;
    expect(rail.getAttribute('aria-label')).toBe('Model and results');
    // The model picker left the page header for the rail.
    expect(target.querySelector('.page header select')).toBeNull();
    expect(rail.querySelector('select')).not.toBeNull();
    expect([...rail.querySelectorAll('.slider')].length).toBe(3);
    expect(text(rail.querySelector('#pg-thresholds'))).toBe('Thresholds');
    expect(rail.querySelector('[role="group"][aria-label="Show on the image"]')).not.toBeNull();
    expect(rail.querySelector('#pg-panel')).not.toBeNull();
    expect(text(rail.querySelector('#pg-timing'))).toContain('Timing');
    const open = [...target.querySelectorAll('.page header button')].find((b) => b.textContent?.includes('Open image'));
    expect(open).toBeDefined();
    // No footer under the image: its count and time are in the image's header, its thresholds in the rail.
    expect(target.querySelector('.vfoot')).toBeNull();
    const facts = [...target.querySelectorAll('.vb-facts .fi')].map((f) => text(f));
    expect(facts[0]).toMatch(/^\d+ shown of [\d,]+\+? candidates$/);
    expect(facts.at(-1)).toMatch(/^[\d.]+ ms$/);
  });

  it("drops the coordinate columns but gives the selected box's coordinates under the list", async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelectorAll('#pg-panel tbody tr').length > 1, 'the detections');
    expect([...target.querySelectorAll('#pg-panel thead th')].map((th) => text(th))).toEqual(['Class', 'Size (px)', 'Confidence']);
    expect(text(target.querySelector('.sel-line'))).toBe('Select a box for its coordinates');
    target.querySelectorAll<HTMLTableRowElement>('#pg-panel tbody tr')[1]!.click();
    flushSync();
    expect(text(target.querySelector('.sel-line'))).toMatch(/^Box x1 \d+ · y1 \d+ · x2 \d+ · y2 \d+$/);
  });

  it('lists the classes found as a tab, each one a show / hide toggle', async () => {
    await run('normal', 'm_hard_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const rows = target.querySelectorAll('#pg-panel tbody tr').length;
    tab('Classes').click();
    flushSync();
    const toggles = [...target.querySelectorAll<HTMLButtonElement>('#pg-panel ul.counts button[aria-pressed]')];
    expect(toggles.length).toBeGreaterThan(1);
    toggles[0]!.click();
    flushSync();
    expect(text(target.querySelector('.vb-facts'))).toContain('1 class hidden');
    tab('Detections').click();
    flushSync();
    expect(target.querySelectorAll('#pg-panel tbody tr').length).toBeLessThan(rows);
  });

  it('selects Detections when the window narrows below 1024 px on the rail-only Classes tab, so a tab stays focusable', async () => {
    const width = window.innerWidth;
    const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });
    try {
      await run('normal', 'm_hard_s');
      await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
      tab('Classes').click();
      flushSync();
      setWidth(820);
      await until(() => !target.querySelector('aside.rail'), 'the panels under the image');
      const tabs = [...target.querySelectorAll<HTMLElement>('[role="tablist"][aria-label="Result views"] [role="tab"]')];
      expect(tabs.map((t) => t.textContent?.replace(/\d+/g, '').trim())).toEqual(['Detections', 'JSON']);
      expect(tabs.map((t) => t.getAttribute('aria-selected'))).toEqual(['true', 'false']);
      expect(tabs.map((t) => t.getAttribute('tabindex'))).toEqual(['0', '-1']);
    } finally {
      setWidth(width);
    }
  });

  it('says an end-to-end model does not use the overlap threshold instead of offering its slider', async () => {
    await run('normal', 'm_shapes_s');
    await until(() => target.querySelector('#pg-panel tbody tr'), 'the detections');
    const rail = target.querySelector('aside.rail')!;
    expect([...rail.querySelectorAll('.no-nms span')].map((s) => text(s))).toEqual(['Overlap (IoU)', 'Not used · end-to-end model']);
    expect([...rail.querySelectorAll('.slider')].length).toBe(2);
  });

  it('clears predicting state and inspection filters when clearing an image mid-prediction', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    let resolvePredict!: (p: Awaited<ReturnType<MockApi['predict']>>) => void;
    const pending = new Promise<Awaited<ReturnType<MockApi['predict']>>>((r) => (resolvePredict = r));
    hub.api = api;
    hub.mode = 'mock';
    vi.spyOn(hub.api, 'predict').mockReturnValueOnce(pending);
    target = document.createElement('div');
    document.body.appendChild(target);
    router.route = parse('#/playground?model=m_hard_s');
    c = mount(Playground, { target });

    // Start a prediction on a sample image: timing panel enters the loading skeleton state.
    (await until(() => target.querySelector<HTMLButtonElement>('button.sample'), 'a sample image')).click();
    await until(() => target.querySelector('.t-sk'), 'predicting skeleton');

    // Clear the image while prediction is still in flight: predicting must reset immediately (no stuck skeleton).
    const clearBtn = target.querySelector<HTMLButtonElement>('button[aria-label="Clear image"]')!;
    clearBtn.click();
    flushSync();
    expect(target.querySelector('.t-sk')).toBeNull();
    expect(text(target.querySelector('section[aria-labelledby="pg-timing"]'))).toContain('Timing appears after a run.');

    // Late resolution of the aborted prediction must stay ignored.
    resolvePredict({
      model_id: 'm_hard_s',
      image: { width: 640, height: 480 },
      boxes: [],
      timing: { preprocess_ms: 1, inference_ms: 2, postprocess_ms: 1, total_ms: 4 },
      device: 'cpu',
      cold: false,
    });
    await new Promise((r) => setTimeout(r, 0));
    flushSync();
    expect(target.querySelector('.t-sk')).toBeNull();
  });
});
