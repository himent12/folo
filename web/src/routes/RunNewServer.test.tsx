// @vitest-environment happy-dom
// New run with what the server can do: mixed precision, the end-to-end head, its estimate, its weights and its
// refusals (the form itself is RunNew.svelte.test.ts, Run on is RunNewCloud.svelte.test.ts).
import { flush as flushSync, tick } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';
import { ApiError, hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { RunEstimate } from '$lib/api/types';
import { parse, router } from '$lib/router';
import RunNew from './RunNew';

/** Port of `mount(RunNew, { target })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof RunNew, o: { target: HTMLElement }) => render(() => <RunNew />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();


let api: MockApi;
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
    { timeout: 3000, interval: 10 },
  );
  return v as T;
}

/** Mounts New run (optionally `?from=` a run) and waits for its form to load. */
async function open(query = '') {
  router.route = parse(`#/runs/new${query}`);
  c = mount(RunNew, { target });
  await until(() => target.querySelector('select#device option[value="gpu:0"]'), 'the device list');
  await until(() => (target.querySelector<HTMLInputElement>('input#name')?.value ? true : null), 'the run name');
}

// Svelte's select binding reads the chosen option with `:checked`, which happy-dom does not match on <option>.
const qs = HTMLSelectElement.prototype.querySelector;
HTMLSelectElement.prototype.querySelector = function (this: HTMLSelectElement, sel: string) {
  return sel === ':checked' ? (this.options[this.selectedIndex] ?? null) : qs.call(this, sel);
} as typeof qs;

function pick(device: string) {
  const sel = target.querySelector<HTMLSelectElement>('select#device')!;
  sel.querySelector<HTMLOptionElement>(`option[value="${device}"]`)!.selected = true;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
}

async function start() {
  target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  await tick();
}

beforeEach(() => {
  // New run keeps a draft for the tab (runnew/draft): every test starts from a fresh tab.
  sessionStorage.clear();
  api = new MockApi('gpu-unplugged', 0);
  // The augmentation preview's images are drawn on a canvas, which this DOM does not have.
  api.imageUrl = async () => 'data:,';
  hub.api = api;
  hub.mode = 'mock';
  target = document.createElement('div');
  document.body.appendChild(target);
  Element.prototype.scrollIntoView ??= () => {};
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

describe('New run: mixed precision', () => {
  const sw = () => target.querySelector<HTMLInputElement>('input#amp');
  const cli = () => target.querySelector('.cli code')!.textContent!;
  const row = () => [...target.querySelectorAll('.summary .rv')].find((r) => r.querySelector('.rv-k')?.textContent === 'Mixed precision');
  const time = () => [...target.querySelectorAll('.summary .est dt')].find((d) => d.textContent === 'Est. time')!.nextElementSibling!.textContent;
  const basis = () => target.querySelector('.summary .est-how > summary')?.textContent;

  it('is off by default; on, the run, the Summary and the command say so; off on the CPU and back on with the GPU', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    const s = await until(sw, 'the switch');
    expect(s.checked).toBe(false);
    expect(s.disabled).toBe(false);
    expect(s.getAttribute('role')).toBe('switch');
    expect(s.getAttribute('aria-describedby')).toBe('amp-hint');
    expect(row()?.querySelector('.rv-v')?.textContent).toBe('Off');
    expect(cli()).not.toContain('--amp');
    await until(() => basis()?.startsWith('Measured'), "the server's estimate");
    const f32 = time();
    s.click();
    flushSync();
    expect(row()?.querySelector('.rv-v')?.textContent).toBe('On');
    expect(cli()).toContain('--amp');
    // The server's estimate follows once the settings settle (folo-s amp was never measured: carried over).
    await until(() => time() !== f32, 'the new estimate');
    // Off on the CPU, and back on with the GPU.
    pick('cpu');
    expect(sw()!.disabled).toBe(true);
    expect(sw()!.checked).toBe(false);
    expect(cli()).not.toContain('--amp');
    pick('auto');
    expect(sw()!.checked).toBe(true);
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect(create.mock.calls[0]![0].amp).toBe(true);
  });

  it("is off where the GPU has no tensor cores through DX12, with the way to Vulkan and the server's own words", async () => {
    hub.api = api = new MockApi('no-tensor-cores', 0);
    await open();
    const s = await until(sw, 'the switch');
    expect(s.disabled).toBe(true);
    expect(s.checked).toBe(false);
    expect(s.getAttribute('aria-describedby')).toBe('amp-why');
    // One action docs/GPU.md confirms (an RTX 4080 SUPER has cooperative matrices on Vulkan), the command whole.
    const remedy = target.querySelector('.amp .remedy')!;
    expect(remedy.querySelector('pre.line')?.textContent).toBe("$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores");
    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    remedy.querySelector<HTMLButtonElement>('button[aria-label="Copy command"]')!.click();
    expect(write).toHaveBeenCalledWith("$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores");
    // The server's reason, verbatim and copyable, folded behind "Driver report".
    const report = remedy.querySelector<HTMLDetailsElement>('details.report')!;
    expect(report.open).toBe(false);
    expect(report.querySelector('pre')?.textContent).toBe((await api.system()).capabilities!.amp.reason);
    report.querySelector<HTMLButtonElement>('button[aria-label="Copy driver report"]')!.click();
    expect(write).toHaveBeenCalledWith((await api.system()).capabilities!.amp.reason);
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect(create.mock.calls[0]![0].amp).toBe(false);
  });

  it('offers no Vulkan command where no backend gives this GPU tensor cores', async () => {
    hub.api = api = new MockApi('no-tensor-cores', 0);
    const sys = api.system.bind(api);
    api.system = async () => {
      const r = await sys();
      return { ...r, gpus: r.gpus.map((g) => ({ ...g, name: 'AMD Radeon RX 7900 XTX' })) };
    };
    await open();
    await until(sw, 'the switch');
    expect(target.querySelector('#amp-why')).not.toBeNull();
    expect(target.querySelector('.amp')?.textContent).not.toContain('FOLO_GPU_BACKEND');
  });

  it('gives the command that turns tensor cores on when the server has them off', async () => {
    hub.api = api = new MockApi('tensor-cores-off', 0);
    await open();
    await until(sw, 'the switch');
    expect(target.querySelector('.amp pre.line')?.textContent).toBe('folo serve --tensor-cores');
  });

  it('is hidden for servers that do not report capabilities, and the run leaves amp out', async () => {
    const sys = api.system.bind(api);
    api.system = async () => ({ ...(await sys()), capabilities: undefined });
    await open();
    expect(sw()).toBeNull();
    expect(row()).toBeUndefined();
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect('amp' in create.mock.calls[0]![0]).toBe(false);
  });
});

describe('New run: the end-to-end head', () => {
  const sw = () => target.querySelector<HTMLInputElement>('input#head');
  const cli = () => target.querySelector('.cli code')!.textContent!;
  const row = () => [...target.querySelectorAll('.summary .rv')].find((r) => r.querySelector('.rv-k')?.textContent === 'Head');

  it('is an Advanced switch, off by default: the run leaves head out (o2m) and the Summary and command say nothing', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    const s = sw()!;
    expect(s.closest('details.advanced')).not.toBeNull();
    expect(s.checked).toBe(false);
    expect(s.getAttribute('aria-describedby')).toBe('head-hint');
    expect(row()).toBeUndefined();
    expect(cli()).not.toContain('--head');
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect('head' in create.mock.calls[0]![0]).toBe(false);
  });

  it('on, sends head o2o, shows it in the Summary and as --head o2o, and the run logs the dual head', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    sw()!.click();
    flushSync();
    expect(row()?.querySelector('.rv-v')?.textContent).toBe('End-to-end (no NMS)');
    expect(cli()).toContain('--head o2o');
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect(create.mock.calls[0]![0].head).toBe('o2o');
    const run = await create.mock.results[0]!.value;
    expect(run.config.head).toBe('o2o');
    // Runs train one at a time: free the slot (the live run and the two waiting ones), then let the worker take it.
    for (const id of ['run_7d1e', 'run_q4n2', 'run_q7c5']) await api.stopRun(id);
    vi.useFakeTimers({ toFake: ['Date'] });
    try {
      vi.setSystemTime(Date.now() + 5000);
      await api.runs();
      vi.setSystemTime(Date.now() + 5000);
      const { lines } = await api.runLogs(run.id, 50);
      expect(lines.some((l) => l.message.startsWith('head o2o: dual assignment'))).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("names o2m in the command when fine-tuning, since the CLI's default is the weights file's head; its row opens Advanced", async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    const sel = target.querySelector<HTMLSelectElement>('select#weights')!;
    const opt = await until(() => sel.querySelector<HTMLOptionElement>('option:not([value="none"])'), 'trained weights');
    opt.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(cli()).toContain('--head o2m');
    sw()!.click();
    flushSync();
    expect(cli()).toContain('--head o2o');
    expect(cli()).not.toContain('--head o2m');
    // Its Summary row opens Advanced, where the switch is.
    const adv = target.querySelector<HTMLDetailsElement>('details.advanced')!;
    expect(adv.open).toBe(false);
    (row() as HTMLButtonElement).click();
    await until(() => adv.open, 'Advanced open');
  });

  it('comes back on in a copy of an o2o run', async () => {
    hub.api = api = new MockApi('normal', 0);
    const base = (await api.runs())[0]!.config;
    const run = await api.createRun({ ...base, name: 'e2e-src', head: 'o2o' });
    await open(`?from=${run.id}`);
    await until(() => sw()?.checked, 'the switch on');
    expect(row()).toBeDefined();
  });

  it('is refused by the mock as by the server when the head is not o2m or o2o', async () => {
    hub.api = api = new MockApi('normal', 0);
    const base = (await api.runs())[0]!.config;
    await expect(api.createRun({ ...base, name: 'bad-head', head: 'nms' as 'o2o' })).rejects.toMatchObject({ status: 400, code: 'invalid_body' });
  });
});

describe('New run: the estimate', () => {
  const timeDd = () => [...target.querySelectorAll('.summary .est dt')].find((d) => d.textContent === 'Est. time')?.nextElementSibling;
  const time = () => timeDd()?.textContent ?? '';
  const basis = () => target.querySelector('.summary .est-how > summary')?.textContent ?? '';
  const rough = () => timeDd()?.classList.contains('rough');

  it("shows the server's estimate as a range, measured on this GPU, and asks again when the settings change", async () => {
    hub.api = api = new MockApi('normal', 0);
    const ask = vi.spyOn(api, 'estimateRun');
    await open();
    await until(() => basis().startsWith('Measured'), "the server's estimate");
    // folo-s at 640, batch 16, f32 is a measured row of the mock: the narrow range of a high confidence.
    expect(basis()).toBe('Measured on this GPU · 226 samples');
    expect(time()).toMatch(/^\d+–\d+ min$/);
    expect(rough()).toBe(false);
    const body = ask.mock.calls.at(-1)![0];
    expect([body.arch, body.imgsz, body.batch, body.amp]).toEqual(['folo-s', 640, 16, false]);
    // The phases: the mosaic epochs and the close-mosaic ones apart.
    expect([...target.querySelectorAll('.summary .how-list dt')].map((d) => d.textContent).slice(0, 2)).toEqual(['90 epochs, mosaic', '10 epochs, no mosaic']);
    // The dock says the same range.
    expect(target.querySelector('.dock-est')?.textContent).toContain(time());
    // It asks again when the settings change, and calls a size it never measured rough.
    const radio = target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-x"]')!;
    radio.click();
    flushSync();
    await until(() => basis().includes('caled'), 'the folo-x estimate');
    // folo-x is 9x folo-s's compute: low confidence, the value one step quieter.
    expect(rough()).toBe(true);
  });

  it("marks the last answer as not current while the new settings' estimate is out, and the memory row states its figure", async () => {
    hub.api = api = new MockApi('normal', 0);
    const real = api.estimateRun.bind(api);
    let release: () => void = () => {};
    const held = new Promise<void>((r) => (release = r));
    let hold = false;
    vi.spyOn(api, 'estimateRun').mockImplementation(async (b) => {
      if (hold) await held;
      return real(b);
    });
    await open();
    await until(() => basis().startsWith('Measured'), "the server's estimate");
    const memDd = () => [...target.querySelectorAll('.summary .est dt')].find((d) => d.textContent === 'Est. GPU memory')?.nextElementSibling;
    const memory = memDd()!.textContent!.split(' / ')[0]!;
    // The breakdown's memory row is the Summary's figure with its source, never a source alone.
    const peak = [...target.querySelectorAll('.summary .how-list dt')].find((d) => d.textContent === 'Peak GPU memory')?.nextElementSibling;
    expect(peak?.textContent).toBe(`${memory} measured`);
    expect(timeDd()?.classList.contains('stale')).toBe(false);

    hold = true;
    target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-n"]')!.click();
    flushSync();
    // The old figures stay, dimmed, and the basis line says the answer is on its way.
    await until(() => basis() === 'Updating…', 'the updating line');
    expect(timeDd()?.classList.contains('stale')).toBe(true);
    expect(memDd()?.classList.contains('stale')).toBe(true);
    expect(target.querySelector('.dock-est .stale')).not.toBeNull();

    release();
    await until(() => basis().startsWith('Measured') || basis().includes('caled'), 'the new estimate');
    expect(timeDd()?.classList.contains('stale')).toBe(false);
    expect(memDd()?.classList.contains('stale')).toBe(false);
    expect(target.querySelector('.dock-est .stale')).toBeNull();
  });

  // #354 review: a Seed field with no seed in it (refused, or empty) was asked about as seed 0, another seed.
  it('leaves the seed out of the estimate while the Seed field holds none, never 0', async () => {
    hub.api = api = new MockApi('normal', 0);
    const ask = vi.spyOn(api, 'estimateRun');
    await open();
    const seed = target.querySelector<HTMLInputElement>('input#seed')!;
    /** Types `v` and returns the next estimate request's body. */
    const asked = async (v: string) => {
      const n = ask.mock.calls.length;
      seed.value = v;
      seed.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      await until(() => ask.mock.calls.length > n, `the estimate for ${JSON.stringify(v)}`);
      return ask.mock.calls.at(-1)![0];
    };
    expect((await asked('7')).seed).toBe(7);
    expect(await asked('9007199254740990.5')).not.toHaveProperty('seed');
    expect((await asked('7')).seed).toBe(7);
    expect(await asked('')).not.toHaveProperty('seed');
  });

  it('falls back to the reference figures on a server without run_estimate, and says so', async () => {
    hub.api = api = new MockApi('no-throughput', 0);
    const ask = vi.spyOn(api, 'estimateRun');
    await open();
    await until(() => basis(), 'the estimate');
    expect(basis()).toBe('Reference benchmarks · not run here yet');
    expect(time()).toMatch(/^\d+–\d+ min$/);
    expect(rough()).toBe(true);
    expect(target.querySelector('.summary .how-note')?.textContent).toContain('does not estimate runs');
    expect(ask).not.toHaveBeenCalled();
    // The latency column is the reference throughout.
    expect(target.querySelector('.lat .here')).toBeNull();
    expect([...target.querySelectorAll('.a-src')].map((s) => s.textContent).filter(Boolean)).toEqual(['reference', 'reference']);
  });

  describe('asks again 300 ms after the settings settle, whether or not the last one was answered', () => {
    afterEach(() => vi.useRealTimers());

    function type(...values: string[]) {
      const el = target.querySelector<HTMLInputElement>('input#epochs')!;
      for (const v of values) {
        el.value = v;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        flushSync();
      }
    }

    /** Opens the form with the server's estimate answered by `answer`, then waits for the first ask (sent at once). */
    async function firstAsk(answer: () => Promise<RunEstimate>) {
      hub.api = api = new MockApi('normal', 0);
      const ask = vi.spyOn(api, 'estimateRun').mockImplementation(answer);
      await open();
      await until(() => ask.mock.calls.length === 1, 'the first estimate');
      return ask;
    }

    async function debounced(ask: MockInstance<MockApi['estimateRun']>) {
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
      type('1', '12', '120');
      expect(ask).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(299);
      expect(ask).toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1);
      expect(ask).toHaveBeenCalledTimes(2);
      expect(ask.mock.calls[1]![0].epochs).toBe(120);
    }

    it('while the first one is still out', async () => {
      const ask = await firstAsk(() => new Promise(() => {}));
      await debounced(ask);
    });

    it('after the server refused the first one', async () => {
      const ask = await firstAsk(() => Promise.reject(new ApiError(500, 'internal', 'estimator crashed')));
      await until(() => target.querySelector('.summary .how-note')?.textContent?.includes('could not estimate'), 'the reference figures');
      await debounced(ask);
    });
  });

  it("marks folo-n's latency as measured on this GPU in its own cell, the others as the reference", async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    const here = await until(() => target.querySelector('.lat .here'), 'the measured latency');
    expect(here.textContent).toBe('5.90 ms');
    const row = here.closest('label')!;
    expect(row.querySelector('input')?.value).toBe('folo-n');
    expect(row.querySelector('.a-src')?.textContent).toBe('measured');
    const s = target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-s"]')!.closest('label')!;
    expect(s.querySelector('.a-src')?.textContent).toBe('reference');
  });
});

describe('New run: DOGFOOD-3', () => {
  const cli = () => target.querySelector('.cli code')!.textContent!;
  async function fineTune() {
    target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-n"]')!.click();
    flushSync();
    const sel = target.querySelector<HTMLSelectElement>('select#weights')!;
    const opt = await until(() => sel.querySelector<HTMLOptionElement>('option[value="m_fast_n"]'), 'fasteners-n-best');
    opt.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
  }

  it("fine-tunes from the weights file in the CLI command: no --model, and the file's path, not the model id (S1)", async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    await fineTune();
    expect(cli()).not.toContain('--model');
    expect(cli()).not.toContain('m_fast_n');
    // This server does not report the file's path: a placeholder, and one line that says what to put there.
    expect(cli()).toContain('--weights <path to fasteners-n-best.folo>');
    expect(target.querySelector('.summary .cli-note')).not.toBeNull();
    expect(target.querySelector('.summary .cli code')?.getAttribute('aria-describedby')).toBe('cli-note');
  });

  it('names the file where the server reports its path (Model.path)', async () => {
    hub.api = api = new MockApi('normal', 0);
    const models = api.models.bind(api);
    vi.spyOn(api, 'models').mockImplementation(async () => (await models()).map((m) => ({ ...m, path: `D:/folo data/models/${m.name}.folo` })));
    await open();
    await fineTune();
    expect(cli()).toContain("--weights 'D:/folo data/models/fasteners-n-best.folo'");
    expect(target.querySelector('.summary .cli-note')).toBeNull();
  });

  it("shows the server's own command where its estimate carries one", async () => {
    hub.api = api = new MockApi('normal', 0);
    const est = api.estimateRun.bind(api);
    vi.spyOn(api, 'estimateRun').mockImplementation(async (b) => ({ ...(await est(b)), cli: `folo train --data D:/x --weights D:/w.folo --name ${b.name}` }));
    await open();
    await fineTune();
    await until(() => cli().includes('D:/w.folo'), "the server's command");
    expect(cli()).toContain(`--name ${target.querySelector<HTMLInputElement>('input#name')!.value}`);
    expect(target.querySelector('.summary .cli-note')).toBeNull();
  });

  it('says "Estimating…" until the server\'s devices load, never a CPU guess or 0 steps (S3)', async () => {
    hub.api = api = new MockApi('normal', 0);
    vi.spyOn(api, 'system').mockReturnValue(new Promise(() => {}));
    router.route = parse('#/runs/new');
    c = mount(RunNew, { target });
    // The dataset loaded, the server's devices still not.
    await until(() => target.querySelector('.dock-est'), 'the dataset');
    const summary = target.querySelector('.summary')!.textContent!;
    expect(summary).toContain('Estimating…');
    expect(summary).not.toContain('CPU is roughly');
    expect(summary).not.toContain('steps / epoch');
    expect(summary).not.toContain('img/s');
    expect(target.querySelector('.dock-est')?.textContent).toContain('Estimating…');
  });

  it("states the GPU memory estimate alone where the adapter's size is not reported (memory_bytes 0), with no fault", async () => {
    hub.api = api = new MockApi('normal', 0);
    const sys = api.system.bind(api);
    vi.spyOn(api, 'system').mockImplementation(async () => {
      const r = await sys();
      return { ...r, gpus: r.gpus.map((g) => ({ ...g, memory_bytes: 0 })) };
    });
    await open();
    const dd = [...target.querySelectorAll('.summary .est dt')].find((d) => d.textContent === 'Est. GPU memory')!.nextElementSibling!;
    expect(dd.textContent).toMatch(/^~[\d.]+ GiB$/);
    const summary = target.querySelector('.summary')!.textContent!;
    expect(summary).not.toContain('unknown');
    expect(summary).not.toContain('reported');
    expect(target.querySelector('.summary .vram')).toBeNull();
  });
});

describe('New run: published weights (server PR #140)', () => {
  const sel = () => target.querySelector<HTMLSelectElement>('select#weights')!;
  function choose(value: string) {
    sel().querySelector<HTMLOptionElement>(`option[value="${value}"]`)!.selected = true;
    sel().dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
  }
  const arch = () => target.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value;

  it('are listed in a group of their own whatever the size, take their size when picked, and drop out with another size', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    expect(arch()).toBe('folo-s');
    const group = await until(() => sel().querySelector('optgroup[label="Published weights"]'), 'the published group');
    expect(group.querySelector('option[value="folo-n-coco"]')).not.toBeNull();
    // The run models of the chosen size are grouped by what their scores are on, outside the published group.
    const own = await until(() => sel().querySelector('option[value="m_shapes_s"]'), 'the run models');
    expect(own.parentElement).not.toBe(group);
    choose('folo-n-coco');
    expect(arch()).toBe('folo-n');
    expect(sel().value).toBe('folo-n-coco');
    expect(target.querySelector('.a-from')).not.toBeNull();
    // Where they come from (the server's own description) one tap away.
    expect(target.querySelector('#weights-about')).toBeNull();
    target.querySelector<HTMLButtonElement>('button[aria-controls="weights-about"]')!.click();
    flushSync();
    expect(target.querySelector('#weights-about')).not.toBeNull();
    const rv = (k: string) => [...target.querySelectorAll('.summary .rv')].find((r) => r.querySelector('.rv-k')?.textContent === k)?.querySelector('.rv-v')?.textContent;
    expect(rv('Model')).toBe('folo-n');
    expect(rv('Start from')).toBe('folo-n-coco');
    // The command names them as the server does, by their published name.
    await until(() => target.querySelector('.cli code')?.textContent?.includes('--weights folo-n-coco'), 'the command');
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect(create.mock.calls[0]![0]).toMatchObject({ arch: 'folo-n', weights: 'folo-n-coco' });
    if (c) unmount(c);
    c = undefined;

    await open();
    await until(() => sel().querySelector('option[value="folo-n-coco"]'), 'the published weights');
    choose('folo-n-coco');
    target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-m"]')!.click();
    flushSync();
    expect(sel().value).toBe('none');
    expect(target.querySelector('.a-from')).toBeNull();
  });
});

describe('New run: the server’s figures and refusals (server PR #140)', () => {
  it("says the server's close-mosaic count for these settings where it sends one", async () => {
    hub.api = api = new MockApi('normal', 0);
    const real = api.estimateRun.bind(api);
    vi.spyOn(api, 'estimateRun').mockImplementation(async (body) => ({ ...(await real(body)), close_mosaic: 3 }));
    await open();
    await until(() => target.querySelector('#close-hint')?.textContent?.includes('last 3 epochs'), "the server's count");
  });

  it('shows a 501 for mixed precision at its field, not as a toast; once the switch is off, only a quiet note stays', async () => {
    hub.api = api = new MockApi('amp-refused', 0);
    const { toast } = await import('$lib/ui/toast');
    const err = vi.spyOn(toast, 'error');
    await open();
    const sw = await until(() => target.querySelector<HTMLInputElement>('input#amp'), 'the switch');
    sw.click();
    flushSync();
    await start();
    expect(err).not.toHaveBeenCalled();
    // The capabilities read again: the switch is off and held off, so the refusal is answered. No alert, no danger
    // line and no button that repeats Start: the field's one sentence and the way back to tensor cores.
    await until(() => target.querySelector('#amp-why'), 'the remedy');
    expect(target.querySelector<HTMLInputElement>('input#amp')!.disabled).toBe(true);
    expect(target.querySelector<HTMLInputElement>('input#amp')!.checked).toBe(false);
    expect(target.querySelector('#amp-error')).toBeNull();
    expect(target.querySelector('.amp [role="alert"]')).toBeNull();
    expect([...target.querySelectorAll('.amp button')].some((b) => /full precision/i.test(b.textContent ?? ''))).toBe(false);
    expect(target.querySelector('.amp pre.line')?.textContent).toBe('folo serve --tensor-cores');
    // The server's whole message, folded and copyable, never inline.
    const report = target.querySelector<HTMLDetailsElement>('.amp details.report')!;
    expect(report.open).toBe(false);
    expect(report.querySelector('pre')?.textContent).toMatch(/^Mixed-precision training \(amp\) needs a GPU with tensor cores/i);
    expect(target.querySelector('#start-why')).toBeNull();
  });

  it('says an estimate refused for mixed precision at the field, before Start', async () => {
    hub.api = api = new MockApi('normal', 0);
    const real = api.estimateRun.bind(api);
    let refuse = true;
    let down = false;
    vi.spyOn(api, 'estimateRun').mockImplementation(async (b) => {
      if (down) throw new TypeError('Failed to fetch');
      if (b.amp && refuse) throw new ApiError(501, 'not_supported', 'mixed-precision training (amp) needs a GPU with tensor cores');
      return real(b);
    });
    await open();
    const sw = await until(() => target.querySelector<HTMLInputElement>('input#amp'), 'the switch');
    sw.click();
    flushSync();
    await until(() => target.querySelector('#amp-error'), 'the refusal at the field');
    expect(target.querySelector('#start-why')?.textContent).toMatch(/^Mixed precision: /);
    sw.click();
    flushSync();
    expect(target.querySelector('#amp-error')).toBeNull();
    expect(target.querySelector('#start-why')).toBeNull();
    // On again, and the server can now: the next estimate's answer lets Start go.
    sw.click();
    flushSync();
    await until(() => target.querySelector('#amp-error'), 'the refusal again');
    refuse = false;
    const seed = target.querySelector<HTMLInputElement>('input#seed')!;
    const reask = (v: string) => {
      seed.value = v;
      seed.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };
    reask('5');
    await until(() => !target.querySelector('#amp-error'), 'the refusal answered');
    expect(target.querySelector('#start-why')).toBeNull();
    // A failed connection says nothing about mixed precision, so it does not keep an old refusal holding Start.
    refuse = true;
    reask('6');
    await until(() => target.querySelector('#amp-error'), 'refused once more');
    down = true;
    reask('7');
    await until(() => !target.querySelector('#amp-error'), 'the refusal let go');
    expect(target.querySelector('#start-why')).toBeNull();
  });

  it('holds Start while the switch is still on after a refusal, and lets go once it is off', async () => {
    hub.api = api = new MockApi('normal', 0);
    vi.spyOn(api, 'createRun').mockRejectedValue(new ApiError(501, 'not_supported', 'mixed-precision training (amp) needs a GPU with tensor cores'));
    await open();
    const sw = await until(() => target.querySelector<HTMLInputElement>('input#amp'), 'the switch');
    sw.click();
    flushSync();
    await start();
    await until(() => target.querySelector('#amp-error'), 'the refusal');
    expect(target.querySelector('#start-why')?.textContent).toMatch(/^Mixed precision: /);
    // The way out is the switch beside it; no button repeats what Start does once it is off.
    expect([...target.querySelectorAll('.amp button')].some((b) => /full precision/i.test(b.textContent ?? ''))).toBe(false);
    sw.click();
    flushSync();
    // Off: the refusal steps down to a quiet note, and Start lets go.
    expect(target.querySelector('#amp-error')).toBeNull();
    expect(target.querySelector('.amp [role="alert"]')).toBeNull();
    expect(target.querySelector('#start-why')).toBeNull();
  });
});

describe('New run from a MuSGD run', () => {
  it('offers MuSGD and keeps it when prefilled from a run that trained with it', async () => {
    const base = (await api.run('run_a2f0')).config;
    const made = await api.createRun({ ...base, name: 'shapes-musgd', device: 'cpu', optimizer: 'musgd' });
    await open(`?from=${made.id}`);
    const sel = target.querySelector<HTMLSelectElement>('select#opt')!;
    expect([...sel.options].map((o) => o.value)).toEqual(['auto', 'sgd', 'adamw', 'musgd']);
    expect(sel.value).toBe('musgd');
    expect(sel.options[sel.selectedIndex]?.textContent?.trim()).toBe('MuSGD');
  });
});

describe('New run: ?weights= prefill, name-only estimate skip, and Escape cancel', () => {
  it('prefills weights and switches arch from ?weights=<id>, and falls back cleanly when missing', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open('?weights=m_fast_n');
    const sel = target.querySelector<HTMLSelectElement>('select#weights')!;
    await until(() => sel.value === 'm_fast_n', 'weights prefilled');
    expect(target.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value).toBe('folo-n');

    // Reopening with an unknown weights id resets to none and notifies once.
    unmount(c!);
    c = undefined;
    sessionStorage.clear();
    const { toast } = await import('$lib/ui/toast');
    toast.items = [];
    await open('?weights=m_missing');
    await until(() => toast.items.some((t) => t.title === 'Linked weights not found'), 'missing weights toast');
    expect(target.querySelector<HTMLSelectElement>('select#weights')!.value).toBe('none');
  });

  it('skips redundant /api/runs/estimate calls when only the run name changes, while updating server CLI --name', async () => {
    hub.api = api = new MockApi('normal', 0);
    const real = api.estimateRun.bind(api);
    const ask = vi.spyOn(api, 'estimateRun').mockImplementation(async (b) => ({
      ...(await real(b)),
      cli: `folo train --data D:/x --name ${b.name}`,
    }));
    await open();
    await until(() => target.querySelector('.cli code')?.textContent?.includes('D:/x'), "the server's command");
    const callsBefore = ask.mock.calls.length;

    const nameInput = target.querySelector<HTMLInputElement>('input#name')!;
    nameInput.value = 'renamed-only';
    nameInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    await new Promise((r) => setTimeout(r, 360));
    flushSync();
    expect(ask.mock.calls.length).toBe(callsBefore);
    expect(target.querySelector('.cli code')?.textContent).toContain('--name renamed-only');
  });

  it('returns to /runs on Escape when no nested dialog or select is active, and stays put otherwise', async () => {
    hub.api = api = new MockApi('normal', 0);
    const go = vi.spyOn(router, 'go').mockImplementation(() => {});
    try {
      await open();
      // Escape while a native <select> has focus must not leave the page.
      const select = target.querySelector<HTMLSelectElement>('select#device')!;
      select.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      expect(go).not.toHaveBeenCalled();

      // Escape while an open <dialog> is present must not leave the page.
      const dlg = document.createElement('dialog');
      dlg.setAttribute('open', '');
      document.body.appendChild(dlg);
      try {
        target.querySelector('input#name')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
        expect(go).not.toHaveBeenCalled();
      } finally {
        dlg.remove();
      }

      // Otherwise Escape returns cleanly to /runs.
      target.querySelector('input#name')!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      expect(go).toHaveBeenCalledWith('/runs');
    } finally {
      go.mockRestore();
    }
  });
});
