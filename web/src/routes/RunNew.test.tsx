// @vitest-environment happy-dom
import { flush as flushSync, tick } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, hub } from '$lib/api/index';
import { RECHECK } from '$lib/api/datasetReads';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router';
import { page } from '$lib/shell/crumbs';
import { undescribedFields } from '$lib/ui/fieldA11y';
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

describe('New run: a device the server cannot use', () => {
  it('says so at the Device field in one line from the adapters it has, lists them again and keeps the form', async () => {
    await open();
    await until(() => target.querySelector('select#device option[value="gpu:1"]'));
    const name = target.querySelector<HTMLInputElement>('input#name')!.value;
    pick('gpu:1');
    const create = vi.spyOn(api, 'createRun');
    await start();
    const msg = await until(() => target.querySelector('#device-error'), 'the device error');
    expect(create).toHaveBeenCalledOnce();
    // One short line from the devices listed now, each named once with its value; the server's words (backends and
    // all) fold into the details.
    await until(() => msg.textContent === 'GPU 1 (gpu:1) is no longer available. Pick Auto, GPU 0 (gpu:0) or CPU.', 'the short line');
    const detail = target.querySelector('.dev-detail')!;
    expect(detail.querySelector('summary')?.textContent).toBe('Technical details');
    expect(detail.textContent).toContain('This machine has gpu:0 (NVIDIA GeForce RTX 4080 SUPER, vulkan), cpu');
    expect(detail.textContent).not.toContain('`');
    // The list is read again: the adapter is gone, the choice stays (marked), and so does the rest of the form.
    const opt = await until(() => {
      const o = target.querySelector<HTMLOptionElement>('select#device option[value="gpu:1"]');
      return o?.textContent?.includes('not found') ? o : null;
    });
    expect(opt.selected).toBe(true);
    expect(target.querySelector<HTMLInputElement>('input#name')!.value).toBe(name);
    // Start says why it is off, briefly and without repeating the field, until another device is picked.
    expect([...target.querySelectorAll<HTMLButtonElement>('button[type="submit"]')].every((b) => b.disabled)).toBe(true);
    expect(target.querySelector('#start-why')?.textContent).toBe('Device: GPU 1 (gpu:1) is not available.');
    // Both Start buttons (summary and phone dock) carry the reason, and the select is tied to its error.
    const starts = [...target.querySelectorAll<HTMLButtonElement>('.start button[type="submit"], .dock button[type="submit"]')];
    expect(starts.map((b) => b.getAttribute('aria-describedby'))).toEqual(['start-why', 'dock-why']);
    const sel = target.querySelector('select#device')!;
    expect(sel.getAttribute('aria-invalid')).toBe('true');
    expect(sel.getAttribute('aria-describedby')).toBe('device-error');
    // The reason is a way to the field.
    target.querySelector<HTMLSelectElement>('select#device')!.blur();
    target.querySelector<HTMLButtonElement>('#start-why button')!.click();
    await until(() => document.activeElement === target.querySelector('select#device'), 'focus on Device');
    // Auto is the way out.
    [...target.querySelectorAll<HTMLButtonElement>('.dev-err button')].find((b) => b.textContent?.trim() === 'Use Auto')!.click();
    flushSync();
    expect(target.querySelector<HTMLSelectElement>('select#device')!.value).toBe('auto');
    expect(target.querySelector('#device-error')).toBeNull();
    expect(target.querySelector('#start-why')).toBeNull();
    expect(target.querySelector('select#device')!.hasAttribute('aria-invalid')).toBe(false);
  });

  // Stress d2-runs: a draft (or a copied run) kept a GPU the server no longer lists; only the server's refusal of the
  // first Start said so.
  it('stops Start before the server for a GPU from the draft that the device list lacks', async () => {
    await open();
    await until(() => target.querySelector('select#device option[value="gpu:1"]'));
    pick('gpu:1');
    expect(target.querySelector('#device-error')).toBeNull();
    unmount(c!);
    c = undefined;
    (api as unknown as { state: { unplugged: boolean } }).state.unplugged = true;
    const create = vi.spyOn(api, 'createRun');
    await open();
    expect(target.querySelector<HTMLSelectElement>('select#device')!.value).toBe('gpu:1');
    const msg = await until(() => target.querySelector('#device-error'), 'the device error');
    expect(msg.textContent).toBe('GPU 1 (gpu:1) is no longer available. Pick Auto, GPU 0 (gpu:0) or CPU.');
    expect(target.querySelector('.dev-detail')).toBeNull();
    expect([...target.querySelectorAll<HTMLButtonElement>('button[type="submit"]')].every((b) => b.disabled)).toBe(true);
    expect(target.querySelector('#start-why')?.textContent).toBe('Device: GPU 1 (gpu:1) is not available.');
    await start();
    expect(create).not.toHaveBeenCalled();
    // A device the list names clears it.
    pick('gpu:0');
    expect(target.querySelector('#device-error')).toBeNull();
    expect(target.querySelector('#start-why')).toBeNull();
  });

  // PR #287 review: a run created over the API keeps its own spelling of the device ("GPU:00", which the trainer reads
  // as gpu:0); its copy is the GPU the list names, not one that is gone.
  it('reads a copied run whose device is spelled another way the trainer accepts as the GPU the list names', async () => {
    const run = api.run.bind(api);
    vi.spyOn(api, 'run').mockImplementation(async (id: string) => {
      const r = await run(id);
      return { ...r, config: { ...r.config, device: ' GPU:00' as typeof r.config.device } };
    });
    await open('?from=run_a2f0');
    await until(() => target.querySelector<HTMLSelectElement>('select#device')!.value === 'gpu:0', 'the copied device');
    expect(target.querySelector('#device-error')).toBeNull();
    expect(target.querySelector('#start-why')?.textContent ?? '').not.toContain('Device');
  });

  it('treats a 501 for a GPU from servers before device_not_found the same way', async () => {
    await open();
    pick('gpu:0');
    vi.spyOn(api, 'createRun').mockRejectedValue(
      new ApiError(501, 'not_implemented', 'device gpu:0 cannot be used by this build', 'use device `cpu` (or `auto`, which falls back to the CPU)'),
    );
    await start();
    const msg = await until(() => target.querySelector('#device-error'));
    expect(msg.textContent).toBe('This server cannot train on GPU 0 (gpu:0). Pick Auto or CPU.');
    expect(target.querySelector('.dev-detail p')?.textContent).toBe(
      'Device gpu:0 cannot be used by this build. Use device cpu (or auto, which falls back to the CPU).',
    );
  });
});

describe('New run after running out of GPU memory', () => {
  it('names what failed and what changed, and marks the changed field', async () => {
    await open('?from=run_a2f0&batch=16');
    const note = await until(() => target.querySelector('.oom-note'));
    // Batch 16 is still forecast over the GPU's memory: the note says so, never "Now batch 16" as if it were the fix.
    expect(note.textContent?.replace(/\s+/g, ' ')).toMatch(/shapes-hard-m ran out of GPU memory at batch 32\. Batch 16 still won't fit: ~26\.\d GiB of 16\.0 GiB\./);
    expect(target.querySelector('#batch [aria-checked="true"]')?.textContent?.trim()).toBe('16');
    const batchField = target.querySelector('#batch')!.closest('.field')!;
    // Beside the new value ("16 · was 32"), not at the far end of the label row.
    expect(batchField.querySelector('.with-was .was')?.textContent).toBe('was 32');
    // 960 px is kept from the failed run; the image size field is not marked.
    expect(target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim()).toBe('960');
    expect(target.querySelector('#imgsz')!.closest('.field')!.querySelector('.was')).toBeNull();
  });
  it('says "Now" with the figure only when the change fits (the run page retry link)', async () => {
    await open('?from=run_a2f0&batch=8');
    const note = await until(() => (target.querySelector('.oom-note')?.textContent?.includes('Now') ? target.querySelector('.oom-note') : null), 'the note');
    expect(note!.textContent?.replace(/\s+/g, ' ')).toMatch(/at batch 32\. Now batch 8 · ~13\.\d GiB\./);
  });
  it("gives the figure and says the fit can't be checked where the GPU doesn't report its memory (memory_bytes 0)", async () => {
    hub.api = api = new MockApi('normal', 0);
    const sys = api.system.bind(api);
    vi.spyOn(api, 'system').mockImplementation(async () => {
      const r = await sys();
      return { ...r, gpus: r.gpus.map((g) => ({ ...g, memory_bytes: 0 })) };
    });
    // Batch 16 is forecast over the mock's 16 GiB elsewhere; with no size reported, neither "fits" nor "won't fit".
    await open('?from=run_a2f0&batch=16');
    const note = await until(() => (target.querySelector('.oom-note')?.textContent?.includes('Now') ? target.querySelector('.oom-note') : null), 'the note');
    const text = note!.textContent!.replace(/\s+/g, ' ');
    expect(text).toMatch(/at batch 32\. Now batch 16 · ~[\d.]+ GiB · this GPU doesn't report its memory, so the fit can't be checked\./);
    expect(text).not.toContain("won't fit");
  });
  it('makes the change that fits the one primary while the settings are forecast not to fit, and Start a secondary', async () => {
    hub.api = api = new MockApi('normal', 0);
    // "Retry with batch 16" as round 22 offered it: folo-m at 960 px, ~26.9 GiB on the mock's 16 GiB GPU.
    await open('?from=run_a2f0&batch=16');
    const fix = await until(() => [...target.querySelectorAll<HTMLElement>('.start .btn.primary')].find((b) => b.textContent?.includes('Use batch 8')), 'the fix');
    expect(fix.textContent).toMatch(/Use batch 8 · ~13\.\d GiB/);
    const startBtn = [...target.querySelectorAll<HTMLButtonElement>('.start button[type="submit"]')][0]!;
    expect(startBtn.className).toContain('secondary');
    expect(startBtn.getAttribute('aria-describedby')).toContain('rail-nofit');
    // The dock (narrow screens) has the fix as its one button.
    expect([...target.querySelectorAll('.dock .btn')].map((b) => b.textContent?.trim())).toEqual(['Use batch 8']);
    // Taking it: batch 8, forecast to fit, and Start is the primary again.
    fix.click();
    flushSync();
    expect(target.querySelector('#batch [aria-checked="true"]')?.textContent?.trim()).toBe('8');
    await until(() => !target.querySelector('#rail-nofit'), 'the fit');
    await until(() => target.querySelector('.start button[type="submit"]')?.className.includes('primary'), 'Start as the primary');
    expect(target.querySelectorAll('.start .btn.primary')).toHaveLength(1);
  });
  it('finds the fit after a change is taken back before its estimate came in (settings A, then B, then A again)', async () => {
    hub.api = api = new MockApi('normal', 0);
    // A's own estimate (folo-m, 960 px, batch 16) is answered at once; every other ask (A's fit, B's estimate) waits
    // until `release`.
    const real = api.estimateRun.bind(api);
    let release!: () => void;
    const held = new Promise<void>((r) => (release = r));
    const isA = (b: { arch: string; imgsz: number; batch: number }) => b.arch === 'folo-m' && b.imgsz === 960 && b.batch === 16;
    let answeredA = false;
    let fitAsks = 0;
    vi.spyOn(api, 'estimateRun').mockImplementation(async (b) => {
      if (isA(b)) return real(b).finally(() => (answeredA = true));
      if (answeredA) fitAsks++;
      await held;
      return real(b);
    });
    const batchTo = (v: string) => {
      [...target.querySelectorAll<HTMLElement>('#batch [role="radio"]')].find((o) => o.textContent?.trim() === v)!.click();
      flushSync();
    };
    // A: folo-m at 960 px, batch 16, answered and forecast not to fit; its fit is being looked for.
    await open('?from=run_a2f0&batch=16');
    await until(() => fitAsks > 0, "A's fit, after A's estimate");
    expect(target.querySelector('.start .btn.primary')?.textContent?.trim()).toBe('Finding settings that fit');
    // B, then A again before B's estimate is in: A's estimate is still the one shown, so it is not asked again.
    batchTo('32');
    batchTo('16');
    release();
    const fix = await until(() => [...target.querySelectorAll<HTMLElement>('.start .btn.primary')].find((b) => b.textContent?.includes('Use batch 8')), 'the fit');
    expect(fix.textContent).toMatch(/Use batch 8 · ~13\.\d GiB/);
  });
  it("ignores an image size the trainer would reject and keeps the run's", async () => {
    await open('?from=run_a2f0&imgsz=333');
    await until(() => target.querySelector('.oom-note'));
    expect(target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim()).toBe('960');
    expect(target.querySelector('#imgsz')!.closest('.field')!.querySelector('.was')).toBeNull();
  });
});

describe('New run: short runs', () => {
  function type(id: string, v: number) {
    const el = target.querySelector<HTMLInputElement>(`input#${id}`)!;
    el.value = String(v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
  }

  function choose(id: string, value: string) {
    const sel = target.querySelector<HTMLSelectElement>(`select#${id}`)!;
    sel.querySelector<HTMLOptionElement>(`option[value="${value}"]`)!.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
  }

  it("warns as the trainer does about a short run from scratch, and offers the epochs that clear it", async () => {
    await open();
    expect(target.querySelector('.short')).toBeNull();
    // fasteners: 1,920 train images at batch 16 are 30 optimizer steps an epoch; 5 epochs are 150, W = 90 of them.
    type('epochs', 5);
    const warn = await until(() => target.querySelector('.summary .short'));
    // One line; the reasons and the fix behind "Why?".
    expect(warn.querySelector('p.callout')?.textContent).toContain('150 optimizer steps');
    expect(warn.querySelector('button.fix')).toBeNull();
    const more = warn.querySelector<HTMLButtonElement>('button.more')!;
    more.click();
    flushSync();
    expect(more.getAttribute('aria-expanded')).toBe('true');
    const text = warn.querySelector(`#${more.getAttribute('aria-controls')}`)!.textContent!.replace(/\s+/g, ' ');
    // The prose names no config keys.
    expect(text).toContain('Warmup takes 90 of the 150 optimizer steps');
    expect(text).not.toMatch(/lr0|warmup_epochs/);
    // The trainer's own count, ceil(200 / 30) = 7, which also clears the warmup warning.
    warn.querySelector<HTMLButtonElement>('button.fix')!.click();
    flushSync();
    expect(target.querySelector<HTMLInputElement>('input#epochs')!.value).toBe('7');
    expect(target.querySelector('.short')).toBeNull();
  });

  it('warns about a mostly-warmup run from trained weights too, but not about its length', async () => {
    await open();
    await until(() => target.querySelectorAll('select#weights option').length > 1, 'weights to start from');
    choose('weights', target.querySelectorAll<HTMLOptionElement>('select#weights option')[1]!.value);
    type('epochs', 5);
    const warn = await until(() => target.querySelector('.summary .short'));
    expect(warn.querySelector('p.callout span')?.textContent?.trim()).toBe('Mostly warmup: 90 of 150 optimizer steps.');
    warn.querySelector<HTMLButtonElement>('button.more')!.click();
    flushSync();
    expect(warn.textContent).not.toContain('Too short');
    expect(warn.textContent).toContain('Warmup takes 90 of the 150 optimizer steps');
    warn.querySelector<HTMLButtonElement>('button.fix')!.click();
    flushSync();
    expect(target.querySelector<HTMLInputElement>('input#epochs')!.value).toBe('6');
    expect(target.querySelector('.short')).toBeNull();
  });

  it('keeps mosaic in a short run: close mosaic is capped at half the run, as the trainer caps it', async () => {
    await open();
    type('epochs', 8);
    // Close mosaic follows the run: at most half of it.
    expect(target.querySelector<HTMLInputElement>('input#close')!.value).toBe('4');
    type('close', 8);
    // Not a warning any more: the trainer keeps mosaic on for half the run, and the field says so.
    expect(target.querySelector('#close-hint')?.textContent).toContain('last 4 epochs');
    type('epochs', 100);
    expect(target.querySelector('#close-hint')?.textContent).not.toContain('4 epochs');
  });
});

describe('New run: the schedule in the Summary', () => {
  it('counts optimizer steps in plain words, with the batches behind them one tap away (ui/InfoButton)', async () => {
    await open();
    // fasteners: 1,920 images at batch 16 are 120 batches, 4 per optimizer step.
    const est = target.querySelector('.summary .est')!;
    expect(est.querySelector('dt')?.textContent).toBe('Schedule');
    // The value is the button: a tap, a click, Enter or Space opens the batches behind it (touch has no hover).
    const steps = est.querySelector<HTMLButtonElement>('dd button.steps')!;
    expect(steps.textContent).toBe('30 optimizer steps / epoch. How the steps are counted');
    // Beside it, the batches that make one optimizer step, never "per step" for a batch.
    expect(est.querySelector('dd.sched-acc')?.textContent).toBe('4 batches per optimizer step · effective batch 64');
    expect(steps.getAttribute('aria-expanded')).toBe('false');
    const why = () => document.getElementById(steps.getAttribute('aria-controls')!)?.textContent?.trim();
    steps.click();
    flushSync();
    expect(steps.getAttribute('aria-expanded')).toBe('true');
    // A line under the row, in the list itself: a popover over the rail would cover its neighbours (round 24).
    const line = document.getElementById(steps.getAttribute('aria-controls')!)!;
    expect(line.tagName).toBe('DD');
    expect(est.contains(line)).toBe(true);
    expect(why()).toBe('120 batches of 16 an epoch, 4 to each optimizer step.');
    // Batch 64 is one batch a step.
    target.querySelector<HTMLButtonElement>('#batch button[role="radio"]:nth-child(4)')!.click();
    flushSync();
    expect(why()).toBe('30 batches of 64 an epoch, one optimizer step each.');
    expect(est.querySelector('dd.sched-acc')).toBeNull();
  });

  it("matches the short-run scenario's run: shapes at batch 32 for 5 epochs suggests its 16 epochs", async () => {
    await open('?dataset=ds_shapes');
    target.querySelector<HTMLButtonElement>('#batch button[role="radio"]:nth-child(3)')!.click();
    const el = target.querySelector<HTMLInputElement>('input#epochs')!;
    el.value = '5';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const warn = await until(() => target.querySelector('.summary .short'));
    expect(warn.querySelector('p.callout span')?.textContent?.trim()).toBe('Too short to learn: 65 optimizer steps.');
    warn.querySelector<HTMLButtonElement>('button.more')!.click();
    flushSync();
    expect(warn.querySelector('button.fix')?.textContent).toBe('Train 16 epochs');
  });
});

describe('New run: the chrome', () => {
  it('owns the create action and has a bottom dock while mounted, so the top bar and tab bar step aside', async () => {
    const before = page.creators;
    await open();
    expect(page.ownsCreate).toBe(true);
    expect(document.documentElement.hasAttribute('data-dock')).toBe(true);
    unmount(c!);
    c = undefined;
    expect(page.creators).toBe(before);
    expect(document.documentElement.hasAttribute('data-dock')).toBe(false);
  });
});

describe('New run: start weights', () => {
  it('never offers fused weights (an export loads for inference only) as a starting point', async () => {
    await open();
    target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-n"]')!.click();
    flushSync();
    const sel = target.querySelector<HTMLSelectElement>('select#weights')!;
    await until(() => sel.querySelector('option:not([value="none"])'), 'trained folo-n weights');
    const ids = [...sel.options].map((o) => o.value);
    expect(ids).toContain('m_fast_n');
    expect(ids).not.toContain('m_fast_n_f16');
  });
});

describe('New run on a phone', () => {
  const width = window.innerWidth;
  const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });
  afterEach(() => setWidth(width));

  it('picks the size from a segmented control, or from the comparison rows, which replace it while open', async () => {
    setWidth(390);
    await open();
    const pick = target.querySelector('#arch-pick')!;
    const compare = target.querySelector<HTMLDetailsElement>('details.a-compare')!;
    expect(compare.open).toBe(false);
    pick.querySelectorAll('button')[2]!.click();
    flushSync();
    expect(compare.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value).toBe('folo-m');
    compare.open = true;
    compare.dispatchEvent(new Event('toggle'));
    flushSync();
    expect(target.querySelector('#arch-pick')).toBeNull();
    expect(target.querySelector('.a-sel')).toBeNull();
    compare.querySelectorAll<HTMLInputElement>('input[name="arch"]')[0]!.click();
    flushSync();
    compare.open = false;
    compare.dispatchEvent(new Event('toggle'));
    flushSync();
    expect(target.querySelector('#arch-pick [aria-checked="true"]')?.textContent?.trim()).toBe('N');
  });
});

describe('New run: image size', () => {
  it("defaults to a small dataset's own size when every training image shares it", async () => {
    const real = api.datasetImages.bind(api);
    vi.spyOn(api, 'datasetImages').mockImplementation(async (id, q) => {
      const r = await real(id, q);
      return { ...r, images: r.images.map((i) => ({ ...i, width: 320, height: 320 })) };
    });
    await open('?dataset=ds_shapes');
    await until(() => target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim() === '320', 'image size 320');
    expect(target.querySelector('#imgsz')!.closest('.field')!.textContent).toContain('images are all 320 px');
  });

  it("drops the size fitted to one dataset when another is picked whose sizes cannot be read", async () => {
    const real = api.datasetImages.bind(api);
    vi.spyOn(api, 'datasetImages').mockImplementation(async (id, q) => {
      if (id !== 'ds_shapes') throw new Error('listing failed');
      const r = await real(id, q);
      return { ...r, images: r.images.map((i) => ({ ...i, width: 320, height: 320 })) };
    });
    await open('?dataset=ds_shapes');
    await until(() => target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim() === '320', 'image size 320');
    const other = target.querySelector<HTMLInputElement>('input[name="dataset"]:not([value="ds_shapes"])')!;
    other.click();
    flushSync();
    await until(() => target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim() === '640', 'back to 640');
  });

  it('lets a size from the link win', async () => {
    const list = vi.spyOn(api, 'datasetImages').mockResolvedValue({ total: 1, images: [{ id: 'img_0', file: 'a.jpg', split: 'train', width: 320, height: 320, boxes: [] }] });
    await open('?imgsz=960');
    // The training images are read (all 320 px), and the link's size stays.
    await until(() => list.mock.results.length, 'the image sizes');
    await list.mock.results.at(-1)!.value;
    await tick();
    flushSync();
    expect(target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim()).toBe('960');
  });
});

describe('New run: the form', () => {
  it('describes every control by the hint or error under it (Batch size and Image size included)', async () => {
    await open();
    target.querySelector<HTMLDetailsElement>('#sec-train details.advanced')!.open = true;
    flushSync();
    expect(undescribedFields(target)).toEqual([]);
    const batch = target.querySelector('#batch')!.getAttribute('aria-describedby')!;
    expect(document.getElementById(batch)?.textContent).toBe('Images per batch. Lower it if GPU memory runs out.');
    expect(target.querySelector('#imgsz')!.getAttribute('aria-describedby')).toBe('imgsz-hint');
    // Invalid, the error replaces the hint.
    const el = target.querySelector<HTMLInputElement>('input#seed')!;
    el.value = '-1';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(undescribedFields(target)).toEqual([]);
    expect(el.getAttribute('aria-describedby')).toBe('seed-error');
  });

  it('ties an invalid field to its message with aria-invalid and aria-describedby', async () => {
    await open();
    const el = target.querySelector<HTMLInputElement>('input#epochs')!;
    expect(el.hasAttribute('aria-invalid')).toBe(false);
    el.value = '0';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(el.getAttribute('aria-invalid')).toBe('true');
    expect(el.getAttribute('aria-describedby')).toBe('epochs-error');
    // The dock's Start says why it is off.
    expect(target.querySelector('#dock-why')?.textContent).toContain(target.querySelector('#epochs-error')!.textContent!);
  });

  it('folds augmentation behind its one-line summary until opened', async () => {
    await open();
    const toggle = target.querySelector<HTMLButtonElement>('.aug-toggle')!;
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelector('#aug-body')?.classList.contains('shut')).toBe(true);
    toggle.click();
    flushSync();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(target.querySelector('#aug-body')?.classList.contains('shut')).toBe(false);
  });

});

describe('New run from a link', () => {
  it("takes the epochs a short run's warning suggests", async () => {
    await open('?from=run_a2f0&epochs=40');
    await until(() => target.querySelector<HTMLInputElement>('input#epochs')?.value === '40', 'epochs 40');
  });

  /** `?from=run_a2f0` as a run created over the API or the CLI with `seed`, as this browser parsed it. */
  async function fromSeed(seed: number) {
    const run = api.run.bind(api);
    vi.spyOn(api, 'run').mockImplementation(async (id: string) => {
      const r = await run(id);
      return { ...r, config: { ...r.config, seed } };
    });
    await open('?from=run_a2f0');
    await until(() => (target.querySelector<HTMLInputElement>('input#name')?.value.includes('retry') ? true : null), 'the copy');
    return target.querySelector<HTMLInputElement>('input#seed')!;
  }

  // Sweep 4 #330: the trainer's seed is a u64, and a copied run's seed over 2^32 − 1 was marked invalid.
  it("copies a run's seed over 2^32 − 1 as it is", async () => {
    const el = await fromSeed(4294967296);
    expect(el.value).toBe('4294967296');
    expect(target.querySelector('#seed-error')).toBeNull();
    expect(target.querySelector('#start-why')?.textContent ?? '').not.toContain('Seed');
  });

  // Over 2^53 − 1 the seed arrives rounded by JSON parsing (9007199254740993 reads as …992): another seed, never
  // offered as the run's. The field is left empty and says why until a seed is typed.
  it("leaves out a run's seed that this browser could not read exactly, and says so", async () => {
    const el = await fromSeed(2 ** 53);
    expect(el.value).toBe('');
    const why = "The copied run's seed is over 9,007,199,254,740,991, more than Studio holds exactly. Enter a seed.";
    expect(target.querySelector('#seed-error')?.textContent).toBe(why);
    expect(target.querySelector('#start-why')?.textContent).toBe(`Seed: ${why}`);
    el.value = '12';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('#seed-error')).toBeNull();
    // Cleared again, it is an empty field like any other, not the copied run's.
    el.value = '';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('#seed-error')?.textContent).toBe('A whole number from 0 to 9,007,199,254,740,991.');
  });

  // #354 review: Start was held, but the CLI command, "the same run from a terminal", copied without `--seed` (seed 0).
  it('holds the CLI command while the seed is missing, as Start, and says why', async () => {
    await fromSeed(2 ** 53);
    const cli = () => target.querySelector('details.cli')!;
    expect(cli().querySelector('button[aria-label^="Copy"]')).toBeNull();
    expect(cli().querySelector('code')).toBeNull();
    expect(cli().querySelector('.cli-note')?.textContent).toBe('Enter a seed first: the command needs one.');
    const el = target.querySelector<HTMLInputElement>('input#seed')!;
    el.value = '12';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(cli().querySelector('button[aria-label="Copy command"]')).not.toBeNull();
    expect(cli().querySelector('code')?.textContent).toContain('--seed 12');
  });

  // #354 review: a reload (the draft) brought back the empty seed without the reason it is empty.
  it("keeps the reason a copied run's seed was left out across a reload, until one is typed", async () => {
    await fromSeed(2 ** 53);
    unmount(c!);
    c = undefined;
    await open('?from=run_a2f0');
    const why = "The copied run's seed is over 9,007,199,254,740,991, more than Studio holds exactly. Enter a seed.";
    expect(target.querySelector<HTMLInputElement>('input#seed')!.value).toBe('');
    expect(target.querySelector('#seed-error')?.textContent).toBe(why);
  });
});

describe('New run: the draft', () => {
  const type = (id: string, v: string) => {
    const el = target.querySelector<HTMLInputElement>(`input#${id}`)!;
    el.value = v;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
  };
  const field = (id: string) => target.querySelector<HTMLInputElement>(`input#${id}`)!.value;
  const dataset = () => target.querySelector<HTMLInputElement>('input[name="dataset"]:checked')?.value;
  const reopen = async (query = '') => {
    unmount(c!);
    c = undefined;
    await open(query);
  };

  it('keeps what was typed for the tab: leaving and coming back restores it, and a start clears it', async () => {
    await open();
    type('name', 'my-draft');
    type('epochs', '7');
    await reopen();
    expect(field('name')).toBe('my-draft');
    expect(field('epochs')).toBe('7');
    vi.spyOn(api, 'createRun');
    await start();
    await until(() => router.route.path.startsWith('/runs/run_'), 'the new run');
    await reopen();
    expect(field('name')).not.toBe('my-draft');
    expect(field('epochs')).toBe('100');
  });

  it("lets a link set its own fields over the draft, and restores that link's own draft whole", async () => {
    await open();
    type('epochs', '7');
    const first = dataset();
    expect(first).not.toBe('ds_shapes');
    // "Train on this dataset": the link's dataset, the draft's epochs.
    await reopen('?dataset=ds_shapes');
    expect(dataset()).toBe('ds_shapes');
    expect(field('epochs')).toBe('7');
    // Another dataset picked under that link: Back to it (or a reload) finds the pick, not the link's.
    target.querySelector<HTMLInputElement>(`input[name="dataset"][value="${first}"]`)!.click();
    flushSync();
    await reopen('?dataset=ds_shapes');
    expect(dataset()).toBe(first);
  });

  // v-data-runs N1: the palette's "New run on X" (or a pasted link, Back / Forward) changes the URL of the open form.
  it('follows its own URL while open, by the same rules as a first open', async () => {
    await open('?dataset=ds_shapes');
    type('epochs', '7');
    const other = [...target.querySelectorAll<HTMLInputElement>('input[name="dataset"]')].find((i) => i.value !== 'ds_shapes')!.value;
    router.route = parse(`#/runs/new?dataset=${other}`);
    await until(() => dataset() === other, 'the linked dataset');
    // The link's dataset (and the name for it), the draft's edits.
    expect(field('epochs')).toBe('7');
    expect(field('name')).not.toMatch(/^shapes-/);
    // Back: the first link again, its dataset again, the edits kept; nothing more is written back to the URL.
    router.route = parse('#/runs/new?dataset=ds_shapes');
    await until(() => dataset() === 'ds_shapes', 'the first link');
    expect(field('epochs')).toBe('7');
    expect(router.route.path + router.route.query).toBe('/runs/new' + 'dataset=ds_shapes');
  });

  // review R4: a draft that is JSON but not a draft of this form (`augment.hsv` null) was trusted, and the form threw.
  it('opens with its defaults over a draft of another shape, and keeps a sound one after', async () => {
    await open();
    type('epochs', '7');
    const kept = JSON.parse(sessionStorage.getItem('folo.runNew.draft')!);
    sessionStorage.setItem('folo.runNew.draft', JSON.stringify({ ...kept, augment: { ...kept.augment, hsv: null } }));
    await reopen();
    expect(field('epochs')).toBe('100');
    expect(JSON.parse(sessionStorage.getItem('folo.runNew.draft')!).augment.hsv).toEqual([0.015, 0.7, 0.4]);
  });
});

describe('New run: a dataset that cannot be trained on (v-data-runs N3, N6; v-shell N9)', () => {
  const starts = () => [...target.querySelectorAll<HTMLButtonElement>('button[type="submit"]')];
  const asked = (spy: { mock: { calls: unknown[][] } }, id: string) => spy.mock.calls.filter((c) => c[0] === id || (c[0] as { dataset_id?: string })?.dataset_id === id).length;

  it('marks a missing folder, keeps Start off with the reason, and asks nothing more of it, also on the next visit', async () => {
    const read = vi.spyOn(api, 'dataset');
    read.mockImplementation(async (id: string) => {
      if (id === 'ds_shapes_hard') throw new ApiError(404, 'path_not_found', 'the folder of dataset shapes-hard is gone');
      return MockApi.prototype.dataset.call(api, id);
    });
    const images = vi.spyOn(api, 'datasetImages');
    const estimate = vi.spyOn(api, 'estimateRun');
    await open('?dataset=ds_shapes_hard');
    const card = target.querySelector('input[name="dataset"][value="ds_shapes_hard"]')!.closest('.card')!;
    expect(card.querySelector('.badge')?.textContent).toBe('Folder missing');
    expect(starts().every((b) => b.disabled)).toBe(true);
    expect(target.querySelector('#start-why')?.textContent).toBe('Dataset: Folder missing: put it back, or pick another dataset.');
    await new Promise((r) => setTimeout(r, 50));
    expect([asked(read, 'ds_shapes_hard'), asked(images, 'ds_shapes_hard'), asked(estimate, 'ds_shapes_hard')]).toEqual([1, 0, 0]);

    // Another visit soon after: the list's mark, not the same failing read again.
    unmount(c!);
    await open('?dataset=ds_shapes_hard');
    await until(() => target.querySelector('.card .badge')?.textContent === 'Folder missing', 'the mark');
    expect(asked(read, 'ds_shapes_hard')).toBe(1);

    // The folder put back: the first read after RECHECK finds it, and Start is on.
    read.mockRestore();
    const now = Date.now();
    vi.spyOn(Date, 'now').mockReturnValue(now + RECHECK + 1);
    unmount(c!);
    await open('?dataset=ds_shapes_hard');
    await until(() => starts().every((b) => !b.disabled), 'Start');
    expect(target.querySelector('.card .badge')).toBeNull();
    vi.restoreAllMocks();
  });

  it("says a draft's or a link's dataset was removed instead of switching to another in silence", async () => {
    const images = vi.spyOn(api, 'datasetImages');
    const estimate = vi.spyOn(api, 'estimateRun');
    vi.spyOn(api, 'runs').mockImplementation(async () => {
      const list = await MockApi.prototype.runs.call(api);
      return [{ ...list[0]!, config: { ...list[0]!.config, dataset_id: 'ds_gone' }, dataset: { name: 'shapes-five', classes: [] } }, ...list.slice(1)];
    });
    await open('?dataset=ds_gone');
    await until(() => target.querySelector('#start-why')?.textContent?.includes('removed'), 'the line');
    expect(target.querySelector('input[name="dataset"]:checked')).toBeNull();
    expect(target.querySelector('#start-why')?.textContent).toBe('Dataset: shapes-five was removed: pick a dataset.');
    expect(target.querySelector('#dataset-error')?.textContent).toBe('shapes-five was removed: pick a dataset.');
    expect([asked(images, 'ds_gone'), asked(estimate, 'ds_gone')]).toEqual([0, 0]);
    // Picking one is all it takes.
    target.querySelector<HTMLInputElement>('input[name="dataset"][value="ds_shapes"]')!.click();
    await until(() => starts().every((b) => !b.disabled), 'Start');
    expect(target.querySelector('#dataset-error')).toBeNull();
  });
});

describe('New run: the keyboard', () => {
  const press = (el: Element, init: KeyboardEventInit = {}) => {
    const e = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, ...init });
    el.dispatchEvent(e);
    return e;
  };

  it('never starts a run on Enter in a field; Ctrl+Enter starts it, as the Start button says', async () => {
    await open();
    const create = vi.spyOn(api, 'createRun');
    // The browser's implicit submission is the default action of Enter in an input: turned off.
    expect(press(target.querySelector('input#epochs')!).defaultPrevented).toBe(true);
    expect(press(target.querySelector('input#name')!).defaultPrevented).toBe(true);
    await tick();
    expect(create).not.toHaveBeenCalled();
    // The Start button's tooltip names the shortcut.
    const tip = target.querySelector('.start .start-tip')!;
    tip.dispatchEvent(new Event('pointerenter'));
    const label = tip.querySelector('button')!.textContent!.trim();
    await until(() => document.getElementById('folo-tooltip')?.textContent?.includes(label), 'the tooltip');
    expect([...document.querySelectorAll('#folo-tooltip kbd')].map((k) => k.textContent)).toEqual(['Ctrl', '↵']);
    press(target.querySelector('input#epochs')!, { ctrlKey: true });
    await until(() => create.mock.calls.length === 1, 'the start');
  });
});

describe('New run: the run name', () => {
  const name = () => target.querySelector<HTMLInputElement>('input#name')!;
  const type = (v: string) => {
    name().value = v;
    name().dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
  };
  const hint = () => target.querySelector('#name-hint')?.textContent;

  it('turns spaces into dashes on leaving the field, saying so while typing, and starts with the tidy name', async () => {
    await open();
    type('my first run');
    expect(hint()).toBe('Saved as my-first-run: letters, digits, dot, dash and underscore.');
    expect(target.querySelector('#name-error')).toBeNull();
    name().dispatchEvent(new Event('blur'));
    flushSync();
    expect(name().value).toBe('my-first-run');
    expect(hint()).toBeUndefined();
    const create = vi.spyOn(api, 'createRun');
    type('my second run');
    await start();
    expect(create.mock.calls[0]![0].name).toBe('my-second-run');
  });

  it('warns about a name another run has, and refuses one over 64 characters', async () => {
    await open();
    const other = (await api.runs())[0]!.name;
    type(other);
    expect(hint()).toBe(`A run called ${other} already exists. Pick another name to tell them apart.`);
    type('x'.repeat(65));
    expect(target.querySelector('#name-error')?.textContent).toBe('At most 64 characters (this one has 65).');
    expect(target.querySelector('#start-why')?.textContent).toBe('Run name: At most 64 characters (this one has 65).');
  });
});

describe('New run while another run trains', () => {
  it('says the run will queue, reads "Queue run", and announces it as queued; on a free machine, as started', async () => {
    const { toast } = await import('$lib/ui/toast');
    const said = vi.spyOn(toast, 'success');
    await open();
    const training = (await api.runs()).find((r) => r.status === 'running' && !r.cloud)!;
    const line = await until(() => target.querySelector('.summary .ahead'), 'the queue line');
    const waiting = (await api.runs()).filter((r) => r.queue_position != null).length;
    expect(line.textContent?.startsWith(`${training.name} is training and ${waiting} runs wait: this run will be 3rd in the queue`)).toBe(true);
    expect(waiting).toBe(2);
    expect(target.querySelector('.start button[type="submit"]')?.textContent?.trim()).toBe('Queue run');
    expect(target.querySelector('.dock button[type="submit"]')?.textContent?.trim()).toBe('Queue run');
    const run = { ...(await api.run(training.id)), id: 'run_q2', name: 'q-two', status: 'queued' as const, queue_position: waiting + 1 };
    vi.spyOn(api, 'createRun').mockResolvedValue(run);
    await start();
    await until(() => said.mock.calls.length, 'the toast');
    expect(said.mock.calls[0]![0]).toBe('Queued q-two');
    expect(said.mock.calls[0]![1]).toMatch(/^3rd in queue · /);

    // Nothing else on this machine: no line, "Start training", and a first place in the queue is a start.
    vi.mocked(api.createRun).mockResolvedValue({ ...run, queue_position: 1 });
    unmount(c!);
    c = undefined;
    sessionStorage.clear();
    said.mockClear();
    const runs = (await api.runs()).filter((r) => !['queued', 'running', 'stopping'].includes(r.status));
    vi.spyOn(api, 'runs').mockResolvedValue(runs);
    await open();
    expect(target.querySelector('.summary .ahead')).toBeNull();
    expect(target.querySelector('.start button[type="submit"]')?.textContent?.trim()).toBe('Start training');
    await start();
    await until(() => said.mock.calls.length, 'the toast');
    expect(said.mock.calls[0]![0]).toBe('Started q-two');
  });
});

describe('New run on a dataset with no training images', () => {
  it('says so at the picker, keeps Start off with the reason, and shows no figures', async () => {
    const real = api.dataset.bind(api);
    vi.spyOn(api, 'dataset').mockImplementation(async (id) => {
      const d = await real(id);
      return id === 'ds_shapes' ? { ...d, splits: { train: 0, val: 0, test: 72 } } : d;
    });
    const asked = vi.spyOn(api, 'estimateRun');
    await open('?dataset=ds_shapes');
    await until(() => target.querySelector('#dataset-error'), 'the dataset error');
    expect(target.querySelector('#dataset-error')!.textContent).toBe(
      'No training images (72 in test). Training needs images in a train split: pick another dataset, or add training images to this one.',
    );
    expect(target.querySelector('#start-why')?.textContent).toBe('Dataset: No training images (72 in test).');
    expect([...target.querySelectorAll<HTMLButtonElement>('button[type="submit"]')].every((b) => b.disabled)).toBe(true);
    // No "0 optimizer steps / epoch" or "under 1 min": dashes, and the server is not asked.
    const est = target.querySelector('.summary .est')!;
    expect([...est.querySelectorAll('dd')].map((d) => d.textContent)).toEqual(['—no estimate', '—no estimate']);
    expect(asked.mock.calls.filter(([c]) => c.dataset_id === 'ds_shapes')).toHaveLength(0);
  });
});

describe('New run: start from a model of another size', () => {
  it('lists the trained models of other sizes, says so when this size has none, and switches the size when one is picked', async () => {
    const real = api.models.bind(api);
    vi.spyOn(api, 'models').mockImplementation(async (...a) => (await real(...a)).filter((m) => m.arch === 'folo-n'));
    await open();
    expect(target.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value).toBe('folo-s');
    const sel = target.querySelector<HTMLSelectElement>('select#weights')!;
    const others = await until(() => [...sel.querySelectorAll<HTMLOptionElement>('optgroup[label="folo-n: switches the size"] option')].filter((o) => o.value).length || null, 'the folo-n models');
    const trained = (await real()).filter((m) => m.arch === 'folo-n' && m.fused !== true && m.source !== 'published').length;
    expect(others).toBe(trained);
    expect(target.querySelector('#weights-hint')?.textContent).toBe(`No trained folo-s models yet. ${trained} trained folo-n models are in the list: picking one switches the size.`);
    const opt = sel.querySelector<HTMLOptionElement>('optgroup[label="folo-n: switches the size"] option')!;
    opt.selected = true;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value).toBe('folo-n');
    expect(sel.value).toBe(opt.value);
    expect(target.querySelector('#weights-error')).toBeNull();
  });
});

describe('New run: an epoch count out of range', () => {
  it('shows no estimate while the field is wrong, and the figures again once it is fixed', async () => {
    await open();
    const est = () => [...target.querySelectorAll('.summary .est > dd.num')].map((d) => d.textContent);
    const type = (v: string) => {
      const el = target.querySelector<HTMLInputElement>('input#epochs')!;
      el.value = v;
      el.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
    };
    for (const bad of ['99999', '0']) {
      type(bad);
      expect(target.querySelector('#epochs-error')).not.toBeNull();
      expect(est()).toEqual(['—no estimate', '—no estimate']);
    }
    type('20');
    await until(() => !est().includes('—no estimate'), 'the figures');
    expect(est()[1]).toMatch(/\d/);
  });
});

describe('New run with no dataset yet', () => {
  it('leads straight to the Add dataset dialog, or to a generated sample set', async () => {
    hub.api = api = new MockApi('empty', 0);
    router.route = parse('#/runs/new');
    c = mount(RunNew, { target });
    await until(() => target.textContent?.includes('Add a dataset first'), 'the empty state');
    const links = [...target.querySelectorAll<HTMLAnchorElement>('a')].map((a) => [a.textContent?.trim(), a.getAttribute('href')]);
    expect(links).toEqual([
      ['Add dataset', '#/datasets?add=folder'],
      ['Generate sample set', '#/datasets?add=synthetic'],
    ]);
  });
});

// Stress s2-runs-7: `?from=` a run that does not exist toasted the server's hint for agents, and the toast outlived the link.
describe('New run from a run that does not exist', () => {
  it('says so in plain words, starts from the defaults, and the toast goes with the page', async () => {
    const { toast } = await import('$lib/ui/toast');
    vi.spyOn(api, 'run').mockRejectedValue(new ApiError(404, 'run_not_found', "run 'run_nope' does not exist", 'list_runs (MCP) or GET /api/runs lists the runs that exist'));
    await open('?from=run_nope');
    const t = await until(() => toast.items.find((x) => x.title === 'Run to copy not found'), 'the toast');
    expect(t.detail).toBe('It was deleted or the link is wrong. The form starts from the defaults.');
    expect(JSON.stringify(toast.items)).not.toContain('MCP');
    unmount(c!);
    c = undefined;
    expect(toast.items.find((x) => x.title === 'Run to copy not found')).toBeUndefined();
  });

  // PR #287 review (sibling of s3-live-3): the form read the datasets once, so one removed in another window stayed
  // picked with Start on.
  it('notices the picked dataset removed in another window, and keeps Start off saying so', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      await open();
      const id = await until(() => target.querySelector<HTMLInputElement>('input[name="dataset"]:checked')?.value, 'a picked dataset');
      const name = (await api.datasets()).find((d) => d.id === id)!.name;
      expect(target.querySelector('#start-why')?.textContent ?? '').not.toContain('removed');
      const list = api.datasets.bind(api);
      vi.spyOn(api, 'datasets').mockImplementation(async () => (await list()).filter((d) => d.id !== id));
      await vi.advanceTimersByTimeAsync(5000);
      const why = await until(() => (target.querySelector('#start-why')?.textContent?.includes('removed') ? target.querySelector('#start-why') : null), 'the removal');
      expect(why.textContent).toContain(`${name} was removed`);
      expect([...target.querySelectorAll<HTMLButtonElement>('button[type="submit"]')].every((b) => b.disabled)).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('leaves an agent-only hint out of any error toast', async () => {
    const { errorDetail } = await import('$lib/ui/toast');
    expect(errorDetail(new ApiError(404, 'model_not_found', 'No model m_x.', 'list_models (MCP) or GET /api/models lists them'))).toBe('No model m_x.');
    expect(errorDetail(new ApiError(409, 'busy', 'a run is active', 'stop it first'))).toBe('A run is active. Stop it first.');
  });
});
