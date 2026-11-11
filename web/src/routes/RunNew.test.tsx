// @vitest-environment happy-dom
import { flush as flushSync, tick } from 'fw';
import { render } from '../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, hub } from '$lib/api/index';
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
