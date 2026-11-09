// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import { parse, router } from '$lib/router.svelte';
import RunNew from './RunNew.svelte';

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
    // One short line from the devices listed now; the server's words (backends and all) fold into the details.
    await until(() => msg.textContent === 'GPU 1 is no longer available. Pick Auto, GPU 0 or CPU.', 'the short line');
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
    expect(target.querySelector('#start-why')?.textContent).toBe('Device: GPU 1 is not available.');
    // Both Start buttons (summary and phone dock) carry the reason, and the select is tied to its error.
    expect(target.querySelector('#dock-why')?.textContent).toBe('Device: GPU 1 is not available.');
    const starts = [...target.querySelectorAll<HTMLButtonElement>('.start button[type="submit"], .dock button[type="submit"]')];
    expect(starts.map((b) => b.getAttribute('aria-describedby'))).toEqual(['start-why', 'dock-why']);
    const sel = target.querySelector('select#device')!;
    expect(sel.getAttribute('aria-invalid')).toBe('true');
    expect(sel.getAttribute('aria-describedby')).toBe('device-error');
    // The reason is a way to the field.
    target.querySelector<HTMLSelectElement>('select#device')!.blur();
    target.querySelector<HTMLButtonElement>('#start-why button')!.click();
    await until(() => document.activeElement === target.querySelector('select#device'), 'focus on Device');
    pick('auto');
    expect(target.querySelector('#device-error')).toBeNull();
    expect(target.querySelector('select#device')!.hasAttribute('aria-invalid')).toBe(false);
  });

  it('offers Auto as the way out', async () => {
    await open();
    await until(() => target.querySelector('select#device option[value="gpu:1"]'));
    pick('gpu:1');
    await start();
    await until(() => target.querySelector('#device-error'));
    const auto = [...target.querySelectorAll<HTMLButtonElement>('.dev-err button')].find((b) => b.textContent?.trim() === 'Use Auto')!;
    auto.click();
    flushSync();
    expect(target.querySelector<HTMLSelectElement>('select#device')!.value).toBe('auto');
    expect(target.querySelector('#device-error')).toBeNull();
    expect(target.querySelector('#start-why')).toBeNull();
  });

  it('treats a 501 for a GPU from servers before device_not_found the same way', async () => {
    await open();
    pick('gpu:0');
    vi.spyOn(api, 'createRun').mockRejectedValue(
      new ApiError(501, 'not_implemented', 'device gpu:0 cannot be used by this build', 'use device `cpu` (or `auto`, which falls back to the CPU)'),
    );
    await start();
    const msg = await until(() => target.querySelector('#device-error'));
    expect(msg.textContent).toBe('This server cannot train on GPU 0. Pick Auto or CPU.');
    expect(target.querySelector('.dev-detail p')?.textContent).toBe(
      'Device gpu:0 cannot be used by this build. Use device cpu (or auto, which falls back to the CPU).',
    );
  });
});

describe('New run after running out of GPU memory', () => {
  it('names what failed and what changed, and marks the changed field', async () => {
    await open('?from=run_a2f0&batch=16');
    const note = await until(() => target.querySelector('.oom-note'));
    expect(note.textContent?.replace(/\s+/g, ' ')).toContain('shapes-hard-m ran out of GPU memory at batch 32. Now batch 16.');
    expect(target.querySelector('#batch [aria-checked="true"]')?.textContent?.trim()).toBe('16');
    const batchField = target.querySelector('#batch')!.closest('.field')!;
    expect(batchField.querySelector('.was')?.textContent).toBe('was 32');
    // 960 px is kept from the failed run; the image size field is not marked.
    expect(target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim()).toBe('960');
    expect(target.querySelector('#imgsz')!.closest('.field')!.querySelector('.was')).toBeNull();
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

  it('warns when warmup takes most of the run, and offers enough epochs', async () => {
    await open();
    expect(target.querySelector('.short')).toBeNull();
    // fasteners: 1,920 train images at batch 16 are 30 optimizer steps an epoch; 5 epochs are 150, 100 of them warmup.
    type('epochs', 5);
    const warn = await until(() => target.querySelector('.summary .short'));
    expect(warn.textContent).toContain("Warmup takes 100 of the run's 150 optimizer steps");
    expect(target.querySelector('.dock-warn')?.textContent).toBe('Mostly warmup');
    warn.querySelector<HTMLButtonElement>('button.fix')!.click();
    flushSync();
    expect(target.querySelector<HTMLInputElement>('input#epochs')!.value).toBe('7');
    expect(target.querySelector('.short')).toBeNull();
  });

  it('keeps mosaic in a short run, and warns when close mosaic is set to cover all of it', async () => {
    await open();
    type('epochs', 8);
    // Close mosaic follows the run: at most half of it.
    expect(target.querySelector<HTMLInputElement>('input#close')!.value).toBe('4');
    expect(target.querySelector('.short')).toBeNull();
    type('close', 8);
    const warn = await until(() => target.querySelector('.summary .short'));
    expect(warn.textContent).toContain('Close mosaic covers all 8 epochs');
    warn.querySelector<HTMLButtonElement>('button.fix')!.click();
    flushSync();
    expect(target.querySelector<HTMLInputElement>('input#close')!.value).toBe('4');
    expect(target.querySelector('.short')).toBeNull();
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

  it('keeps 640 for larger images', async () => {
    await open('?dataset=ds_shapes');
    await new Promise((r) => setTimeout(r, 400));
    flushSync();
    expect(target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim()).toBe('640');
  });

  it('lets a size from the link win', async () => {
    vi.spyOn(api, 'datasetImages').mockResolvedValue({ total: 1, images: [{ id: 'img_0', file: 'a.jpg', split: 'train', width: 320, height: 320, boxes: [] }] });
    await open('?imgsz=960');
    await new Promise((r) => setTimeout(r, 400));
    flushSync();
    expect(target.querySelector('#imgsz [aria-checked="true"]')?.textContent?.trim()).toBe('960');
  });
});
