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
  it('says so at the Device field with the adapters it has, lists them again and keeps the form', async () => {
    await open();
    await until(() => target.querySelector('select#device option[value="gpu:1"]'));
    const name = target.querySelector<HTMLInputElement>('input#name')!.value;
    pick('gpu:1');
    const create = vi.spyOn(api, 'createRun');
    await start();
    const msg = await until(() => target.querySelector('#device-error'), 'the device error');
    expect(create).toHaveBeenCalledOnce();
    expect(msg.textContent).toContain('Device gpu:1 is not available');
    expect(msg.textContent).toContain('This machine has gpu:0 (NVIDIA GeForce RTX 4080 SUPER, vulkan), cpu');
    expect(msg.textContent).not.toContain('`');
    // The list is read again: the adapter is gone, the choice stays (marked), and so does the rest of the form.
    const opt = await until(() => {
      const o = target.querySelector<HTMLOptionElement>('select#device option[value="gpu:1"]');
      return o?.textContent?.includes('not found') ? o : null;
    });
    expect(opt.selected).toBe(true);
    expect(target.querySelector<HTMLInputElement>('input#name')!.value).toBe(name);
    // Start says why it is off, briefly, until another device is picked.
    expect([...target.querySelectorAll<HTMLButtonElement>('button[type="submit"]')].every((b) => b.disabled)).toBe(true);
    expect(target.querySelector('#start-why')?.textContent).toBe('Device: gpu:1 is not on this machine; pick another.');
    // Both Start buttons (summary and phone dock) carry the reason, and the select is tied to its error.
    expect(target.querySelector('#dock-why')?.textContent).toBe('Device: gpu:1 is not on this machine; pick another.');
    const starts = [...target.querySelectorAll<HTMLButtonElement>('.start button[type="submit"], .dock button[type="submit"]')];
    expect(starts.map((b) => b.getAttribute('aria-describedby'))).toEqual(['start-why', 'dock-why']);
    const sel = target.querySelector('select#device')!;
    expect(sel.getAttribute('aria-invalid')).toBe('true');
    expect(sel.getAttribute('aria-describedby')).toBe('device-error');
    pick('auto');
    expect(target.querySelector('#device-error')).toBeNull();
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
    expect(msg.textContent).toBe('Device gpu:0 cannot be used by this build. Use device cpu (or auto, which falls back to the CPU).');
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
