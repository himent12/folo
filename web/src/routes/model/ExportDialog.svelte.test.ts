// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HttpApi } from '$lib/api/client';
import { hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import type { Model } from '$lib/api/types';
import { toast } from '$lib/ui/toast.svelte';
import ExportDialog from './ExportDialog.svelte';

const model: Model = {
  id: 'm_x',
  name: 'shapes-n-best',
  arch: 'folo-n',
  imgsz: 640,
  classes: ['a'],
  params: 2_624_064,
  flops_g: 6.5,
  bytes: 5_400_000,
  created: '2026-11-01T00:00:00Z',
  run_id: null,
  metrics: null,
};

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;

async function until<T>(get: () => T | null | undefined | false, what: string): Promise<T> {
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

const buttons = () => [...target.ownerDocument.querySelectorAll<HTMLButtonElement>('dialog button')];
const button = (text: string) => buttons().find((b) => b.textContent?.trim().startsWith(text));

beforeEach(() => {
  // happy-dom has no modal dialogs: opening is enough for the content to render.
  HTMLDialogElement.prototype.showModal ??= function (this: HTMLDialogElement) {
    this.open = true;
  };
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
  vi.restoreAllMocks();
});

describe('ExportDialog', () => {
  it("offers the f16 file as the primary download, named as the server's file", async () => {
    const api = new HttpApi('http://folo.test');
    // A second export of the same model: the server names it -2, and the download follows.
    vi.spyOn(api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best-f16-2.folo', bytes: 5_300_000 });
    vi.spyOn(api, 'system').mockResolvedValue({ cpu: { name: 'cpu', threads: 1, util: 0 }, memory: { used_bytes: 0, total_bytes: 1 }, gpus: [] });
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    button('Export')!.click();
    const dl = await until(() => button('Download'), 'the download button');
    expect(dl.classList.contains('primary')).toBe(true);
    expect(button('Done')!.classList.contains('primary')).toBe(false);
    const clicked = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.href).toBe('http://folo.test/api/models/m_x/weights?format=f16');
      expect(this.download).toBe('shapes-n-best-f16-2.folo');
    });
    dl.click();
    expect(clicked).toHaveBeenCalledOnce();
  });

  it('says "Export finished" once: no toast on top of the open dialog', async () => {
    hub.api = new MockApi('normal', 0);
    vi.spyOn(hub.api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best-f16.folo', bytes: 5_300_000 });
    const success = vi.spyOn(toast, 'success');
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    button('Export')!.click();
    await until(() => button('Done'), 'the finished state');
    expect(target.ownerDocument.querySelector('dialog')?.textContent).toContain('Export finished');
    expect(success).not.toHaveBeenCalled();
    // Without the toast, a live region in the dialog announces it.
    const live = target.ownerDocument.querySelectorAll('dialog [role="status"]');
    expect(live).toHaveLength(1);
    expect(live[0]!.textContent).toBe('Export finished: shapes-n-best-f16.folo, 5.1 MiB.');
  });

  it('says plainly that ONNX cannot be downloaded, and shows where it is', async () => {
    hub.api = new MockApi('normal', 0);
    vi.spyOn(hub.api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best.onnx', bytes: 10_700_000 });
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    flushSync();
    button('Export')!.click();
    await until(() => target.ownerDocument.querySelector('dialog .where code'), 'the path');
    expect(button('Download')).toBeUndefined();
    expect(button('Done')!.classList.contains('primary')).toBe(true);
    expect(target.ownerDocument.querySelector('dialog .note')?.textContent).toMatch(/cannot download ONNX/);
  });

  it('sends the ONNX switches the server says it takes', async () => {
    hub.api = new MockApi('normal', 0);
    const run = vi.spyOn(hub.api, 'exportModel');
    c = mount(ExportDialog, { target, props: { model: { ...model, id: 'm_fast_n' }, open: true } });
    flushSync();
    const checks = () => [...target.ownerDocument.querySelectorAll<HTMLInputElement>('dialog [role="group"] input[type="checkbox"]')];
    // Hidden until ONNX is picked, also on a server that has them.
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(checks()).toHaveLength(0);
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    const boxes = await until(() => (checks().length === 3 ? checks() : null), 'the switches');
    const names = boxes.map((b) => b.closest('label')?.querySelector('.sw-name')?.textContent);
    expect(names).toEqual(['Decode boxes in the graph', 'Dynamic input size', 'Half precision (f16)']);
    boxes[0]!.click();
    boxes[2]!.click();
    flushSync();
    button('Export')!.click();
    await until(() => target.ownerDocument.querySelector('dialog .where code'), 'the path');
    expect(run).toHaveBeenCalledWith('m_fast_n', 'onnx', { decode: true, dynamic: false, half: true });
    expect(target.ownerDocument.querySelector('dialog .where code')?.textContent).toMatch(/-f16\.onnx$/);
  });

  it('hides the ONNX switches from servers without them, and never sends them', async () => {
    hub.api = new MockApi('normal', 0);
    vi.spyOn(hub.api, 'system').mockResolvedValue({ cpu: { name: 'cpu', threads: 1, util: 0 }, memory: { used_bytes: 0, total_bytes: 1 }, gpus: [], features: [] });
    const run = vi.spyOn(hub.api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best.onnx', bytes: 10_700_000 });
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(target.ownerDocument.querySelectorAll('dialog input[type="checkbox"]')).toHaveLength(0);
    button('Export')!.click();
    await until(() => target.ownerDocument.querySelector('dialog .where code'), 'the path');
    expect(run).toHaveBeenCalledWith('m_x', 'onnx', undefined);
  });
});
