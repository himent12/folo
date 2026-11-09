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
  it('offers the f16 file as a download from the weights endpoint', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best-f16.folo', bytes: 5_300_000 });
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    button('Export')!.click();
    const dl = await until(() => button('Download'), 'the download button');
    const clicked = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      expect(this.href).toBe('http://folo.test/api/models/m_x/weights?format=f16');
      expect(this.download).toBe('shapes-n-best-f16.folo');
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
    expect(live[0]!.textContent).toBe('Export finished: shapes-n-best-f16.folo, 5.05 MiB.');
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
    expect(target.ownerDocument.querySelector('dialog .note')?.textContent).toMatch(/cannot download ONNX/);
  });
});
