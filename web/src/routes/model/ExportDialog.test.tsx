// @vitest-environment happy-dom
import { flush as flushSync } from 'fw';
import { render as renderFw } from '../../../framework/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HttpApi } from '$lib/api/client';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import type { Model } from '$lib/api/types';
import { toast } from '$lib/ui/toast';
import ExportDialog, { type ExportDialogProps } from './ExportDialog';

/** Port of `mount(ExportDialog, { target, props })`: renders into `target`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof ExportDialog, o: { target: HTMLElement; props: ExportDialogProps }) => renderFw(() => <ExportDialog {...o.props} />, o.target);
const unmount = (c: { unmount(): void }) => c.unmount();

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

/** `/api/system` without optional features; tests add the ones they need. */
const SYS = { cpu: { name: 'cpu', threads: 1, util: 0 }, memory: { used_bytes: 0, total_bytes: 1 }, gpus: [] };

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

/** The mock answering as a server would (its exports and features), not as the demo, which creates no file. */
const asServer = (api: MockApi) => Object.defineProperty(api, 'mock', { value: false }) as MockApi;

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
    hub.api = asServer(new MockApi('normal', 0));
    vi.spyOn(hub.api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best-f16.folo', bytes: 5_300_000 });
    const success = vi.spyOn(toast, 'success');
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    button('Export')!.click();
    await until(() => button('Done'), 'the finished state');
    expect(target.ownerDocument.querySelector('dialog')?.textContent).toContain('Export finished');
    expect(success).not.toHaveBeenCalled();
    // Without the toast, a live region in the dialog announces it.
    const live = target.ownerDocument.querySelectorAll('dialog p[role="status"]');
    expect(live).toHaveLength(1);
    expect(live[0]!.textContent).toBe('Export finished: shapes-n-best-f16.folo, 5.1 MiB.');
  });

  it('says plainly that ONNX cannot be downloaded, and shows where it is', async () => {
    hub.api = asServer(new MockApi('normal', 0));
    // An older server: no ONNX download.
    vi.spyOn(hub.api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches'] });
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
    hub.api = asServer(new MockApi('normal', 0));
    // A server with the switches but no ONNX download: the export writes the file on the server.
    vi.spyOn(hub.api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches'] });
    const run = vi.spyOn(hub.api, 'exportModel');
    c = mount(ExportDialog, { target, props: { model: { ...model, id: 'm_fast_n' }, open: true } });
    flushSync();
    const checks = () => [...target.ownerDocument.querySelectorAll<HTMLInputElement>('dialog [role="group"] input[type="checkbox"]')];
    // There and off until ONNX is picked (models F17: picking it does not grow the dialog under the pointer).
    const off = await until(() => (checks().length === 3 ? checks() : null), 'the switches');
    expect(off.every((b) => b.disabled)).toBe(true);
    const height = target.ownerDocument.querySelector('dialog fieldset')!.childElementCount;
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    const boxes = await until(() => (checks().every((b) => !b.disabled) ? checks() : null), 'the switches on');
    expect(target.ownerDocument.querySelector('dialog fieldset')!.childElementCount).toBe(height);
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
    hub.api = asServer(new MockApi('normal', 0));
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

describe('ExportDialog: ONNX download (weights_onnx)', () => {
  const pickOnnx = async () => {
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    return until(() => {
      const all = [...target.ownerDocument.querySelectorAll<HTMLInputElement>('dialog [role="group"] input[type="checkbox"]')];
      return all.length === 3 && all.every((b) => !b.disabled) ? all : null;
    }, 'the switches');
  };

  it('downloads the graph straight away, the button naming the file and the switches in the query', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches', 'weights_onnx'] });
    const exp = vi.spyOn(api, 'exportModel');
    const get = vi.spyOn(api, 'onnxFile').mockResolvedValue(new File([new Uint8Array(2048)], 'shapes-n-best-f16.onnx'));
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    const boxes = await pickOnnx();
    await until(() => button('Download shapes-n-best.onnx'), 'the download button');
    boxes[2]!.click(); // half
    flushSync();
    const dl = await until(() => button('Download shapes-n-best-f16.onnx'), 'the f16 name');
    expect(dl.classList.contains('primary')).toBe(true);
    expect(button('Export')).toBeUndefined();
    let saved = '';
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved = this.download;
    });
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    dl.click();
    await until(() => button('Done'), 'the finished state');
    expect(get).toHaveBeenCalledWith('m_x', { decode: false, dynamic: false, half: true });
    expect(exp).not.toHaveBeenCalled();
    expect(saved).toBe('shapes-n-best-f16.onnx');
    const text = target.ownerDocument.querySelector('dialog .done')?.textContent ?? '';
    expect(text).toContain('Downloaded');
    expect(text).toContain('shapes-n-best-f16.onnx');
    expect(text).toContain('Nothing was written on the server.');
    expect(target.ownerDocument.querySelector('dialog [role="status"]')?.textContent).toBe('Downloaded shapes-n-best-f16.onnx, 2.0 KiB.');
  });

});

describe('ExportDialog: the demo creates no file', () => {
  const dialogText = () => target.ownerDocument.querySelector('dialog')?.textContent?.replace(/\s+/g, ' ') ?? '';
  const links = () => [...target.ownerDocument.querySelectorAll<HTMLAnchorElement>('dialog a')];

  for (const format of ['folo-f16', 'onnx'] as const) {
    it(`says so before and after, for ${format}: a neutral preview, no saved file, no Download, a way to a server`, async () => {
      hub.api = new MockApi('normal', 0);
      const clicked = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
      const success = vi.spyOn(toast, 'success');
      c = mount(ExportDialog, { target, props: { model: { ...model, id: 'm_fast_n', name: 'fasteners-n-best' }, open: true } });
      flushSync();
      if (format === 'onnx') target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
      // The server lists weights_onnx, but the demo has no graph: never "Download <file>".
      await new Promise((r) => setTimeout(r, 20));
      flushSync();
      expect(dialogText()).toContain('Demo: no file is created.');
      expect(button('Download')).toBeUndefined();
      button('Export')!.click();
      await until(() => button('Done'), 'the preview');
      const text = dialogText();
      expect(text).toContain('Demo preview: no file was created');
      expect(text).toContain(format === 'onnx' ? 'fasteners-n-best.onnx' : '.folo');
      expect(text).not.toMatch(/Export finished|Saved on the server|Downloaded/);
      expect(target.ownerDocument.querySelector('dialog .ok:not(.quiet)')).toBeNull();
      expect(button('Download')).toBeUndefined();
      const connect = links().find((a) => a.textContent?.includes('Connect to a server'))!;
      expect(connect.classList.contains('primary')).toBe(true);
      expect(connect.getAttribute('href')).toMatch(/settings$/);
      expect(clicked).not.toHaveBeenCalled();
      expect(success).not.toHaveBeenCalled();
      expect(target.ownerDocument.querySelector('dialog [role="status"]')?.textContent).toBe('Demo preview: no file was created.');
    });
  }
});
