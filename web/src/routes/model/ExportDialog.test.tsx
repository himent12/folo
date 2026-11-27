// @vitest-environment happy-dom
import { flush as flushSync, signal } from 'fw';
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
  it('downloads the f16 file straight away, the button naming it, and writes nothing on the server', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue(SYS);
    const exp = vi.spyOn(api, 'exportModel');
    const get = vi.spyOn(api, 'f16File').mockResolvedValue(new File([new Uint8Array(4096)], 'shapes-n-best-f16.folo'));
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    const dl = await until(() => button('Download shapes-n-best-f16.folo'), 'the download button');
    expect(dl.classList.contains('primary')).toBe(true);
    let saved = '';
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved = this.download;
    });
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    dl.click();
    await until(() => button('Done'), 'the finished state');
    expect(get).toHaveBeenCalledWith('m_x');
    // models F1: each Export used to write <name>-f16-2.folo, -3, ... on the server, then download another file.
    expect(exp).not.toHaveBeenCalled();
    expect(saved).toBe('shapes-n-best-f16.folo');
    expect(button('Download')).toBeUndefined();
    expect(target.ownerDocument.querySelector('dialog .done')?.textContent).toContain('Nothing was written on the server.');
  });

  it('says "Export finished" once: no toast on top of the open dialog', async () => {
    hub.api = asServer(new MockApi('normal', 0));
    // An older server: ONNX is written on the server.
    vi.spyOn(hub.api, 'system').mockResolvedValue({ ...SYS, features: [] });
    vi.spyOn(hub.api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best.onnx', bytes: 10_700_000 });
    const success = vi.spyOn(toast, 'success');
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    flushSync();
    (await until(() => {
      const b = button('Export');
      return b && !b.disabled ? b : null;
    }, 'Export ready')).click();
    await until(() => button('Done'), 'the finished state');
    expect(target.ownerDocument.querySelector('dialog')?.textContent).toContain('Export finished');
    expect(success).not.toHaveBeenCalled();
    // Without the toast, a live region in the dialog announces it.
    const live = target.ownerDocument.querySelectorAll('dialog p[role="status"]');
    expect(live).toHaveLength(1);
    expect(live[0]!.textContent).toBe('Export finished: shapes-n-best.onnx, 10.2 MiB.');
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
    (await until(() => {
      const b = button('Export');
      return b && !b.disabled ? b : null;
    }, 'Export ready')).click();
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

  it('halves the ONNX size estimate on weights_onnx-only servers when Half precision is checked', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue({ ...SYS, features: ['weights_onnx'] });
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    const boxes = await pickOnnx();
    const est = () => [...target.ownerDocument.querySelectorAll('dialog .opt')][1]!.querySelector('.est')!.textContent!.trim();
    const before = est();
    boxes[2]!.click(); // half
    flushSync();
    const after = est();
    expect(after).not.toBe(before);
    expect(before).toMatch(/10\.3 MiB/);
    expect(after).toMatch(/5\.1 MiB/);
  });

  it('resets stale server capabilities if system() fails on a subsequent open', async () => {
    const api = new HttpApi('http://folo.test');
    const sysSpy = vi.spyOn(api, 'system').mockResolvedValueOnce({ ...SYS, features: ['weights_onnx'] }).mockRejectedValueOnce(new Error('offline'));
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    await pickOnnx();
    expect(target.ownerDocument.querySelectorAll('dialog input[type="checkbox"]')).toHaveLength(3);
    // Simulate a server switch where /api/system now fails: stale weights_onnx checkboxes must clear.
    hub.epoch++;
    flushSync();
    await until(() => sysSpy.mock.calls.length === 2 && target.ownerDocument.querySelectorAll('dialog input[type="checkbox"]').length === 0, 'cleared switches');
  });

  it('waits for slow /api/system before allowing ONNX export so weights_onnx does not fall back to server export', async () => {
    const api = new HttpApi('http://folo.test');
    let resolveSystem!: (s: typeof SYS & { features: string[] }) => void;
    vi.spyOn(api, 'system').mockImplementation(
      () =>
        new Promise((res) => {
          resolveSystem = res;
        }),
    );
    const exp = vi.spyOn(api, 'exportModel');
    const get = vi.spyOn(api, 'onnxFile').mockResolvedValue(new File([new Uint8Array(1024)], 'shapes-n-best.onnx'));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();

    // FOLO f16 stays ready immediately even while /api/system is still pending.
    expect(button('Download shapes-n-best-f16.folo')?.disabled).toBe(false);

    // Switching to ONNX while /api/system is pending holds the primary action instead of writing to the server.
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    flushSync();
    const pendingBtn = button('Checking server…');
    expect(pendingBtn).toBeDefined();
    expect(pendingBtn!.disabled).toBe(true);
    pendingBtn!.click();
    flushSync();
    expect(exp).not.toHaveBeenCalled();

    // Once /api/system resolves with weights_onnx + switches, ONNX downloads directly instead of POST /export.
    resolveSystem({ ...SYS, features: ['export_onnx_switches', 'weights_onnx'] });
    const dl = await until(() => {
      const b = button('Download shapes-n-best.onnx');
      return b && !b.disabled ? b : null;
    }, 'direct ONNX download button');
    expect(target.ownerDocument.querySelectorAll('dialog [role="group"] input[type="checkbox"]')).toHaveLength(3);
    dl.click();
    await until(() => button('Done'), 'finished download');
    expect(get).toHaveBeenCalledWith('m_x', { decode: false, dynamic: false, half: false });
    expect(exp).not.toHaveBeenCalled();
  });

  it('blocks ONNX export and offers Retry when /api/system fails, recovering to direct ONNX download on Retry', async () => {
    const api = new HttpApi('http://folo.test');
    const sysSpy = vi
      .spyOn(api, 'system')
      .mockRejectedValueOnce(new Error('502 Bad Gateway'))
      .mockResolvedValueOnce({ ...SYS, features: ['export_onnx_switches', 'weights_onnx'] });
    const exp = vi.spyOn(api, 'exportModel');
    const get = vi.spyOn(api, 'onnxFile').mockResolvedValue(new File([new Uint8Array(1024)], 'shapes-n-best.onnx'));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();

    // Switch to ONNX after /api/system failed: Export must stay disabled (never POST /export) and show Retry.
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    const retryBtn = await until(() => button('Retry'), 'capability Retry button');
    expect(retryBtn.disabled).toBe(false);
    const expBtn = button('Export')!;
    expect(expBtn.disabled).toBe(true);
    expBtn.click();
    flushSync();
    expect(exp).not.toHaveBeenCalled();

    // FOLO f16 remains immediately downloadable even while ONNX capability check failed.
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="folo-f16"]')!.click();
    flushSync();
    expect(button('Download shapes-n-best-f16.folo')?.disabled).toBe(false);

    // Back to ONNX -> click Retry -> recovers and downloads ONNX directly.
    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    flushSync();
    button('Retry')!.click();
    const dl = await until(() => {
      const b = button('Download shapes-n-best.onnx');
      return b && !b.disabled ? b : null;
    }, 'direct ONNX download after retry');
    expect(sysSpy).toHaveBeenCalledTimes(2);
    dl.click();
    await until(() => button('Done'), 'finished download');
    expect(get).toHaveBeenCalledTimes(1);
    expect(exp).not.toHaveBeenCalled();
  });

  it('moves keyboard focus to the done action when export finishes and back to the format radio on Export another', async () => {
    hub.api = new MockApi('normal', 0);
    c = mount(ExportDialog, { target, props: { model: { ...model, id: 'm_fast_n', name: 'fasteners-n-best' }, open: true } });
    flushSync();
    const expBtn = button('Export')!;
    expBtn.focus();
    expBtn.click();
    await until(() => button('Done'), 'done state');
    await until(() => target.ownerDocument.activeElement?.textContent?.includes('Connect to a server'), 'focus on primary done action');
    expect(target.ownerDocument.activeElement).not.toBe(target.ownerDocument.body);

    button('Export another')!.click();
    await until(() => target.ownerDocument.activeElement?.matches('input[name="export-format"]:checked'), 'focus back on checked format radio');
  });
});

describe('ExportDialog: closing while the file is on its way', () => {
  /** A download that ends when the test says. */
  function deferred() {
    let resolve!: (f: File) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<File>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    return { promise, resolve, reject };
  }
  const dialog = () => target.ownerDocument.querySelector('dialog')!;

  async function startThenClose(api: HttpApi) {
    vi.spyOn(api, 'system').mockResolvedValue(SYS);
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    (await until(() => button('Download shapes-n-best-f16.folo'), 'the download button')).click();
    flushSync();
    expect(dialog().textContent).toContain('Preparing');
    button('Cancel')!.click();
    flushSync();
    // happy-dom's close() may not fire the event the browser fires.
    if (!dialog().open) dialog().dispatchEvent(new Event('close'));
    flushSync();
  }

  it('still saves the file, and says so in a toast (models F6)', async () => {
    const api = new HttpApi('http://folo.test');
    const d = deferred();
    vi.spyOn(api, 'f16File').mockReturnValue(d.promise);
    const saved: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved.push(this.download);
    });
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    const success = vi.spyOn(toast, 'success');
    await startThenClose(api);
    d.resolve(new File([new Uint8Array(2048)], 'shapes-n-best-f16.folo'));
    await until(() => saved.length > 0, 'the save');
    expect(saved).toEqual(['shapes-n-best-f16.folo']);
    expect(success).toHaveBeenCalledWith('Downloaded shapes-n-best-f16.folo', '2.0 KiB');
  });

  it('says in a toast when it fails after closing', async () => {
    const api = new HttpApi('http://folo.test');
    const d = deferred();
    vi.spyOn(api, 'f16File').mockReturnValue(d.promise);
    const failed = vi.spyOn(toast, 'error');
    await startThenClose(api);
    d.reject(new Error('disk full'));
    await until(() => failed.mock.calls.length > 0, 'the error toast');
    expect(failed.mock.calls[0]![1]).toBe('Could not export shapes-n-best');
  });

  it('resets out of running and suppresses stale completion if hub.epoch changes mid-export', async () => {
    const api = new HttpApi('http://folo.test');
    const d = deferred();
    vi.spyOn(api, 'system').mockResolvedValue(SYS);
    vi.spyOn(api, 'f16File').mockReturnValue(d.promise);
    const saved: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved.push(this.download);
    });
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    const success = vi.spyOn(toast, 'success');
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model, open: true } });
    flushSync();
    (await until(() => button('Download shapes-n-best-f16.folo'), 'the download button')).click();
    flushSync();
    expect(dialog().textContent).toContain('Preparing');

    // Server switch: dialog resets immediately to idle (not wedged in running), and stale settlement is ignored.
    hub.epoch++;
    flushSync();
    expect(dialog().textContent).not.toContain('Preparing');
    expect(target.ownerDocument.querySelector<HTMLFieldSetElement>('dialog fieldset')!.disabled).toBe(false);
    d.resolve(new File([new Uint8Array(2048)], 'shapes-n-best-f16.folo'));
    await new Promise((r) => setTimeout(r, 20));
    flushSync();
    expect(saved).toEqual([]);
    expect(success).not.toHaveBeenCalled();
    expect(dialog().textContent).not.toContain('Downloaded');
  });

  it('switching props.model mid-export resets the new model dialog while letting Model A finish via toast', async () => {
    const api = new HttpApi('http://folo.test');
    const d = deferred();
    vi.spyOn(api, 'system').mockResolvedValue(SYS);
    vi.spyOn(api, 'f16File').mockReturnValue(d.promise);
    const saved: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      saved.push(this.download);
    });
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    const success = vi.spyOn(toast, 'success');
    hub.api = api;
    const current = signal<Model>(model);
    const open = signal(true);
    c = renderFw(() => <ExportDialog model={current.get()} bind:open={open} />, target);
    flushSync();
    (await until(() => button('Download shapes-n-best-f16.folo'), 'Model A download')).click();
    flushSync();
    expect(dialog().textContent).toContain('Preparing');

    // Switch to Model B while Model A is still downloading: Model B's dialog resets to idle.
    current.set({ ...model, id: 'm_y', name: 'fasteners-s-best' });
    flushSync();
    expect(dialog().textContent).not.toContain('Preparing');
    expect(button('Download fasteners-s-best-f16.folo')).toBeDefined();

    // Model A settles: downloads and toasts Model A without overwriting Model B's open dialog.
    d.resolve(new File([new Uint8Array(2048)], 'shapes-n-best-f16.folo'));
    await until(() => saved.length > 0, 'Model A save');
    expect(saved).toEqual(['shapes-n-best-f16.folo']);
    expect(success).toHaveBeenCalledWith('Downloaded shapes-n-best-f16.folo', '2.0 KiB');
    expect(dialog().textContent).not.toContain('Downloaded');
    expect(button('Download fasteners-s-best-f16.folo')).toBeDefined();
  });

  it('passes an AbortSignal to hub.api.system() and aborts it on dialog close and epoch change (#309)', () => {
    const api = new HttpApi('http://folo.test');
    const signals: (AbortSignal | undefined)[] = [];
    vi.spyOn(api, 'system').mockImplementation((s) => {
      signals.push(s);
      return new Promise(() => {});
    });
    hub.api = api;
    const open = signal(true);
    c = renderFw(() => <ExportDialog model={model} bind:open={open} />, target);
    flushSync();
    expect(signals).toHaveLength(1);
    expect(signals[0]?.aborted).toBe(false);

    // Server epoch change aborts the previous capability probe and starts a fresh one.
    hub.epoch++;
    flushSync();
    expect(signals[0]?.aborted).toBe(true);
    expect(signals).toHaveLength(2);
    expect(signals[1]?.aborted).toBe(false);

    // Closing the dialog aborts the in-flight probe cleanly.
    open.set(false);
    flushSync();
    expect(signals[1]?.aborted).toBe(true);
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

describe('ExportDialog: dynamic ONNX axes & CLI commands', () => {
  it('wires independent dynamic_batch / dynamic_hw through POST /export when export_dynamic_axes is advertised', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches', 'export_dynamic_axes', 'weights_onnx'] });
    const exp = vi.spyOn(api, 'exportModel').mockResolvedValue({ path: 'D:/folo/exports/shapes-n-best-decoded.onnx', bytes: 10_200_000 });
    const dl = vi.spyOn(api, 'onnxFile');
    hub.api = api;
    c = mount(ExportDialog, { target, props: { model: { ...model, path: 'D:/folo/models/shapes-n-best.folo' }, open: true } });
    flushSync();

    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    const axesGroup = await until(
      () => target.ownerDocument.querySelector('[role="radiogroup"][aria-label="Dynamic ONNX axes"]'),
      'dynamic axes selector',
    );
    // Static defaults to direct download on weights_onnx servers.
    expect(button('Download shapes-n-best.onnx')).toBeDefined();

    // Pick Batch (--dynamic-batch): switches from direct download to POST /export so partial axes are not dropped.
    const batchBtn = [...axesGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Batch')!;
    batchBtn.click();
    flushSync();
    expect(button('Download')).toBeUndefined();
    const expBtn = await until(() => button('Export'), 'Export button for split axes');

    // Also check Decode boxes in the graph.
    const decodeBox = target.ownerDocument.querySelectorAll<HTMLInputElement>('dialog [role="group"] input[type="checkbox"]')[0]!;
    decodeBox.click();
    flushSync();

    expBtn.click();
    await until(() => target.ownerDocument.querySelector('dialog .where code'), 'saved path');
    expect(dl).not.toHaveBeenCalled();
    expect(exp).toHaveBeenCalledWith('m_x', 'onnx', {
      decode: true,
      dynamic: false,
      dynamic_batch: true,
      dynamic_hw: false,
      half: false,
    });
    expect(target.ownerDocument.querySelector('dialog .note')?.textContent).toMatch(/Split dynamic-axis ONNX exports/);
    expect(target.ownerDocument.querySelector('dialog .note')?.textContent).not.toMatch(/cannot download ONNX files yet/);
  });

  it('keeps CLI-only dynamic axis preview independent of server export switches when export_dynamic_axes is absent', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches', 'weights_onnx'] });
    const get = vi.spyOn(api, 'onnxFile').mockResolvedValue(new File([new Uint8Array(1024)], 'shapes-n-best.onnx'));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:x');
    hub.api = api;
    c = mount(ExportDialog, {
      target,
      props: { model: { ...model, path: 'D:/folo/models/shapes-n-best.folo' }, open: true },
    });
    flushSync();

    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    const dynBox = await until(
      () => {
        const boxes = target.ownerDocument.querySelectorAll<HTMLInputElement>('dialog [role="group"] input[type="checkbox"]');
        return boxes.length === 3 && !boxes[1]!.disabled ? boxes[1]! : null;
      },
      'dynamic checkbox ready',
    );
    expect(dynBox.checked).toBe(false);

    // Open CLI commands and pick Batch (--dynamic-batch): only the CLI preview changes; server switch stays static.
    target.ownerDocument.querySelector<HTMLButtonElement>('[data-export-cli] button.cli-toggle')!.click();
    flushSync();
    const cliAxes = await until(
      () => target.ownerDocument.querySelector('[role="radiogroup"][aria-label="CLI dynamic ONNX axes"]'),
      'CLI dynamic axes picker',
    );
    [...cliAxes.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'Batch')!.click();
    flushSync();

    const exportPre = target.ownerDocument.querySelector('pre[aria-label="Export CLI command"]')!;
    expect(exportPre.textContent).toBe('folo export --model D:/folo/models/shapes-n-best.folo --format onnx --dynamic-batch');
    expect(dynBox.checked).toBe(false);

    // Download still sends dynamic: false (never silently promoted to dynamic=true by CLI preview).
    button('Download shapes-n-best.onnx')!.click();
    await until(() => button('Done'), 'finished download');
    expect(get).toHaveBeenCalledWith('m_x', { decode: false, dynamic: false, half: false });
  });

  it('shows copy-ready export and predict CLI snippets matching selected ONNX options', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches', 'export_dynamic_axes'] });
    hub.api = api;
    c = mount(ExportDialog, {
      target,
      props: { model: { ...model, path: 'D:/folo/models/shapes-n-best.folo' }, open: true },
    });
    flushSync();

    target.ownerDocument.querySelector<HTMLInputElement>('input[value="onnx"]')!.click();
    const axesGroup = await until(
      () => target.ownerDocument.querySelector('[role="radiogroup"][aria-label="Dynamic ONNX axes"]'),
      'dynamic axes selector',
    );
    const hwBtn = [...axesGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === 'H×W')!;
    hwBtn.click();
    flushSync();

    // Expand CLI commands disclosure.
    target.ownerDocument.querySelector<HTMLButtonElement>('[data-export-cli] button.cli-toggle')!.click();
    flushSync();

    const exportPre = await until(() => target.ownerDocument.querySelector('pre[aria-label="Export CLI command"]'), 'export CLI');
    expect(exportPre.textContent).toBe('folo export --model D:/folo/models/shapes-n-best.folo --format onnx --dynamic-hw');

    const predictPre = target.ownerDocument.querySelector('pre[aria-label="Predict CLI command"]')!;
    expect(predictPre.textContent).toContain('--tile 640 --tile-overlap 0.2');

    // Switch predict snippet to TTA and ByteTrack.
    const predGroup = target.ownerDocument.querySelector('[role="radiogroup"][aria-label="Predict snippet mode"]')!;
    const pickPred = (label: string) => {
      [...predGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label)!.click();
      flushSync();
    };
    pickPred('TTA');
    expect(predictPre.textContent).toBe('folo predict --model D:/folo/models/shapes-n-best.folo <images> --tta');
    pickPred('ByteTrack');
    expect(predictPre.textContent).toBe('folo predict --model D:/folo/models/shapes-n-best.folo <frames_dir> --track');

    // Verify Quantize (int8) selective-f32 CLI presets.
    const quantPre = target.ownerDocument.querySelector('pre[aria-label="Quantize CLI command"]')!;
    expect(quantPre.textContent).toBe(
      'folo quantize --model D:/folo/models/shapes-n-best.folo --data <dataset> --f32-policy auto --max-f32-macs 0.05 --sqnr-threshold 35',
    );
    const quantGroup = target.ownerDocument.querySelector('[role="radiogroup"][aria-label="Quantize selective-f32 policy"]')!;
    const pickQuant = (label: string) => {
      [...quantGroup.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent?.trim() === label)!.click();
      flushSync();
    };
    pickQuant('10% f32');
    expect(quantPre.textContent).toContain('--max-f32-macs 0.10 --sqnr-threshold 40');
    pickQuant('P5 tail');
    expect(quantPre.textContent).toBe(
      'folo quantize --model D:/folo/models/shapes-n-best.folo --data <dataset> --f32-policy p5-tail',
    );
    pickQuant('Full int8');
    expect(quantPre.textContent).toBe(
      'folo quantize --model D:/folo/models/shapes-n-best.folo --data <dataset> --f32-policy none',
    );
  });

  it('describes end-to-end detections without external NMS when decoding an o2o model to ONNX', async () => {
    const api = new HttpApi('http://folo.test');
    vi.spyOn(api, 'system').mockResolvedValue({ ...SYS, features: ['export_onnx_switches', 'export_dynamic_axes'] });
    hub.api = api;
    c = mount(ExportDialog, {
      target,
      props: { model: { ...model, head: 'o2o' }, open: true },
    });
    flushSync();

    const switches = await until(() => target.ownerDocument.querySelector('dialog [role="group"][aria-label="ONNX options"]'), 'ONNX options');
    const text = switches.textContent ?? '';
    expect(text).toContain('end-to-end detections [batch, K, 6] (up to 300 per image)');
    expect(text).toContain('no external NMS');
    expect(text).not.toContain('NMS still runs outside');
  });
});
