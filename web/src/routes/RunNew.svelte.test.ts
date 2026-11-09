// @vitest-environment happy-dom
import { flushSync, mount, tick, unmount } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from 'vitest';
import { ApiError, hub } from '$lib/api/index.svelte';
import { MockApi } from '$lib/api/mock';
import type { RunEstimate } from '$lib/api/types';
import { parse, router } from '$lib/router.svelte';
import { page } from '$lib/shell/crumbs.svelte';
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
    expect(target.querySelector('#dock-why')?.textContent).toBe('Device: GPU 1 (gpu:1) is not available.');
    // In the dock the reason takes the estimate's place (one row of chrome), and it is an ink link behind a danger mark.
    expect(target.querySelector('.dock .dock-est')).toBeNull();
    for (const id of ['#start-why', '#dock-why']) {
      const why = target.querySelector(id)!;
      expect(why.firstElementChild?.tagName.toLowerCase()).toBe('svg');
      expect(why.querySelector('button.why-go')).not.toBeNull();
    }
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
    // One condition, one severity: the meter and the line are danger; the figure, a metric, stays in ink.
    expect(target.querySelector('#rail-nofit')?.className).toContain('danger');
    expect(target.querySelector('.summary .vram i')?.className).toContain('danger');
    const fig = [...target.querySelectorAll('.summary .est dd')].find((d) => /GiB/.test(d.textContent ?? ''))!;
    expect(fig.textContent).toMatch(/~26\.\d GiB/);
    expect(fig.className).not.toMatch(/danger|warn/);
    // The dock (narrow screens) carries the figure against the GPU's, the verdict and the fix as its one button.
    const dock = target.querySelector('.dock')!;
    expect(dock.querySelector('.dock-est')?.textContent?.replace(/\s+/g, ' ')).toMatch(/~26\.\d GiB \/ 16\.0 GiB.*Likely won't fit/);
    expect([...dock.querySelectorAll('.btn')].map((b) => b.textContent?.trim())).toEqual(['Use batch 8']);
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
  it('keeps Start the primary when the settings fit', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    await until(() => target.querySelector('.summary .vram i'), 'the meter');
    expect(target.querySelector('#rail-nofit')).toBeNull();
    expect(target.querySelector('.start button[type="submit"]')?.className).toContain('primary');
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
    // One line, a callout without a fill; the reasons and the fix behind "Why?".
    expect(warn.querySelector('p.callout')?.textContent?.replace(/\s+/g, ' ').trim()).toBe('Too short to learn: 150 steps. Why?');
    expect(warn.querySelector('button.fix')).toBeNull();
    expect(warn.querySelector('.short-why')).toBeNull();
    const more = warn.querySelector<HTMLButtonElement>('button.more')!;
    expect(more.textContent).toBe('Why?');
    more.click();
    flushSync();
    expect(more.getAttribute('aria-expanded')).toBe('true');
    const why = warn.querySelector(`#${more.getAttribute('aria-controls')}`)!;
    const text = why.textContent!.replace(/[ \t\n]+/g, ' ');
    expect(text).toContain('5 epochs × 30 steps. From scratch, a detector needs a few hundred');
    // "0" never ends up alone on a line, and the prose names no config keys.
    expect(text).toContain('expect an mAP near 0.');
    expect(text).toContain('Warmup takes 90 of the 150 steps');
    expect(text).not.toMatch(/lr0|warmup_epochs/);
    expect(target.querySelector('.dock-warn')?.textContent).toBe('Too short to learn');
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
    expect(warn.querySelector('p.callout span')?.textContent?.trim()).toBe('Mostly warmup: 90 of 150 steps.');
    warn.querySelector<HTMLButtonElement>('button.more')!.click();
    flushSync();
    expect(warn.textContent).not.toContain('Too short');
    expect(warn.textContent).toContain('Warmup takes 90 of the 150 steps');
    expect(target.querySelector('.dock-warn')?.textContent).toBe('Mostly warmup');
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
    expect(target.querySelector('#close-hint')?.textContent).toBe('At most half the run: mosaic closes for the last 4 epochs.');
    type('epochs', 100);
    expect(target.querySelector('#close-hint')?.textContent).toBe('Train the last N epochs without mosaic to settle the boxes.');
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
    expect(steps.textContent).toBe('30 steps / epoch. How the steps are counted');
    expect(steps.getAttribute('aria-expanded')).toBe('false');
    const why = () => document.getElementById(steps.getAttribute('aria-controls')!)?.textContent?.trim();
    steps.click();
    flushSync();
    expect(steps.getAttribute('aria-expanded')).toBe('true');
    // A line under the row, in the list itself: a popover over the rail would cover its neighbours (round 24).
    const line = document.getElementById(steps.getAttribute('aria-controls')!)!;
    expect(line.tagName).toBe('DD');
    expect(est.contains(line)).toBe(true);
    expect(why()).toBe('120 batches of 16; 4 per optimizer step.');
    // Batch 64 is one batch a step.
    target.querySelector<HTMLButtonElement>('#batch button[role="radio"]:nth-child(4)')!.click();
    flushSync();
    expect(why()).toBe('30 batches of 64, one step each.');
  });

  it("matches the short-run scenario's run: shapes at batch 32 for 5 epochs suggests its 16 epochs", async () => {
    await open('?dataset=ds_shapes');
    target.querySelector<HTMLButtonElement>('#batch button[role="radio"]:nth-child(3)')!.click();
    const el = target.querySelector<HTMLInputElement>('input#epochs')!;
    el.value = '5';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    const warn = await until(() => target.querySelector('.summary .short'));
    expect(warn.querySelector('p.callout span')?.textContent?.trim()).toBe('Too short to learn: 65 steps.');
    warn.querySelector<HTMLButtonElement>('button.more')!.click();
    flushSync();
    expect(warn.querySelector('button.fix')?.textContent).toBe('Train 16 epochs');
  });
});

describe('New run: mixed precision', () => {
  const sw = () => target.querySelector<HTMLInputElement>('input#amp');
  const cli = () => target.querySelector('.cli code')!.textContent!;
  const row = () => [...target.querySelectorAll('.summary .rv')].find((r) => r.querySelector('.rv-k')?.textContent === 'Mixed precision');
  const time = () => [...target.querySelectorAll('.summary .est dt')].find((d) => d.textContent === 'Est. time')!.nextElementSibling!.textContent;
  const basis = () => target.querySelector('.summary .est-how > summary')?.textContent;

  it('is off by default; on, the run, the Summary and the command say so, and the estimate is faster', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    const s = await until(sw, 'the switch');
    expect(s.checked).toBe(false);
    expect(s.disabled).toBe(false);
    expect(s.getAttribute('role')).toBe('switch');
    // A field on the Training grid: the label's line carries the switch; the hint (folo-s, measured: about 1.6×
    // faster) under it describes the switch.
    const field = s.closest('.fields > .field')!;
    expect(field.querySelector('.switch')?.textContent?.trim()).toBe('Mixed precision');
    expect(field.querySelector('#amp-hint')?.textContent).toBe('f16 on tensor cores: about 1.6× faster, 40% less memory.');
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
    // Only the control is disabled: the label stays a field label, not disabled text.
    expect(s.closest('.field')?.classList.contains('sw-field')).toBe(true);
    expect(s.getAttribute('aria-describedby')).toBe('amp-why');
    // One plain sentence and one action docs/GPU.md confirms (an RTX 4080 SUPER has cooperative matrices on Vulkan):
    // the shared remedy, the shell named, the command whole on one copyable line.
    expect(target.querySelector('#amp-why')?.textContent).toBe('DirectX 12 does not reach this GPU’s tensor cores; restart the server on Vulkan.');
    const remedy = target.querySelector('.amp .remedy')!;
    expect(remedy.querySelector('.shell')?.textContent).toBe('In PowerShell:');
    expect(remedy.querySelector('pre.line')?.textContent).toBe("$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores");
    expect(remedy.querySelector('pre.line')?.hasAttribute('data-scrolls')).toBe(true);
    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    remedy.querySelector<HTMLButtonElement>('button[aria-label="Copy command"]')!.click();
    expect(write).toHaveBeenCalledWith("$env:FOLO_GPU_BACKEND='vulkan'; folo serve --tensor-cores");
    // The server's reason, verbatim and copyable, folded behind "Driver report".
    const report = remedy.querySelector<HTMLDetailsElement>('details.report')!;
    expect(report.open).toBe(false);
    expect(report.querySelector('summary')?.textContent).toBe('Driver report');
    expect(report.querySelector('pre')?.textContent).toBe((await api.system()).capabilities!.amp.reason);
    expect(report.querySelector('pre')?.textContent).toContain('NVIDIA GeForce RTX 4080 SUPER (Dx12) has no cooperative matrices in wgpu');
    report.querySelector<HTMLButtonElement>('button[aria-label="Copy driver report"]')!.click();
    expect(write).toHaveBeenCalledWith((await api.system()).capabilities!.amp.reason);
    const create = vi.spyOn(api, 'createRun');
    await start();
    expect(create.mock.calls[0]![0].amp).toBe(false);
  });

  it('says to train in f32 where no backend gives this GPU tensor cores', async () => {
    hub.api = api = new MockApi('no-tensor-cores', 0);
    const sys = api.system.bind(api);
    api.system = async () => {
      const r = await sys();
      return { ...r, gpus: r.gpus.map((g) => ({ ...g, name: 'AMD Radeon RX 7900 XTX' })) };
    };
    await open();
    await until(sw, 'the switch');
    expect(target.querySelector('#amp-why')?.textContent).toBe('This GPU has no tensor cores FOLO can use. Train in f32 on this GPU.');
    expect(target.querySelector('#amp-why')?.textContent).not.toContain('FOLO_GPU_BACKEND');
  });

  it('treats Auto on a server without a usable GPU as the CPU, with no GPU advice', async () => {
    hub.api = api = new MockApi('no-tensor-cores', 0);
    const sys = api.system.bind(api);
    api.system = async () => {
      const r = await sys();
      return { ...r, capabilities: { ...r.capabilities!, device: 'cpu' } };
    };
    await open();
    await until(sw, 'the switch');
    expect(target.querySelector('#amp-why')?.textContent).toBe('This server has no usable GPU, so runs train on the CPU in f32.');
  });

  it('says how to turn tensor cores on when the server has them off', async () => {
    hub.api = api = new MockApi('tensor-cores-off', 0);
    await open();
    await until(sw, 'the switch');
    expect(target.querySelector('#amp-why')?.textContent).toBe('Tensor cores are off on this server; restart it with them on:');
    expect(target.querySelector('.amp .shell')).toBeNull();
    expect(target.querySelector('.amp pre.line')?.textContent).toBe('folo serve --tensor-cores');
  });

  it('is off on the CPU, and comes back on with the GPU', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    const s = await until(sw, 'the switch');
    s.click();
    flushSync();
    pick('cpu');
    expect(sw()!.disabled).toBe(true);
    expect(sw()!.checked).toBe(false);
    expect(target.querySelector('#amp-why')?.textContent).toBe('The CPU has no tensor cores. Pick a GPU to train in mixed precision.');
    expect(target.querySelector('.amp .dev-detail')).toBeNull();
    expect(cli()).not.toContain('--amp');
    pick('auto');
    expect(sw()!.checked).toBe(true);
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
    expect(s.closest('.field')?.querySelector('.switch')?.textContent?.trim()).toBe('End-to-end (no NMS)');
    // One line, with the cost measured on the shapes gate (docs/BENCHMARKS.md: 0.015 mAP50-95 behind NMS).
    expect(s.getAttribute('aria-describedby')).toBe('head-hint');
    expect(target.querySelector('#head-hint')?.textContent).toBe('No NMS step at inference; about 0.015 lower mAP50-95 than NMS on the same model.');
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

  it("names o2m in the command when fine-tuning, since the CLI's default is the weights file's head", async () => {
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
  });

  it('opens Advanced from its Summary row', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    sw()!.click();
    flushSync();
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

describe('New run: the top bar', () => {
  it('owns the create action while mounted, so the top bar steps aside', async () => {
    const before = page.creators;
    await open();
    expect(page.ownsCreate).toBe(true);
    unmount(c!);
    c = undefined;
    expect(page.creators).toBe(before);
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

  it('names a run without weights "From scratch", in the select and in the Summary', async () => {
    await open();
    expect(target.querySelector('select#weights option[value="none"]')?.textContent).toBe('From scratch');
    const rv = [...target.querySelectorAll('.summary .rv')].find((r) => r.querySelector('.rv-k')?.textContent === 'Start from');
    expect(rv?.querySelector('.rv-v')?.textContent).toBe('From scratch');
  });
});

describe('New run on a phone', () => {
  const width = window.innerWidth;
  const setWidth = (w: number) => (window as unknown as { happyDOM: { setViewport(v: { width: number }): void } }).happyDOM.setViewport({ width: w });
  afterEach(() => setWidth(width));

  it('picks the size from a segmented control and folds the full comparison behind "Compare sizes"', async () => {
    setWidth(390);
    await open();
    const pick = target.querySelector('#arch-pick')!;
    expect([...pick.querySelectorAll('button')].map((b) => b.textContent?.trim())).toEqual(['N', 'S', 'M', 'L', 'X']);
    expect(target.querySelector('.a-sel')?.textContent).toBe('folo-s The default: fast and accurate.');
    expect(target.querySelector('.a-specs')?.textContent?.replace(/\s+/g, ' ').trim()).toBe('9.46M params · 21.6 GFLOPs');
    // Latency is the architecture's reference, said so, with where it was measured under it.
    expect(target.querySelector('.a-lat')?.textContent).toBe('Reference latency 7.87 ms');
    expect(target.querySelectorAll('.a-pick .a-specs')[1]?.textContent).toBe('RTX 4080 SUPER · 640 px · f32 · end to end');
    const compare = target.querySelector<HTMLDetailsElement>('details.a-compare')!;
    expect(compare.open).toBe(false);
    expect(compare.querySelectorAll('.arch')).toHaveLength(5);
    pick.querySelectorAll('button')[2]!.click();
    flushSync();
    expect(target.querySelector('.a-sel')?.textContent).toBe('folo-m More accuracy on busy scenes.');
    expect(target.querySelector('.a-lat')?.textContent).toBe('Reference latency not measured yet');
    expect(compare.querySelector<HTMLInputElement>('input[name="arch"]:checked')?.value).toBe('folo-m');
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

describe('New run: the form', () => {
  it('ties an invalid field to its message with aria-invalid and aria-describedby', async () => {
    await open();
    const el = target.querySelector<HTMLInputElement>('input#epochs')!;
    expect(el.hasAttribute('aria-invalid')).toBe(false);
    el.value = '0';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(el.getAttribute('aria-invalid')).toBe('true');
    expect(el.getAttribute('aria-describedby')).toBe('epochs-error');
    expect(target.querySelector('#epochs-error')?.textContent).toBe('A whole number from 1 to 10,000.');
    expect(target.querySelector('.dock .dock-est')).toBeNull();
    expect(target.querySelector('#dock-why')?.textContent).toBe('Epochs: A whole number from 1 to 10,000.');
  });

  it('shows unmeasured latency as a dash and the size letters in uppercase', async () => {
    await open();
    const rows = [...target.querySelectorAll('.arch')];
    expect(rows.map((r) => r.querySelector('.a-size')?.textContent)).toEqual(['N', 'S', 'M', 'L', 'X']);
    expect(rows.slice(2).map((r) => r.querySelectorAll('.a-v')[2]?.textContent?.trim())).toEqual(['—', '—', '—']);
    // The dash is explained once, under the table, not in every row (DESIGN_DIRECTION 23).
    expect(target.querySelector('.arches')?.textContent).not.toContain('not measured');
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

  it('marks the page as having a bottom dock while the form is shown', async () => {
    await open();
    expect(document.documentElement.hasAttribute('data-dock')).toBe(true);
    unmount(c!);
    c = undefined;
    expect(document.documentElement.hasAttribute('data-dock')).toBe(false);
  });
});

describe('New run from a link', () => {
  it("takes the epochs a short run's warning suggests", async () => {
    await open('?from=run_a2f0&epochs=40');
    await until(() => target.querySelector<HTMLInputElement>('input#epochs')?.value === '40', 'epochs 40');
  });
});

describe('New run: the estimate', () => {
  const timeDd = () => [...target.querySelectorAll('.summary .est dt')].find((d) => d.textContent === 'Est. time')?.nextElementSibling;
  const time = () => timeDd()?.textContent ?? '';
  const basis = () => target.querySelector('.summary .est-how > summary')?.textContent ?? '';
  const rough = () => timeDd()?.classList.contains('rough');

  it("shows the server's estimate as a range, says it was measured on this GPU, and lists its phases", async () => {
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
    const rows = [...target.querySelectorAll('.summary .how-list dt')].map((d) => d.textContent);
    expect(rows).toEqual(['90 epochs, mosaic', '10 epochs, no mosaic', 'Validation', 'First epoch', 'Peak GPU memory']);
    expect(target.querySelector('.summary .how-note')?.textContent).toContain('these very settings');
    // The dock says the same range and calls memory GPU memory.
    expect(target.querySelector('.dock-est')?.textContent).toContain(time());
    expect(target.querySelector('.dock-est')?.textContent).toContain('GPU memory');
    expect(target.querySelector('.dock-est')?.textContent).not.toContain('VRAM');
  });

  it('asks again when the settings change, and calls a size it never measured rough', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    await until(() => basis().startsWith('Measured'), "the server's estimate");
    const radio = target.querySelector<HTMLInputElement>('input[name="arch"][value="folo-x"]')!;
    radio.click();
    flushSync();
    await until(() => basis().includes('caled'), 'the folo-x estimate');
    expect(basis()).toBe('Wide range: scaled from folo-s at 640 px');
    // folo-x is 9x folo-s's compute: low confidence, the value one step quieter.
    expect(rough()).toBe(true);
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
    // The caption under the table is one meta line, not a paragraph explaining each row.
    expect([...target.querySelectorAll('p.ref .ui-meta > span')].map((s) => s.textContent)).toEqual([
      'Latency at 640 px, end to end',
      'reference: RTX 4080 SUPER, f32',
      '—: not measured yet',
    ]);
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
    expect(target.querySelector('.summary .cli-note')?.textContent).toBe(
      'Put the path of fasteners-n-best.folo on the server in its place: this server does not report it.',
    );
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

  it("states the GPU memory estimate alone where the adapter's size is not reported (memory_bytes 0)", async () => {
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
    expect(summary.split('This GPU’s memory size isn’t reported.')).toHaveLength(2);
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

  it('lists them in a group of their own whatever the size, with what they are as the tooltip', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    expect(arch()).toBe('folo-s');
    const group = await until(() => sel().querySelector('optgroup[label="Published weights"]'), 'the published group');
    const opt = group.querySelector<HTMLOptionElement>('option[value="folo-n-coco"]')!;
    // A score never without what it is on: published weights' split is not known here.
    expect(opt.textContent).toBe('folo-n-coco · mAP 0.314 · split unknown');
    expect(opt.title).toMatch(/^folo-n trained from scratch on COCO/);
    // The run models of the chosen size are grouped by what their scores are on, outside the published group.
    const own = await until(() => sel().querySelector('option[value="m_shapes_s"]'), 'the run models');
    expect(own.parentElement?.getAttribute('label')).toBe('Validation split of shapes');
    expect(own.textContent).toBe('shapes-s-best · mAP 0.854 on shapes');
  });

  it('take their size when picked, say so, and drop out when another size is picked', async () => {
    hub.api = api = new MockApi('normal', 0);
    await open();
    await until(() => sel().querySelector('option[value="folo-n-coco"]'), 'the published weights');
    choose('folo-n-coco');
    expect(arch()).toBe('folo-n');
    expect(sel().value).toBe('folo-n-coco');
    expect(target.querySelector('.a-from')?.textContent?.replace(/\s+/g, ' ').trim()).toBe(
      "folo-n, from folo-n-coco. The run trains the weights' size; another size starts from scratch.",
    );
    expect(target.querySelector('#weights-hint')?.textContent).toMatch(/^folo-n trained from scratch on COCO/);
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
    expect(target.querySelector('#close-hint')?.textContent).toBe('At most half the run: mosaic closes for the last 3 epochs.');
  });

  it('shows a 501 for mixed precision at its field, not as a toast, with what works now', async () => {
    hub.api = api = new MockApi('amp-refused', 0);
    const { toast } = await import('$lib/ui/toast.svelte');
    const err = vi.spyOn(toast, 'error');
    await open();
    const sw = await until(() => target.querySelector<HTMLInputElement>('input#amp'), 'the switch');
    sw.click();
    flushSync();
    await start();
    const msg = await until(() => target.querySelector('#amp-error'), 'the refusal');
    expect(msg.getAttribute('role')).toBe('alert');
    expect(msg.textContent).toMatch(/^Start was refused: Mixed-precision training \(amp\) needs a GPU with tensor cores/);
    expect(err).not.toHaveBeenCalled();
    // The capabilities read again: the switch is off with the way to turn tensor cores on, and Start trains in f32.
    await until(() => target.querySelector('#amp-why'), 'the remedy');
    expect(target.querySelector('#amp-why')?.textContent).toBe('Tensor cores are off on this server; restart it with them on:');
    expect(target.querySelector<HTMLInputElement>('input#amp')!.disabled).toBe(true);
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
    expect(target.querySelector('#start-why')?.textContent).toBe('Mixed precision: Not available on this server now.');
    expect(target.querySelector('#amp-hint')?.textContent).toBe('Turn it off to train in f32.');
    sw.click();
    flushSync();
    expect(target.querySelector('#amp-error')).toBeNull();
    expect(target.querySelector('#start-why')).toBeNull();
  });
});
