// @vitest-environment happy-dom
// New run CLI command preview: GPU VRAM auto-batch preview (`--set auto_batch=true`) without mutating POST /api/runs payloads.
import { flush as flushSync, signal, tick } from 'fw';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { hub } from '$lib/api/index';
import { MockApi } from '$lib/api/mock';
import { prefs } from '$lib/prefs';
import { parse, router } from '$lib/router';
import { splitCliFlags, withAutoBatch } from '$lib/trainCli';
import { toast } from '$lib/ui/toast';
import { render, type Rendered } from '../../../framework/testing';
import RunNew from '../RunNew';
import CliCommand from './CliCommand';

let target: HTMLElement;
let mounted: Rendered | undefined;

beforeEach(() => {
  sessionStorage.clear();
  toast.items = [];
  target = document.createElement('div');
  document.body.appendChild(target);
  Element.prototype.scrollIntoView ??= () => {};
});

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  target?.remove();
  vi.restoreAllMocks();
  toast.items = [];
});

describe('CliCommand auto-batch preview', () => {
  it('appends --set auto_batch=true cleanly and idempotently in withAutoBatch', () => {
    const baseParts = ['folo train', '--data D:/shapes', '--batch 16'];
    const baseLine = baseParts.join(' ');
    const first = withAutoBatch(baseParts, baseLine);
    expect(first.line).toBe('folo train --data D:/shapes --batch 16 --set auto_batch=true');
    expect(first.parts.at(-1)).toBe('--set auto_batch=true');
    // Calling twice never duplicates the flag
    const second = withAutoBatch(first.parts, first.line);
    expect(second.line).toBe(first.line);
    expect(second.parts).toEqual(first.parts);

    // Flag text inside a quoted dataset path must NOT trick withAutoBatch into omitting the real override
    const trickyLine = "folo train --data 'D:/datasets --set auto_batch=true backup' --batch 16";
    const trickyParts = splitCliFlags(trickyLine);
    expect(trickyParts).toEqual(['folo train', "--data 'D:/datasets --set auto_batch=true backup'", '--batch 16']);
    const withOverride = withAutoBatch(trickyParts, trickyLine);
    expect(withOverride.line).toBe(`${trickyLine} --set auto_batch=true`);
    expect(withOverride.parts.at(-1)).toBe('--set auto_batch=true');
  });

  it('toggles --set auto_batch=true in preview & clipboard copy and disables on CPU', async () => {
    const onCpu = signal(false);
    const baseParts = ['folo train', '--data D:/shapes', '--model folo-s', '--epochs 100', '--batch 16', '--imgsz 640'];
    const baseLine = baseParts.join(' ');

    mounted = render(
      () => <CliCommand cliParts={baseParts} cli={baseLine} cliPlaceholder={null} onCpu={onCpu.get()} />,
      target,
    );
    flushSync();

    const toggle = target.querySelector<HTMLInputElement>('[data-auto-batch-toggle]')!;
    expect(toggle.checked).toBe(false);
    expect(toggle.disabled).toBe(false);
    expect(target.querySelector('.cli code')?.textContent).not.toContain('--set auto_batch=true');

    toggle.checked = true;
    toggle.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelector('.cli code')?.textContent).toContain('--set auto_batch=true');
    expect(target.querySelector('[data-auto-batch-note]')?.textContent).toContain('CLI-only');

    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    target.querySelector<HTMLButtonElement>('button[aria-label="Copy command"]')!.click();
    await Promise.resolve();
    expect(write).toHaveBeenCalledWith(`${baseLine} --set auto_batch=true`);

    // Switching to CPU disables the toggle and omits --set auto_batch=true
    onCpu.set(true);
    flushSync();
    expect(toggle.disabled).toBe(true);
    expect(toggle.checked).toBe(false);
    expect(target.querySelector('.cli code')?.textContent).not.toContain('--set auto_batch=true');
  });

  it('never alters POST /api/runs TrainConfig payload when Auto-batch VRAM CLI preview is checked in RunNew', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/runs/new');

    mounted = render(() => <RunNew />, target);
    await vi.waitFor(() => {
      flushSync();
      const t = target.querySelector<HTMLInputElement>('[data-auto-batch-toggle]');
      const hasGpu = !!target.querySelector('select#device option[value="gpu:0"]');
      const hasDataset = !!target.querySelector('input[name="dataset"]:checked');
      if (!t || t.disabled || !hasGpu || !hasDataset) {
        throw new Error('waiting for RunNew GPU and dataset');
      }
    });

    const toggle = target.querySelector<HTMLInputElement>('[data-auto-batch-toggle]')!;
    toggle.checked = true;
    toggle.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();
    expect(target.querySelector('.cli code')?.textContent).toContain('--set auto_batch=true');

    const create = vi.spyOn(api, 'createRun');
    target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    await tick();
    expect(create).toHaveBeenCalledOnce();
    const payload = create.mock.calls[0]![0];
    expect(typeof payload.batch).toBe('number');
    expect(payload.batch).toBe(16);
    expect('auto_batch' in payload).toBe(false);
  });

  it('syncs cloud --cloud/--offer/--max-preempt-retries CLI preview, disables Auto-batch VRAM, and sends max_preempt_retries only to createRun', async () => {
    const api = new MockApi('normal', 0);
    api.imageUrl = async () => 'data:,';
    hub.api = api;
    hub.mode = 'mock';
    router.route = parse('#/runs/new');

    const estimateSpy = vi.spyOn(api, 'estimateRun');
    const createSpy = vi.spyOn(api, 'createRun');

    mounted = render(() => <RunNew />, target);
    await vi.waitFor(() => {
      flushSync();
      const runon = target.querySelector<HTMLSelectElement>('select#runon');
      const hasCloudOpt = !!runon?.querySelector('option[value="cc_4d2a"]');
      const hasDataset = !!target.querySelector('input[name="dataset"]:checked');
      if (!hasCloudOpt || !hasDataset) throw new Error('waiting for RunNew cloud connection');
    });

    const runon = target.querySelector<HTMLSelectElement>('select#runon')!;
    runon.value = 'cc_4d2a';
    runon.dispatchEvent(new Event('change', { bubbles: true }));
    flushSync();

    await vi.waitFor(() => {
      flushSync();
      const preempt = target.querySelector<HTMLInputElement>('input#preempt');
      const code = target.querySelector('.cli code')?.textContent ?? '';
      if (!preempt || !code.includes('--cloud cc_4d2a')) throw new Error('waiting for cloud offer & spot auto-resume field');
    });

    // Auto-batch VRAM is disabled on cloud runs (`--set` conflicts with `--cloud`).
    const autoBatchToggle = target.querySelector<HTMLInputElement>('[data-auto-batch-toggle]')!;
    expect(autoBatchToggle.disabled).toBe(true);
    expect(target.querySelector('.cli code')?.textContent).not.toContain('--set auto_batch=true');
    expect(target.querySelector('.cli code')?.textContent).not.toContain('--max-preempt-retries');

    // Invalid retry budget (e.g. blank or 21) holds the CLI command instead of silently dropping --max-preempt-retries.
    const preemptInput = target.querySelector<HTMLInputElement>('input#preempt')!;
    preemptInput.value = '21';
    preemptInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('.cli code')).toBeNull();
    expect(target.querySelector('.cli .cli-note')?.textContent).toMatch(/0.*20/);

    // Invalid cloud time limit (e.g. 0 hours) also holds the CLI command instead of emitting `--max-hours 0`.
    preemptInput.value = '3';
    preemptInput.dispatchEvent(new Event('input', { bubbles: true }));
    const hoursInput = target.querySelector<HTMLInputElement>('input#hours')!;
    hoursInput.value = '0';
    hoursInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
    expect(target.querySelector('.cli code')).toBeNull();
    hoursInput.value = '12';
    hoursInput.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();

    // Explicit serverUrl is forwarded via --server (with FOLO_TOKEN hint), and cloud --data passes dataset ID.
    const prevServer = prefs.serverUrl;
    prefs.serverUrl = 'http://gpu-box:7878';
    try {
      // Change Spot auto-resume to 0 (disable automatic preemption recovery).
      preemptInput.value = '0';
      preemptInput.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();
      const cloudCode = target.querySelector('.cli code')?.textContent ?? '';
      expect(cloudCode).toContain('--max-preempt-retries 0');
      expect(cloudCode).toContain('--server http://gpu-box:7878');
      expect(cloudCode).toMatch(/--data ds_\w+/);
      expect(target.querySelector('.cli')?.textContent).toContain('FOLO_TOKEN');

      // Submit: createRun receives max_preempt_retries: 0, while estimateRun never receives it.
      target.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      await tick();
      expect(createSpy).toHaveBeenCalledOnce();
      expect(createSpy.mock.calls[0]![0].compute?.connection_id).toBe('cc_4d2a');
      expect(createSpy.mock.calls[0]![1]).toEqual({ max_preempt_retries: 0 });
      for (const call of estimateSpy.mock.calls) {
        expect('max_preempt_retries' in (call[0] as unknown as Record<string, unknown>)).toBe(false);
      }
    } finally {
      prefs.serverUrl = prevServer;
    }
  });
});
