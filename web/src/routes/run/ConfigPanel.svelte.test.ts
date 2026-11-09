// @vitest-environment happy-dom
// Run › Config with `Run.resolved` (docs/API.md): what the run trains with, the request beside it where it differs.
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveTrain } from '$lib/api/mock/runs';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import ConfigPanel from './ConfigPanel.svelte';

const config: TrainConfig = {
  name: 'fasteners-s',
  dataset_id: 'ds_fasteners',
  arch: 'folo-s',
  epochs: 100,
  batch: 16,
  imgsz: 640,
  device: 'gpu:0',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
};

let target: HTMLElement;
let c: ReturnType<typeof mount> | undefined;
afterEach(() => {
  if (c) unmount(c);
  c = undefined;
  target.remove();
});

function rows(resolved: ResolvedTrain | null, cfg: TrainConfig = config) {
  if (c) {
    unmount(c);
    target.remove();
  }
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(ConfigPanel, { target, props: { config: cfg, resolved, compact: true } });
  flushSync();
  const dts = [...target.querySelectorAll('dt')];
  return Object.fromEntries(dts.map((dt) => [dt.textContent?.trim(), dt.nextElementSibling as HTMLElement]));
}

describe('ConfigPanel', () => {
  it('shows the resolved optimizer, LR, accumulation and warmup as the truth', () => {
    // fasteners: 1920 images at batch 16 over 100 epochs is 3000 iterations: AdamW at 0.002 · 5 / (4 + 8).
    const r = rows(resolveTrain(config, 1920, 8, 120));
    expect(r['Optimizer']?.textContent).toBe('AdamW (auto)');
    // A named optimizer is a name, in sans (Geist Mono's struck W reads as "₩").
    expect(r['Optimizer']?.querySelector('.mono')).toBeNull();
    expect(r['Initial LR']?.textContent).toBe('0.000833 (requested 0.01)');
    // A batch and an optimizer step are two rows, never "per step" for a batch.
    expect(r['Batch']?.textContent).toBe('16');
    expect(r['Effective batch']?.textContent).toBe('64 (4 × 16)');
    expect(r['Warmup']?.textContent).toBe('360 optimizer steps (3 epochs)');
    expect(r['LR decay']?.textContent).toBe('Linear to 1%');
    expect(r['Betas']?.textContent).toBe('0.9 · 0.999');
    expect(r['Weight decay']?.textContent).toBe('0.0005');
  });

  it('keeps the request as today without a resolution', () => {
    const r = rows(null);
    expect(r['Optimizer']?.textContent).toBe('Auto (AdamW for short runs, else SGD)');
    expect(r['Optimizer']?.querySelector('.mono')).toBeNull();
    expect(r['Initial LR']?.textContent).toBe('0.01');
    expect(r['Batch']?.textContent).toBe('16');
    expect(r['Warmup']).toBeUndefined();
  });

  it('shows the capped close mosaic, the precision and the head (DOGFOOD-3 S2, S4)', () => {
    const short = { ...config, epochs: 15, amp: true, head: 'o2o' as const };
    const r = rows(resolveTrain(short, 1920, 8, 120), short);
    expect(r['Close mosaic']?.textContent).toBe('last 7 epochs (requested 10)');
    expect(r['Close mosaic']?.querySelector('.ui-note')?.textContent).toBe('(requested 10)');
    expect(r['Precision']?.textContent).toBe('Mixed precision (f16)');
    expect(r['Head']?.textContent).toBe('End-to-end (no NMS)');
    // The defaults are said too.
    const d = rows(null);
    expect(d['Close mosaic']?.textContent).toBe('last 10 epochs');
    expect(d['Precision']?.textContent).toBe('f32');
    expect(d['Head']?.textContent).toBe('NMS (one-to-many)');
  });
});

describe("ConfigPanel: the run's CLI command (servers that list run_cli)", () => {
  const line = "folo train --data D:/data/fasteners --model folo-s --name 'Bob''s run'";
  function mountWith(props: Record<string, unknown>) {
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(ConfigPanel, { target, props: { config, ...props } });
    flushSync();
  }

  it('shows the server’s command under "In PowerShell:" on one copyable line, in every layout', async () => {
    for (const layout of [{}, { compact: true }, { essential: true }]) {
      if (c) {
        unmount(c);
        target.remove();
      }
      mountWith({ cli: line, ...layout });
      expect(target.querySelector('.cli .shell')?.textContent).toBe('In PowerShell:');
      expect(target.querySelector('.cli pre.line')?.textContent).toBe(line);
      expect(target.querySelector('.cli pre.line')?.getAttribute('aria-label')).toBe('PowerShell command');
    }
    // Copy gives `cli` exactly, quotes and all.
    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    target.querySelector<HTMLButtonElement>('.cli button[aria-label="Copy command"]')!.click();
    await Promise.resolve();
    expect(write).toHaveBeenCalledWith(line);
  });

  it('shows none where the server has none', () => {
    mountWith({ cli: null });
    expect(target.querySelector('.cli')).toBeNull();
  });
});

describe('ConfigPanel: the work before the reference', () => {
  function mountWith(props: Record<string, unknown>) {
    target = document.createElement('div');
    document.body.appendChild(target);
    c = mount(ConfigPanel, { target, props: { config, resolved: resolveTrain(config, 1920, 8, 120), ...props } });
    flushSync();
    const lists = [...target.querySelectorAll('dl')];
    /** The settings of each list, the folded ones marked by their disclosure's title. */
    return lists.map((dl) => ({
      fold: dl.closest('details')?.querySelector('summary')?.textContent ?? null,
      group: dl.querySelector('.group')?.textContent ?? null,
      keys: [...dl.querySelectorAll('dt')].map((dt) => dt.textContent?.trim()),
    }));
  }

  it('shows a phone the six settings that say what will train, the rest folded', () => {
    const lists = mountWith({ essential: true, resolved: null });
    expect(lists[0]).toEqual({ fold: null, group: null, keys: ['Dataset', 'Model', 'Epochs', 'Batch', 'Image size', 'Device'] });
    expect(lists.slice(1).map((l) => l.fold)).toEqual(['More training settings', 'Augmentation']);
    expect(target.querySelectorAll('details[open]')).toHaveLength(0);
  });

  it('balances the two columns: setup on the left, the optimizer over augmentation on the right', () => {
    const lists = mountWith({});
    expect(lists.map((l) => l.group)).toEqual(['Training', 'Optimizer', 'Augmentation']);
    const left = lists[0]!.keys.length;
    const right = lists[1]!.keys.length + lists[2]!.keys.length;
    // The right column carries one heading more: within a few rows of the left, never a hollow of six.
    expect(Math.abs(left - right)).toBeLessThanOrEqual(3);
  });
});
