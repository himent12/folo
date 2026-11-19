// @vitest-environment happy-dom
// Run › Config with `Run.resolved` (docs/API.md): what the run trains with, the request beside it where it differs.
import { flush as flushSync } from 'fw';
import { render, type Rendered } from '../../../framework/testing';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveTrain } from '$lib/api/mock/runs';
import type { ResolvedTrain, TrainConfig } from '$lib/api/types';
import ConfigPanel, { type ConfigPanelProps } from './ConfigPanel';

/** Port of `mount(ConfigPanel, { target, props })`; `unmount(c)` is `c.unmount()`. */
const mount = (_: typeof ConfigPanel, o: { target: HTMLElement; props: Record<string, unknown> }) => render(() => <ConfigPanel {...(o.props as unknown as ConfigPanelProps)} />, o.target);
const unmount = (c: Rendered) => c.unmount();

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
let c: Rendered | undefined;
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
  // Keyed by the label alone (its first text), not the explanation under it.
  return Object.fromEntries(dts.map((dt) => [dt.firstChild?.textContent?.trim(), dt.nextElementSibling as HTMLElement]));
}

describe('ConfigPanel', () => {
  // Sweep 4 #330: a seed over 2^53 − 1 (a run made over the API or the CLI) arrives rounded by JSON parsing; the panel
  // showed the rounded number as the run's seed.
  it('shows a seed as it is, and one too large to read exactly as the bound it is over', () => {
    expect(rows(null, { ...config, seed: 4294967296 })['Seed']?.textContent).toBe('4294967296');
    expect(rows(null, { ...config, seed: 2 ** 53 })['Seed']?.textContent).toBe('Over 9,007,199,254,740,991');
  });

  it('shows the resolved optimizer, LR, accumulation and warmup as the truth', () => {
    // fasteners: 1920 images at batch 16 over 100 epochs; auto (rule 5) is MuSGD at 0.01 for every run.
    const r = rows(resolveTrain(config, 1920, 8, 120));
    expect(r['Optimizer']?.textContent).toBe('MuSGD (auto)');
    expect(r['Initial LR']?.textContent).toBe('0.01');
    // A batch and an optimizer step are two rows, never "per step" for a batch.
    expect(r['Batch']?.textContent).toBe('16');
    expect(r['Effective batch']?.textContent).toBe('64 (4 × 16)');
    expect(r['Warmup']?.textContent).toBe('360 optimizer steps (3 epochs)');
    expect(r['LR decay']?.textContent).toBe('Linear to 1%');
    expect(r['Momentum']?.textContent).toBe('0.937');
    expect(r['Weight decay']?.textContent).toBe('0.0005');
  });

  it('sets the architecture and the device in sans: names, not identifiers (design/README.md, Type)', () => {
    const r = rows(null, { ...config, weights: 'D:/runs/shapes/best.folo' });
    expect(r['Model']?.textContent).toBe('folo-s');
    expect(r['Device']?.textContent).toBe('gpu:0');
    const mono = (k: string) => !!r[k]?.querySelector('.mono');
    expect([mono('Model'), mono('Device'), mono('Dataset')]).toEqual([false, false, false]);
  });

  it('keeps the request as today without a resolution', () => {
    const r = rows(null);
    // One short value; what Auto picks is a line under the label, never a long value wrapping in the value column.
    expect(r['Optimizer']?.textContent).toBe('Auto');
    expect(r['Optimizer']?.previousElementSibling?.querySelector('.sub')).not.toBeNull();
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

  it('shows the server’s command on one copyable line in every layout, and none where the server has none', async () => {
    for (const layout of [{}, { compact: true }, { essential: true }]) {
      if (c) {
        unmount(c);
        target.remove();
      }
      mountWith({ cli: line, ...layout });
      expect(target.querySelector('.cli pre.line')?.textContent).toBe(line);
      expect(target.querySelector('.cli pre.line')?.getAttribute('aria-label')).toBe('PowerShell command');
    }
    // Copy gives `cli` exactly, quotes and all.
    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    target.querySelector<HTMLButtonElement>('.cli button[aria-label="Copy command"]')!.click();
    await Promise.resolve();
    expect(write).toHaveBeenCalledWith(line);
    unmount(c!);
    target.remove();
    mountWith({ cli: null });
    expect(target.querySelector('.cli')).toBeNull();
  });

  it("puts the run's id and Copy command in the header (design/UNIFIED.md §4 Run), Copy command where the server has one", async () => {
    mountWith({ cli: line, runId: 'run_7d1e' });
    const header = target.querySelector('#config header')!;
    expect(header.querySelector('.run-id')?.textContent).toBe('run_7d1e');
    const write = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText: write }, configurable: true });
    const copy = [...header.querySelectorAll<HTMLButtonElement>('button')].find((b) => (b.getAttribute('aria-label') ?? b.textContent?.trim()) === 'Copy command')!;
    copy.click();
    await Promise.resolve();
    expect(write).toHaveBeenCalledWith(line);
    unmount(c!);
    target.remove();
    mountWith({ cli: null, runId: 'run_7d1e' });
    const labels = [...target.querySelectorAll('#config header button')].map((b) => b.getAttribute('aria-label') ?? b.textContent?.trim());
    expect(labels).toEqual(['Copy JSON']);
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
      keys: [...dl.querySelectorAll('dt')].map((dt) => dt.firstChild?.textContent?.trim()),
    }));
  }

  it('shows a phone the six settings that say what will train, the rest folded', () => {
    const lists = mountWith({ essential: true, resolved: null });
    expect(lists[0]).toEqual({ fold: null, group: null, keys: ['Dataset', 'Model', 'Epochs', 'Batch', 'Image size', 'Device'] });
    expect(lists.slice(1).map((l) => l.fold)).toEqual(['More training settings', 'Augmentation']);
    expect(target.querySelectorAll('details[open]')).toHaveLength(0);
  });
});
