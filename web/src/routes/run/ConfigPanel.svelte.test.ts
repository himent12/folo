// @vitest-environment happy-dom
// Run › Config with `Run.resolved` (docs/API.md): what the run trains with, the request beside it where it differs.
import { flushSync, mount, unmount } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
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
  target.remove();
});

function rows(resolved: ResolvedTrain | null) {
  target = document.createElement('div');
  document.body.appendChild(target);
  c = mount(ConfigPanel, { target, props: { config, resolved, compact: true } });
  flushSync();
  const dts = [...target.querySelectorAll('dt')];
  return Object.fromEntries(dts.map((dt) => [dt.textContent?.trim(), dt.nextElementSibling as HTMLElement]));
}

describe('ConfigPanel', () => {
  it('shows the resolved optimizer, LR, accumulation and warmup as the truth', () => {
    // fasteners: 1920 images at batch 16 over 100 epochs is 3000 iterations: AdamW at 0.002 · 5 / (4 + 8).
    const r = rows(resolveTrain(config, 1920, 8, 120));
    expect(r['Optimizer']?.textContent).toBe('AdamW(auto)');
    expect(r['Optimizer']?.querySelector('.mono')?.textContent).toBe('AdamW');
    expect(r['Initial LR']?.textContent).toBe('0.000833(requested 0.01)');
    expect(r['Batch × accumulate']?.textContent).toBe('16 × 4 = 64');
    expect(r['Warmup']?.textContent).toBe('360 steps(3 epochs)');
    expect(r['LR decay']?.textContent).toBe('Linear to 1%');
    expect(r['Betas']?.textContent).toBe('0.9 · 0.999');
    expect(r['Weight decay']?.textContent).toBe('0.0005');
  });

  it('keeps the request as today without a resolution', () => {
    const r = rows(null);
    expect(r['Optimizer']?.textContent).toBe('auto');
    expect(r['Optimizer']?.querySelector('.mono')).toBeNull();
    expect(r['Initial LR']?.textContent).toBe('0.01');
    expect(r['Batch']?.textContent).toBe('16');
    expect(r['Warmup']).toBeUndefined();
  });
});
