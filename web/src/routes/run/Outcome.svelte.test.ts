// @vitest-environment happy-dom
import { flushSync, mount, unmount } from 'svelte';
import { describe, expect, it, vi } from 'vitest';
import type { Run, TrainConfig } from '$lib/api/types';
import Outcome from './Outcome.svelte';

const config: TrainConfig = {
  name: 'shapes-hard-m',
  dataset_id: 'ds_shapes_hard',
  arch: 'folo-m',
  epochs: 100,
  batch: 32,
  imgsz: 960,
  device: 'gpu:0',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
};

const failed = (error: string, c: Partial<TrainConfig> = {}): Run => ({
  id: 'run_a2f0',
  name: 'shapes-hard-m',
  status: 'failed',
  config: { ...config, ...c },
  created: '2026-11-07T06:00:00Z',
  started: '2026-11-07T06:00:03Z',
  finished: '2026-11-07T06:16:00Z',
  epoch: 23,
  epochs: 100,
  step: 2875,
  steps_per_epoch: 125,
  eta_s: null,
  best: { epoch: 22, map50_95: 0.41 },
  device: 'NVIDIA GeForce RTX 4080 SUPER (vulkan)',
  model_id: null,
  error,
});

function render(run: Run, onresume = vi.fn()) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(Outcome, { target, props: { run, elapsed: 960, onresume } });
  flushSync();
  const buttons = [...target.querySelectorAll<HTMLElement>('a, button')].filter((b) => b.closest('.c-acts'));
  return {
    target,
    buttons,
    labels: buttons.map((b) => b.textContent?.trim()),
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

const OOM = 'the GPU ran out of memory (wgpu backend: Out of Memory)';

describe('Outcome', () => {
  it('says plainly that a run ran out of GPU memory and offers the smaller settings, half the batch first', () => {
    const r = render(failed(OOM));
    expect(r.target.querySelector('.c-title')?.textContent).toBe('Ran out of GPU memory');
    expect(r.target.querySelector('.c-msg')?.textContent).toContain('batch 32 at 960 px');
    expect(r.labels).toEqual(['Retry with batch 16', 'Retry at 640 px', 'Retry with folo-s']);
    // The likeliest fix is the primary action; each opens New run from this run with the fix applied.
    expect(r.buttons[0]?.className).toContain('primary');
    expect(r.buttons[1]?.className).not.toContain('primary');
    expect(r.buttons[0]?.getAttribute('href')).toBe('#/runs/new?from=run_a2f0&batch=16');
    expect(r.buttons[2]?.getAttribute('href')).toBe('#/runs/new?from=run_a2f0&arch=folo-s');
    // No Resume: the same settings would run out again. The driver's report sits behind a disclosure.
    expect(r.labels).not.toContain('Resume');
    expect(r.target.querySelector('.c-detail code')?.textContent).toBe('wgpu backend: Out of Memory');
    expect(r.target.querySelector('.c-msg')?.textContent).not.toContain('wgpu');
    r.done();
  });

  it('leads with a smaller image size once the batch is already small', () => {
    const r = render(failed(OOM, { batch: 2 }));
    expect(r.labels).toEqual(['Retry at 640 px', 'Retry with folo-s', 'Retry with batch 1']);
    expect(r.buttons[0]?.getAttribute('href')).toBe('#/runs/new?from=run_a2f0&imgsz=640');
    r.done();
  });

  it('recognises the older out-of-memory wording', () => {
    const r = render(failed('gpu:0 out of memory: tried to allocate 2.25 GiB (15.21 GiB of 15.99 GiB in use)'));
    expect(r.target.querySelector('.c-title')?.textContent).toBe('Ran out of GPU memory');
    expect(r.target.querySelector('.c-detail code')?.textContent).toContain('tried to allocate 2.25 GiB');
    r.done();
  });

  it('keeps any other failure in full with Resume', () => {
    const onresume = vi.fn();
    const r = render(failed('dataset folder D:/data/shapes-hard is gone'), onresume);
    expect(r.target.querySelector('.c-title')?.textContent).toBe('Failed at epoch 23');
    expect(r.target.querySelector('.c-msg')?.textContent).toBe('dataset folder D:/data/shapes-hard is gone');
    expect(r.labels).toEqual(['Resume', 'New run like this']);
    r.buttons[0]?.click();
    expect(onresume).toHaveBeenCalledOnce();
    r.done();
  });
});
