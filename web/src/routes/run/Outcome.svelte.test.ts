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

function render(run: Run, onresume = vi.fn(), warnings: string[] = []) {
  const target = document.createElement('div');
  document.body.appendChild(target);
  const c = mount(Outcome, { target, props: { run, elapsed: 960, onresume, warnings } });
  flushSync();
  const inside = (sel: string) => [...target.querySelectorAll<HTMLElement>('a, button')].filter((b) => b.closest(sel));
  const buttons = inside('.c-acts');
  return {
    target,
    buttons,
    labels: buttons.map((b) => b.textContent?.trim()),
    /** The actions behind the disclosure. */
    more: () => inside('.c-detail').map((b) => b.textContent?.trim()),
    done: () => {
      unmount(c);
      target.remove();
    },
  };
}

const OOM = 'the GPU ran out of memory (wgpu backend: Out of Memory)';

describe('Outcome', () => {
  it('says plainly that a run ran out of GPU memory: one line, the likeliest fix, the rest behind More fixes', () => {
    const r = render(failed(OOM));
    expect(r.target.querySelector('.c-title')?.textContent).toBe('Ran out of GPU memory');
    expect(r.target.querySelector('.c-msg')?.textContent).toContain('Batch 32 at 960 px');
    // One action (half the batch, primary) and the disclosure; each fix opens New run from this run with it applied.
    expect(r.labels).toEqual(['Retry with batch 16', 'More fixes']);
    expect(r.buttons[0]?.className).toContain('primary');
    expect(r.buttons[0]?.getAttribute('href')).toBe('#/runs/new?from=run_a2f0&batch=16');
    expect(r.more()).toEqual(['Retry at 640 px', 'Retry with folo-s']);
    const detail = r.target.querySelector<HTMLElement>('.c-detail')!;
    expect(detail.hidden).toBe(true);
    r.buttons[1]!.click();
    flushSync();
    expect(detail.hidden).toBe(false);
    expect(r.buttons[1]!.getAttribute('aria-expanded')).toBe('true');
    expect(detail.querySelector('a')?.getAttribute('href')).toBe('#/runs/new?from=run_a2f0&imgsz=640');
    // No Resume: the same settings would run out again. The driver's report is behind the disclosure.
    expect(r.labels).not.toContain('Resume');
    expect(r.target.querySelector('.c-detail code')?.textContent).toBe('wgpu backend: Out of Memory');
    expect(r.target.querySelector('.c-msg')?.textContent).not.toContain('wgpu');
    r.done();
  });

  it('leads with a smaller image size once the batch is already small', () => {
    const r = render(failed(OOM, { batch: 2 }));
    expect(r.labels).toEqual(['Retry at 640 px', 'More fixes']);
    expect(r.more()).toEqual(['Retry with folo-s', 'Retry with batch 1']);
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
    expect(r.target.querySelector('.c-msg')?.textContent?.trim()).toBe('dataset folder D:/data/shapes-hard is gone');
    expect(r.labels).toEqual(['Resume', 'More fixes']);
    expect(r.more()).toEqual(['New run like this']);
    r.buttons[0]?.click();
    expect(onresume).toHaveBeenCalledOnce();
    r.done();
  });

  it('says a too-short run in one line with the epochs that clear it, the reasons in plain words behind Why?', () => {
    const SHORT =
      'this run takes only 125 optimizer steps (5 epochs × 25: 800 training images, batch 32); a detector trained from scratch usually needs a few hundred steps to learn anything, so expect an mAP near 0. Train for at least 8 epochs, add images, or start from pretrained weights';
    const WARMUP = "warmup takes 75 of the run's 125 optimizer steps, so the learning rate reaches lr0 only in the second half of the run; lower warmup_epochs (now 3) or train for more epochs";
    const run = { ...failed(''), status: 'completed' as const, error: null, epochs: 5, config: { ...config, epochs: 5 } };
    const r = render(run, vi.fn(), [SHORT, WARMUP]);
    const note = r.target.querySelector<HTMLElement>('[aria-label="Schedule warning"]')!;
    expect(note.querySelector('.c-title')?.textContent).toBe('Too short to learn:');
    expect(note.querySelector('.c-msg')?.textContent?.trim()).toBe('125 optimizer steps');
    const acts = [...note.querySelectorAll<HTMLElement>('.c-acts a, .c-acts button')];
    expect(acts.map((b) => b.textContent?.trim())).toEqual(['Train 8 epochs', 'Why?']);
    expect(acts[0]?.getAttribute('href')).toBe('#/runs/new?from=run_a2f0&epochs=8');
    const why = note.querySelector('.c-detail')?.textContent ?? '';
    expect(why).toContain('reaches its initial value');
    expect(why).not.toMatch(/lr0|warmup_epochs/);
    r.done();
  });
});
