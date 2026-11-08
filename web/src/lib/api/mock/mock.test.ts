import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { image, specs, stats, total } from './datasets';
import { MockApi } from './index';
import { epochAt, progressAt, stepAt } from './runs';

const NOW = Date.parse('2026-11-08T14:30:00Z');

describe('mock datasets', () => {
  it('generates the same image for the same index', () => {
    const s = specs[0]!;
    expect(image(s, 42)).toEqual(image(s, 42));
    expect(image(s, 42)).not.toEqual(image(s, 43));
  });

  it('keeps boxes inside the image', () => {
    for (const s of specs) {
      for (let i = 0; i < 200; i++) {
        for (const b of image(s, i).boxes) {
          expect(b.x1).toBeGreaterThanOrEqual(0);
          expect(b.y1).toBeGreaterThanOrEqual(0);
          expect(b.x2).toBeLessThanOrEqual(s.width);
          expect(b.y2).toBeLessThanOrEqual(s.height);
          expect(b.x2).toBeGreaterThan(b.x1);
          expect(b.class).toBeLessThan(s.classes.length);
        }
      }
    }
  });

  it('computes stats that agree with the images', () => {
    const s = specs[1]!;
    const st = stats(s);
    expect(st.class_counts.reduce((a, b) => a + b, 0)).toBe(st.boxes);
    expect(st.boxes_per_image_hist.counts.reduce((a, b) => a + b, 0)).toBe(total(s));
    expect(st.box_size_hist.edges.length).toBe(st.box_size_hist.counts.length + 1);
  });
});

describe('mock runs', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => vi.useRealTimers());

  it('lists runs newest first with one live run part way through', async () => {
    const api = new MockApi('normal', 0);
    const runs = await api.runs();
    expect(runs.map((r) => r.created)).toEqual([...runs.map((r) => r.created)].sort().reverse());
    const live = runs.filter((r) => r.status === 'running');
    expect(live).toHaveLength(1);
    expect(live[0]!.epoch).toBeGreaterThan(20);
    expect(live[0]!.epoch).toBeLessThan(60);
  });

  it('serves a metric history consistent with the run', async () => {
    const api = new MockApi('normal', 0);
    const run = await api.run('run_7d1e');
    const m = await api.runMetrics('run_7d1e');
    expect(m.epochs).toHaveLength(run.epoch - 1);
    expect(m.steps.at(-1)!.step).toBeLessThanOrEqual(run.step);
    expect(m.epochs.at(-1)!.val.map50_95).toBeGreaterThan(m.epochs[0]!.val.map50_95);
  });

  it('streams steps and epochs as time passes', async () => {
    const api = new MockApi('normal', 0);
    const steps: number[] = [];
    const epochs: number[] = [];
    const off = api.runEvents('run_7d1e', { step: (s) => steps.push(s.step), epoch: (e) => epochs.push(e.epoch) });
    vi.advanceTimersByTime(7000);
    off();
    expect(steps.length).toBeGreaterThan(50);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
    expect(epochs.length).toBeGreaterThanOrEqual(1);
  });

  it('stops a running run', async () => {
    const api = new MockApi('normal', 0);
    const r = await api.stopRun('run_7d1e');
    expect(r.status).toBe('stopping');
    vi.advanceTimersByTime(2000);
    expect((await api.run('run_7d1e')).status).toBe('stopped');
  });

  it('fails every call in the error scenario', async () => {
    const api = new MockApi('error', 0);
    await expect(api.runs()).rejects.toMatchObject({ code: 'server_unavailable' });
  });

  it('produces plausible curves', () => {
    const api = new MockApi('normal', 0);
    void api;
    const prof = {
      id: 'x',
      name: 'x',
      config: {
        name: 'x', dataset_id: 'ds_shapes', arch: 'folo-n' as const, epochs: 50, batch: 16, imgsz: 640, device: 'auto' as const,
        optimizer: 'auto' as const, lr0: 0.01, weights: null, seed: 0,
        augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4] as [number, number, number], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
      },
      classes: 3, classEase: [1, 1, 1], stepsPerEpoch: 10, epochSeconds: 1, mapMax: 0.7, tau: 8, imgsPerS: 300,
      createdAt: NOW, startedAt: NOW, device: 'cpu',
    };
    expect(epochAt(prof, 49).val.map50_95).toBeGreaterThan(epochAt(prof, 2).val.map50_95);
    expect(stepAt(prof, 499).loss.total).toBeLessThan(stepAt(prof, 5).loss.total);
    expect(progressAt(prof, NOW + 25_000).status).toBe('running');
    expect(progressAt(prof, NOW + 60_000).status).toBe('completed');
  });
});
