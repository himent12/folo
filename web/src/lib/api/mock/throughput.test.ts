import { describe, expect, it } from 'vitest';
import { ApiError } from '../client';
import type { TrainConfig } from '../types';
import { MockApi } from './index';

const base: TrainConfig = {
  name: 'e',
  dataset_id: 'ds_shapes',
  arch: 'folo-n',
  epochs: 50,
  batch: 16,
  imgsz: 640,
  device: 'auto',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
  amp: false,
};

describe('mock throughput (docs/API.md)', () => {
  it('lists measured rows first, then the reference for every size at 640 and 320 px', async () => {
    const t = await new MockApi('normal', 0).throughput();
    const firstRef = t.training.findIndex((r) => r.source === 'reference');
    expect(t.training.slice(0, firstRef).every((r) => r.source === 'measured')).toBe(true);
    expect(t.training.slice(firstRef).every((r) => r.source === 'reference' && r.samples === 0 && r.updated === null)).toBe(true);
    // A measured row replaces the reference with the same key on this adapter.
    expect(t.training.filter((r) => r.device === 'gpu:0' && r.arch === 'folo-n' && r.imgsz === 640 && r.batch === 16 && !r.amp)).toHaveLength(1);
    for (const arch of ['folo-m', 'folo-x'] as const) expect(t.inference.some((r) => r.arch === arch && r.imgsz === 320 && r.device === 'cpu')).toBe(true);
  });

  it('estimates from the exact measured row with a narrow range, and responds to every setting', async () => {
    const api = new MockApi('normal', 0);
    const e = await api.estimateRun(base);
    expect([e.source, e.confidence, e.basis.exact]).toEqual(['measured', 'high', true]);
    expect(e.est_total_s_high / e.est_total_s).toBeLessThan(1.2);
    expect(e.phases.map((p) => [p.phase, p.epochs])).toEqual([
      ['mosaic', 40],
      ['close_mosaic', 10],
    ]);
    // est_total_s = startup + Σ epochs × epoch_s.
    expect(e.est_total_s).toBeCloseTo(e.startup_s + e.phases.reduce((s, p) => s + p.epochs * p.epoch_s, 0), 0);
    const total = (c: Partial<TrainConfig>) => api.estimateRun({ ...base, ...c }).then((x) => x.est_total_s);
    expect(await total({ epochs: 100 })).toBeGreaterThan(e.est_total_s);
    expect(await total({ arch: 'folo-l' })).toBeGreaterThan(e.est_total_s);
    expect(await total({ imgsz: 320 })).toBeLessThan(e.est_total_s);
    expect(await total({ amp: true })).not.toBe(e.est_total_s);
    // Another batch keeps the rate (the reference has no batch dependence) but not the exact basis, and moves memory.
    const b32 = await api.estimateRun({ ...base, batch: 32 });
    expect(b32.basis.exact).toBe(false);
    expect(b32.est_peak_vram_bytes!).toBeGreaterThan(e.est_peak_vram_bytes!);
  });

  it('falls back to the reference on the CPU, with the wide range of another machine', async () => {
    const e = await new MockApi('normal', 0).estimateRun({ ...base, device: 'cpu' });
    expect([e.source, e.confidence, e.est_peak_vram_bytes]).toEqual(['reference', 'low', null]);
    expect(e.est_total_s_high / e.est_total_s).toBeCloseTo(3);
  });

  it('checks the config as POST /api/runs does, and answers 404 on a server without the feature', async () => {
    await expect(new MockApi('normal', 0).estimateRun({ ...base, dataset_id: 'nope' })).rejects.toMatchObject({ code: 'dataset_not_found' });
    await expect(new MockApi('normal', 0).estimateRun({ ...base, epochs: 0 })).rejects.toMatchObject({ code: 'invalid_config' });
    await expect(new MockApi('no-throughput', 0).estimateRun(base)).rejects.toBeInstanceOf(ApiError);
    expect((await new MockApi('no-throughput', 0).system()).features).not.toContain('throughput');
    expect((await new MockApi('normal', 0).system()).features).toEqual(expect.arrayContaining(['throughput', 'run_estimate', 'benchmark']));
  });

  it('adds a probe to the table as measured samples, and refuses one while a run it created is active', async () => {
    const api = new MockApi('normal', 0);
    const r = await api.benchmark({});
    expect(r.training).toMatchObject({ arch: 'folo-n', imgsz: 320, amp: false, source: 'measured', probes: 1 });
    const t = await api.throughput();
    expect(t.training[0]).toMatchObject({ arch: 'folo-n', imgsz: 320, amp: false, probes: 1 });
    await expect(api.benchmark({ budget_s: 11 })).rejects.toMatchObject({ code: 'invalid_body' });
    await api.createRun({ ...base, name: 'busy' });
    await expect(api.benchmark({})).rejects.toMatchObject({ status: 409, code: 'run_active' });
  });
});
