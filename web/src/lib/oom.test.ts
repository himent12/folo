import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import type { TrainConfig } from '$lib/api/types';
import { referenceVram } from '$lib/reference';
import { failureCause, fitFix, fitFixes, fitsGpu, isOom, oomCalibration, oomDetail, oomFixes, retryPath, smallerBatches, useLabel, vramForecaster, type VramOf } from './oom';

const cfg = (c: Partial<TrainConfig>) => ({ batch: 16, imgsz: 640, arch: 'folo-s' as const, device: 'auto' as const, ...c });

describe('isOom', () => {
  it('recognises the run error, older wordings and the API code', () => {
    expect(isOom('the GPU ran out of memory (wgpu backend: Out of Memory)')).toBe(true);
    expect(isOom('gpu:0 out of memory: tried to allocate 2.25 GiB (15.21 GiB of 15.99 GiB in use)')).toBe(true);
    expect(isOom('GPU out of memory: tried to allocate 1.19 GiB')).toBe(true);
    expect(isOom(new ApiError(507, 'gpu_out_of_memory', 'the GPU ran out of memory (x)', 'lower `batch`'))).toBe(true);
    expect(isOom(new ApiError(500, 'internal', 'wgpu backend: OutOfMemory'))).toBe(true);
  });

  it('leaves other failures alone', () => {
    expect(isOom('dataset folder is gone')).toBe(false);
    // Mentioning memory is not enough: a config error, the host running out, a training failure.
    expect(isOom('invalid config: out of memory? no')).toBe(false);
    expect(isOom(new ApiError(400, 'invalid_config', 'batch 4096 would run out of memory'))).toBe(false);
    expect(isOom('io error: failed to decode image: out of memory (host)')).toBe(false);
    expect(isOom('training failed: the loader ran out of memory')).toBe(false);
    expect(isOom(new ApiError(404, 'device_not_found', 'device gpu:3 is not available'))).toBe(false);
    expect(isOom(null)).toBe(false);
    expect(isOom(undefined)).toBe(false);
  });
});

describe('oomDetail', () => {
  it('takes the driver report out of the parentheses', () => {
    expect(oomDetail('the GPU ran out of memory (wgpu backend: Out of Memory)')).toBe('wgpu backend: Out of Memory');
  });
  it('keeps an older message whole', () => {
    expect(oomDetail('gpu:0 out of memory: tried to allocate 2.25 GiB')).toBe('gpu:0 out of memory: tried to allocate 2.25 GiB');
    expect(oomDetail('the GPU ran out of memory')).toBe(null);
    expect(oomDetail(null)).toBe(null);
  });
});

describe('oomFixes', () => {
  it('halves a large batch first, then a smaller image, then a smaller model', () => {
    expect(oomFixes(cfg({ batch: 32, imgsz: 960, arch: 'folo-m' })).map((f) => f.label)).toEqual(['Retry with batch 16', 'Retry at 640 px', 'Retry with folo-s']);
  });
  it('leads with the image size once the batch is small', () => {
    expect(oomFixes(cfg({ batch: 4 })).map((f) => f.kind)).toEqual(['imgsz', 'arch', 'batch']);
    expect(oomFixes(cfg({ batch: 1 })).map((f) => f.kind)).toEqual(['imgsz', 'arch']);
  });
  it('rounds an odd batch down and steps off a size New run does not offer', () => {
    const [b, s] = oomFixes(cfg({ batch: 7, imgsz: 1024 }));
    expect(b?.query).toEqual({ batch: '3' });
    expect(s?.query).toEqual({ imgsz: '960' });
  });
  it('falls back to the CPU when nothing smaller is left', () => {
    expect(oomFixes(cfg({ batch: 1, imgsz: 320, arch: 'folo-n' })).map((f) => f.kind)).toEqual(['cpu']);
    expect(oomFixes(cfg({ batch: 1, imgsz: 320, arch: 'folo-n', device: 'cpu' }))).toEqual([]);
  });
  it('builds the New run link', () => {
    const [fix] = oomFixes(cfg({ batch: 64 }));
    expect(retryPath('run_a2f0', fix)).toBe('/runs/new?from=run_a2f0&batch=32');
    expect(retryPath('run_a2f0')).toBe('/runs/new?from=run_a2f0');
  });
});

const GiB = 2 ** 30;
const full = (c: Partial<TrainConfig>): TrainConfig => ({
  name: 'r',
  dataset_id: 'ds',
  arch: 'folo-m',
  epochs: 100,
  batch: 32,
  imgsz: 960,
  device: 'auto',
  optimizer: 'auto',
  lr0: 0.01,
  weights: null,
  augment: { mosaic: 1, mixup: 0, hsv: [0.015, 0.7, 0.4], fliplr: 0.5, scale: 0.5, translate: 0.1, close_mosaic: 10 },
  seed: 0,
  ...c,
});
/** The reference model, as New run's local estimate. */
const reference: VramOf = async (c) => ({ bytes: referenceVram(c.arch, c.imgsz, c.batch, c.amp ?? false), measured: false });

describe('fitFixes: a suggested fix must fix it', () => {
  it('halves down to the largest batch the forecast fits with 10% to spare, and says what it costs', async () => {
    // folo-m at 960 px: batch 16 is forecast at ~26.9 GiB, batch 8 at ~13.6 GiB, on a 16 GiB GPU.
    const [first] = await fitFixes(full({}), 16 * GiB, reference);
    expect(first).toMatchObject({ kind: 'batch', label: 'Retry with batch 8', query: { batch: '8' }, fits: true });
    expect(first!.vram! / GiB).toBeCloseTo(13.6, 1);
    expect(referenceVram('folo-m', 960, 16, false) / GiB).toBeCloseTo(26.9, 1);
    expect(fitsGpu(first!.vram!, 16 * GiB)).toBe(true);
  });
  it('gives a smaller image or model the batch that fits it, and drops a fix that fits at no batch', async () => {
    const fixes = await fitFixes(full({}), 16 * GiB, reference);
    expect(fixes.map((f) => f.label)).toEqual(['Retry with batch 8', 'Retry at 640 px, batch 16', 'Retry with folo-s, batch 16']);
    expect(fixes.every((f) => f.fits && fitsGpu(f.vram!, 16 * GiB))).toBe(true);
    expect(fixes[1]!.query).toEqual({ imgsz: '640', batch: '16' });
  });
  it('keeps the headroom: a forecast between 90% and 100% of the memory does not fit', async () => {
    const tight: VramOf = async (c) => ({ bytes: c.batch >= 16 ? 15 * GiB : 7 * GiB, measured: false });
    expect((await fitFixes(full({ batch: 32, imgsz: 320, arch: 'folo-n' }), 16 * GiB, tight))[0]).toMatchObject({ label: 'Retry with batch 8', fits: true });
  });
  it("scales the reference by the failed run's correction, but never a measured peak", async () => {
    const low = full({ arch: 'folo-n', batch: 32, imgsz: 640 });
    const k = oomCalibration(low, 16 * GiB);
    expect(k).toBeGreaterThan(1);
    const [ref] = await fitFixes(low, 16 * GiB, reference, k);
    expect(ref!.vram).toBeCloseTo(referenceVram('folo-n', 640, Number(ref!.query.batch), false) * k, 0);
    const measured: VramOf = async (c) => ({ bytes: c.batch * 0.5 * GiB, measured: true });
    expect((await fitFixes(low, 16 * GiB, measured, k))[0]).toMatchObject({ query: { batch: '16' }, vram: 8 * GiB });
  });
  it('leads with the CPU when nothing on the GPU fits', async () => {
    const huge: VramOf = async () => ({ bytes: 99 * GiB, measured: false });
    const fixes = await fitFixes(full({}), 16 * GiB, huge);
    expect(fixes[0]).toMatchObject({ kind: 'cpu', fits: true });
    expect(fixes.slice(1).every((f) => f.fits === false)).toBe(true);
  });
  it('without the GPU memory, offers the fixes unforecast', async () => {
    expect(await fitFixes(full({}), null, reference)).toEqual(oomFixes(full({})));
  });
  it('halves to 1', () => {
    expect(smallerBatches(32)).toEqual([16, 8, 4, 2, 1]);
    expect(smallerBatches(1)).toEqual([]);
  });
});

describe("fitFix: New run's one change", () => {
  it('is the first fix that fits, named as an action', async () => {
    const f = await fitFix(full({ batch: 16 }), 16 * GiB, reference);
    expect(f && useLabel(f)).toBe('Use batch 8');
    expect(useLabel({ kind: 'imgsz', label: 'Retry at 640 px, batch 16', query: {} })).toBe('Use 640 px, batch 16');
    expect(useLabel({ kind: 'cpu', label: 'Retry on the CPU', query: {} })).toBe('Use the CPU');
  });
  it('is never the CPU', async () => {
    expect(await fitFix(full({}), 16 * GiB, async () => ({ bytes: 99 * GiB, measured: false }))).toBeNull();
  });
});

describe('vramForecaster', () => {
  const sys = (features: string[]) => ({ features }) as unknown as Parameters<typeof vramForecaster>[1];
  it("asks the server where it estimates runs, and falls back to the reference when it cannot", async () => {
    const api = { estimateRun: async () => ({ est_peak_vram_bytes: 5 * GiB, vram_source: 'measured' }) } as unknown as Parameters<typeof vramForecaster>[0];
    expect(await vramForecaster(api, sys(['run_estimate']))(full({}))).toEqual({ bytes: 5 * GiB, measured: true });
    expect(await vramForecaster(api, sys([]))(full({}))).toEqual({ bytes: referenceVram('folo-m', 960, 32, false), measured: false });
    const broken = { estimateRun: async () => Promise.reject(new Error('down')) } as unknown as Parameters<typeof vramForecaster>[0];
    expect((await vramForecaster(broken, sys(['run_estimate']))(full({})))?.measured).toBe(false);
    expect(await vramForecaster(api, sys(['run_estimate']))(full({ device: 'cpu' }))).toBeNull();
  });
});

describe('failureCause', () => {
  it('names an out-of-memory failure, nothing else', () => {
    expect(failureCause({ status: 'failed', error: 'the GPU ran out of memory (x)' })).toBe('Out of GPU memory');
    expect(failureCause({ status: 'failed', error: 'disk full' })).toBe(null);
    expect(failureCause({ status: 'completed', error: null })).toBe(null);
  });
});
