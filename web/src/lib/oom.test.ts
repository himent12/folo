import { describe, expect, it } from 'vitest';
import { ApiError } from '$lib/api/client';
import type { TrainConfig } from '$lib/api/types';
import { failureCause, isOom, oomDetail, oomFixes, retryPath } from './oom';

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

describe('failureCause', () => {
  it('names an out-of-memory failure, nothing else', () => {
    expect(failureCause({ status: 'failed', error: 'the GPU ran out of memory (x)' })).toBe('Out of GPU memory');
    expect(failureCause({ status: 'failed', error: 'disk full' })).toBe(null);
    expect(failureCause({ status: 'completed', error: null })).toBe(null);
  });
});
