import { describe, expect, it } from 'vitest';
import type { Gpu, Run } from '$lib/api/types';
import { deviceLine, runGpu } from './device';

const gpus: Gpu[] = [
  { ordinal: 0, name: 'NVIDIA GeForce RTX 4080 SUPER', backend: 'vulkan', memory_bytes: 16e9, util: 0.93 },
  { ordinal: 1, name: 'NVIDIA GeForce RTX 3060', backend: 'vulkan', memory_bytes: 12e9, util: 0.12 },
];
const run = (device: string, label: string) => ({ device: label, config: { device } }) as unknown as Pick<Run, 'device' | 'config'>;

describe('runGpu / deviceLine', () => {
  it('never attaches GPU load to a CPU run', () => {
    expect(runGpu(run('cpu', 'AMD Ryzen 9 7950X (cpu)'), gpus)).toBeNull();
    expect(deviceLine(run('cpu', 'AMD Ryzen 9 7950X (cpu)'), gpus)).toBe('Ryzen 9 7950X (CPU)');
  });

  it('uses the GPU with the requested ordinal', () => {
    expect(runGpu(run('gpu:1', 'NVIDIA GeForce RTX 3060 (vulkan)'), gpus)?.ordinal).toBe(1);
    expect(deviceLine(run('gpu:1', 'NVIDIA GeForce RTX 3060 (vulkan)'), gpus)).toBe('RTX 3060 · 12 % busy');
  });

  it('matches auto runs by adapter name, and stays silent when unsure', () => {
    expect(runGpu(run('auto', 'NVIDIA GeForce RTX 4080 SUPER (vulkan)'), gpus)?.ordinal).toBe(0);
    expect(runGpu(run('auto', 'Some other adapter'), gpus)).toBeNull();
    expect(runGpu(run('gpu:7', 'x'), gpus)).toBeNull();
  });
});
