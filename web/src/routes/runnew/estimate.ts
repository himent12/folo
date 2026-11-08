// Rough training estimates for the New run form. Calibrated loosely against the mock and an RTX 4080-class GPU;
// they are hints ("about 9 minutes", "about 7 GiB"), labelled as estimates in the UI.

import type { Arch } from '$lib/api/types';
import { e2eMs, modelSize } from '$lib/benchmarks';

// Sizes and latency are measured where docs/BENCHMARKS.md has them (n, s); `ms` is absent otherwise.
export const arches: { arch: Arch; size: string; params: number; flops: number; ms?: number; note: string }[] = [
  { arch: 'folo-n', size: 'n', params: modelSize['folo-n']!.params, flops: modelSize['folo-n']!.gflops, ms: e2eMs['folo-n'], note: 'Fastest. Edge devices, quick experiments.' },
  { arch: 'folo-s', size: 's', params: modelSize['folo-s']!.params, flops: modelSize['folo-s']!.gflops, ms: e2eMs['folo-s'], note: 'The sensible default: fast and accurate.' },
  { arch: 'folo-m', size: 'm', params: 20_050_000, flops: 68.0, note: 'More accuracy on harder, busier scenes.' },
  { arch: 'folo-l', size: 'l', params: 25_310_000, flops: 86.9, note: 'Best accuracy; needs a strong GPU.' },
];

const gpuImgsPerS: Record<Arch, number> = { 'folo-n': 520, 'folo-s': 320, 'folo-m': 170, 'folo-l': 130, 'folo-x': 70 };
const gibPerImage: Record<Arch, number> = { 'folo-n': 0.11, 'folo-s': 0.19, 'folo-m': 0.36, 'folo-l': 0.47, 'folo-x': 0.8 };

export interface Estimate {
  stepsPerEpoch: number;
  imgsPerS: number;
  epochSeconds: number;
  totalSeconds: number;
  vramBytes: number;
}

export function estimate(arch: Arch, trainImages: number, epochs: number, batch: number, imgsz: number, cpu: boolean): Estimate {
  const area = (imgsz / 640) ** 2;
  const imgsPerS = (gpuImgsPerS[arch] / area) * (cpu ? 1 / 14 : 1);
  const epochSeconds = trainImages / imgsPerS + Math.min(30, trainImages * 0.002 + 1);
  const params = arches.find((a) => a.arch === arch)?.params ?? 10_000_000;
  const vram = 0.6 * 2 ** 30 + params * 16 + batch * area * gibPerImage[arch] * 2 ** 30;
  return {
    stepsPerEpoch: Math.max(1, Math.ceil(trainImages / Math.max(1, batch))),
    imgsPerS,
    epochSeconds,
    totalSeconds: epochSeconds * epochs,
    vramBytes: cpu ? 0 : vram,
  };
}

/**
 * The factor that makes the estimate agree with a run that ran out of GPU memory: when the estimate for its settings
 * (`failedBytes`) fit in `gpuBytes`, it was low, so estimates are scaled until those settings exceed the memory by 5%.
 * 1 when the estimate already said they would not fit.
 */
export function oomScale(failedBytes: number, gpuBytes: number): number {
  return failedBytes > 0 && failedBytes < gpuBytes * 1.05 ? (gpuBytes * 1.05) / failedBytes : 1;
}
