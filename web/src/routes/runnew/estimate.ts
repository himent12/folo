// Rough training estimates for the New run form. Calibrated loosely against the mock and an RTX 4080-class GPU;
// they are hints ("about 9 minutes", "about 7 GiB"), labelled as estimates in the UI.

import type { Arch } from '$lib/api/types';
import { e2eMs, modelSize } from '$lib/benchmarks';

// `size` is the letter shown in the model list, uppercase (a lowercase l reads as I or | in Geist).
// Sizes and latency are measured where docs/BENCHMARKS.md has them (n, s); `ms` is absent otherwise. The other sizes'
// params and GFLOPs are docs/MODEL.md's table (few classes, 640 px).
export const arches: { arch: Arch; size: string; params: number; flops: number; ms?: number; note: string }[] = [
  { arch: 'folo-n', size: 'N', params: modelSize['folo-n']!.params, flops: modelSize['folo-n']!.gflops, ms: e2eMs['folo-n'], note: 'Fastest; edge devices, quick tries.' },
  { arch: 'folo-s', size: 'S', params: modelSize['folo-s']!.params, flops: modelSize['folo-s']!.gflops, ms: e2eMs['folo-s'], note: 'The default: fast and accurate.' },
  { arch: 'folo-m', size: 'M', params: 20_050_000, flops: 68.0, note: 'More accuracy on busy scenes.' },
  { arch: 'folo-l', size: 'L', params: 25_310_000, flops: 86.9, note: 'High accuracy; a strong GPU.' },
  { arch: 'folo-x', size: 'X', params: 56_880_000, flops: 195.3, note: 'Best accuracy; the most memory.' },
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

/** docs/MODEL.md §8.3: gradients accumulate up to a nominal batch of 64. */
const NOMINAL_BATCH = 64;

export interface Warmup {
  /** Optimizer steps per epoch (`spe`; the trainer steps once per accumulated window, not per batch). */
  stepsPerEpoch: number;
  /** Optimizer steps in the run. */
  steps: number;
  /** Warmup steps, `W`. */
  warmup: number;
}

/**
 * The trainer's warmup (docs/MODEL.md §8.3 and §8.4): `nb = ceil(N / batch)` batches an epoch, accumulated
 * `round(64 / batch)` at a time into `spe = ceil(nb / accumulate)` optimizer steps, and
 * `W = min(max(round(3 · spe), 100), (epochs − 1) · spe)` of them warming up.
 */
export function warmup(trainImages: number, epochs: number, batch: number): Warmup {
  const nb = Math.ceil(Math.max(0, trainImages) / Math.max(1, batch));
  const accumulate = Math.max(Math.round(NOMINAL_BATCH / Math.max(1, batch)), 1);
  const spe = Math.ceil(nb / accumulate);
  const e = Math.max(0, Math.floor(epochs));
  return { stepsPerEpoch: spe, steps: e * spe, warmup: Math.min(Math.max(Math.round(3 * spe), 100), Math.max(e - 1, 0) * spe) };
}

export type ShortRun =
  | { kind: 'warmup'; warmup: number; steps: number; epochs: number }
  | { kind: 'mosaic'; close: number; epochs: number; suggest: number };

/**
 * Why a short run would mostly not train as configured: warmup takes more than half its optimizer steps (`epochs`
 * is then the fewest epochs that bring it to half), or close mosaic turns mosaic off for the whole run (`suggest` is
 * a close-mosaic value that leaves it on for at least half). Warmup first: more epochs also answers mosaic.
 */
export function shortRun(trainImages: number, epochs: number, batch: number, closeMosaic: number, mosaic: number): ShortRun | null {
  if (!(trainImages > 0) || !Number.isInteger(epochs) || epochs < 1) return null;
  const w = warmup(trainImages, epochs, batch);
  if (w.warmup * 2 > w.steps) {
    let e = epochs + 1;
    while (e < 100_000) {
      const n = warmup(trainImages, e, batch);
      if (n.warmup * 2 <= n.steps) break;
      e++;
    }
    return { kind: 'warmup', warmup: w.warmup, steps: w.steps, epochs: e };
  }
  if (mosaic > 0 && closeMosaic >= epochs) return { kind: 'mosaic', close: closeMosaic, epochs, suggest: autoCloseMosaic(epochs) };
  return null;
}

/** Close mosaic's default: 10 epochs, but never more than half the run, so a short run still sees mosaic. */
export function autoCloseMosaic(epochs: number): number {
  return Math.min(10, Math.floor(Math.max(0, epochs) / 2));
}

/**
 * The image size to train at by default: the dataset's own size when every training image shares it and it is
 * smaller than 640 px (training above it only upscales), rounded up to the trainer's multiple of 32; otherwise 640.
 * `sizes` are the training images' `[width, height]`; null when they are not all known.
 */
export function defaultImgsz(sizes: readonly (readonly [number, number])[] | null): number {
  const first = sizes?.[0];
  if (!sizes || !first || sizes.some(([w, h]) => w !== first[0] || h !== first[1])) return 640;
  const side = Math.ceil(Math.max(first[0], first[1]) / 32) * 32;
  return side >= 64 && side < 640 ? side : 640;
}
