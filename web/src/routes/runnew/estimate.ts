// Training estimates for the New run form where the server has none (`run_estimate` missing from /api/system's
// features) or while its answer loads: docs/BENCHMARKS.md's reference figures, the same ones the server falls back
// to, carried over to the run's settings the same way (lib/reference.ts). Shown as a range, labelled as reference.

import { closeMosaicEpochs } from '$lib/api/progress';
import type { Arch } from '$lib/api/types';
import { e2eMs } from '$lib/benchmarks';
import { archSpec, REF_STARTUP_S, referenceRates, referenceSpread, referenceVram } from '$lib/reference';

// `size` is the letter shown in the model list, uppercase (a lowercase l reads as I or | in Geist).
// Sizes and latency are measured where docs/BENCHMARKS.md has them (n, s); `ms` is absent otherwise. The other sizes'
// params and GFLOPs are docs/MODEL.md's table (few classes, 640 px).
export const arches: { arch: Arch; size: string; params: number; flops: number; ms?: number; note: string }[] = [
  { arch: 'folo-n', size: 'N', params: archSpec['folo-n'].params, flops: archSpec['folo-n'].gflops, ms: e2eMs['folo-n'], note: 'Fastest; edge devices, quick tries.' },
  { arch: 'folo-s', size: 'S', params: archSpec['folo-s'].params, flops: archSpec['folo-s'].gflops, ms: e2eMs['folo-s'], note: 'The default: fast and accurate.' },
  { arch: 'folo-m', size: 'M', params: archSpec['folo-m'].params, flops: archSpec['folo-m'].gflops, note: 'More accuracy on busy scenes.' },
  { arch: 'folo-l', size: 'L', params: archSpec['folo-l'].params, flops: archSpec['folo-l'].gflops, note: 'High accuracy; a strong GPU.' },
  { arch: 'folo-x', size: 'X', params: archSpec['folo-x'].params, flops: archSpec['folo-x'].gflops, note: 'Best accuracy; the most memory.' },
];

/**
 * Mixed precision (`amp`) against f32, measured on the training step (docs/BENCHMARKS.md, Mixed-precision training:
 * batch 16 at 640 on an RTX 4080 SUPER): speed for the sizes measured, and the peak memory, about 40% less for both.
 */
export const AMP_SPEEDUP: Partial<Record<Arch, number>> = { 'folo-n': 1.24, 'folo-s': 1.61 };

export interface Estimate {
  /** Batches an epoch (`ceil(N / batch)`); the optimizer steps are fewer when batches accumulate (`warmup`). */
  stepsPerEpoch: number;
  /** Training images per second with mosaic (the headline rate). */
  imgsPerS: number;
  epochSeconds: number;
  totalSeconds: number;
  /** The range around `totalSeconds` (docs/API.md's spread for a reference estimate). */
  lowSeconds: number;
  highSeconds: number;
  vramBytes: number;
  /** As the server's `RunEstimate.phases`: the mosaic epochs, then the close-mosaic ones. */
  phases: { phase: 'mosaic' | 'close_mosaic'; epochs: number; train_imgs_per_s: number; epoch_s: number }[];
  valImages: number;
  valImgsPerS: number;
  startupS: number;
}

export interface EstimateOptions {
  /** Validation images an epoch (the val split; the training split when it has none, as the trainer does). */
  valImages?: number;
  /** Epochs without mosaic at the end (already capped at half the run; every epoch when mosaic is 0). */
  closeEpochs?: number;
  /** Whether this server's device is the reference machine's (an RTX 4080 SUPER, or its i7-14700KF on the CPU). */
  sameMachine?: boolean;
}

/**
 * The server's reference estimate (docs/API.md, POST /api/runs/estimate): each epoch trains at its phase's rate and
 * validates at the validation rate; the first adds the startup. `amp`: mixed precision on the GPU.
 */
export function estimate(arch: Arch, trainImages: number, epochs: number, batch: number, imgsz: number, cpu: boolean, amp = false, opts: EstimateOptions = {}): Estimate {
  const mixed = amp && !cpu;
  const r = referenceRates(arch, imgsz, mixed, cpu);
  const e = Number.isFinite(epochs) && epochs > 0 ? Math.floor(epochs) : 0;
  const close = Math.max(0, Math.min(e, opts.closeEpochs ?? 0));
  const val = (opts.valImages ?? 0) > 0 ? opts.valImages! : trainImages;
  const valS = val / r.val;
  const phases = [
    { phase: 'mosaic' as const, epochs: e - close, train_imgs_per_s: r.mosaic, epoch_s: trainImages / r.mosaic + valS },
    { phase: 'close_mosaic' as const, epochs: close, train_imgs_per_s: r.close, epoch_s: trainImages / r.close + valS },
  ].filter((p) => p.epochs > 0);
  const train = phases.reduce((t, p) => t + p.epochs * p.epoch_s, 0);
  const total = REF_STARTUP_S + train;
  const [lo, hi] = referenceSpread(opts.sameMachine ?? false);
  return {
    stepsPerEpoch: Math.max(1, Math.ceil(trainImages / Math.max(1, batch))),
    imgsPerS: r.mosaic,
    epochSeconds: e > 0 ? train / e : 0,
    totalSeconds: total,
    lowSeconds: total * (1 + lo),
    highSeconds: total * (1 + hi),
    vramBytes: cpu ? 0 : referenceVram(arch, imgsz, batch, mixed),
    phases,
    valImages: val,
    valImgsPerS: r.val,
    startupS: REF_STARTUP_S,
  };
}

/** Lives with the other out-of-memory rules (lib/oom); here for the form's tests. */
export { oomScale } from '$lib/oom';

/** docs/MODEL.md §8.3: gradients accumulate up to a nominal batch of 64. */
const NOMINAL_BATCH = 64;

export interface Warmup {
  /** Optimizer steps per epoch (`spe`; the trainer steps once per accumulated window, not per batch). */
  stepsPerEpoch: number;
  /** Batches accumulated into one optimizer step, `round(64 / batch)` (at least 1). */
  accumulate: number;
  /** Optimizer steps in the run, `T`. */
  steps: number;
  /** Warmup steps, `W`. */
  warmup: number;
}

/** The trainer's default `warmup_epochs` (docs/MODEL.md §8.3); the form does not change it. */
export const WARMUP_EPOCHS = 3;

/**
 * docs/MODEL.md §8.4 (`folo_nn::sched::warmup_iters_floor` with a new run's `warmup_floor` 0): `W = 0` when
 * `warmup_epochs <= 0`, else `min(round(warmup_epochs · spe), max(epochs − 1, 0) · spe)`.
 */
export function warmupIters(warmupEpochs: number, epochs: number, spe: number): number {
  if (!(warmupEpochs > 0)) return 0;
  return Math.min(Math.round(warmupEpochs * spe), Math.max(epochs - 1, 0) * spe);
}

/**
 * The trainer's schedule for a run (docs/MODEL.md §8.3 and §8.4): `nb = ceil(N / batch)` batches an epoch, accumulated
 * `round(64 / batch)` at a time into `spe = ceil(nb / accumulate)` optimizer steps, `W` of them warming up.
 */
export function warmup(trainImages: number, epochs: number, batch: number): Warmup {
  const nb = Math.ceil(Math.max(0, trainImages) / Math.max(1, batch));
  const accumulate = Math.max(Math.round(NOMINAL_BATCH / Math.max(1, batch)), 1);
  const spe = Math.ceil(nb / accumulate);
  const e = Math.max(0, Math.floor(epochs));
  return { stepsPerEpoch: spe, accumulate, steps: e * spe, warmup: warmupIters(WARMUP_EPOCHS, e, spe) };
}

/** Runs from scratch with fewer optimizer steps than this get a warning (`folo_vision::train::SHORT_RUN_STEPS`). */
export const SHORT_RUN_STEPS = 200;

export interface ShortRun extends Warmup {
  /** From scratch with fewer than `SHORT_RUN_STEPS` optimizer steps: expect an mAP near 0. */
  scratch: boolean;
  /** Warmup takes more than half the run: the learning rate reaches lr0 only in its second half. */
  warmupHeavy: boolean;
  /**
   * The epochs to suggest: for a short run from scratch the trainer's own, `ceil(200 / spe)` (`SHORT_RUN_STEPS
   * .div_ceil(spe)` in folo_vision::train::schedule_warnings, so the run page suggests the same), else the fewest that
   * clear the warmup warning.
   */
  epochs: number;
}

/**
 * The trainer's short-run warnings (`folo_vision::train::schedule_warnings`, docs/MODEL.md §8.4), before the run
 * starts: a run from scratch (`fromScratch`, no start weights) with fewer than 200 optimizer steps, or a warmup longer
 * than half the run. Null when neither applies.
 */
export function shortRun(trainImages: number, epochs: number, batch: number, fromScratch: boolean): ShortRun | null {
  if (!(trainImages > 0) || !Number.isInteger(epochs) || epochs < 1) return null;
  const check = (e: number) => {
    const w = warmup(trainImages, e, batch);
    return { ...w, scratch: fromScratch && w.steps < SHORT_RUN_STEPS, warmupHeavy: w.warmup * 2 > w.steps };
  };
  const now = check(epochs);
  if (!now.scratch && !now.warmupHeavy) return null;
  if (now.scratch) return { ...now, epochs: Math.ceil(SHORT_RUN_STEPS / Math.max(1, now.stepsPerEpoch)) };
  let e = epochs + 1;
  for (; e < 100_000; e++) {
    const n = check(e);
    if (!n.scratch && !n.warmupHeavy) break;
  }
  return { ...now, epochs: e };
}

// The trainer's cap of close mosaic, kept with the run's progress (one formula for New run, Run and the mock).
export { closeMosaicEpochs };

/** Close mosaic's default: the trainer's 10 epochs under its own cap. */
export function autoCloseMosaic(epochs: number): number {
  return closeMosaicEpochs(10, epochs);
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
