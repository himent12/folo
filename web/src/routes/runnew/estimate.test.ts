import { describe, expect, it } from 'vitest';
import { AMP_SPEEDUP, autoCloseMosaic, closeMosaicEpochs, defaultImgsz, estimate, oomScale, shortRun, warmup, warmupIters } from './estimate';

const GiB = 2 ** 30;

describe('oomScale', () => {
  it('leaves the estimate alone when it already said the failed settings would not fit', () => {
    expect(oomScale(27 * GiB, 16 * GiB)).toBe(1);
  });
  it('scales a low estimate until the failed settings exceed the memory by 5%', () => {
    const k = oomScale(8 * GiB, 16 * GiB);
    expect(k).toBeCloseTo(2.1);
    expect(8 * GiB * k).toBeGreaterThan(16 * GiB);
  });
  it('does nothing without an estimate', () => {
    expect(oomScale(0, 16 * GiB)).toBe(1);
  });
});

describe('warmupIters', () => {
  it("matches the engine's own examples (folo_nn::sched)", () => {
    expect(warmupIters(3, 30, 13)).toBe(100); // docs/MODEL.md §10.4: T = 390, the full floor
    expect(warmupIters(0, 30, 13)).toBe(0);
    expect(warmupIters(3, 1, 13)).toBe(0); // epochs − 1 = 0
    expect(warmupIters(3, 10, 13)).toBe(39); // T = 130: floor 39, 3 epochs = 39
    expect(warmupIters(3, 12, 3)).toBe(10); // T = 36: floor 10
    expect(warmupIters(0.5, 37, 9)).toBe(99); // T = 333: floor 99
    expect(warmupIters(0.5, 334, 1)).toBe(100); // T = 334: the full floor
    expect(warmupIters(3, 4, 3)).toBe(9); // 3 epochs of 4, capped at epochs − 1
  });
});

describe('warmup', () => {
  it("follows docs/MODEL.md's worked example (800 images, batch 16, 10 epochs)", () => {
    // nb = 50, accumulate = 4, spe = 13, T = 130; F = min(100, 39) = 39, so W = min(max(39, 39), 9 · 13) = 39.
    expect(warmup(800, 10, 16)).toEqual({ stepsPerEpoch: 13, accumulate: 4, steps: 130, warmup: 39 });
  });
  it('counts optimizer steps, not batches: a large batch accumulates nothing', () => {
    expect(warmup(1000, 100, 64).stepsPerEpoch).toBe(16);
    expect(warmup(1000, 100, 128).stepsPerEpoch).toBe(8);
  });
  it('never warms up past the last epoch, and not at all in a one-epoch run', () => {
    expect(warmup(64, 5, 16)).toEqual({ stepsPerEpoch: 1, accumulate: 4, steps: 5, warmup: 3 });
    expect(warmup(64, 1, 16).warmup).toBe(0);
  });
  it('is three epochs on a large dataset', () => {
    expect(warmup(100_000, 100, 16).warmup).toBe(Math.round(3 * Math.ceil(6250 / 4)));
  });
});

describe('shortRun', () => {
  it('says nothing about a normal run', () => {
    expect(shortRun(1920, 100, 16, true)).toBeNull();
  });
  it('flags a run from scratch under 200 optimizer steps, with the fewest epochs that reach them', () => {
    // 800 images at batch 16: spe = 13; 10 epochs are 130 steps (W = 39). 200 steps take 16 epochs.
    expect(shortRun(800, 10, 16, true)).toEqual({ stepsPerEpoch: 13, accumulate: 4, steps: 130, warmup: 39, scratch: true, warmupHeavy: false, epochs: 16 });
    // From trained weights a short run is fine.
    expect(shortRun(800, 10, 16, false)).toBeNull();
    expect(shortRun(800, 16, 16, true)).toBeNull();
  });
  it('flags a warmup longer than half the run, from weights too', () => {
    // 1,920 images at batch 16: spe = 30. 5 epochs are 150 steps, W = min(max(90, 45), 120) = 90 > 75.
    expect(shortRun(1920, 5, 16, false)).toEqual({ stepsPerEpoch: 30, accumulate: 4, steps: 150, warmup: 90, scratch: false, warmupHeavy: true, epochs: 6 });
    // Three warmup epochs are at most half the run from 6 epochs on, whatever the dataset (the floor stays under 30%).
    expect(shortRun(1920, 6, 16, false)).toBeNull();
    expect(shortRun(64, 4, 16, false)).toMatchObject({ warmup: 3, steps: 4, warmupHeavy: true, epochs: 6 });
    expect(shortRun(64, 20, 16, false)).toBeNull();
  });
  it("suggests the trainer's epochs for a short run from scratch, ceil(200 / spe), so the run page says the same", () => {
    // From scratch as well: 150 steps; 6 epochs clear warmup (180 steps) but 200 steps need 7.
    expect(shortRun(1920, 5, 16, true)).toMatchObject({ scratch: true, warmupHeavy: true, epochs: 7 });
    // The short-run scenario's run (800 images, batch 32: 25 batches, 2 per step): 65 steps, 16 epochs.
    expect(shortRun(800, 5, 32, true)).toMatchObject({ stepsPerEpoch: 13, accumulate: 2, steps: 65, warmup: 39, epochs: 16 });
    // 50 steps an epoch: the trainer's 4 epochs, even though warmup still takes over half of those 200 steps.
    expect(shortRun(3200, 3, 64, true)).toMatchObject({ stepsPerEpoch: 50, scratch: true, epochs: 4 });
    expect(shortRun(3200, 4, 64, true)).toMatchObject({ scratch: false, warmupHeavy: true, epochs: 6 });
  });
  it('ignores invalid input', () => {
    expect(shortRun(0, 10, 16, true)).toBeNull();
    expect(shortRun(1920, Number.NaN, 16, true)).toBeNull();
  });
});

describe('close mosaic', () => {
  it("is capped at half the run, as the trainer does (docs/MODEL.md §8.6's examples)", () => {
    // close_mosaic = 10 closes mosaic for 10 epochs of 20 or 30, 6 of 12, 4 of 8 and none of 1.
    expect([20, 30, 12, 8, 1].map((e) => closeMosaicEpochs(10, e))).toEqual([10, 10, 6, 4, 0]);
    expect(closeMosaicEpochs(3, 100)).toBe(3);
  });
  it('defaults to 10 epochs under the same cap', () => {
    expect([1, 2, 5, 20, 100].map(autoCloseMosaic)).toEqual([0, 1, 2, 10, 10]);
  });
});

describe('defaultImgsz', () => {
  it("trains at a smaller dataset's own size", () => {
    expect(defaultImgsz([[320, 320], [320, 320]])).toBe(320);
    expect(defaultImgsz([[300, 200]])).toBe(320);
  });
  it('stays at 640 for larger, mixed or unknown sizes', () => {
    expect(defaultImgsz([[1024, 768]])).toBe(640);
    expect(defaultImgsz([[320, 320], [480, 480]])).toBe(640);
    expect(defaultImgsz([])).toBe(640);
    expect(defaultImgsz(null)).toBe(640);
  });
});

describe('estimate with mixed precision', () => {
  it("uses the measured speed-up, carries the nearest size's to the others, and trims the activation memory", () => {
    const f32 = estimate('folo-s', 1000, 10, 16, 640, false);
    const amp = estimate('folo-s', 1000, 10, 16, 640, false, true);
    expect(amp.imgsPerS / f32.imgsPerS).toBeCloseTo(AMP_SPEEDUP['folo-s']!);
    expect(amp.vramBytes).toBeLessThan(f32.vramBytes);
    // folo-m has no amp row: the nearest by compute (folo-s at 640) carries over, with its ratio.
    expect(estimate('folo-m', 1000, 10, 16, 640, false, true).imgsPerS / estimate('folo-m', 1000, 10, 16, 640, false).imgsPerS).toBeCloseTo(AMP_SPEEDUP['folo-s']!);
  });
  it('changes nothing on the CPU', () => {
    expect(estimate('folo-n', 1000, 10, 16, 640, true, true)).toEqual(estimate('folo-n', 1000, 10, 16, 640, true));
  });
});

describe('estimate from the reference figures (docs/BENCHMARKS.md)', () => {
  it("reproduces the document's rows: folo-n at 640 trains at 208 img/s, and its step peak of 3474 MiB", () => {
    const e = estimate('folo-n', 1000, 10, 16, 640, false);
    expect(e.imgsPerS).toBeCloseTo(208);
    expect(e.vramBytes / 2 ** 20).toBeCloseTo(3474, -1);
  });
  it('adds the startup once and the close-mosaic epochs at their own rate', () => {
    const all = estimate('folo-n', 2000, 10, 16, 640, false, false, { valImages: 200 });
    const closed = estimate('folo-n', 2000, 10, 16, 640, false, false, { valImages: 200, closeEpochs: 5 });
    expect(all.totalSeconds).toBeCloseTo(6 + 10 * (2000 / 208 + 200 / 275));
    expect(closed.totalSeconds).toBeLessThan(all.totalSeconds);
  });
  it('validates the training split when there is no val split', () => {
    expect(estimate('folo-n', 2000, 10, 16, 640, false, false, { valImages: 0 }).totalSeconds).toBeCloseTo(6 + 10 * (2000 / 208 + 2000 / 275));
  });
  it("gives the reference's range: -30% / +60% on the reference machine, -60% / +200% elsewhere", () => {
    const here = estimate('folo-s', 1000, 10, 16, 640, false, false, { sameMachine: true });
    expect([here.lowSeconds / here.totalSeconds, here.highSeconds / here.totalSeconds]).toEqual([0.7, 1.6].map((x) => expect.closeTo(x)));
    const other = estimate('folo-s', 1000, 10, 16, 640, false);
    expect([other.lowSeconds / other.totalSeconds, other.highSeconds / other.totalSeconds]).toEqual([0.4, 3].map((x) => expect.closeTo(x)));
  });
  it('slows smaller images less than their compute (to the 0.55 within 4x)', () => {
    const big = estimate('folo-s', 1000, 10, 16, 640, false).imgsPerS;
    const small = estimate('folo-s', 1000, 10, 16, 480, false).imgsPerS;
    expect(small / big).toBeCloseTo((640 / 480) ** (2 * 0.55));
  });
});
