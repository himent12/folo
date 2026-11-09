import { describe, expect, it } from 'vitest';
import { autoCloseMosaic, defaultImgsz, oomScale, shortRun, warmup } from './estimate';

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

describe('warmup', () => {
  it("follows docs/MODEL.md's worked example (800 images, batch 16, 10 epochs)", () => {
    // nb = 50, accumulate = 4, spe = 13; W = min(max(39, 100), 9 · 13) = 100.
    expect(warmup(800, 10, 16)).toEqual({ stepsPerEpoch: 13, steps: 130, warmup: 100 });
  });
  it('counts optimizer steps, not batches: a large batch accumulates nothing', () => {
    expect(warmup(1000, 100, 64).stepsPerEpoch).toBe(16);
    expect(warmup(1000, 100, 128).stepsPerEpoch).toBe(8);
  });
  it('never warms up past the last epoch, and not at all in a one-epoch run', () => {
    expect(warmup(64, 5, 16)).toEqual({ stepsPerEpoch: 1, steps: 5, warmup: 4 });
    expect(warmup(64, 1, 16).warmup).toBe(0);
  });
  it('is three epochs on a large dataset', () => {
    expect(warmup(100_000, 100, 16).warmup).toBe(Math.round(3 * Math.ceil(6250 / 4)));
  });
});

describe('shortRun', () => {
  it('says nothing about a normal run', () => {
    expect(shortRun(1920, 100, 16, 10, 1)).toBeNull();
  });
  it('flags a run that is mostly warmup, with the fewest epochs that bring warmup to half', () => {
    // 1,920 images at batch 16: spe = 30, W = 100 from 5 epochs on; 100 ≤ 15 · E needs E ≥ 7.
    expect(shortRun(1920, 5, 16, 2, 1)).toEqual({ kind: 'warmup', warmup: 100, steps: 150, epochs: 7 });
    expect(shortRun(1920, 7, 16, 3, 1)).toBeNull();
    // A tiny dataset steps once an epoch: warmup is 100 steps, so 200 epochs.
    expect(shortRun(64, 20, 16, 10, 1)).toMatchObject({ kind: 'warmup', warmup: 19, steps: 20, epochs: 200 });
  });
  it('flags close mosaic covering the whole run, with a value that leaves mosaic on', () => {
    expect(shortRun(100_000, 10, 16, 10, 1)).toEqual({ kind: 'mosaic', close: 10, epochs: 10, suggest: 5 });
    // Without mosaic there is nothing to close.
    expect(shortRun(100_000, 10, 16, 10, 0)).toBeNull();
  });
  it('ignores invalid input', () => {
    expect(shortRun(0, 10, 16, 0, 1)).toBeNull();
    expect(shortRun(1920, Number.NaN, 16, 0, 1)).toBeNull();
  });
});

describe('autoCloseMosaic', () => {
  it('is 10 epochs, at most half the run', () => {
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
