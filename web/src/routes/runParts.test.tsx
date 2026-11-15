import { describe, expect, it } from 'vitest';
import { startFrom } from './RunCompare';
import { epochsDone, lrParts, mosaicOffText, stepEpoch } from './run/readings';

describe('Run metrics: mosaicOffText from Run.epoch (stress s3-live-1)', () => {
  // A 3-epoch run with close_mosaic capped to 1: the trainer closes mosaic after epoch 2, so epoch 3 alone trains
  // without it. `Run.epoch` is the epoch in progress while live and the one it failed in for a failed run.
  const say = (status: string, epoch: number, live: boolean) => mosaicOffText(1, 3, epochsDone(status, epoch), live).cap;
  it('does not call mosaic off while epoch 2 of 3 still trains with it', () => {
    expect(say('running', 2, true)).toBe('Mosaic off after epoch 2');
    expect(say('stopping', 2, true)).toBe('Mosaic off after epoch 2');
    expect(say('running', 3, true)).toBe('Mosaic off · final epoch');
  });
  it('does not claim a mosaic-free epoch a run that failed in epoch 2 never trained', () => {
    expect(say('failed', 2, false)).toBe('Mosaic off after epoch 2');
    expect(say('failed', 3, false)).toBe('Last 1 epoch: mosaic off');
  });
  it('reads the finished epochs of completed and stopped runs as they are', () => {
    expect(say('completed', 3, false)).toBe('Last 1 epoch: mosaic off');
    expect(say('stopped', 1, false)).toBe('Mosaic off after epoch 2');
    expect(epochsDone('queued', 0)).toBe(0);
  });
});

describe('Compare: startFrom', () => {
  const models = [{ id: 'm_shapes_n', name: 'shapes-n' }];
  it('names the model a run was fine-tuned from, or says it started from scratch', () => {
    expect(startFrom('m_shapes_n', models)).toBe('shapes-n');
    expect(startFrom(null, models)).toBeNull();
    expect(startFrom('', models)).toBeNull();
  });
  it('shows a weights path by its file name, and an unknown id as given', () => {
    expect(startFrom('D:/runs/shapes-n/best.folo', models)).toBe('best.folo');
    expect(startFrom('C:\\runs\\x\\last.folo', models)).toBe('last.folo');
    expect(startFrom('m_gone', undefined)).toBe('m_gone');
  });
});

describe('Run metrics: lrParts', () => {
  it('splits a learning rate into a short value and its power of ten', () => {
    expect(lrParts(3.42e-4)).toEqual({ value: '3.42', unit: '× 10⁻⁴', exp: '−4' });
    expect(lrParts(0.01)).toEqual({ value: '1.00', unit: '× 10⁻²', exp: '−2' });
    expect(lrParts(25)).toEqual({ value: '2.50', unit: '× 10¹', exp: '1' });
    expect(lrParts(null)).toEqual({ value: '—' });
    expect(lrParts(0)).toEqual({ value: '0' });
  });
});

describe('Run metrics: mosaicOffText', () => {
  it("names the hatched end of the bar with the trainer's capped count, for where the run is", () => {
    // Ahead of the phase: when it starts, never "mosaic off" as if it were the current state.
    expect(mosaicOffText(10, 100, 38, true).cap).toBe('Mosaic off after epoch 90');
    expect(mosaicOffText(10, 100, 38, false).cap).toBe('Mosaic off after epoch 90');
    // In it, live.
    expect(mosaicOffText(10, 100, 93, true).cap).toBe('Mosaic off · final 10 epochs');
    expect(mosaicOffText(1, 20, 19, true).cap).toBe('Mosaic off · final epoch');
    // Past it.
    expect(mosaicOffText(7, 15, 15, false).cap).toBe('Last 7 epochs: mosaic off');
    expect(mosaicOffText(1, 20, 20, false).full).toBe('Last 1 epoch: mosaic off');
    // Stopped early by the trainer: the phase it never reached is not promised, one it ended in is not "the last 10".
    expect(mosaicOffText(10, 2000, 256, false, true).cap).toBe('Stopped early, before mosaic off');
    expect(mosaicOffText(10, 100, 95, false, true).cap).toBe('Mosaic off from epoch 91');
  });
});

describe('Run metrics: stepEpoch', () => {
  it("names a per-step reading's epoch, partial only past a finished run's last epoch", () => {
    expect(stepEpoch('running', 23.4, 23)).toBe('ep 24');
    expect(stepEpoch('failed', 23.4, 23)).toBe('ep 24 (partial)');
    expect(stepEpoch('stopped', 23.4, 23)).toBe('ep 24 (partial)');
    expect(stepEpoch('completed', 100, 100)).toBe('ep 100');
    expect(stepEpoch('failed', 0.2, 0)).toBe('ep 1 (partial)');
  });
});
