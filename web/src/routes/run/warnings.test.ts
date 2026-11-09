import { describe, expect, it } from 'vitest';
import type { LogLine } from '$lib/api/types';
import { plain, scheduleTitle, scheduleWarnings, sentences, shortCause, suggestedEpochs } from './warnings';

const SHORT =
  'this run takes only 125 optimizer steps (5 epochs × 25: 800 training images, batch 32); a detector trained from scratch usually needs a few hundred steps to learn anything, so expect an mAP near 0. Train for at least 8 epochs, add images, or start from pretrained weights';
const WARMUP = "warmup takes 75 of the run's 125 optimizer steps, so the learning rate reaches lr0 only in the second half of the run; lower warmup_epochs (now 3) or train for more epochs";
const line = (level: LogLine['level'], message: string): LogLine => ({ t: '2026-11-08T14:00:00Z', level, message });

describe('scheduleWarnings', () => {
  it('finds the trainer’s schedule warnings after the first line, once each, in order', () => {
    const logs = [line('info', 'FOLO 0.1.0  folo-n  device cpu'), line('warn', SHORT), line('warn', WARMUP), line('info', 'epoch 1/5'), line('warn', SHORT)];
    expect(scheduleWarnings(logs)).toEqual([SHORT, WARMUP]);
  });

  it('ignores other warnings and the same words at another level', () => {
    expect(scheduleWarnings([line('warn', 'data loader stalled 140 ms'), line('info', SHORT)])).toEqual([]);
  });
});

describe('sentences', () => {
  it('turns an engine message into capitalised sentences', () => {
    expect(sentences(WARMUP)).toBe(
      "Warmup takes 75 of the run's 125 optimizer steps, so the learning rate reaches lr0 only in the second half of the run. Lower warmup_epochs (now 3) or train for more epochs.",
    );
    expect(sentences(SHORT)).toMatch(/^This run takes only 125 optimizer steps \(5 epochs × 25: 800 training images, batch 32\)\. A detector .* pretrained weights\.$/);
    expect(sentences('')).toBe('');
  });
});

describe('suggestedEpochs', () => {
  it('is the engine rule, ⌈200 / steps per epoch⌉, from the warning numbers', () => {
    expect(suggestedEpochs([SHORT])).toBe(8);
    // 30 steps an epoch: 200 / 30 = 6.67, so 7 epochs (the engine's div_ceil), whatever the sentence says.
    expect(suggestedEpochs(['this run takes only 90 optimizer steps (3 epochs × 30: 480 training images, batch 16); …'])).toBe(7);
  });

  it('reads the epochs the trainer suggests, or null', () => {
    expect(suggestedEpochs(['this run takes only 125 optimizer steps (5 epochs × 25: …); … Train for at least 8 epochs, add images, or start from pretrained weights'])).toBe(8);
    expect(suggestedEpochs(["warmup takes 70 of the run's 125 optimizer steps, …; lower warmup_epochs (now 3) or train for more epochs"])).toBeNull();
  });
});

describe('plain', () => {
  it('says the engine warnings without config keys', () => {
    expect(plain(WARMUP)).toBe(
      "Warmup takes 75 of the run's 125 optimizer steps, so the learning rate reaches its initial value only in the second half of the run. Shorten the warmup (now 3 epochs) or train for more epochs.",
    );
    expect(plain(SHORT)).toMatch(/^This run takes only 125 optimizer steps \(25 per epoch: 800 training images, batch 32\)\. A detector /);
    for (const w of [SHORT, WARMUP]) expect(plain(w)).not.toMatch(/lr0|warmup_epochs/);
  });
});

describe('shortCause', () => {
  it('is the run optimizer steps, or the warmup share when that is the only warning', () => {
    expect(shortCause([SHORT, WARMUP])).toBe('125 optimizer steps');
    expect(shortCause([WARMUP])).toBe('warmup takes 75 of 125 optimizer steps');
    expect(shortCause([])).toBe('');
  });
});

describe('scheduleTitle', () => {
  it('says too short only when the engine does; a warmup-only warning is "Mostly warmup"', () => {
    const short = 'this run takes only 65 optimizer steps (5 epochs × 13: …); … Train for at least 16 epochs, add images, or start from pretrained weights';
    const warm = "warmup takes 300 of the run's 500 optimizer steps, so the learning rate reaches lr0 only in the second half of the run; lower warmup_epochs (now 3) or train for more epochs";
    expect(scheduleTitle([short, warm])).toBe('Too short to learn:');
    expect(scheduleTitle([warm])).toBe('Mostly warmup:');
  });
});
