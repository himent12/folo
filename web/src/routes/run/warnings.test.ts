import { describe, expect, it } from 'vitest';
import type { LogLine } from '$lib/api/types';
import { minimumEpochs, plain, scheduleTitle, scheduleWarnings, sentences, shortCause, suggestedEpochs } from './warnings';

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

describe('minimumEpochs', () => {
  it('is the engine rule, ⌈200 / steps per epoch⌉, from the warning numbers', () => {
    expect(minimumEpochs([SHORT])).toBe(8);
    // 30 steps an epoch: 200 / 30 = 6.67, so 7 epochs (the engine's div_ceil), whatever the sentence says.
    expect(minimumEpochs(['this run takes only 90 optimizer steps (3 epochs × 30: 480 training images, batch 16); …'])).toBe(7);
  });

  it('reads the epochs the trainer suggests, or null', () => {
    expect(minimumEpochs(['this run takes only 125 optimizer steps (5 epochs × 25: …); … Train for at least 8 epochs, add images, or start from pretrained weights'])).toBe(8);
    expect(minimumEpochs(["warmup takes 70 of the run's 125 optimizer steps, …; lower warmup_epochs (now 3) or train for more epochs"])).toBeNull();
  });
});

describe('suggestedEpochs', () => {
  it('suggests a clearly larger, round count, never one epoch more (v-data-runs N4)', () => {
    // 15 epochs of 13 steps: the minimum is 16; the suggestion is 30, twice the run.
    expect(suggestedEpochs(['this run takes only 195 optimizer steps (15 epochs × 13: 200 training images, batch 16); …'], 15)).toBe(30);
    // 60 epochs of 3 steps (the minimum 67): 150, not 67 or 120.
    expect(suggestedEpochs(['this run takes only 180 optimizer steps (60 epochs × 3: 40 training images, batch 16); …'], 60)).toBe(150);
    // The minimum above twice the run wins, rounded up: 5 epochs with a minimum of 8 → 10.
    expect(suggestedEpochs([SHORT], 5)).toBe(10);
    expect(suggestedEpochs([WARMUP], 5)).toBeNull();
  });
});

describe('plain', () => {
  it('says the engine warnings without config keys, the steps per epoch from the numbers', () => {
    expect(plain(SHORT)).toMatch(/^This run takes only 125 optimizer steps \(25 per epoch: 800 training images, batch 32\)\. A detector /);
    for (const w of [SHORT, WARMUP]) expect(plain(w)).not.toMatch(/lr0|warmup_epochs/);
  });
});

describe('shortCause and scheduleTitle', () => {
  it('lead with the short run, and with the warmup share only when that is the only warning', () => {
    expect(shortCause([SHORT, WARMUP])).toBe('125 optimizer steps');
    expect(shortCause([WARMUP])).toBe('warmup takes 75 of 125 optimizer steps');
    expect(shortCause([])).toBe('');
    expect(scheduleTitle([SHORT, WARMUP])).toBe('Too short to learn:');
    expect(scheduleTitle([WARMUP])).toBe('Mostly warmup:');
  });

  it('do not say a run that scored well was too short to learn, nor that its mAP is near 0', () => {
    expect(scheduleTitle([SHORT], true)).toBe('More epochs would likely improve this:');
    expect(scheduleTitle([WARMUP], true)).toBe('Mostly warmup:');
    expect(plain(SHORT, true)).not.toMatch(/mAP near 0|learn anything/);
    expect(plain(SHORT, true)).toContain('usually needs a few hundred steps before it stops improving');
  });
});
