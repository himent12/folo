import { describe, expect, it } from 'vitest';
import type { LogLine } from '$lib/api/types';
import { scheduleWarnings, sentences } from './warnings';

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
  it('reads the epochs the trainer suggests, or null', async () => {
    const { suggestedEpochs } = await import('./warnings');
    expect(suggestedEpochs(['this run takes only 125 optimizer steps (5 epochs × 25: …); … Train for at least 8 epochs, add images, or start from pretrained weights'])).toBe(8);
    expect(suggestedEpochs(["warmup takes 70 of the run's 125 optimizer steps, …; lower warmup_epochs (now 3) or train for more epochs"])).toBeNull();
  });
});
