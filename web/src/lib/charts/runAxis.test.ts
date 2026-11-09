import { describe, expect, it } from 'vitest';
import { defaultAxis, lengthRatio, runLength, runX } from './runAxis';

// The demo comparison: a live run at 38 of 100 epochs, one completed at 150, one stopped at 61, one failed at 23.4.
const live = { status: 'running', epoch: 38, epochs: 100 };
const done = { status: 'completed', epoch: 150, epochs: 150 };
const stopped = { status: 'stopped', epoch: 61, epochs: 100 };
const failed = { status: 'failed', epoch: 23.4, epochs: 100 };

describe('run comparison axis', () => {
  it('measures a live run by its plan and an ended one by the epochs it ran', () => {
    expect(runLength(live)).toBe(100);
    expect(runLength(done)).toBe(150);
    expect(runLength(stopped)).toBe(61);
    expect(runLength(failed)).toBe(23);
    expect(runLength({ status: 'failed', epoch: 0, epochs: 50 })).toBe(50);
  });

  it('defaults to "% of run" only when the lengths differ by more than 3x', () => {
    expect(lengthRatio([live, done, stopped, failed])).toBeCloseTo(150 / 23);
    expect(defaultAxis([live, done, stopped, failed])).toBe('progress');
    expect(defaultAxis([live, done, stopped])).toBe('epoch');
    expect(defaultAxis([done, { status: 'completed', epoch: 50, epochs: 50 }])).toBe('epoch');
    expect(defaultAxis([done, { status: 'completed', epoch: 49, epochs: 49 }])).toBe('progress');
    expect(defaultAxis([failed])).toBe('epoch');
  });

  it('stretches each ended run over the whole axis on "% of run"', () => {
    expect(runX(23, failed, 'progress')).toBe(100);
    expect(runX(38, live, 'progress')).toBe(38);
    expect(runX(23, failed, 'epoch')).toBe(23);
  });
});
