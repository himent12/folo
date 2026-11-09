import { describe, expect, it } from 'vitest';
import { defaultAxis, lengthRatio, runEnd, runLength, runX } from './runAxis';

// The demo comparison: a live run at 38 of 100 epochs, one completed at 150, one stopped at 61, one failed at 23.4.
const live = { status: 'running', epoch: 38, epochs: 100 };
const done = { status: 'completed', epoch: 150, epochs: 150 };
const stopped = { status: 'stopped', epoch: 61, epochs: 100 };
const failed = { status: 'failed', epoch: 23.4, epochs: 100 };

describe('run comparison axis', () => {
  it('measures every run by its plan, whatever its status', () => {
    expect(runLength(live)).toBe(100);
    expect(runLength(done)).toBe(150);
    expect(runLength(stopped)).toBe(100);
    expect(runLength(failed)).toBe(100);
    expect(runLength({ status: 'failed', epoch: 0, epochs: 50 })).toBe(50);
    expect(runLength({ status: 'completed', epoch: 12, epochs: 0 })).toBe(12);
  });

  it('defaults to "By progress" only when the plans differ by more than 3x', () => {
    expect(lengthRatio([live, done, stopped, failed])).toBeCloseTo(1.5);
    expect(defaultAxis([live, done, stopped, failed])).toBe('epoch');
    expect(defaultAxis([done, { status: 'completed', epoch: 50, epochs: 50 }])).toBe('epoch');
    expect(defaultAxis([done, { status: 'completed', epoch: 49, epochs: 49 }])).toBe('progress');
    // An early failure is not a short plan: it does not switch the axis.
    expect(defaultAxis([done, { status: 'failed', epoch: 3, epochs: 150 }])).toBe('epoch');
    expect(defaultAxis([failed])).toBe('epoch');
  });

  it('puts every run on % of its planned epochs, so an early failure ends early', () => {
    expect(runX(23, failed, 'progress')).toBe(23);
    expect(runX(61, stopped, 'progress')).toBe(61);
    expect(runX(38, live, 'progress')).toBe(38);
    expect(runX(150, done, 'progress')).toBe(100);
    expect(runX(23, failed, 'epoch')).toBe(23);
  });

  it('marks how each line ends', () => {
    expect(runEnd('running')).toBe('live');
    expect(runEnd('stopping')).toBe('live');
    expect(runEnd('stopped')).toBe('stopped');
    expect(runEnd('failed')).toBe('failed');
    expect(runEnd('completed')).toBeNull();
    expect(runEnd('queued')).toBeNull();
  });
});
