import { describe, expect, it } from 'vitest';
import { runHorizon } from './horizon';

describe('runHorizon', () => {
  it('ends finished runs at their last epoch', () => {
    expect(runHorizon(23.4, 100, 'observed')).toEqual([0, 24]);
    expect(runHorizon(150, 150, 'observed')).toEqual([0, 150]);
  });
  it('ends a live run within one epoch of its data, never past the plan', () => {
    // Validated through epoch 37, steps into epoch 38: the axis ends at 38, not 45.
    expect(runHorizon(37.6, 100, 'observed')).toEqual([0, 38]);
    expect(runHorizon(38, 100, 'observed')).toEqual([0, 38]);
    expect(runHorizon(99.5, 100, 'observed')).toEqual([0, 100]);
    expect(runHorizon(100.2, 100, 'observed')).toEqual([0, 100]);
  });
  it('handles a run stopped before its first epoch finished', () => {
    expect(runHorizon(0.5, 100, 'observed')).toEqual([0, 1]);
  });
  it('shows the full plan on request or before any data', () => {
    expect(runHorizon(38, 100, 'full')).toEqual([0, 100]);
    expect(runHorizon(0, 100, 'observed')).toEqual([0, 100]);
  });
});
