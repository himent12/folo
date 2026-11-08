import { describe, expect, it } from 'vitest';
import { runHorizon } from './horizon';

describe('runHorizon', () => {
  it('ends finished runs at their last epoch', () => {
    expect(runHorizon(23.4, 100, false, 'observed')).toEqual([0, 24]);
  });
  it('gives live runs stable headroom without passing the plan', () => {
    expect(runHorizon(38, 100, true, 'observed')).toEqual([0, 45]);
    expect(runHorizon(39, 100, true, 'observed')).toEqual([0, 45]);
    expect(runHorizon(97, 100, true, 'observed')).toEqual([0, 100]);
  });
  it('shows the full plan on request or before any data', () => {
    expect(runHorizon(38, 100, true, 'full')).toEqual([0, 100]);
    expect(runHorizon(0, 100, true, 'observed')).toEqual([0, 100]);
  });
});
