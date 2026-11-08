import { describe, expect, it } from 'vitest';
import { runProgress, stepsPerEpoch } from './progress';
import type { Step } from './types';

const step = (step: number, epoch: number) => ({ step, epoch }) as Step;

describe('runProgress', () => {
  it('divides steps by the planned steps', () => {
    expect(runProgress({ step: 6, epoch: 1, epochs: 3, steps_per_epoch: 12, status: 'running' })).toBeCloseTo(6 / 36);
  });
  it('does not read 100% while the server has not counted steps per epoch yet', () => {
    expect(runProgress({ step: 5, epoch: 1, epochs: 3, steps_per_epoch: 0, status: 'running' })).toBe(0);
    expect(runProgress({ step: 30, epoch: 3, epochs: 3, steps_per_epoch: 0, status: 'running' })).toBeCloseTo(2 / 3);
    expect(runProgress({ step: 0, epoch: 3, epochs: 3, steps_per_epoch: 0, status: 'completed' })).toBe(1);
  });
  it('clamps to 0–1', () => {
    expect(runProgress({ step: 40, epoch: 3, epochs: 3, steps_per_epoch: 12, status: 'running' })).toBe(1);
  });
});

describe('stepsPerEpoch', () => {
  it("prefers the server's figure", () => {
    expect(stepsPerEpoch({ steps_per_epoch: 12 }, [step(1, 1), step(2, 2)])).toBe(12);
  });
  it('infers it from the first finished epoch while the server says 0', () => {
    const steps = [1, 2, 3, 4, 5].map((n) => step(n, n <= 3 ? 1 : 2));
    expect(stepsPerEpoch({ steps_per_epoch: 0 }, steps)).toBe(3);
  });
  it('is 0 while nothing tells', () => {
    expect(stepsPerEpoch({ steps_per_epoch: 0 }, [step(1, 1), step(2, 1)])).toBe(0);
    expect(stepsPerEpoch({ steps_per_epoch: 0 })).toBe(0);
  });
});
