import { describe, expect, it } from 'vitest';
import { failedTitle, failureSummary } from './failure';

describe('a failed run in plain words', () => {
  it('names the epoch it failed in, from its steps when the server gave the last finished one', () => {
    expect(failedTitle({ epoch: 1, step: 25, steps_per_epoch: 13 })).toBe('Failed in epoch 2');
    expect(failedTitle({ epoch: 2, step: 25, steps_per_epoch: 13 })).toBe('Failed in epoch 2');
    expect(failedTitle({ epoch: 0, step: 0, steps_per_epoch: 0 })).toBe('Failed before training started');
  });

  it('says what went wrong by its code, and leaves an unknown one to the message', () => {
    expect(failureSummary('data_error', 'bccd-copy')).toBe('Files of dataset bccd-copy went missing or changed during training.');
    expect(failureSummary('training_failed', 'x')).toBe('Training diverged or produced no usable batch.');
    expect(failureSummary(undefined, 'x')).toBeNull();
    expect(failureSummary('some_new_code', 'x')).toBeNull();
  });
});
