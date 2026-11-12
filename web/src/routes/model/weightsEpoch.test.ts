import { describe, expect, it } from 'vitest';
import type { Run } from '$lib/api/types';
import { weightsEpoch } from './weightsEpoch';

const run = (status: Run['status'], best: number | null, epochs = 5) => ({ id: 'run_a', status, epochs, best: best === null ? null : { epoch: best, map50_95: 0.4 } });

describe('weightsEpoch', () => {
  it('names the epoch the weights are from, and the same weights when best is last', () => {
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, run('completed', 1))).toBe('best epoch 1 of 5');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('completed', 1))).toBe('last epoch, 5 of 5');
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, run('completed', 5))).toBe('best epoch 5 of 5, same weights as last');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('completed', 5))).toBe('last epoch, 5 of 5, same weights as best');
    expect(weightsEpoch({ id: 'm_a-last', run_id: 'run_a' }, run('stopped', 2))).toBe('last finished epoch');
  });

  it('says nothing it does not know', () => {
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, null)).toBeNull();
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_b' }, run('completed', 1))).toBeNull();
    expect(weightsEpoch({ id: 'm_a', run_id: 'run_a' }, run('completed', null))).toBeNull();
  });
});
